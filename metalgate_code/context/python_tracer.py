"""Python-specific tracer using the ty language server and tree-sitter.

LSP communication is handled by
:class:`~metalgate_code.context.ty_lsp_client.TyLspClient`.

``find_symbol`` uses LSP ``workspace/symbol`` — project files only, not
site-packages.  For third-party symbols, use ``goto_definition`` from a
usage site.

Tree-sitter is used for:
  - ``get_source`` — line-based source extraction from scope nodes
  - ``get_file_outline`` — fast outline extraction (no LSP round-trip)
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language

from metalgate_code.context.cache import _CACHE_MISS, CodeCache
from metalgate_code.context.tracer_base import (
    _MAX_CALLERS,
    Tracer,
    TreeSitterConfig,
    _lsp_symbol_kind_to_str,
    _name_col_on_line,
    _parse_hover_base,
    _path_to_uri,
    _uri_to_path,
)
from metalgate_code.context.tracer_base import (
    _ts_call_positions as _ts_call_positions_impl,
)
from metalgate_code.context.tracer_base import (
    _ts_find_function_and_calls as _ts_find_function_and_calls_impl,
)
from metalgate_code.context.tracer_base import (
    _ts_find_function_containing as _ts_find_function_containing_impl,
)
from metalgate_code.context.tracer_base import (
    _ts_find_scope_at_line as _ts_find_scope_at_line_impl,
)
from metalgate_code.context.ty_lsp_client import (
    LocalTyLspClient,
    SandboxTyLspClient,
    TyLspClient,
)
from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

logger = logging.getLogger("metalgate_code")

_STDLIB_MARKERS = ("typeshed", "/stdlib/", "/builtins.pyi")

# Tree-sitter Python language — shared across all parses.
# A fresh Parser is created per call (Parser is not thread-safe).
_TS_LANGUAGE = Language(tspython.language())


# Re-exports for backward compatibility (tests import these from here).
_LSP_SYMBOL_KINDS = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum_member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type_parameter",
}


# Python-specific tree-sitter helpers


def _ts_find_identifier_in_scope(
    source_bytes: bytes, line: int, name: str
) -> Optional[tuple[int, int]]:
    """Find the closest ``identifier`` node matching *name* within the scope
    containing *line* (1-based).

    Returns (line_1based, col_0based) or None.  Only actual identifier nodes
    are matched — not strings, comments, or keywords.  The closest match to
    *line* wins (used as a fallback when the name isn't on the given line).
    """
    # Need a Tracer instance for _ts_find_scope_node, but this is a module-level
    # function.  We'll parse directly here since this is Python-specific.
    from tree_sitter import Parser

    tree = Parser(_TS_LANGUAGE).parse(source_bytes)
    root = tree.root_node

    # Find the scope containing *line* (containing=True).
    best_scope = None
    best_size = None
    scope_kinds = ("function_definition", "class_definition")

    def find_scope(node):
        nonlocal best_scope, best_size
        if node.type in scope_kinds:
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            if start <= line <= end:
                size = end - start
                if best_scope is None or size < best_size:
                    best_scope = node
                    best_size = size
        for child in node.children:
            find_scope(child)

    find_scope(root)
    if best_scope is None:
        return None

    # Walk the scope's subtree to find the closest matching identifier.
    best: Optional[tuple[int, int]] = None
    best_dist = float("inf")

    def find_ident(node):
        nonlocal best, best_dist
        if node.type == "identifier" and node.text == name.encode("utf-8"):
            n_line = node.start_point[0] + 1
            n_col = node.start_point[1]
            dist = abs(n_line - line)
            if dist < best_dist:
                best = (n_line, n_col)
                best_dist = dist
        for child in node.children:
            find_ident(child)

    find_ident(best_scope)
    return best


def _ts_is_stub_function(source_bytes: bytes, line: int) -> bool:
    """True if the function starting on *line* (1-based) is a stub.

    A stub body contains only a docstring and one of:
    - ``raise NotImplementedError``
    - ``pass``
    - ``...``

    Uses tree-sitter AST inspection, not string matching.
    """
    from tree_sitter import Parser

    tree = Parser(_TS_LANGUAGE).parse(source_bytes)
    root = tree.root_node

    # Find the function_definition starting on *line*.
    fn = None
    for node in root.children:
        if node.type == "function_definition" and node.start_point[0] + 1 == line:
            fn = node
            break
    if fn is None:
        # Search recursively.
        def find_fn(node):
            if node.type == "function_definition" and node.start_point[0] + 1 == line:
                return node
            for child in node.children:
                result = find_fn(child)
                if result is not None:
                    return result
            return None

        fn = find_fn(root)
    if fn is None:
        return False

    body = fn.child_by_field_name("body")
    if body is None:
        return False

    # Inspect each top-level statement in the body.
    for child in body.children:
        if child.type == "expression_statement":
            # Docstring (string) is allowed in stubs — skip it.
            if any(s.type == "string" for s in child.children):
                continue
            # Ellipsis (``...``) is a stub body.
            if any(s.type == "ellipsis" for s in child.children):
                return True
            # Any other expression is concrete.
            return False
        elif child.type == "pass_statement":
            return True
        elif child.type == "raise_statement":
            # ``raise NotImplementedError`` is a stub; any other raise is concrete.
            text = child.text.decode("utf-8", errors="replace") if child.text else ""
            return "NotImplementedError" in text
        else:
            # return, assignment, etc. → concrete.
            return False

    return False


def _is_stdlib_path(path: str) -> bool:
    """True if *path* points to a stdlib/typeshed definition.

    These are noise in callee results — ``isinstance``, ``getattr``, ``len``,
    ``super``, ``logger.info``, etc. all resolve here.
    """
    return any(marker in path for marker in _STDLIB_MARKERS)


def _parse_hover(hover: object) -> tuple[str, str]:
    """Extract (signature, docstring) from an LSP hover response.

    Delegates to :func:`_parse_hover_base` — Python (ty) does not require
    any language-specific post-processing.
    """
    return _parse_hover_base(hover)


# Re-exports for backward compatibility
#
# Tests import these module-level functions from python_tracer.  They are
# now implemented as methods on Tracer, but we provide thin wrappers that
# delegate to the shared implementation using the Python TreeSitterConfig.

_PY_CONFIG = TreeSitterConfig(
    language=_TS_LANGUAGE,
    function_kinds=("function_definition",),
    scope_kinds=("function_definition", "class_definition"),
    call_node_type="call",
    member_node_type="attribute",
    member_field_name="attribute",
)


# Backward-compatible module-level wrappers for tests that import these
# as module-level functions.  They delegate to the shared implementations
# in tracer_base, passing the Python TreeSitterConfig.


def _ts_find_scope_at_line(source_bytes: bytes, line: int) -> Optional[tuple[int, int]]:
    """Backward-compatible wrapper — delegates to tracer_base._ts_find_scope_at_line."""
    return _ts_find_scope_at_line_impl(_PY_CONFIG, source_bytes, line)


def _ts_find_function_containing(
    source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str]]]:
    """Backward-compatible wrapper — delegates to tracer_base._ts_find_function_containing."""
    return _ts_find_function_containing_impl(_PY_CONFIG, source_bytes, line)


def _ts_call_positions(
    source_bytes: bytes, start_line: int, end_line: int
) -> list[tuple[int, int]]:
    """Backward-compatible wrapper — delegates to tracer_base._ts_call_positions."""
    return _ts_call_positions_impl(_PY_CONFIG, source_bytes, start_line, end_line)


def _ts_find_function_and_calls(
    source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str], list[tuple[int, int]]]]:
    """Backward-compatible wrapper — delegates to tracer_base._ts_find_function_and_calls."""
    return _ts_find_function_and_calls_impl(_PY_CONFIG, source_bytes, line)


# PythonTracer


class PythonTracer(Tracer):
    """Python-specific tracer using ty language server and tree-sitter."""

    _ts_config = _PY_CONFIG

    _def_keywords = ("async def ", "def ", "class ")

    def __init__(
        self,
        root: str,
        backend,
        cache: CodeCache,
    ) -> None:
        super().__init__(root, backend, cache)
        self._lsp: Optional[TyLspClient] = None
        self._lsp_lock = threading.Lock()  # guards _lsp creation
        # Serializes all LSP requests.  ty is single-threaded; concurrent
        # requests cause "content modified" errors.  RLock (not Lock) so
        # _resolve can call find_symbol while already holding the lock.
        self._lsp_request_lock = threading.RLock()
        self._ms: Optional[MicrosandboxBackend] = None

    @property
    def ms(self) -> MicrosandboxBackend:
        """The MicrosandboxBackend (validated once, then cached).

        Raises RuntimeError if the backend is not a MicrosandboxBackend.
        Callers that don't require the sandbox should check
        :meth:`_is_sandbox` first.
        """
        if self._ms is not None:
            return self._ms
        if not isinstance(self.backend, MicrosandboxBackend):
            raise RuntimeError("PythonTracer requires a MicrosandboxBackend")
        self._ms = self.backend
        return self._ms

    # Tracer interface

    def get_file_outline(self, file: str) -> list[dict]:
        """Parse *file* and return every class/function/method with
        name, kind, line, end_line, signature."""
        cached = self.cache.get_outline(file)
        if cached is not None:
            return cached

        try:
            source = self._read_file(file)
        except OSError:
            logger.warning("Failed to read %s for outline", file, exc_info=True)
            return []

        result = self._ts_outline(source, file)
        self.cache.set_outline(file, result)
        return result

    def _ts_outline(self, source: str, file: str) -> list[dict]:
        """Extract outline using tree-sitter (no LSP round-trip needed).

        Walks the AST collecting function_definition and class_definition
        nodes.  For classes, recurses into the body block to find methods.
        """
        source_bytes = source.encode("utf-8", errors="replace")
        tree = self._ts_parse(source_bytes)
        root = tree.root_node
        result: list[dict] = []

        def walk(node, parent_class: Optional[str] = None):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    return

                name = name_node.text.decode("utf-8", errors="replace")
                params_node = node.child_by_field_name("parameters")
                param_str = (
                    params_node.text.decode("utf-8", errors="replace")
                    if params_node
                    else "..."
                )
                is_async = any(child.type == "async" for child in node.children)
                prefix = "async def " if is_async else "def "

                result.append(
                    {
                        "name": name,
                        "kind": "method" if parent_class else "function",
                        "class": parent_class,
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "signature": f"{prefix}{name}{param_str}",
                    }
                )

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    return

                name = name_node.text.decode("utf-8", errors="replace")

                # Extract base classes from the argument_list, if any.
                bases = ""
                for child in node.children:
                    if child.type == "argument_list":
                        bases = child.text.decode("utf-8", errors="replace").strip("()")

                result.append(
                    {
                        "name": name,
                        "kind": "class",
                        "class": None,
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "signature": (
                            f"class {name}({bases})" if bases else f"class {name}"
                        ),
                    }
                )
                # Recurse into the class body block to find methods.
                for child in node.children:
                    if child.type == "block":
                        for sub in child.children:
                            walk(sub, parent_class=name)
            else:
                for child in node.children:
                    walk(child, parent_class)

        walk(root)
        for sym in result:
            sym["file"] = file
        return result

    def goto_definition(
        self, file: str, line: int, name: Optional[str] = None
    ) -> Optional[dict]:
        """Resolve the symbol *name* on *line* of *file* to its definition.

        If *name* is None, the first identifier on *line* is used.
        Results are cached.
        """
        if name is None:
            name = self._first_name_on_line(file, line)
            if name is None:
                return None

        cached = self.cache.get_definition(file, line, name)
        if cached is not _CACHE_MISS and cached is not None:
            return cached

        result = self._resolve(file, line, name)
        if result is not None:
            self.cache.set_definition(file, line, name, result)
        return result

    def get_callers(self, file: str, line: int) -> list[dict]:
        """Find every place in the project that references the symbol on *line* of *file*.

        Uses LSP textDocument/references.  For each reference, determines the
        enclosing function/method name via outline lookup.
        """
        # Read the file once and derive col + name from the same source.
        try:
            source = self._read_file(file)
        except OSError:
            return []

        lines = source.splitlines()
        if line < 1 or line > len(lines):
            return []

        col = self._def_name_col_from_lines(lines, line)
        if col is None:
            return []
        sym_name = self._def_name_from_lines(lines, line)

        lsp = self._get_lsp()
        guest_file = self._to_guest_path(file)
        uri = _path_to_uri(guest_file)

        with self._lsp_request_lock:
            self._did_open(lsp, uri, source)

            try:
                refs = lsp.references(uri, line - 1, col, include_declaration=False)
            except Exception:
                logger.warning(
                    "LSP references failed for %s:%d", file, line, exc_info=True
                )
                return [
                    {
                        "file": file,
                        "line": line,
                        "name": sym_name or "",
                        "caller": "",
                        "context": "",
                        "note": (
                            "LSP references request failed. The symbol may still "
                            "be referenced — try again."
                        ),
                    }
                ]

        results: list[dict] = []
        seen: set[tuple[str, int]] = set()

        for r in refs:
            ref_uri = r.get("uri", "")
            if not ref_uri:
                continue
            ref_file = self._to_host_path(_uri_to_path(ref_uri))
            ref_range = r.get("range", {})
            ref_line = ref_range.get("start", {}).get("line", 0) + 1

            # Skip the definition itself.
            if ref_file == file and ref_line == line:
                continue

            key = (ref_file, ref_line)
            if key in seen:
                continue
            seen.add(key)

            # Find the enclosing function/method name for this reference.
            caller_name = ""
            try:
                ref_outline = self.get_file_outline(ref_file)
                best = self._find_symbol_at_line(ref_outline, ref_line)
                if best:
                    caller_name = best["name"]
            except Exception:
                logger.warning(
                    "get_file_outline failed for %s", ref_file, exc_info=True
                )

            # Read the referencing line for context.
            context_text = ""
            try:
                ref_source = self._read_file(ref_file, limit=max(ref_line, 1) + 1)
                ref_lines = ref_source.splitlines()
                if 0 < ref_line <= len(ref_lines):
                    context_text = ref_lines[ref_line - 1].strip()
            except OSError:
                pass

            results.append(
                {
                    "file": ref_file,
                    "line": ref_line,
                    "name": sym_name or "",
                    "caller": caller_name,
                    "context": context_text,
                }
            )
            if len(results) >= _MAX_CALLERS:
                break

        if not results:
            return [
                {
                    "file": file,
                    "line": line,
                    "name": sym_name or "",
                    "caller": "",
                    "context": "",
                    "note": (
                        "No static callers found. This symbol may be called "
                        "via dynamic dispatch, framework callbacks, or from "
                        "site-packages not indexed by the language server."
                    ),
                }
            ]

        return results

    def get_callees(self, file: str, line: int) -> list[dict]:
        """Find every symbol called by the function on *line* of *file*.

        Uses tree-sitter to find call positions within the function body,
        then resolves each to its definition via LSP textDocument/definition.
        Stdlib/builtins are filtered out.  Results are deduplicated by name,
        preferring concrete implementations over abstract stubs.
        """
        try:
            source = self._read_file(file)
        except OSError:
            return []

        # Single tree walk: find the function and collect all call positions.
        func_info = self._ts_find_function_and_calls(
            source.encode("utf-8", errors="replace"), line
        )
        if func_info is None:
            return []

        start_line, end_line, func_name, positions = func_info

        lsp = self._get_lsp()
        guest_file = self._to_guest_path(file)
        uri = _path_to_uri(guest_file)
        results: list[dict] = []
        seen: set[tuple] = set()

        with self._lsp_request_lock:
            self._did_open(lsp, uri, source)

            for call_line, call_col in positions:
                try:
                    defs = lsp.definition(uri, call_line - 1, call_col)
                except Exception:
                    logger.warning(
                        "LSP definition failed at %s:%d:%d",
                        file,
                        call_line,
                        call_col,
                        exc_info=True,
                    )
                    continue

                if not defs:
                    continue

                if isinstance(defs, dict):
                    defs = [defs]

                for d in defs:
                    info = self._extract_def_info(d)
                    if info is None:
                        continue
                    d_file, d_line, d_col, d_uri = info

                    # Skip stdlib/builtins — noise without codebase value.
                    if _is_stdlib_path(d_file):
                        continue

                    key = (d_file, d_line)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Look up name/kind/signature from the definition's outline.
                    d_name = ""
                    d_kind = ""
                    d_sig = ""
                    try:
                        outline = self.get_file_outline(d_file)
                        sym = self._find_symbol_at_line(outline, d_line, exact=True)
                        if sym:
                            d_name = sym["name"]
                            d_kind = sym["kind"]
                            d_sig = sym.get("signature", "")
                    except Exception:
                        logger.warning(
                            "get_file_outline failed for %s", d_file, exc_info=True
                        )

                    results.append(
                        {
                            "name": d_name,
                            "kind": d_kind,
                            "file": d_file,
                            "line": d_line,
                            "signature": d_sig,
                        }
                    )

        # Deduplicate by name, preferring concrete over stub implementations.
        return self._dedup_callees(results)

    def _dedup_callees(self, results: list[dict]) -> list[dict]:
        """Deduplicate callees by name, preferring concrete over stub implementations.

        When the same method appears via both its abstract declaration (a stub
        that raises ``NotImplementedError`` or uses ``pass``/``...``) and its
        concrete implementation, keep the concrete one.  Callees with empty
        names are kept as-is (no dedup possible).
        """
        by_name: dict[str, dict] = {}
        unnamed: list[dict] = []
        for r in results:
            rname = r.get("name", "")
            if not rname:
                unnamed.append(r)
                continue
            existing = by_name.get(rname)
            if existing is None:
                by_name[rname] = r
            elif self._callee_is_stub(existing) and not self._callee_is_stub(r):
                by_name[rname] = r
        return unnamed + list(by_name.values())

    def _callee_is_stub(self, callee: dict) -> bool:
        """True if *callee* points to a stub/abstract method (pass/.../raise
        NotImplementedError).  Uses tree-sitter to inspect the function body.
        """
        file = callee.get("file", "")
        line = callee.get("line", 0)
        if not file or not line:
            return False
        try:
            source = self._read_file(file)
        except OSError:
            return False
        return _ts_is_stub_function(source.encode("utf-8", errors="replace"), line)

    def find_symbol(self, name: str) -> list[dict]:
        """Search for *name* across the project via LSP ``workspace/symbol``.

        ty's ``workspace/symbol`` indexes first-party project files only —
        it does not search site-packages.  For third-party symbols, use
        ``goto_definition`` from a usage site.
        """
        results: list[dict] = []
        seen: set[tuple] = set()

        try:
            lsp = self._get_lsp()
            with self._lsp_request_lock:
                symbols = lsp.workspace_symbol(name)
            for sym in symbols:
                location = sym.get("location", {})
                sym_uri = location.get("uri", "")
                if not sym_uri:
                    continue
                sym_file = self._to_host_path(_uri_to_path(sym_uri))
                sym_range = location.get("range", {})
                sym_line = sym_range.get("start", {}).get("line", 0) + 1

                key = (sym_file, sym_line)
                if key in seen:
                    continue
                seen.add(key)

                kind = _lsp_symbol_kind_to_str(sym.get("kind", 0))

                results.append(
                    {
                        "name": sym.get("name", name),
                        "kind": kind,
                        "file": sym_file,
                        "line": sym_line,
                    }
                )
        except Exception:
            logger.warning("workspace/symbol failed for %r", name, exc_info=True)

        if not results:
            return [
                {
                    "name": name,
                    "kind": "",
                    "file": "",
                    "line": 0,
                    "note": (
                        "No project symbols found. This symbol may exist only "
                        "in installed packages — use goto_definition from a "
                        "usage site to resolve it."
                    ),
                }
            ]

        return results

    # Private helpers

    def _get_lsp(self) -> TyLspClient | LocalTyLspClient:
        """Get or lazily create the ty LSP client (double-checked locking)."""
        if self._lsp is not None:
            return self._lsp

        with self._lsp_lock:
            if self._lsp is not None:
                return self._lsp

            if self._is_sandbox:
                sb = self.ms._ensure_sandbox_sync()
                guest_root = self.ms._to_guest_path(str(self.root))
                root_uri = _path_to_uri(guest_root)

                venv_bin = self.ms.venv_bin
                venv_env = self.ms.venv_env
                python_path = f"{venv_bin}/python" if venv_bin else None

                self._lsp = SandboxTyLspClient(
                    sb,
                    root_uri,
                    python_path=python_path,
                    venv_bin=venv_bin,
                    venv_env=venv_env,
                )
            else:
                root_uri = _path_to_uri(str(self.root))
                self._lsp = LocalTyLspClient(root_uri, python_path=sys.executable)

            self._lsp.start()
            return self._lsp

    def _resolve(self, file: str, line: int, name: str) -> Optional[dict]:
        """Resolve *name* at *line* in *file* to its definition via LSP.

        1. Find the column of *name* on *line* (or in the enclosing scope).
        2. Call LSP textDocument/definition at that position.
        3. If LSP can't resolve, fall back to workspace/symbol search by name.
        4. Get hover info for signature/docstring.
        5. Determine kind from the definition file's outline.
        """
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None

            lsp = self._get_lsp()
            guest_file = self._to_guest_path(file)
            uri = _path_to_uri(guest_file)

            # Find the column of *name* — first on the given line, then in
            # the enclosing scope via tree-sitter.
            col = _name_col_on_line(lines[line - 1], name)
            if col is None:
                found = _ts_find_identifier_in_scope(
                    source.encode("utf-8", errors="replace"), line, name
                )
                if found is not None:
                    line, col = found

            with self._lsp_request_lock:
                self._did_open(lsp, uri, source)

                if col is not None:
                    d_file, d_line, d_col, d_uri = self._lsp_definition(
                        lsp, uri, line - 1, col
                    )
                else:
                    d_file, d_line, d_col, d_uri = None, 0, 0, None

                # Fall back to workspace/symbol if LSP couldn't resolve.
                if d_uri is None:
                    if name:
                        sym_results = self.find_symbol(name)
                        if sym_results and sym_results[0].get("file"):
                            sr = sym_results[0]
                            d_file = sr["file"]
                            d_line = sr["line"]
                            d_col = 0
                            d_uri = _path_to_uri(self._to_guest_path(d_file))
                    if d_uri is None:
                        return None

                # Get hover info for signature/docstring.
                signature = ""
                docstring = ""
                if col is not None:
                    try:
                        hover = lsp.hover(uri, line - 1, col)
                        signature, docstring = _parse_hover(hover)
                    except Exception:
                        logger.warning(
                            "hover failed for %s:%d", file, line, exc_info=True
                        )

            # Determine kind from the definition file's outline.
            kind = "unknown"
            if d_file:
                try:
                    outline = self.get_file_outline(d_file)
                    sym = self._find_symbol_at_line(outline, d_line, exact=True)
                    if sym:
                        kind = sym["kind"]
                        if not signature:
                            signature = sym.get("signature", "")
                except Exception:
                    logger.warning(
                        "get_file_outline failed for %s", d_file, exc_info=True
                    )

            return {
                "name": name,
                "kind": kind,
                "file": d_file,
                "line": d_line,
                "col": d_col,
                "signature": signature,
                "docstring": docstring,
            }
        except Exception:
            logger.warning(
                "goto_definition failed for %s:%d", file, line, exc_info=True
            )
            return None

    def _lsp_definition(
        self,
        lsp: TyLspClient,
        uri: str,
        line_0: int,
        col: int,
    ) -> tuple[Optional[str], int, int, Optional[str]]:
        """Call LSP textDocument/definition and return
        (file, line_1based, col, uri) for the first result.
        """
        defs = lsp.definition(uri, line_0, col)
        if not defs:
            return None, 0, 0, None
        if isinstance(defs, dict):
            defs = [defs]
        info = self._extract_def_info(defs[0])
        if info is None:
            return None, 0, 0, None
        return info
