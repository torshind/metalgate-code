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
import re
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from metalgate_code.context.cache import _CACHE_MISS, CodeCache
from metalgate_code.context.tracer_base import _MAX_CALLERS, Tracer
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

# LSP SymbolKind enum values → human-readable names.
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


# Tree-sitter helpers
#
# All tree-sitter functions take raw bytes and return 1-based line numbers
# (matching LSP convention).  Column numbers are 0-based.


def _ts_parse(source_bytes: bytes):
    """Parse *source_bytes* with a fresh Parser (thread-safe)."""
    return Parser(_TS_LANGUAGE).parse(source_bytes)


def _ts_find_scope_node(
    source_bytes: bytes,
    line: int,
    *,
    kinds: tuple[str, ...] = ("function_definition", "class_definition"),
    containing: bool = False,
):
    """Find the tightest (smallest) AST node of the given *kinds* matching *line*.

    *containing*=False → match nodes whose def starts on *line* (start == line).
    *containing*=True  → match nodes whose body contains *line* (start <= line <= end).

    Returns the tree-sitter node, or None.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node
    best = None
    best_size = None

    def visit(node):
        nonlocal best, best_size
        if node.type in kinds:
            start = node.start_point[0] + 1  # 1-based
            end = node.end_point[0] + 1
            if (containing and start <= line <= end) or (
                not containing and start == line
            ):
                size = end - start
                if best is None or size < best_size:
                    best = node
                    best_size = size
        for child in node.children:
            visit(child)

    visit(root)
    return best


def _ts_find_scope_at_line(source_bytes: bytes, line: int) -> Optional[tuple[int, int]]:
    """Return (start_0based, end_1based_exclusive) of the scope starting on *line*.

    The tuple is suitable for slicing ``source.splitlines()``:
    ``lines[start:end]`` gives the full scope body.
    """
    node = _ts_find_scope_node(source_bytes, line)
    if node is None:
        return None
    return (node.start_point[0], node.end_point[0] + 1)


def _ts_find_function_containing(
    source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str]]]:
    """Return (start_1based, end_1based, name) of the innermost function
    whose body contains *line*, or None.
    """
    node = _ts_find_scope_node(
        source_bytes, line, kinds=("function_definition",), containing=True
    )
    if node is None:
        return None
    name_node = node.child_by_field_name("name")
    name = name_node.text.decode("utf-8", errors="replace") if name_node else None
    return (node.start_point[0] + 1, node.end_point[0] + 1, name)


def _ts_call_positions(
    source_bytes: bytes, start_line: int, end_line: int
) -> list[tuple[int, int]]:
    """Return (line_1based, col_0based) of every function call in [start_line, end_line].

    For each ``call`` node, the position targets the callable name:
      - ``foo()``   → position of ``foo``
      - ``obj.m()`` → position of ``m`` (the attribute, not the object)
      - other       → position of the function expression

    Decorators and class definitions are naturally excluded — they are not
    ``call`` nodes within a function body.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node
    positions: list[tuple[int, int]] = []

    def visit(node):
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                # For obj.m(), target the attribute name "m", not "obj".
                if func_node.type == "attribute":
                    attr_node = func_node.child_by_field_name("attribute")
                    pos_node = attr_node if attr_node is not None else func_node
                else:
                    pos_node = func_node
                line_1 = pos_node.start_point[0] + 1
                col_0 = pos_node.start_point[1]
                if start_line <= line_1 <= end_line:
                    positions.append((line_1, col_0))
        for child in node.children:
            visit(child)

    visit(root)
    return positions


def _ts_find_function_and_calls(
    source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str], list[tuple[int, int]]]]:
    """Find the innermost function containing *line* and all call positions within it.

    Parses once, then does two passes over the tree:
      1. Find the innermost function_definition containing *line*.
      2. Collect all call positions within that function's line range.

    Returns (start_line, end_line, func_name, positions) or None.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node

    # Pass 1: find innermost function containing *line*.
    best = None
    best_size = None
    best_name = None

    def find_fn(node):
        nonlocal best, best_size, best_name
        if node.type == "function_definition":
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            if start <= line <= end:
                size = end - start
                if best is None or size < best_size:
                    best = (start, end)
                    best_size = size
                    name_node = node.child_by_field_name("name")
                    best_name = (
                        name_node.text.decode("utf-8", errors="replace")
                        if name_node
                        else None
                    )
        for child in node.children:
            find_fn(child)

    find_fn(root)
    if best is None:
        return None

    start_line, end_line = best
    positions: list[tuple[int, int]] = []

    # Pass 2: collect call positions within the function's line range.
    def collect_calls(node):
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                if func_node.type == "attribute":
                    attr_node = func_node.child_by_field_name("attribute")
                    pos_node = attr_node if attr_node is not None else func_node
                else:
                    pos_node = func_node
                cl = pos_node.start_point[0] + 1
                col = pos_node.start_point[1]
                if start_line <= cl <= end_line:
                    positions.append((cl, col))
        for child in node.children:
            collect_calls(child)

    collect_calls(root)
    return (start_line, end_line, best_name, positions)


def _name_col_on_line(line_text: str, name: str, occurrence: int = 0) -> Optional[int]:
    """Column (0-based) of the *occurrence*-th whole-word match of *name*.

    By default returns the first occurrence.  Pass *occurrence* > 0 to
    resolve later references on the same line (e.g. ``foo(foo)``).
    """
    idx = 0
    for m in re.finditer(rf"\b{re.escape(name)}\b", line_text):
        if idx == occurrence:
            return m.start()
        idx += 1
    return None


def _ts_find_identifier_in_scope(
    source_bytes: bytes, line: int, name: str
) -> Optional[tuple[int, int]]:
    """Find the closest ``identifier`` node matching *name* within the scope
    containing *line* (1-based).

    Returns (line_1based, col_0based) or None.  Only actual identifier nodes
    are matched — not strings, comments, or keywords.  The closest match to
    *line* wins (used as a fallback when the name isn't on the given line).
    """
    best_scope = _ts_find_scope_node(source_bytes, line, containing=True)
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


def _uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to a filesystem path, decoding percent-encoding."""
    if uri.startswith("file://"):
        return urllib.parse.unquote(uri[7:])
    return urllib.parse.unquote(uri)


def _path_to_uri(path: str) -> str:
    """Convert a filesystem path to a ``file://`` URI."""
    if path.startswith("file://"):
        return path
    return "file://" + str(Path(path))


def _is_stdlib_path(path: str) -> bool:
    """True if *path* points to a stdlib/typeshed definition.

    These are noise in callee results — ``isinstance``, ``getattr``, ``len``,
    ``super``, ``logger.info``, etc. all resolve here.
    """
    return any(marker in path for marker in _STDLIB_MARKERS)


def _ts_is_stub_function(source_bytes: bytes, line: int) -> bool:
    """True if the function starting on *line* (1-based) is a stub.

    A stub body contains only a docstring and one of:
    - ``raise NotImplementedError``
    - ``pass``
    - ``...``

    Uses tree-sitter AST inspection, not string matching.
    """
    fn = _ts_find_scope_node(source_bytes, line, kinds=("function_definition",))
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
            text = child.text.decode("utf-8", errors="replace")
            return "NotImplementedError" in text
        else:
            # return, assignment, etc. → concrete.
            return False

    return False


def _lsp_symbol_kind_to_str(kind_num: int) -> str:
    """Map an LSP SymbolKind number to a human-readable kind string."""
    return _LSP_SYMBOL_KINDS.get(kind_num, "unknown")


def _parse_hover(hover: object) -> tuple[str, str]:
    """Extract (signature, docstring) from an LSP hover response.

    LSP hover ``contents`` may be:
    - MarkupContent (dict with ``value``)
    - a plain string
    - a list of MarkedString entries

    The first non-empty line is treated as the signature; the rest as the
    docstring.  Markdown code fences are stripped if present.
    """
    if not hover or not isinstance(hover, dict):
        return "", ""
    contents = hover.get("contents", {})

    # Normalize the three possible contents shapes into a single string.
    if isinstance(contents, dict):
        value = contents.get("value", "")
    elif isinstance(contents, str):
        value = contents
    elif isinstance(contents, list):
        # MarkedString list — join all string/dict entries.
        parts: list[str] = []
        for entry in contents:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                val = entry.get("value", "")
                if isinstance(val, str):
                    parts.append(val)
        value = "\n".join(p for p in parts if p)
    else:
        value = ""

    if not value:
        return "", ""

    # Strip markdown code fences if present.
    lines = str(value).strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    if not lines:
        return "", ""

    signature = lines[0].strip()
    docstring = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return signature, docstring


class PythonTracer(Tracer):
    """Python-specific tracer using ty language server and tree-sitter."""

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
        self._open_docs: set[str] = set()  # URIs already opened via did_open

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

    @property
    def _is_sandbox(self) -> bool:
        """True when the backend is a MicrosandboxBackend."""
        return isinstance(self.backend, MicrosandboxBackend)

    # Path translation (host ↔ guest)

    def _to_guest_path(self, file: str) -> str:
        """Translate a host path to a guest path (sandbox only).

        For non-sandbox backends, paths are already host paths and are
        returned unchanged.
        """
        if self._is_sandbox:
            return self.ms._resolve_guest_path(file)
        return file

    def _to_host_path(self, path: str) -> str:
        """Translate a guest path to a host path (sandbox only).

        For non-sandbox backends, paths are already host paths and are
        returned unchanged.
        """
        if self._is_sandbox:
            return self.ms._to_host_path(path)
        return path

    # LSP document lifecycle

    def _did_open(self, lsp: TyLspClient, uri: str, source: str) -> None:
        """Open *uri* in the LSP server if not already open.

        did_open must be called once per document; repeated calls cause
        server errors.  Always called within _lsp_request_lock.
        """
        if uri in self._open_docs:
            return
        self._open_docs.add(uri)
        lsp.did_open(uri, source)

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
        tree = _ts_parse(source_bytes)
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

    def get_source(self, file: str, line: int, context: int = 60) -> dict:
        """Return the full source of the function/class starting on *line*.

        If tree-sitter finds a scope starting on *line*, the entire scope body
        is returned and *context* is ignored.  Otherwise, a fallback window
        of *context* lines centred around *line* is returned.
        """
        try:
            source = self._read_file(file)
            all_lines = source.splitlines()

            source_bytes = source.encode("utf-8", errors="replace")
            scope = _ts_find_scope_at_line(source_bytes, line)

            if scope:
                start, end = scope
                fallback = False
            else:
                logger.warning(
                    "get_source: no scope found at %s:%d, falling back to "
                    "context window of %d lines",
                    file,
                    line,
                    context,
                )
                centre = line - 1
                start = max(0, centre - context // 2)
                end = min(len(all_lines), centre + (context + 1) // 2)
                fallback = True

            snippet = all_lines[start:end]
            return {
                "file": file,
                "start_line": start + 1,
                "end_line": end,
                "source": "\n".join(snippet),
                "fallback": fallback,
            }
        except OSError as exc:
            return {
                "file": file,
                "start_line": 0,
                "end_line": 0,
                "source": "",
                "fallback": True,
                "error": str(exc),
            }

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
        func_info = _ts_find_function_and_calls(
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

    def _find_symbol_at_line(
        self, outline: list[dict], line: int, *, exact: bool = False
    ) -> Optional[dict]:
        """Find the symbol at *line* in *outline*.

        If *exact*, return the symbol whose ``line`` matches exactly.
        Otherwise, return the tightest enclosing scope (smallest range
        containing *line*).
        """
        best = None
        best_size = float("inf")
        for sym in outline:
            if exact:
                if sym["line"] == line:
                    return sym
            elif sym["line"] <= line <= sym["end_line"]:
                size = sym["end_line"] - sym["line"]
                if size < best_size:
                    best = sym
                    best_size = size
        return best

    def _extract_def_info(self, d: dict) -> Optional[tuple[str, int, int, str]]:
        """Extract (file, line_1based, col, uri) from a single LSP
        definition dict.  Returns None if the dict has no valid uri.
        """
        d_uri = d.get("uri", "")
        if not d_uri:
            return None
        d_file = self._to_host_path(_uri_to_path(d_uri))
        d_range = d.get("range", {})
        d_line = d_range.get("start", {}).get("line", 0) + 1
        d_col = d_range.get("start", {}).get("character", 0)
        return d_file, d_line, d_col, d_uri

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

    def _first_name_on_line(self, file: str, line: int) -> Optional[str]:
        """Return the first identifier-like token on *line* of *file*."""
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            text = lines[line - 1]
            for m in re.finditer(r"\b[a-zA-Z_]\w*\b", text):
                return m.group()
        except OSError:
            logger.warning(
                "Failed to read %s for _first_name_on_line", file, exc_info=True
            )
        return None

    def _def_name_col_from_lines(self, lines: list[str], line: int) -> Optional[int]:
        """Column of the name token on a def/class line.

        For ``def foo()`` returns the column of ``foo``.  Returns None if
        the line is not a def/class definition.
        """
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        for kw in ("async def ", "def ", "class "):
            if stripped.startswith(kw):
                return indent + len(kw)
        return None

    def _def_name_from_lines(self, lines: list[str], line: int) -> Optional[str]:
        """Extract the symbol name from a def/class line.

        For ``def foo(x):`` returns ``foo``.  The name is everything after
        the keyword up to ``(``, ``:``, or whitespace.
        """
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        for kw in ("async def ", "def ", "class "):
            if stripped.startswith(kw):
                rest = stripped[len(kw) :]
                # Name ends at '(' ':' or whitespace.
                for i, ch in enumerate(rest):
                    if ch in "((: \t":
                        return rest[:i] if i > 0 else None
                return rest.rstrip()
        return None
