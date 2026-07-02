"""Python-specific tracer using ty language server and tree-sitter.

Replaces the former jedi/parso-based implementation with Astral's ``ty``
language server, run as a subprocess inside the microsandbox VM.  LSP
communication is handled by :class:`~metalgate_code.context.ty_lsp_client.TyLspClient`.

``find_symbol`` uses LSP ``workspace/symbol`` — project files only, not
site-packages (confirmed from ty 0.0.55 source).  For third-party symbols,
use ``goto_definition`` from a usage site.

Tree-sitter is retained for:
  - ``get_source`` — line-based source extraction from tree-sitter scope nodes
  - ``get_file_outline`` — fast outline extraction (no LSP round-trip needed)
"""

from __future__ import annotations

import logging
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from metalgate_code.context.cache import _CACHE_MISS, CodeCache
from metalgate_code.context.tracer_base import _MAX_CALLERS, Tracer
from metalgate_code.context.ty_lsp_client import TyLspClient
from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

logger = logging.getLogger("metalgate_code")

_STDLIB_MARKERS = ("typeshed", "/stdlib/", "/builtins.pyi")

# Tree-sitter language — parsers are created per-call for thread safety.
_TS_LANGUAGE = Language(tspython.language())

# LSP SymbolKind enum (subset relevant to Python).
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


def _ts_parse(source_bytes: bytes):
    """Parse *source_bytes* with a fresh per-call Parser (thread-safe)."""
    return Parser(_TS_LANGUAGE).parse(source_bytes)


def _ts_find_scope_at_line(source_bytes: bytes, line: int) -> Optional[tuple[int, int]]:
    """Return (start_line_0based, end_line_1based_exclusive) of the tightest
    function/class whose definition starts on *line* (1-based), or ``None``.

    The return tuple is suitable for slicing ``source.splitlines()``:
    ``lines[start:end]`` gives the full scope body.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node
    best = None
    best_size = None

    def visit(node):
        nonlocal best, best_size
        if node.type in ("function_definition", "class_definition"):
            start = node.start_point[0] + 1  # 1-based
            end = node.end_point[0] + 1
            if start == line:
                size = end - start
                if best is None or size < best_size:
                    best = (node.start_point[0], node.end_point[0] + 1)
                    best_size = size
        for child in node.children:
            visit(child)

    visit(root)
    return best


def _ts_find_function_containing(
    source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str]]]:
    """Return (start_line_1based, end_line_1based_inclusive, func_name) of the
    innermost function whose body contains *line* (1-based), or ``None``.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node
    best = None
    best_size = None
    best_name = None

    def visit(node):
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
            visit(child)

    visit(root)
    if best is None:
        return None
    return (best[0], best[1], best_name)


def _ts_call_positions(
    source_bytes: bytes, start_line: int, end_line: int
) -> list[tuple[int, int]]:
    """Return (line_1based, col_0based) of every function call in [start_line, end_line].

    Walks the tree-sitter AST to find ``call`` nodes.  For each call:
      - ``foo()``    → position of ``foo`` (identifier)
      - ``obj.m()``  → position of ``m`` (attribute's identifier)
      - other        → position of the function expression

    Decorators, class definitions, and the function's own definition line
    are naturally excluded — they are not ``call`` nodes within the
    function body.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node
    positions: list[tuple[int, int]] = []

    def visit(node):
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                if func_node.type == "attribute":
                    attr_node = func_node.child_by_field_name("attribute")
                    if attr_node is not None:
                        pos_node = attr_node
                    else:
                        pos_node = func_node
                else:
                    pos_node = func_node
                line = pos_node.start_point[0] + 1
                col = pos_node.start_point[1]
                if start_line <= line <= end_line:
                    positions.append((line, col))
        for child in node.children:
            visit(child)

    visit(root)
    return positions


def _name_col_on_line(line_text: str, name: str, occurrence: int = 0) -> Optional[int]:
    """Column of the *occurrence*-th (0-based) whole-word match of *name*.

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

    Returns (line_1based, col_0based) or ``None``.  Uses tree-sitter to walk
    the AST of the innermost function (or class) containing *line*, so only
    actual identifier nodes are matched — not strings, comments, or keywords.
    The closest match to *line* wins.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node

    # Find the tightest function or class containing *line*.
    best_scope = None
    best_scope_size = None

    def find_scope(node):
        nonlocal best_scope, best_scope_size
        if node.type in ("function_definition", "class_definition"):
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            if start <= line <= end:
                size = end - start
                if best_scope is None or size < best_scope_size:
                    best_scope = node
                    best_scope_size = size
        for child in node.children:
            find_scope(child)

    find_scope(root)
    if best_scope is None:
        return None

    # Find the closest identifier node matching *name* within that scope.
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
    """Return True if *path* points to a stdlib/typeshed definition.

    These are noise in callee results — ``isinstance``, ``getattr``, ``len``,
    ``super``, ``logger.info``, etc. all resolve here.
    """
    return any(marker in path for marker in _STDLIB_MARKERS)


def _ts_is_stub_function(source_bytes: bytes, line: int) -> bool:
    """Return True if the function starting on *line* (1-based) is a stub.

    A stub is a function whose body contains only a docstring and one of:
    - ``raise NotImplementedError``
    - ``pass``
    - ``...``

    Uses tree-sitter to inspect the AST, not string matching.
    """
    tree = _ts_parse(source_bytes)
    root = tree.root_node

    def find_fn(node):
        if node.type == "function_definition":
            start = node.start_point[0] + 1
            if start == line:
                return node
        for child in node.children:
            found = find_fn(child)
            if found is not None:
                return found
        return None

    fn = find_fn(root)
    if fn is None:
        return False

    body = fn.child_by_field_name("body")
    if body is None:
        return False

    for child in body.children:
        if child.type == "expression_statement":
            # Docstring (string) is allowed in stubs.
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
            # ``raise NotImplementedError`` — check the raised name.
            text = child.text.decode("utf-8", errors="replace")
            if "NotImplementedError" in text:
                return True
            return False
        else:
            # Any other statement (return, assignment, etc.) is concrete.
            return False

    return False


def _dedup_callees(results: list[dict]) -> list[dict]:
    """Deduplicate callee results by name, keeping the most specific implementation.

    When the same method appears via both its abstract declaration (a stub
    that raises ``NotImplementedError`` or uses ``pass``/``...``) and its
    concrete implementation, keep the concrete one.  Callees with empty
    names are kept as-is.
    """
    deduped: list[dict] = []
    by_name: dict[str, dict] = {}
    for r in results:
        rname = r.get("name", "")
        if not rname:
            deduped.append(r)
            continue
        existing = by_name.get(rname)
        if existing is None:
            by_name[rname] = r
            deduped.append(r)
        elif _callee_is_stub(existing) and not _callee_is_stub(r):
            # Replace the abstract entry with the concrete one.
            deduped[deduped.index(existing)] = r
            by_name[rname] = r
    return deduped


def _callee_is_stub(callee: dict) -> bool:
    """Whether a callee result points to a stub/abstract method.

    Uses tree-sitter to inspect the function body at the callee's location.
    """
    file = callee.get("file", "")
    line = callee.get("line", 0)
    if not file or not line:
        return False
    try:
        with open(file, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError:
        return False
    return _ts_is_stub_function(source.encode("utf-8", errors="replace"), line)


def _lsp_symbol_kind_to_str(kind_num: int) -> str:
    """Map an LSP SymbolKind number to a human-readable kind string."""
    return _LSP_SYMBOL_KINDS.get(kind_num, "unknown")


def _parse_hover(hover: object) -> tuple[str, str]:
    """Extract (signature, docstring) from an LSP hover response.

    LSP hover ``contents`` may be ``MarkupContent`` (dict with ``value``),
    a plain string, or a list of ``MarkedString``.  The first non-empty
    line is treated as the signature; the remainder as the docstring.
    """
    if not hover or not isinstance(hover, dict):
        return "", ""
    contents = hover.get("contents", {})
    value = ""
    if isinstance(contents, dict):
        value = contents.get("value", "")
    elif isinstance(contents, str):
        value = contents
    elif isinstance(contents, list):
        # MarkedString list — join all string entries.
        parts: list[str] = []
        for entry in contents:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                val = entry.get("value", "")
                if isinstance(val, str):
                    parts.append(val)
        value = "\n".join(p for p in parts if p)
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
        self._lsp_lock = threading.Lock()
        # Serializes all LSP requests (did_open + references/definition/hover).
        # The ty server is single-threaded; concurrent requests from parallel
        # tool calls cause "content modified" errors and timeouts.  RLock so
        # _resolve can call find_symbol while already holding the lock.
        self._lsp_request_lock = threading.RLock()
        self._ms: Optional[MicrosandboxBackend] = None
        # LSP document lifecycle: track which URIs are already open.
        self._open_docs: set[str] = set()

    @property
    def ms(self) -> MicrosandboxBackend:
        """The MicrosandboxBackend instance (validated once, then cached)."""
        if self._ms is not None:
            return self._ms
        if not isinstance(self.backend, MicrosandboxBackend):
            raise RuntimeError("PythonTracer requires a MicrosandboxBackend")
        self._ms = self.backend
        return self._ms

    # ------------------------------------------------------------------ #
    # LSP document lifecycle management
    # ------------------------------------------------------------------ #

    def _did_open(self, lsp: TyLspClient, uri: str, source: str) -> None:
        """Open *uri* in the LSP server if not already open.

        LSP ``textDocument/didOpen`` is meant to be called once per document.
        Repeated calls for the same URI are skipped to avoid server errors.
        Always called within ``_lsp_request_lock`` so no separate lock needed.
        """
        if uri in self._open_docs:
            return
        self._open_docs.add(uri)
        lsp.did_open(uri, source)

    def _get_lsp(self) -> TyLspClient:
        """Get or lazily create the ty LSP client (thread-safe).

        The venv (``venv_bin`` / ``venv_env``) is established by the backend
        during sandbox boot via :class:`~metalgate_code.factory.venv_manager.VenvManager`.
        The tracer consumes it directly — it never manipulates or discovers
        the venv itself.  ``python_path`` is derived from ``venv_bin`` when
        available.
        """
        if self._lsp is not None:
            return self._lsp

        with self._lsp_lock:
            if self._lsp is not None:
                return self._lsp

            sb = self.ms._ensure_sandbox_sync()
            guest_root = self.ms._to_guest_path(str(self.root))
            root_uri = _path_to_uri(guest_root)

            venv_bin = self.ms.venv_bin
            venv_env = self.ms.venv_env
            python_path = f"{venv_bin}/python" if venv_bin else None

            self._lsp = TyLspClient(
                sb,
                root_uri,
                python_path=python_path,
                venv_bin=venv_bin,
                venv_env=venv_env,
            )
            self._lsp.start()
            return self._lsp

    # ------------------------------------------------------------------ #
    # Tracer interface
    # ------------------------------------------------------------------ #

    def get_file_outline(self, file: str) -> list[dict]:
        """Parse *file* and return every class/function/method with name, kind, line, end_line, signature."""
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
        """Extract outline using tree-sitter."""
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
                param_str = "..."
                if params_node:
                    param_str = params_node.text.decode("utf-8", errors="replace")

                is_async = False
                for child in node.children:
                    if child.type == "async":
                        is_async = True
                        break

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
        """Resolve the symbol *name* on *line* of *file* to its definition."""
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

        If tree-sitter finds a scope (function or class definition) starting
        on *line*, the entire scope body is returned and *context* is ignored.

        If no scope is found at *line*, a fallback window of *context* lines
        centred around *line* is returned instead, and a warning is logged.
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
        """Find every place in the project that references the symbol on *line* of *file*."""
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
        guest_file = self.ms._resolve_guest_path(file)
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
            ref_file = self.ms._to_host_path(_uri_to_path(ref_uri))
            ref_range = r.get("range", {})
            ref_line = ref_range.get("start", {}).get("line", 0) + 1

            if ref_file == file and ref_line == line:
                continue

            key = (ref_file, ref_line)
            if key in seen:
                continue
            seen.add(key)

            caller_name = ""
            try:
                ref_outline = self.get_file_outline(ref_file)
                best = None
                best_size = float("inf")
                for sym in ref_outline:
                    if sym["line"] <= ref_line <= sym["end_line"]:
                        size = sym["end_line"] - sym["line"]
                        if size < best_size:
                            best = sym
                            best_size = size
                if best:
                    caller_name = best["name"]
            except Exception:
                logger.warning(
                    "get_file_outline failed for %s", ref_file, exc_info=True
                )

            # Read the referencing line for context
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
        """Find every symbol called by the function on *line* of *file*, resolved to definitions."""
        try:
            source = self._read_file(file)
        except OSError:
            return []

        func_info = _ts_find_function_containing(
            source.encode("utf-8", errors="replace"), line
        )
        if func_info is None:
            return []

        start_line, end_line, func_name = func_info
        positions = _ts_call_positions(
            source.encode("utf-8", errors="replace"), start_line, end_line
        )

        lsp = self._get_lsp()
        guest_file = self.ms._resolve_guest_path(file)
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
                    d_uri = d.get("uri", "")
                    if not d_uri:
                        continue
                    d_file = self.ms._to_host_path(_uri_to_path(d_uri))

                    # Skip stdlib/builtins — they add noise without aiding
                    # codebase understanding.
                    if _is_stdlib_path(d_file):
                        continue

                    d_range = d.get("range", {})
                    d_line = d_range.get("start", {}).get("line", 0) + 1

                    key = (d_file, d_line)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Look up name/kind from the definition file's outline
                    d_name = ""
                    d_kind = ""
                    d_sig = ""
                    try:
                        outline = self.get_file_outline(d_file)
                        for sym in outline:
                            if sym["line"] == d_line:
                                d_name = sym["name"]
                                d_kind = sym["kind"]
                                d_sig = sym.get("signature", "")
                                break
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

        # Deduplicate by name: when the same method appears via both its
        # abstract declaration and concrete implementation, keep the
        # concrete one (detected via AST-based stub inspection).
        return _dedup_callees(results)

    def find_symbol(self, name: str) -> list[dict]:
        """Search for *name* across the project via LSP ``workspace/symbol``.

        ty's ``workspace/symbol`` indexes first-party project files only —
        it does not search site-packages (confirmed from ty 0.0.55 source).
        For third-party symbols, use ``goto_definition`` from a usage site,
        which ty resolves directly to the site-packages definition.
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
                sym_file = self.ms._to_host_path(_uri_to_path(sym_uri))
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

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _resolve(self, file: str, line: int, name: str) -> Optional[dict]:
        """Resolve *name* at *line* in *file* to its definition via LSP.

        ty resolves usage sites directly to the actual definition —
        first-party or site-packages.  When ty can't resolve (e.g.
        conditional imports, type annotations in signatures), falls
        back to ``workspace/symbol`` search by name.
        """
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None

            lsp = self._get_lsp()
            guest_file = self.ms._resolve_guest_path(file)
            uri = _path_to_uri(guest_file)

            # Try to resolve at the given line first.
            col = _name_col_on_line(lines[line - 1], name)

            # If the name isn't on the given line, find it in the enclosing
            # scope via tree-sitter so we can give the LSP the right position.
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

                if d_uri is None:
                    # LSP couldn't resolve (e.g. name not on the given line,
                    # conditional import, type annotation).  Fall back to
                    # workspace symbol search by name.
                    if name:
                        sym_results = self.find_symbol(name)
                        if sym_results and sym_results[0].get("file"):
                            sr = sym_results[0]
                            d_file = sr["file"]
                            d_line = sr["line"]
                            d_col = 0
                            d_uri = _path_to_uri(self.ms._resolve_guest_path(d_file))
                    if d_uri is None:
                        return None

                # Get hover info for signature/docstring (only when we
                # have a valid position in the source file).
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

            # Determine kind from the definition file's outline
            kind = "unknown"
            if d_file:
                try:
                    outline = self.get_file_outline(d_file)
                    for sym in outline:
                        if sym["line"] == d_line:
                            kind = sym["kind"]
                            if not signature:
                                signature = sym.get("signature", "")
                            break
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
        """Call LSP definition and return (file, line_1based, col, uri)."""
        defs = lsp.definition(uri, line_0, col)
        if not defs:
            return None, 0, 0, None
        if isinstance(defs, dict):
            defs = [defs]
        d = defs[0]
        d_uri = d.get("uri", "")
        if not d_uri:
            return None, 0, 0, None
        d_file = self.ms._to_host_path(_uri_to_path(d_uri))
        d_range = d.get("range", {})
        d_line = d_range.get("start", {}).get("line", 0) + 1
        d_col = d_range.get("start", {}).get("character", 0)
        return d_file, d_line, d_col, d_uri

    def _first_name_on_line(self, file: str, line: int) -> Optional[str]:
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

    def _def_name_col(self, file: str, line: int) -> Optional[int]:
        """Column of the name token on a def/class line."""
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            return self._def_name_col_from_lines(lines, line)
        except OSError:
            logger.warning("Failed to read %s for _def_name_col", file, exc_info=True)
            return None

    def _def_name_col_from_lines(self, lines: list[str], line: int) -> Optional[int]:
        """Column of the name token on a def/class line, from pre-read lines."""
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        for kw in ("async def ", "def ", "class "):
            if stripped.startswith(kw):
                return indent + len(kw)
        return None

    def _def_name(self, file: str, line: int) -> Optional[str]:
        """Extract the symbol name from a def/class line."""
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            return self._def_name_from_lines(lines, line)
        except OSError:
            logger.warning("Failed to read %s for _def_name", file, exc_info=True)
            return None

    def _def_name_from_lines(self, lines: list[str], line: int) -> Optional[str]:
        """Extract the symbol name from a def/class line, from pre-read lines."""
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        for kw in ("async def ", "def ", "class "):
            if stripped.startswith(kw):
                rest = stripped[len(kw) :]
                # name is up to '(' or ':' or whitespace
                for i, ch in enumerate(rest):
                    if ch in "((: \t":
                        return rest[:i] if i > 0 else None
                return rest.rstrip()
        return None
