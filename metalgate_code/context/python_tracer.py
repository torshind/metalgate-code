"""Python-specific tracer using ty language server and tree-sitter.

Replaces the former jedi/parso-based implementation with Astral's ``ty``
language server, run as a subprocess inside the microsandbox VM.  LSP
communication is handled by :class:`~metalgate_code.context.ty_lsp_client.TyLspClient`.

Tree-sitter is retained for:
  - ``find_symbol`` — fast exact symbol search across project and site-packages
  - ``get_source`` — line-based source extraction from tree-sitter scope nodes
  - ``get_file_outline`` — fast outline extraction (no LSP round-trip needed)
"""

from __future__ import annotations

import io
import logging
import re
import threading
import tokenize
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

_PYCACHE = "__pycache__"
_VENV_DIR = ".venv"
_STDLIB_MARKERS = ("typeshed", "/stdlib/", "/logging/", "/builtins.pyi")

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


class _TreeCache:
    """Short-lived in-memory cache for tree-sitter parse results.

    Keyed by (file, mtime).  Avoids redundant re-parses within a single
    tool invocation that reads the same file multiple times.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: Optional[tuple[str, float, object]] = None

    def get(self, file: str, mtime: float):
        with self._lock:
            if self._entry is not None:
                cached_file, cached_mtime, tree = self._entry
                if cached_file == file and cached_mtime == mtime:
                    return tree
            return None

    def set(self, file: str, mtime: float, tree) -> None:
        with self._lock:
            self._entry = (file, mtime, tree)


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


def _call_positions(
    source: str, start_line: int, end_line: int, func_name: str | None = None
) -> list[tuple[int, int]]:
    """Return (line, col) of every NAME token followed by '(' in [start_line, end_line].

    If *func_name* is given, skip the token when it is *func_name* on the
    function's own definition line (avoids treating ``def foo(...):`` as a
    call to ``foo``).

    Also skips:
      - Decorator lines (``@decorator(...)``) — the ``@`` prefix means the
        name is not a call within the function body.
      - Class definition lines (``class Foo(Bar)``) — the class name and
        base class list are not function calls.
    """
    positions: list[tuple[int, int]] = []
    _skip = {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.COMMENT,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
    }
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return positions

    paren_depth = 0
    in_class_def = False  # True between 'class' keyword and its ':' at depth 0

    for i, tok in enumerate(tokens):
        if tok.string == "(":
            paren_depth += 1
        elif tok.string == ")":
            paren_depth -= 1

        if tok.type != tokenize.NAME:
            continue
        tok_line = tok.start[0]
        if not (start_line <= tok_line <= end_line):
            continue
        if func_name is not None and tok.string == func_name and tok_line == start_line:
            continue
        # Skip decorators: if the previous token is '@', this is a decorator
        # application, not a call.
        if i > 0 and tokens[i - 1].string == "@":
            continue
        # Track class definition context: from 'class' keyword to ':' at
        # paren depth 0.  Names inside this region (class name + bases) are
        # not function calls.
        if tok.string == "class":
            in_class_def = True
            continue
        if in_class_def and tok.string == ":" and paren_depth == 0:
            in_class_def = False
            continue
        if in_class_def:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].type in _skip:
            j += 1
        if j < len(tokens) and tokens[j].string == "(":
            positions.append((tok.start[0], tok.start[1]))

    return positions


def _name_col_on_line(
    line_text: str, name: str, occurrence: int = 0
) -> Optional[int]:
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
        parts = []
        for entry in contents:
            if isinstance(entry, str):
                parts.append(entry)
            elif isinstance(entry, dict):
                parts.append(entry.get("value", ""))
        value = "\n".join(p for p in parts if p)
    if not value:
        return "", ""
    # Strip markdown code fences if present.
    lines = value.strip().splitlines()
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
        self._ms: Optional[MicrosandboxBackend] = None
        # LSP document lifecycle: track which URIs are already open.
        self._open_docs: set[str] = set()
        self._open_docs_lock = threading.Lock()
        # In-memory tree-sitter parse cache (single-entry, mtime-keyed).
        self._tree_cache = _TreeCache()

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
        """
        with self._open_docs_lock:
            if uri in self._open_docs:
                return
            self._open_docs.add(uri)
        lsp.did_open(uri, source)

    def _did_close(self, lsp: TyLspClient, uri: str) -> None:
        """Close *uri* in the LSP server if currently open."""
        with self._open_docs_lock:
            if uri not in self._open_docs:
                return
            self._open_docs.discard(uri)
        try:
            lsp.notify(
                "textDocument/didClose",
                {"textDocument": {"uri": uri}},
            )
        except Exception:
            logger.warning("did_close failed for %s", uri, exc_info=True)

    # ------------------------------------------------------------------ #
    # In-memory tree-sitter parse cache
    # ------------------------------------------------------------------ #

    def _cached_parse(self, file: str, source: str):
        """Parse *source* with tree-sitter, using an in-memory cache.

        The cache is keyed by (file, mtime) and stores a single entry —
        enough to avoid redundant re-parses when multiple methods read the
        same file within one tool invocation.
        """
        import os

        try:
            mtime = os.path.getmtime(file)
        except OSError:
            mtime = 0.0
        cached = self._tree_cache.get(file, mtime)
        if cached is not None:
            return cached
        tree = _ts_parse(source.encode("utf-8", errors="replace"))
        self._tree_cache.set(file, mtime, tree)
        return tree

    def _get_lsp(self) -> TyLspClient:
        """Get or lazily create the ty LSP client (thread-safe)."""
        if self._lsp is not None:
            return self._lsp

        with self._lsp_lock:
            if self._lsp is not None:
                return self._lsp

            sb = self.ms._ensure_sandbox_sync()
            guest_root = self.ms._to_guest_path(str(self.root))
            root_uri = _path_to_uri(guest_root)
            python_path = self._detect_python()

            self._lsp = TyLspClient(sb, root_uri, python_path=python_path)
            self._lsp.start()
            return self._lsp

    def _detect_python(self) -> Optional[str]:
        """Detect the Python interpreter path inside the sandbox."""
        if self.backend is None:
            return None
        try:
            result = self.backend.execute("uv run which python")
            if result.exit_code is not None and result.exit_code == 0:
                return result.output.strip()
            result = self.backend.execute("which python")
            if result.exit_code is not None and result.exit_code == 0:
                return result.output.strip()
        except Exception:
            logger.warning("Failed to detect Python interpreter", exc_info=True)
        return None

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
        if cached is not _CACHE_MISS:
            return cached

        result = self._resolve(file, line, name)
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

            snippet = all_lines[start:end]
            return {
                "file": file,
                "start_line": start + 1,
                "end_line": end,
                "source": "\n".join(snippet),
            }
        except OSError as exc:
            return {
                "file": file,
                "start_line": 0,
                "end_line": 0,
                "source": "",
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
        self._did_open(lsp, uri, source)

        try:
            refs = lsp.references(uri, line - 1, col, include_declaration=False)
        except Exception:
            logger.warning("LSP references failed for %s:%d", file, line, exc_info=True)
            return []

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
                logger.warning("get_file_outline failed for %s", ref_file, exc_info=True)

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
        positions = _call_positions(source, start_line, end_line, func_name=func_name)

        lsp = self._get_lsp()
        guest_file = self.ms._resolve_guest_path(file)
        uri = _path_to_uri(guest_file)
        self._did_open(lsp, uri, source)
        results: list[dict] = []
        seen: set[tuple] = set()

        for call_line, call_col in positions:
            try:
                defs = lsp.definition(uri, call_line - 1, call_col)
            except Exception:
                logger.warning(
                    "LSP definition failed at %s:%d:%d",
                    file, call_line, call_col, exc_info=True,
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

        return results

    def find_symbol(self, name: str) -> list[dict]:
        """Search for *name* across the project and installed packages.

        Uses the LSP ``workspace/symbol`` request for project-scoped results.
        If that returns nothing (ty only indexes the workspace root, not
        PYTHONPATH/site-packages), falls back to scanning project import
        statements and resolving each via LSP ``definition`` to find the
        symbol in third-party packages.
        """
        results: list[dict] = []
        seen: set[tuple] = set()

        # 1. Try workspace/symbol — fast, covers project files.
        try:
            lsp = self._get_lsp()
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

        if results:
            return results

        # 2. Fall back: scan project files for import statements that
        #    reference *name*, then resolve via LSP definition to find
        #    the symbol's definition in third-party packages.
        return self._find_symbol_via_imports(name)

    def _find_symbol_via_imports(self, name: str) -> list[dict]:
        """Find *name* in third-party packages by scanning project imports.

        Reads each project .py file, looks for ``import name`` or
        ``from ... import name`` statements, then uses LSP ``definition``
        to resolve the symbol to its actual definition location.
        """
        try:
            lsp = self._get_lsp()
        except Exception:
            logger.warning("LSP unavailable for find_symbol_via_imports", exc_info=True)
            return []

        results: list[dict] = []
        seen: set[tuple] = set()
        # Word-boundary pattern to avoid substring false positives.
        name_re = re.compile(rf"\b{re.escape(name)}\b")

        # Scan project .py files for import lines mentioning *name*
        for py_file in self.root.rglob("*.py"):
            if _PYCACHE in py_file.parts or _VENV_DIR in py_file.parts:
                continue
            try:
                source = self._read_file(str(py_file))
            except OSError:
                continue

            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                # Match: from X import name  /  import name
                # Use word-boundary regex to avoid substring false positives.
                if not (
                    (stripped.startswith("from ") and name_re.search(stripped))
                    or (stripped.startswith("import ") and name_re.search(stripped))
                ):
                    continue

                # Resolve via LSP definition
                col = _name_col_on_line(line, name)
                if col is None:
                    continue
                try:
                    guest_file = self.ms._to_guest_path(str(py_file))
                    uri = _path_to_uri(guest_file)
                    self._did_open(lsp, uri, source)
                    defs = lsp.definition(uri, i - 1, col)
                except Exception:
                    logger.warning(
                        "LSP definition failed for import at %s:%d",
                        py_file, i, exc_info=True,
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
                    d_line = d.get("range", {}).get("start", {}).get("line", 0) + 1

                    key = (d_file, d_line)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Skip if it just points back to the same import line
                    # (path may be host-absolute or project-relative)
                    if d_line == i and (
                        str(py_file) == d_file or d_file.endswith(str(py_file))
                    ):
                        continue

                    # Get kind from outline
                    d_kind = ""
                    try:
                        outline = self.get_file_outline(d_file)
                        for sym in outline:
                            if sym["line"] == d_line:
                                d_kind = sym["kind"]
                                break
                    except Exception:
                        logger.warning(
                            "get_file_outline failed for %s", d_file, exc_info=True
                        )

                    results.append(
                        {
                            "name": name,
                            "kind": d_kind,
                            "file": d_file,
                            "line": d_line,
                        }
                    )
                    if len(results) >= _MAX_CALLERS:
                        return results

        return results

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _resolve(self, file: str, line: int, name: str) -> Optional[dict]:
        """Resolve *name* at *line* in *file* to its definition via LSP.

        If ty resolves to an import statement in the same file (e.g.
        ``from bar import Bar`` instead of the actual class definition),
        delegates to :meth:`find_symbol` which resolves imports to their
        third-party definitions.
        """
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            col = _name_col_on_line(lines[line - 1], name)
            if col is None:
                return None

            lsp = self._get_lsp()
            guest_file = self.ms._resolve_guest_path(file)
            uri = _path_to_uri(guest_file)
            self._did_open(lsp, uri, source)

            d_file, d_line, d_col, d_uri = self._lsp_definition(
                lsp, uri, line - 1, col
            )
            if d_uri is None:
                return None

            # If ty resolved to an import line in the same file, delegate
            # to find_symbol which resolves imports to third-party defs.
            if d_file and d_file.endswith(file) and 1 <= d_line <= len(lines):
                import_line = lines[d_line - 1].strip()
                if import_line.startswith(("import ", "from ")):
                    sym_results = self.find_symbol(name)
                    for sr in sym_results:
                        sr_file = sr.get("file", "")
                        if sr_file and not sr_file.endswith(file):
                            d_file = sr_file
                            d_line = sr.get("line", 0)
                            d_col = 0
                            d_uri = _path_to_uri(
                                self.ms._resolve_guest_path(sr_file)
                            )
                            break

            # Get hover info for signature/docstring
            signature = ""
            docstring = ""
            try:
                hover = lsp.hover(uri, line - 1, col)
                signature, docstring = _parse_hover(hover)
            except Exception:
                logger.warning("hover failed for %s:%d", file, line, exc_info=True)

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
            logger.warning("Failed to read %s for _first_name_on_line", file, exc_info=True)
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

    def _def_name_col_from_lines(
        self, lines: list[str], line: int
    ) -> Optional[int]:
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

    def _def_name_from_lines(
        self, lines: list[str], line: int
    ) -> Optional[str]:
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
