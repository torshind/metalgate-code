"""Abstract base class and shared helpers for language-specific Tracers.

Shared utilities (used by both GoTracer and PythonTracer):
  - LSP constants and helpers (``_LSP_SYMBOL_KINDS``, ``_lsp_symbol_kind_to_str``,
    ``_uri_to_path``, ``_path_to_uri``, ``_parse_hover_base``)
  - Text helpers (``_name_col_on_line``)
  - Tree-sitter configuration (``TreeSitterConfig``) and shared AST helpers
    (``_ts_find_scope_node``, ``_ts_find_scope_at_line``, ``_ts_find_function_containing``,
    ``_ts_call_positions``, ``_ts_find_function_and_calls``)
  - Path translation (``_to_guest_path``, ``_to_host_path``, ``_is_sandbox``, ``ms``)
  - LSP lifecycle (``_did_open``)
  - Shared Tracer methods (``get_source``, ``_find_symbol_at_line``,
    ``_first_name_on_line``, ``_extract_def_info``, ``_def_name_col_from_lines``,
    ``_def_name_from_lines``)
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from deepagents.backends.protocol import SandboxBackendProtocol
from tree_sitter import Language, Parser

from metalgate_code.context.cache import CodeCache

logger = logging.getLogger("metalgate_code")

_MAX_CALLERS = 50
_CALLERS_TIMEOUT = 15.0
_CALLERS_WORKERS = os.environ.get("CALLERS_WORKERS", 4)

# LSP SymbolKind enum values → human-readable names (shared across languages).
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


# Module-level shared utilities


def _lsp_symbol_kind_to_str(kind_num: int) -> str:
    """Map an LSP SymbolKind number to a human-readable kind string."""
    return _LSP_SYMBOL_KINDS.get(kind_num, "unknown")


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


def _uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to a filesystem path, decoding percent-encoding."""
    if uri.startswith("file://"):
        return urllib.parse.unquote(uri[7:])
    return urllib.parse.unquote(uri)


def _path_to_uri(path: str) -> str:
    """Convert a filesystem path to a ``file://`` URI.

    Does NOT resolve symlinks — the backend stores the raw path and path
    translation (host ↔ guest) relies on exact string matching.  Resolving
    would break on macOS where /tmp → /private/tmp.
    """
    if path.startswith("file://"):
        return path
    return "file://" + str(Path(path))


def _parse_hover_base(hover: object) -> tuple[str, str]:
    """Extract (signature, docstring) from an LSP hover response.

    LSP hover ``contents`` may be:
    - MarkupContent (dict with ``value``)
    - a plain string
    - a list of MarkedString entries

    The first non-empty line is treated as the signature; the rest as the
    docstring.  Markdown code fences are stripped if present.

    Language-specific post-processing (e.g. stripping gopls pkg.go.dev links)
    should be done by the caller before invoking this function, or by
    wrapping it.
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


# Tree-sitter configuration


@dataclass(frozen=True)
class TreeSitterConfig:
    """Language-specific tree-sitter parameters.

    Captures the three dimensions in which Go and Python tree-sitter helpers
    differ, allowing the shared algorithms in ``Tracer`` to be parameterized.

    Attributes:
        language: The tree-sitter ``Language`` object for this language.
        function_kinds: Node types that represent function/method definitions.
        scope_kinds: Node types that represent scope-defining constructs
            (functions, methods, types, classes).
        call_node_type: The tree-sitter node type for call expressions
            (``"call_expression"`` for Go, ``"call"`` for Python).
        member_node_type: The tree-sitter node type for member access
            expressions (``"selector_expression"`` for Go, ``"attribute"``
            for Python).
        member_field_name: The tree-sitter field name for the accessed member
            (``"field"`` for Go, ``"attribute"`` for Python).
    """

    language: Language
    function_kinds: tuple[str, ...]
    scope_kinds: tuple[str, ...]
    call_node_type: str
    member_node_type: str
    member_field_name: str


# Shared tree-sitter helpers (module-level)
#
# All tree-sitter functions take raw bytes and return 1-based line numbers
# (matching LSP convention).  Column numbers are 0-based.


def _ts_parse(config: TreeSitterConfig, source_bytes: bytes):
    """Parse *source_bytes* with a fresh Parser (thread-safe)."""
    return Parser(config.language).parse(source_bytes)


def _ts_find_scope_node(
    config: TreeSitterConfig,
    source_bytes: bytes,
    line: int,
    *,
    containing: bool = False,
    kinds: Optional[tuple[str, ...]] = None,
):
    """Find the tightest (smallest) AST node of the given *kinds* matching *line*.

    *containing*=False → match nodes whose def starts on *line* (start == line).
    *containing*=True  → match nodes whose body contains *line* (start <= line <= end).

    Returns the tree-sitter node, or None.
    """
    if kinds is None:
        kinds = config.scope_kinds
    tree = _ts_parse(config, source_bytes)
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


def _ts_find_scope_at_line(
    config: TreeSitterConfig, source_bytes: bytes, line: int
) -> Optional[tuple[int, int]]:
    """Return (start_0based, end_1based_exclusive) of the scope starting on *line*.

    The tuple is suitable for slicing ``source.splitlines()``:
    ``lines[start:end]`` gives the full scope body.
    """
    node = _ts_find_scope_node(config, source_bytes, line)
    if node is None:
        return None
    return (node.start_point[0], node.end_point[0] + 1)


def _ts_find_function_containing(
    config: TreeSitterConfig, source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str]]]:
    """Return (start_1based, end_1based, name) of the innermost function
    whose body contains *line*, or None.
    """
    node = _ts_find_scope_node(
        config,
        source_bytes,
        line,
        kinds=config.function_kinds,
        containing=True,
    )
    if node is None:
        return None
    name_node = node.child_by_field_name("name")
    name = name_node.text.decode("utf-8", errors="replace") if name_node else None
    return (node.start_point[0] + 1, node.end_point[0] + 1, name)


def _ts_call_positions(
    config: TreeSitterConfig,
    source_bytes: bytes,
    start_line: int,
    end_line: int,
) -> list[tuple[int, int]]:
    """Return (line_1based, col_0based) of every call expression in [start_line, end_line].

    For each call node, the position targets the callable name:
      - ``foo()``   → position of ``foo``
      - ``obj.m()`` → position of ``m`` (the member, not the object)
      - other       → position of the function expression
    """
    tree = _ts_parse(config, source_bytes)
    root = tree.root_node
    positions: list[tuple[int, int]] = []

    def visit(node):
        if node.type == config.call_node_type:
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                # For member access expressions (obj.m()), target the
                # member name "m", not the object/qualifier.
                if func_node.type == config.member_node_type:
                    member_node = func_node.child_by_field_name(
                        config.member_field_name
                    )
                    pos_node = member_node if member_node is not None else func_node
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
    config: TreeSitterConfig, source_bytes: bytes, line: int
) -> Optional[tuple[int, int, Optional[str], list[tuple[int, int]]]]:
    """Find the innermost function containing *line* and all call positions within it.

    Parses once, then does two passes over the tree:
      1. Find the innermost function/method containing *line*.
      2. Collect all call positions within that function's line range.

    Returns (start_line, end_line, func_name, positions) or None.
    """
    tree = _ts_parse(config, source_bytes)
    root = tree.root_node

    # Pass 1: find innermost function containing *line*.
    best = None
    best_size = None
    best_name = None

    def find_fn(node):
        nonlocal best, best_size, best_name
        if node.type in config.function_kinds:
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
        if node.type == config.call_node_type:
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                if func_node.type == config.member_node_type:
                    member_node = func_node.child_by_field_name(
                        config.member_field_name
                    )
                    pos_node = member_node if member_node is not None else func_node
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


# Tracer base class


class Tracer(ABC):
    """Abstract base for language-specific code navigation.

    Subclasses must set the ``_ts_config`` class attribute to a
    :class:`TreeSitterConfig` and implement the abstract methods.
    """

    # Set by subclasses.
    _ts_config: TreeSitterConfig

    def __init__(
        self,
        root: str,
        backend: SandboxBackendProtocol,
        cache: CodeCache,
    ) -> None:
        # Do NOT resolve symlinks — the backend stores the raw path and
        # path translation (host ↔ guest) relies on exact string matching.
        # Resolving would break on macOS where /tmp → /private/tmp.
        self.root = Path(root)
        self.cache = cache
        self.backend = backend
        # URIs already opened via did_open.  Subclasses share this set
        # so _did_open can live in the base class.
        self._open_docs: set[str] = set()

    # File reading

    def _read_file(self, file: str, limit: int = 10000) -> str:
        """Read file content using backend if available, otherwise local filesystem."""
        if self.backend is not None:
            result = self.backend.read(file, offset=0, limit=limit)
            if result.error is None and result.file_data is not None:
                return result.file_data["content"]
        return Path(file).read_text(encoding="utf-8", errors="ignore")

    def _read_file_bytes(self, file: str, limit: int = 10000) -> bytes:
        """Read file content as bytes using backend if available."""
        return self._read_file(file, limit=limit).encode("utf-8", errors="ignore")

    # Path translation (host ↔ guest)
    #
    # When the backend is a MicrosandboxBackend, the agent passes sandbox
    # paths like ``/workspace/orders.go``.  These must be translated to host
    # paths (e.g. ``/Users/foo/project/orders.go``) before sending to the LSP
    # server.  Conversely, LSP response URIs contain host paths that must be
    # translated back to sandbox paths for the agent.
    #
    # For non-sandbox backends (e.g. LocalShellBackend), paths are already host
    # paths and are returned unchanged.

    @property
    def _is_sandbox(self) -> bool:
        """True when the backend is a MicrosandboxBackend."""
        # Lazy import to avoid circular dependencies at module load time.
        from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

        return isinstance(self.backend, MicrosandboxBackend)

    @property
    def ms(self):
        """The MicrosandboxBackend (validated once, then cached).

        Raises RuntimeError if the backend is not a MicrosandboxBackend.
        Callers that don't require the sandbox should check
        :meth:`_is_sandbox` first.
        """
        from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

        if not isinstance(self.backend, MicrosandboxBackend):
            raise RuntimeError(f"{type(self).__name__} requires a MicrosandboxBackend")
        return self.backend

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

    def _did_open(self, lsp, uri: str, source: str) -> None:
        """Open *uri* in the LSP server if not already open.

        did_open must be called once per document; repeated calls cause
        server errors.  Always called within _lsp_request_lock.
        """
        if uri in self._open_docs:
            return
        self._open_docs.add(uri)
        lsp.did_open(uri, source)

    # Shared tree-sitter helpers (delegate to module-level functions)

    def _ts_parse(self, source_bytes: bytes):
        """Parse *source_bytes* with a fresh Parser (thread-safe)."""
        return _ts_parse(self._ts_config, source_bytes)

    def _ts_find_scope_node(
        self,
        source_bytes: bytes,
        line: int,
        *,
        containing: bool = False,
        kinds: Optional[tuple[str, ...]] = None,
    ):
        """Find the tightest (smallest) AST node of the given *kinds* matching *line*.

        *containing*=False → match nodes whose def starts on *line* (start == line).
        *containing*=True  → match nodes whose body contains *line* (start <= line <= end).

        Returns the tree-sitter node, or None.
        """
        return _ts_find_scope_node(
            self._ts_config,
            source_bytes,
            line,
            containing=containing,
            kinds=kinds,
        )

    def _ts_find_scope_at_line(
        self, source_bytes: bytes, line: int
    ) -> Optional[tuple[int, int]]:
        """Return (start_0based, end_1based_exclusive) of the scope starting on *line*.

        The tuple is suitable for slicing ``source.splitlines()``:
        ``lines[start:end]`` gives the full scope body.
        """
        return _ts_find_scope_at_line(self._ts_config, source_bytes, line)

    def _ts_find_function_containing(
        self, source_bytes: bytes, line: int
    ) -> Optional[tuple[int, int, Optional[str]]]:
        """Return (start_1based, end_1based, name) of the innermost function
        whose body contains *line*, or None.
        """
        return _ts_find_function_containing(self._ts_config, source_bytes, line)

    def _ts_call_positions(
        self, source_bytes: bytes, start_line: int, end_line: int
    ) -> list[tuple[int, int]]:
        """Return (line_1based, col_0based) of every call expression in [start_line, end_line].

        For each call node, the position targets the callable name:
          - ``foo()``   → position of ``foo``
          - ``obj.m()`` → position of ``m`` (the member, not the object)
          - other       → position of the function expression
        """
        return _ts_call_positions(self._ts_config, source_bytes, start_line, end_line)

    def _ts_find_function_and_calls(
        self, source_bytes: bytes, line: int
    ) -> Optional[tuple[int, int, Optional[str], list[tuple[int, int]]]]:
        """Find the innermost function containing *line* and all call positions within it.

        Parses once, then does two passes over the tree:
          1. Find the innermost function/method containing *line*.
          2. Collect all call positions within that function's line range.

        Returns (start_line, end_line, func_name, positions) or None.
        """
        return _ts_find_function_and_calls(self._ts_config, source_bytes, line)

    # Shared Tracer methods

    def get_source(self, file: str, line: int, context: int = 60) -> dict:
        """Return the full source of the function/class starting on *line*.

        If tree-sitter finds a scope starting on *line*, the entire scope body
        is returned and *context* is ignored.  Otherwise, a fallback window
        of *context* lines centred around *line* is returned.
        """
        try:
            source_bytes = self._read_file_bytes(file)
            source = source_bytes.decode("utf-8", errors="replace")
            all_lines = source.splitlines()

            scope = self._ts_find_scope_at_line(source_bytes, line)

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

        The file path is translated to the agent's coordinate system via
        :meth:`_uri_to_result_path`.
        """
        d_uri = d.get("uri", "")
        if not d_uri:
            return None
        d_file = self._uri_to_result_path(d_uri)
        d_range = d.get("range", {})
        d_line = d_range.get("start", {}).get("line", 0) + 1
        d_col = d_range.get("start", {}).get("character", 0)
        return d_file, d_line, d_col, d_uri

    def _uri_to_result_path(self, uri: str) -> str:
        """Convert an LSP ``file://`` URI to a path in the agent's coordinate system.

        For sandbox backends, this is the sandbox path (e.g. ``/workspace/...``).
        For non-sandbox backends, this is the host path.

        Subclasses override this when the LSP server runs in a different
        coordinate system than the agent.  The default implementation converts
        the URI to a host path (correct for Python's ty, which runs in the
        sandbox and returns sandbox paths).
        """
        return self._to_host_path(_uri_to_path(uri))

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
        """Column of the name token on a definition line.

        For ``def foo()`` returns the column of ``foo``.  Returns None if
        the line is not a definition line.

        Subclasses with receiver parameters (e.g. Go methods) should
        override this to skip the receiver.
        """
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        for kw in self._def_keywords:
            if stripped.startswith(kw):
                return indent + len(kw)
        return None

    def _def_name_from_lines(self, lines: list[str], line: int) -> Optional[str]:
        """Extract the symbol name from a definition line.

        For ``def foo(x):`` returns ``foo``.  The name is everything after
        the keyword up to ``(``, ``:``, or whitespace.

        Subclasses with receiver parameters (e.g. Go methods) should
        override this to skip the receiver.
        """
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        for kw in self._def_keywords:
            if stripped.startswith(kw):
                rest = stripped[len(kw) :]
                # Name ends at '(' ':' or whitespace.
                for i, ch in enumerate(rest):
                    if ch in "((: \t":
                        return rest[:i] if i > 0 else None
                return rest.rstrip()
        return None

    # Subclasses must set this class attribute.
    _def_keywords: tuple[str, ...]

    # public interface — every subclass must implement these five methods

    @abstractmethod
    def get_file_outline(self, file: str) -> list[dict]:
        """Return every class/function/method defined in *file*."""
        ...

    @abstractmethod
    def goto_definition(
        self, file: str, line: int, name: Optional[str] = None
    ) -> Optional[dict]:
        """Resolve the symbol *name* on *line* of *file* to its definition."""
        ...

    @abstractmethod
    def get_callers(self, file: str, line: int) -> list[dict]:
        """Find every place in the project that references the symbol on *line* of *file*."""
        ...

    @abstractmethod
    def get_callees(self, file: str, line: int) -> list[dict]:
        """Find every symbol called by the function on *line* of *file*."""
        ...

    @abstractmethod
    def find_symbol(self, name: str) -> list[dict]:
        """Search for *name* across the project."""
        ...
