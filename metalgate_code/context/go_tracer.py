"""Go-specific tracer using tree-sitter-go and gopls LSP.

LSP communication is handled by
:class:`~metalgate_code.context.gopls_lsp_client.GoplsLspClient`.

``find_symbol`` uses LSP ``workspace/symbol`` when gopls is available, and
falls back to scanning cached tree-sitter outlines (exact, case-insensitive
match) when it is not.  For third-party symbols, use ``goto_definition`` from
a usage site.

Tree-sitter is used for:
  - ``get_source`` — line-based source extraction from scope nodes
  - ``get_file_outline`` — fast outline extraction (no LSP round-trip)
  - ``get_callees`` — finding call positions within a function body
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from metalgate_code.context.cache import _CACHE_MISS
from metalgate_code.context.gopls_lsp_client import GoplsLspClient
from metalgate_code.context.tracer_base import (
    _MAX_CALLERS,
    Tracer,
    TreeSitterConfig,
    _lsp_symbol_kind_to_str,
    _name_col_on_line,
    _path_to_uri,
    _uri_to_path,
)

logger = logging.getLogger("metalgate_code")

# Tree-sitter Go language — shared across all parses.
_TS_GO_LANGUAGE = Language(tsgo.language())

# Markers for Go stdlib source paths (GOROOT).  Used to skip outline lookups
# for stdlib definitions — they are noise and the files live outside the
# sandbox, so reading them via the backend would fail.
_GO_STDLIB_MARKERS = (
    "/libexec/src/",  # Homebrew: /opt/homebrew/Cellar/go/X/libexec/src/
    "/go/src/",  # Official installer: /usr/local/go/src/
    "/sdk/go*/src/",  # Multiple-version installs
)


def _is_go_stdlib_path(path: str) -> bool:
    """True if *path* points to the Go stdlib source tree (GOROOT)."""
    return any(marker in path for marker in _GO_STDLIB_MARKERS)


# Tree-sitter helpers
#
# All tree-sitter functions take raw bytes and return 1-based line numbers
# (matching LSP convention).  Column numbers are 0-based.


def _ts_col_for_name(source_bytes: bytes, line: int, name: str) -> Optional[int]:
    """Find the 0-based column of *name* on *line* (1-based) using tree-sitter.

    For selector expressions like ``log.New`` or ``c.Next``, returns the
    column of the **field** (the member after the last ``.``), not the
    qualifier.  This is the position gopls needs to resolve the member
    definition rather than the package or receiver.

    Falls back to ``_name_col_on_line`` (regex) if tree-sitter can't find
    the node (e.g. the name is inside a comment or string).
    """
    tree = Parser(_TS_GO_LANGUAGE).parse(source_bytes)
    root = tree.root_node

    is_selector = "." in name
    member = name.rsplit(".", 1)[-1] if is_selector else name

    def visit(node):
        # For selector expressions, find the field identifier.
        if is_selector and node.type == "selector_expression":
            field_node = node.child_by_field_name("field")
            if field_node is not None:
                node_line = field_node.start_point[0] + 1
                if node_line == line and field_node.text == member.encode():
                    return field_node.start_point[1]
            # Also try matching the field as a direct child identifier.
            for child in node.children:
                if (
                    child.type == "field_identifier"
                    and child.start_point[0] + 1 == line
                    and child.text == member.encode()
                ):
                    return child.start_point[1]
        # For plain identifiers, match the name directly.
        elif not is_selector and node.type == "identifier":
            node_line = node.start_point[0] + 1
            if node_line == line and node.text == name.encode():
                return node.start_point[1]
        for child in node.children:
            result = visit(child)
            if result is not None:
                return result
        return None

    col = visit(root)
    if col is not None:
        return col

    # Fallback to regex for edge cases (comments, strings, etc.)
    lines = source_bytes.decode("utf-8", errors="replace").splitlines()
    if 1 <= line <= len(lines):
        col = _name_col_on_line(lines[line - 1], name)
        if col is not None and is_selector:
            col = col + len(name) - len(member)
    return col


def _ts_go_collect_outline(node, result: list) -> None:
    """Recursively walk tree-sitter Go tree, appending dicts for every symbol."""
    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        if name_node is None:
            return

        name = name_node.text.decode("utf-8", errors="replace")
        param_str = (
            params_node.text.decode("utf-8", errors="replace") if params_node else "..."
        )

        result.append(
            {
                "name": name,
                "kind": "function",
                "class": None,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "signature": f"func {name}{param_str}",
            }
        )
        for child in node.children:
            _ts_go_collect_outline(child, result)

    elif node.type == "method_declaration":
        name_node = node.child_by_field_name("name")
        recv_node = node.child_by_field_name("receiver")
        params_node = node.child_by_field_name("parameters")
        if name_node is None:
            return

        name = name_node.text.decode("utf-8", errors="replace")
        recv_type = "..."
        if recv_node:
            recv_text = recv_node.text.decode("utf-8", errors="replace")
            recv_type = recv_text.strip("()")

        param_str = (
            params_node.text.decode("utf-8", errors="replace") if params_node else "..."
        )

        result.append(
            {
                "name": name,
                "kind": "method",
                "class": recv_type,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "signature": f"func ({recv_type}) {name}{param_str}",
            }
        )
        for child in node.children:
            _ts_go_collect_outline(child, result)

    elif node.type == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                if name_node and type_node:
                    kind = (
                        "struct"
                        if type_node.type == "struct_type"
                        else "interface"
                        if type_node.type == "interface_type"
                        else "type"
                    )
                    name = name_node.text.decode("utf-8", errors="replace")
                    result.append(
                        {
                            "name": name,
                            "kind": kind,
                            "class": None,
                            "line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "signature": f"type {name} {kind}",
                        }
                    )
                    for sub in type_node.children:
                        _ts_go_collect_outline(sub, result)

    else:
        for child in node.children:
            _ts_go_collect_outline(child, result)


def _parse_hover(hover: object) -> tuple[str, str]:
    """Extract (signature, docstring) from a gopls LSP hover response.

    gopls-specific post-processing (stripping pkg.go.dev links and ``---``
    separators) is applied before delegating to :func:`_parse_hover_base`.
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

    raw = str(value).strip()

    # gopls auto-generates a pkg.go.dev link for every symbol, even when
    # there is no doc comment.  Strip it first so it doesn't interfere
    # with code-fence detection below.
    raw = re.sub(
        r"\n*---\n*\[.*? on pkg\.go\.dev\]\(.*?\)\s*$",
        "",
        raw,
    ).strip()

    # Strip markdown code fences if present.
    # gopls wraps the signature in ```go ... ``` fences.  The closing
    # fence is NOT necessarily the last line — there may be docstring
    # text after it.  Remove the opening fence line, then find and
    # remove the closing fence line wherever it is.
    lines = raw.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        # Find the closing fence (a line that is just ```)
        for i, ln in enumerate(lines):
            if ln.strip() == "```":
                del lines[i]
                break
    if not lines:
        return "", ""

    signature = lines[0].strip()
    docstring = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    # Strip leading "---" separators that gopls inserts between the
    # signature fence and the docstring.
    docstring = re.sub(r"^(---\s*\n*)+", "", docstring).strip()

    return signature, docstring


class GoTracer(Tracer):
    """Go-specific tracer using tree-sitter-go and gopls LSP."""

    _ts_config = TreeSitterConfig(
        language=_TS_GO_LANGUAGE,
        function_kinds=("function_declaration", "method_declaration"),
        scope_kinds=(
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ),
        call_node_type="call_expression",
        member_node_type="selector_expression",
        member_field_name="field",
    )

    _def_keywords = ("func ", "type ")

    def __init__(
        self,
        root: str,
        backend,
        cache,
    ) -> None:
        super().__init__(root, backend, cache)
        self._lsp: Optional[GoplsLspClient] = None
        self._lsp_lock = threading.Lock()  # guards _lsp creation
        # Serializes all LSP requests.  gopls is single-threaded; concurrent
        # requests cause "content modified" errors.  RLock (not Lock) so
        # _resolve can call find_symbol while already holding the lock.
        self._lsp_request_lock = threading.RLock()

    # Path translation (sandbox ↔ host)
    #
    # gopls runs on the HOST as a local subprocess.  When the backend is
    # a MicrosandboxBackend, the agent passes sandbox paths like
    # ``/workspace/orders.go``.  These must be translated to host paths
    # (e.g. ``/Users/foo/project/orders.go``) before sending to gopls.
    # Conversely, gopls response URIs contain host paths that must be
    # translated back to sandbox paths for the agent.

    def _to_host_uri(self, file: str) -> str:
        """Create a ``file://`` URI using the host path (for gopls)."""
        return _path_to_uri(self._to_host_path(file))

    def _uri_to_sandbox_path(self, uri: str) -> str:
        """Convert a gopls ``file://`` URI to a sandbox path for the agent."""
        return self._to_guest_path(_uri_to_path(uri))

    def _uri_to_result_path(self, uri: str) -> str:
        """Convert a gopls ``file://`` URI to a sandbox path for the agent.

        gopls runs on the host and returns host paths in its URIs.  The
        agent works with sandbox paths, so we translate host → guest.
        """
        return self._uri_to_sandbox_path(uri)

    # LSP document lifecycle

    def _get_lsp(self) -> GoplsLspClient:
        """Get or lazily create the gopls LSP client (double-checked locking)."""
        if self._lsp is not None:
            return self._lsp

        with self._lsp_lock:
            if self._lsp is not None:
                return self._lsp

            root_uri = _path_to_uri(str(self.root))
            self._lsp = GoplsLspClient(root_uri, cwd=str(self.root))
            self._lsp.start()
            return self._lsp

    # Tracer interface

    def get_file_outline(self, file: str) -> list[dict]:
        """Parse *file* and return every func/method/struct/interface with
        name, kind, line, end_line, signature."""
        cached = self.cache.get_outline(file)
        if cached is not None:
            return cached

        try:
            source_bytes = self._read_file_bytes(file)
        except OSError:
            logger.warning("Failed to read %s for outline", file, exc_info=True)
            return []

        result = self._ts_outline(source_bytes, file)
        self.cache.set_outline(file, result)
        return result

    def _ts_outline(self, source_bytes: bytes, file: str) -> list[dict]:
        """Extract outline using tree-sitter (no LSP round-trip needed)."""
        tree = self._ts_parse(source_bytes)
        result: list[dict] = []
        _ts_go_collect_outline(tree.root_node, result)
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
        if cached is not _CACHE_MISS:
            return cached

        result = self._resolve(file, line, name)
        if result is not None:
            self.cache.set_definition(file, line, name, result)
        return result

    def get_callers(self, file: str, line: int) -> list[dict]:
        """Find every place in the project that **directly** calls the symbol
        on *line* of *file*.

        Uses LSP call hierarchy (prepareCallHierarchy + incomingCalls) for
        one level only — no transitive expansion.  Each result points at the
        actual call site, not the caller's definition.
        """
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
        uri = self._to_host_uri(file)

        with self._lsp_request_lock:
            self._did_open(lsp, uri, source)

            try:
                items = lsp.prepare_call_hierarchy(uri, line - 1, col)
            except Exception:
                logger.warning(
                    "LSP prepareCallHierarchy failed for %s:%d",
                    file,
                    line,
                    exc_info=True,
                )
                return []

            if not items:
                return []

            seen_sites: set[tuple[str, int]] = set()
            results: list[dict] = []

            for item in items:
                try:
                    incoming = lsp.incoming_calls(item)
                except Exception:
                    logger.warning(
                        "LSP incomingCalls failed for %s:%d",
                        file,
                        line,
                        exc_info=True,
                    )
                    continue

                for call in incoming:
                    from_item = call.get("from", {})
                    from_uri = from_item.get("uri", "")
                    if not from_uri:
                        continue
                    from_file = self._uri_to_sandbox_path(from_uri)

                    # Use fromRanges for the actual call site.
                    from_ranges = call.get("fromRanges", [])
                    if not from_ranges:
                        from_ranges = [from_item.get("range", {})]

                    for rng in from_ranges:
                        ref_line = rng.get("start", {}).get("line", 0) + 1

                        # Skip the original definition.
                        if from_file == file and ref_line == line:
                            continue

                        site_key = (from_file, ref_line)
                        if site_key in seen_sites:
                            continue
                        seen_sites.add(site_key)

                        # Find the enclosing function/method name.
                        caller_name = ""
                        try:
                            ref_outline = self.get_file_outline(from_file)
                            sym = self._find_symbol_at_line(ref_outline, ref_line)
                            if sym:
                                caller_name = sym["name"]
                        except Exception:
                            logger.warning(
                                "get_file_outline failed for %s",
                                from_file,
                                exc_info=True,
                            )

                        # Read the referencing line for context.
                        context_text = ""
                        try:
                            ref_source = self._read_file(
                                from_file, limit=max(ref_line, 1) + 1
                            )
                            ref_lines = ref_source.splitlines()
                            if 0 < ref_line <= len(ref_lines):
                                context_text = ref_lines[ref_line - 1].strip()
                        except OSError:
                            pass

                        results.append(
                            {
                                "file": from_file,
                                "line": ref_line,
                                "name": sym_name or "",
                                "caller": caller_name,
                                "context": context_text,
                            }
                        )
                        if len(results) >= _MAX_CALLERS:
                            break

                    if len(results) >= _MAX_CALLERS:
                        break

        return results

    def get_callees(self, file: str, line: int) -> list[dict]:
        """Find every symbol called by the function on *line* of *file*.

        Uses tree-sitter to find call positions within the function body,
        then resolves each to its definition via LSP textDocument/definition.
        Results are deduplicated by (file, line).
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
        uri = self._to_host_uri(file)
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

                    key = (d_file, d_line)
                    if key in seen:
                        continue
                    seen.add(key)

                    d_name = ""
                    d_kind = ""
                    d_sig = ""

                    if not _is_go_stdlib_path(d_file):
                        # Project file — look up name/kind/signature from
                        # the definition's tree-sitter outline.
                        try:
                            outline = self.get_file_outline(d_file)
                            sym = self._find_symbol_at_line(outline, d_line, exact=True)
                            if sym:
                                d_name = sym["name"]
                                d_kind = sym["kind"]
                                d_sig = sym.get("signature", "")
                        except Exception:
                            logger.warning(
                                "get_file_outline failed for %s",
                                d_file,
                                exc_info=True,
                            )

                    if not d_name:
                        # Stdlib, builtin, or outline miss — use hover at
                        # the definition position for name/kind/signature.
                        try:
                            hover = lsp.hover(d_uri, d_line - 1, d_col)
                            sig, _ = _parse_hover(hover)
                            if sig:
                                d_sig = sig
                                if sig.startswith("func ("):
                                    d_kind = "method"
                                    # Extract method name from "func (recv) Name(..."
                                    m = re.match(r"func\s*\([^)]*\)\s*(\w+)", sig)
                                    if m:
                                        d_name = m.group(1)
                                elif sig.startswith("func "):
                                    d_kind = "function"
                                    m = re.match(r"func\s+(\w+)", sig)
                                    if m:
                                        d_name = m.group(1)
                        except Exception:
                            logger.warning(
                                "hover failed for callee at %s:%d",
                                d_file,
                                d_line,
                                exc_info=True,
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
        """Search for *name* across the project.

        Uses LSP ``workspace/symbol`` when gopls is available.  Falls back
        to scanning cached tree-sitter outlines (exact, case-insensitive
        match) when gopls is not installed.
        """
        results = self._find_symbol_lsp(name)
        if results is not None and len(results) > 0:
            return results
        return self._find_symbol_ts(name)

    def _find_symbol_lsp(
        self, name: str, *, scoped: bool = True
    ) -> Optional[list[dict]]:
        """LSP-based symbol search, or None if gopls is unavailable.

        Results are filtered to exact, case-insensitive name matches.
        When *scoped* is True (default), results are further filtered to the
        project root directory — gopls ``workspace/symbol`` does fuzzy
        matching and indexes every loaded module (stdlib, module cache, etc.),
        so we must filter both by name and by path.

        When *scoped* is False, the project-root filter is skipped.  This is
        used by ``goto_definition`` as a fallback for selector expressions
        (e.g. ``log.New``) where the member lives in stdlib.
        """
        name_lower = name.lower()
        results: list[dict] = []
        seen: set[tuple] = set()
        root_prefix = str(self.root) + os.sep

        try:
            lsp = self._get_lsp()
            with self._lsp_request_lock:
                symbols = lsp.workspace_symbol(name)
            for sym in symbols:
                sym_name = sym.get("name", "")

                # Exact, case-insensitive match only — gopls does fuzzy matching.
                if sym_name.lower() != name_lower:
                    continue

                location = sym.get("location", {})
                sym_uri = location.get("uri", "")
                if not sym_uri:
                    continue
                sym_file = self._uri_to_sandbox_path(sym_uri)

                # Filter to the project root — workspace/symbol indexes every
                # module gopls has loaded, including other projects, stdlib,
                # and the Go module cache.
                if scoped and not sym_file.startswith(root_prefix):
                    continue

                sym_range = location.get("range", {})
                sym_line = sym_range.get("start", {}).get("line", 0) + 1

                key = (sym_file, sym_line)
                if key in seen:
                    continue
                seen.add(key)

                kind = _lsp_symbol_kind_to_str(sym.get("kind", 0))

                results.append(
                    {
                        "name": sym_name,
                        "kind": kind,
                        "file": sym_file,
                        "line": sym_line,
                    }
                )
        except FileNotFoundError:
            # gopls not installed — fall back to tree-sitter search.
            return None
        except Exception:
            logger.warning("workspace/symbol failed for %r", name, exc_info=True)
            return None

        return results

    def _find_symbol_ts(self, name: str) -> list[dict]:
        """Exact, case-insensitive symbol search via cached tree-sitter outlines."""
        name_lower = name.lower()
        results: list[dict] = []
        seen: set[tuple] = set()

        go_files = self._glob_go_files()
        for go_file in go_files:
            try:
                symbols = self.get_file_outline(str(go_file))
            except Exception:
                logger.warning(
                    "Failed to get outline for %s in find_symbol",
                    go_file,
                    exc_info=True,
                )
                continue
            for sym in symbols:
                if sym["name"].lower() != name_lower:
                    continue
                key = (sym["file"], sym["line"], sym["name"])
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "name": sym["name"],
                        "kind": sym["kind"],
                        "file": sym["file"],
                        "line": sym["line"],
                    }
                )

        return results

    def _glob_go_files(self) -> list[Path]:
        """Find all .go files under root using backend if available."""
        if self.backend is not None:
            result = self.backend.glob("**/*.go", path=str(self.root))
            if result.error is None and result.matches is not None:
                return [Path(m["path"]) for m in result.matches]
        return list(self.root.rglob("*.go"))

    # Private helpers

    def _resolve(self, file: str, line: int, name: str) -> Optional[dict]:
        """Resolve *name* at *line* in *file* to its definition via LSP.

        Uses tree-sitter to find the precise column of *name* on *line*.
        For selector expressions (``log.New``, ``c.Next``), the column
        targets the **field** (the member after the last ``.``), so gopls
        resolves the member definition rather than the package import or
        receiver variable.

        This is the same approach used by ``get_callees``: call
        ``textDocument/definition`` at the correct position and take the
        first result.  Hover is performed at the **definition** position
        (not the call site) to get the correct signature/docstring.

        Transient LSP failures (e.g. gopls still indexing on a cold start)
        are retried once after a short delay.
        """
        source = self._read_file(file)
        lines = source.splitlines()
        if line < 1 or line > len(lines):
            return None

        uri = self._to_host_uri(file)

        is_selector = "." in name
        member_name = name.rsplit(".", 1)[-1] if is_selector else name

        # Use tree-sitter to find the precise column — same approach
        # as get_callees.  For selectors, this targets the field node
        # so gopls resolves the member, not the qualifier/receiver.
        col = _ts_col_for_name(source.encode("utf-8", errors="replace"), line, name)
        if col is None:
            return None

        # _get_lsp() is inside the lock so gopls startup is fully serialized.
        # Without this, concurrent cold-start callers race on _get_lsp() and
        # the losing threads crash the server during initialization.
        with self._lsp_request_lock:
            lsp = self._get_lsp()
            self._did_open(lsp, uri, source)

            # Call LSP textDocument/definition at the precise column.
            # For selectors, the column is on the field (member), so
            # gopls resolves the function/method definition directly —
            # same mechanism get_callees uses successfully.
            # Retry once: gopls may return empty while it is still indexing
            # on a cold start.
            defs = None
            for attempt in (1, 2):
                try:
                    defs = lsp.definition(uri, line - 1, col)
                except Exception:
                    if attempt == 2:
                        logger.warning(
                            "LSP definition failed at %s:%d:%d",
                            file,
                            line,
                            col,
                            exc_info=True,
                        )
                        return None
                    time.sleep(0.5)
                    continue
                if defs:
                    break
                if attempt == 1:
                    time.sleep(0.5)

            if not defs:
                return None
            if isinstance(defs, dict):
                defs = [defs]

            info = self._extract_def_info(defs[0])
            if info is None:
                return None
            d_file, d_line, d_col, d_uri = info

            # Get hover info at the **call site** position — not the
            # definition position.  For stdlib symbols the definition file
            # lives outside the workspace, so gopls cannot hover there.
            # Hovering at the call site (the position we just resolved from)
            # returns the same signature/docstring and works for both
            # project and stdlib symbols.
            signature = ""
            docstring = ""
            try:
                hover = lsp.hover(uri, line - 1, col)
                signature, docstring = _parse_hover(hover)
            except Exception:
                logger.warning(
                    "hover failed at %s:%d:%d",
                    file,
                    line,
                    col,
                    exc_info=True,
                )

        # Determine kind from the definition file's outline.
        # Skip outline lookup for stdlib — those files live outside the
        # sandbox and can't be read via the backend.  For stdlib, infer
        # the kind from the hover signature instead.
        kind = "unknown"
        if d_file and not _is_go_stdlib_path(d_file):
            try:
                outline = self.get_file_outline(d_file)
                sym = self._find_symbol_at_line(outline, d_line, exact=True)
                if sym:
                    kind = sym["kind"]
                    if not signature:
                        signature = sym.get("signature", "")
            except Exception:
                logger.warning("get_file_outline failed for %s", d_file, exc_info=True)

        # Fallback / override: infer kind from the hover signature when
        # the outline lookup missed, returned "unknown", or returned a
        # container type ("interface"/"struct"/"type") that doesn't match
        # the actual symbol being resolved.  This happens for interface
        # methods: gopls points at the ``type X interface`` line, so the
        # outline returns kind="interface", but the hover signature is
        # the method signature (e.g. "Render(http.ResponseWriter) error").
        if kind in ("unknown", "interface", "struct", "type") and signature:
            if signature.startswith("func ("):
                kind = "method"
            elif signature.startswith("func "):
                kind = "function"
            elif not signature.startswith(("type ", "var ", "const ", "package ")):
                # Interface method signatures from gopls hover don't have
                # a "func" prefix, e.g. "Render(http.ResponseWriter) error".
                # If the signature doesn't look like a type/var/const/package
                # declaration, treat it as a method.
                kind = "method"

        return {
            "name": member_name,
            "kind": kind,
            "file": d_file,
            "line": d_line,
            "col": col + 1,  # call-site column (1-based, member position)
            "signature": signature,
            "docstring": docstring,
        }

    # Go-specific overrides for definition-line parsing

    def _def_name_col_from_lines(self, lines: list[str], line: int) -> Optional[int]:
        """Column of the name token on a func/type line.

        For ``func foo()`` returns the column of ``foo``.  For methods,
        skips the receiver: ``func (r *Type) Name()`` → column of ``Name``.
        Returns None if the line is not a func/type definition.
        """
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        for kw in ("func ", "type "):
            if stripped.startswith(kw):
                rest = stripped[len(kw) :]
                prefix = len(kw)
                # For methods, skip receiver: func (r *Type) Name(...)
                if rest.startswith("("):
                    # Count nested parentheses to handle func (f func()) Name()
                    depth = 1
                    close = 1
                    while close < len(rest) and depth > 0:
                        if rest[close] == "(":
                            depth += 1
                        elif rest[close] == ")":
                            depth -= 1
                        close += 1
                    prefix += close
                    rest = rest[close:]
                # Skip whitespace before name
                ws = len(rest) - len(rest.lstrip())
                prefix += ws
                rest = rest.lstrip()
                name_match = re.match(r"(\w+)", rest)
                if name_match:
                    return indent + prefix + name_match.start()
        return None

    def _def_name_from_lines(self, lines: list[str], line: int) -> Optional[str]:
        """Extract the symbol name from a func/type line.

        For ``func foo(x int)`` returns ``foo``.  For methods, skips the
        receiver.  Returns None if the line is not a func/type definition.
        """
        if line < 1 or line > len(lines):
            return None
        raw = lines[line - 1]
        stripped = raw.lstrip()
        for kw in ("func ", "type "):
            if stripped.startswith(kw):
                rest = stripped[len(kw) :]
                # For methods, skip receiver: func (r *Type) Name(...)
                if rest.startswith("("):
                    depth = 1
                    close = 1
                    while close < len(rest) and depth > 0:
                        if rest[close] == "(":
                            depth += 1
                        elif rest[close] == ")":
                            depth -= 1
                        close += 1
                    rest = rest[close:]
                rest = rest.lstrip()
                # Name ends at '(' or whitespace.
                name_match = re.match(r"(\w+)", rest)
                if name_match:
                    return name_match.group(1)
        return None
