"""Go-specific tracer using tree-sitter-go and gopls CLI."""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from metalgate_code.context.cache import _CACHE_MISS, CodeCache
from metalgate_code.context.tracer_base import Tracer, _CALLERS_TIMEOUT, _MAX_CALLERS

logger = logging.getLogger("metalgate_code")

_TS_GO_LANGUAGE = Language(tsgo.language())
_TS_GO_PARSER = Parser(_TS_GO_LANGUAGE)


def _ts_go_extract_symbols(source_bytes: bytes, file: str) -> list[dict]:
    """Extract function/method/struct/interface names from Go source using tree-sitter."""
    tree = _TS_GO_PARSER.parse(source_bytes)
    root = tree.root_node
    results: list[dict] = []

    def walk(node, parent_type: str | None = None):
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                results.append(
                    {
                        "name": name_node.text.decode("utf-8", errors="replace"),
                        "kind": "function",
                        "file": file,
                        "line": name_node.start_point[0] + 1,
                    }
                )
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                results.append(
                    {
                        "name": name_node.text.decode("utf-8", errors="replace"),
                        "kind": "method",
                        "file": file,
                        "line": name_node.start_point[0] + 1,
                    }
                )
        elif node.type == "type_declaration":
            # type_declaration contains type_spec which has name and type
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
                        results.append(
                            {
                                "name": name_node.text.decode("utf-8", errors="replace"),
                                "kind": kind,
                                "file": file,
                                "line": name_node.start_point[0] + 1,
                            }
                        )
        for child in node.children:
            walk(child, parent_type)

    walk(root)
    return results


def _ts_go_collect_outline(node, result: list, parent_struct: str | None = None) -> None:
    """Recursively walk tree-sitter Go tree, appending dicts for every symbol."""
    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        if name_node is None:
            return

        param_str = "..."
        if params_node:
            param_str = params_node.text.decode("utf-8", errors="replace")

        result.append(
            {
                "name": name_node.text.decode("utf-8", errors="replace"),
                "kind": "function",
                "class": None,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "signature": f"func {name_node.text.decode('utf-8', errors='replace')}{param_str}",
            }
        )
        for child in node.children:
            _ts_go_collect_outline(child, result, parent_struct)

    elif node.type == "method_declaration":
        name_node = node.child_by_field_name("name")
        recv_node = node.child_by_field_name("receiver")
        params_node = node.child_by_field_name("parameters")
        if name_node is None:
            return

        recv_type = "..."
        if recv_node:
            recv_text = recv_node.text.decode("utf-8", errors="replace")
            recv_type = recv_text.strip("()")

        param_str = "..."
        if params_node:
            param_str = params_node.text.decode("utf-8", errors="replace")

        result.append(
            {
                "name": name_node.text.decode("utf-8", errors="replace"),
                "kind": "method",
                "class": recv_type,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "signature": (
                    f"func ({recv_type}) {name_node.text.decode('utf-8', errors='replace')}"
                    f"{param_str}"
                ),
            }
        )
        for child in node.children:
            _ts_go_collect_outline(child, result, parent_struct)

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
                    result.append(
                        {
                            "name": name_node.text.decode("utf-8", errors="replace"),
                            "kind": kind,
                            "class": None,
                            "line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "signature": f"type {name_node.text.decode('utf-8', errors='replace')} {kind}",
                        }
                    )
                    for sub in type_node.children:
                        _ts_go_collect_outline(sub, result, name_node.text.decode("utf-8", errors="replace"))

    else:
        for child in node.children:
            _ts_go_collect_outline(child, result, parent_struct)


def _ts_go_find_function_at(root_node, line: int):
    """Return the innermost function/method node whose body contains *line* (1-based)."""
    best = None
    best_size = None

    def visit(node):
        nonlocal best, best_size
        if node.type in ("function_declaration", "method_declaration"):
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            if start <= line <= end:
                size = end - start
                if best is None or size < best_size:
                    best = node
                    best_size = size
        for child in node.children:
            visit(child)

    visit(root_node)
    return best


def _ts_go_find_scope_at_line(root_node, line: int):
    """Return the tightest function/method/struct/interface node whose definition line == *line* (1-based)."""
    result = None
    best_size = None

    def visit(node):
        nonlocal result, best_size
        if node.type in ("function_declaration", "method_declaration", "type_declaration"):
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            if start == line:
                size = end - start
                if result is None or size < best_size:
                    result = node
                    best_size = size
        for child in node.children:
            visit(child)

    visit(root_node)
    return result


def _gopls_cmd(
    subcommand: str,
    file: str,
    line: int,
    col: int,
    cwd: str | None = None,
    timeout: float = _CALLERS_TIMEOUT,
) -> list[dict]:
    """Run a gopls CLI subcommand and return parsed results.

    ``definition`` supports ``-json`` (gopls v0.22+).  ``references`` and
    ``call_hierarchy`` return plain text, so we parse that instead.
    """
    loc = f"{file}:{line}:{col}"
    use_json = subcommand == "definition"
    cmd = ["gopls", subcommand, "-json", loc] if use_json else ["gopls", subcommand, loc]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except FileNotFoundError:
        logger.warning("gopls not found in PATH")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"gopls {subcommand} timed out after {timeout}s")
        return []
    if proc.returncode != 0:
        logger.debug(f"gopls {subcommand} failed: {proc.stderr}")
        return []

    if use_json:
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
    return _parse_gopls_text(subcommand, proc.stdout)


def _parse_gopls_text(subcommand: str, text: str) -> list[dict]:
    """Parse plain-text gopls output into a list of normalised dicts."""
    results: list[dict] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue

        if subcommand == "references":
            # /path/file.go:line:col-endcol   or   /path/file.go:line:col
            m = re.match(r"^(.+):(\d+):(\d+)(?:-\d+)?$", line)
            if m:
                results.append(
                    {
                        "file": m.group(1),
                        "line": int(m.group(2)),
                        "col": int(m.group(3)),
                        "name": "",
                    }
                )

        elif subcommand == "call_hierarchy":
            # caller[N]: ranges L:C-EC in FILE from/to function NAME in FILE:L:C-EC
            if line.startswith("caller"):
                func_m = re.search(r"function\s+(\w+)\s+in\s+(.+?):\d+:\d+", line)
                range_m = re.search(
                    r"ranges\s+(\d+):(\d+)-\d+\s+in\s+(.+?)\s+from/to", line
                )
                if func_m and range_m:
                    results.append(
                        {
                            "file": range_m.group(3),
                            "line": int(range_m.group(1)),
                            "col": int(range_m.group(2)),
                            "name": func_m.group(1),
                        }
                    )
    return results


def _gopls_item_to_dict(item: dict) -> dict | None:
    """Normalize a gopls result (JSON or plain-text) into our standard dict format."""
    # Already normalized (from plain-text parser) — ensure all expected keys
    if item.get("file"):
        return {
            "name": item.get("name", ""),
            "kind": item.get("kind", ""),
            "file": item["file"],
            "line": item.get("line", 0),
            "col": item.get("col", 0),
            "signature": item.get("signature", ""),
        }

    # JSON format from gopls definition
    span = item.get("span", {})
    uri = span.get("uri", "")
    # Strip file:// prefix if present
    if uri.startswith("file://"):
        uri = uri[7:]
    start = span.get("start", {})
    line = start.get("line", 0) + 1  # gopls uses 0-based lines
    col = start.get("column", 0)

    description = item.get("description", "")
    name = item.get("name", "")
    kind = item.get("kind", "")

    # gopls definition JSON doesn't include name/kind fields; parse from description
    if not name and description:
        m = re.match(r"^(?:func|type|var|const)\s+(?:\([^)]+\)\s+)?(\w+)", description)
        if m:
            name = m.group(1)

    if not kind and description:
        if description.startswith("func "):
            kind = "function"
        elif description.startswith("type "):
            kind = "struct" if "struct" in description else "type"
        elif description.startswith("var "):
            kind = "var"
        elif description.startswith("const "):
            kind = "const"

    signature = description.split("\n")[0] if description else ""

    return {
        "name": name,
        "kind": kind,
        "file": uri,
        "line": line,
        "col": col,
        "signature": signature,
    }


class GoTracer(Tracer):
    """Go-specific tracer using tree-sitter-go and gopls CLI."""

    def __init__(
        self,
        root: str,
        backend,
        cache: CodeCache,
    ) -> None:
        super().__init__(root, backend, cache)

    def _glob_go_files(self) -> list[Path]:
        """Find all .go files under root using backend if available."""
        if self.backend is not None:
            result = self.backend.glob("**/*.go", path=str(self.root))
            if result.error is None and result.matches is not None:
                return [Path(m["path"]) for m in result.matches]
        return list(self.root.rglob("*.go"))

    def get_file_outline(self, file: str) -> list[dict]:
        """Parse *file* and return every func/method/struct/interface with name, kind, line, end_line, signature."""
        cached = self.cache.get_outline(file)
        if cached is not None:
            return cached

        try:
            source_bytes = self._read_file_bytes(file)
            tree = _TS_GO_PARSER.parse(source_bytes)
        except Exception:
            return []

        result: list[dict] = []
        _ts_go_collect_outline(tree.root_node, result)

        for sym in result:
            sym["file"] = file

        self.cache.set_outline(file, result)
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
        """Return the full source of the function/method/struct/interface starting on *line*."""
        try:
            source = self._read_file(file)
            all_lines = source.splitlines()

            source_bytes = source.encode("utf-8", errors="ignore")
            tree = _TS_GO_PARSER.parse(source_bytes)
            node = _ts_go_find_scope_at_line(tree.root_node, line)

            if node:
                start = node.start_point[0]
                end = node.end_point[0] + 1
            else:
                # Fallback: return *context* lines centred on *line* (1-based).
                centre = line - 1  # convert to 0-based index
                start = max(0, centre - context // 2)
                end = min(len(all_lines), centre + (context + 1) // 2)

            snippet = all_lines[start:end]
            return {
                "file": file,
                "start_line": start + 1,
                "end_line": end,
                "source": "\n".join(snippet),
            }
        except Exception as exc:
            return {
                "file": file,
                "start_line": 0,
                "end_line": 0,
                "source": "",
                "error": str(exc),
            }

    def get_callers(self, file: str, line: int) -> list[dict]:
        """Find every place in the project that references the symbol on *line* of *file*."""
        col = self._def_name_col(file, line)
        if col is None:
            return []

        # gopls expects 1-based columns; _def_name_col returns 0-based
        items = _gopls_cmd("call_hierarchy", file, line, col + 1, cwd=str(self.root))
        results = []
        for item in items:
            normalized = _gopls_item_to_dict(item)
            if normalized is None:
                continue
            ref_file = normalized["file"]
            ref_line = normalized["line"]
            if ref_file == file and ref_line == line:
                continue

            # Find the innermost enclosing scope at the reference line
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
                pass

            results.append(
                {
                    "file": ref_file,
                    "line": ref_line,
                    "name": normalized["name"],
                    "caller": caller_name,
                    "context": normalized["signature"],
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

        source_bytes = source.encode("utf-8", errors="ignore")
        tree = _TS_GO_PARSER.parse(source_bytes)
        func_node = _ts_go_find_function_at(tree.root_node, line)
        if func_node is None:
            return []

        start_line = func_node.start_point[0] + 1
        end_line = func_node.end_point[0] + 1

        # Find call expressions within the function body
        call_positions = self._find_call_positions(source, start_line, end_line)

        results: list[dict] = []
        seen: set[tuple] = set()

        for call_line, call_col in call_positions:
            # tree-sitter columns are 0-based; gopls expects 1-based
            items = _gopls_cmd(
                "definition",
                file,
                call_line,
                call_col + 1,
                cwd=str(self.root),
            )
            for item in items:
                normalized = _gopls_item_to_dict(item)
                if normalized is None:
                    continue
                key = (normalized["file"], normalized["line"])
                if key in seen:
                    continue
                seen.add(key)
                results.append(normalized)

        return results

    def find_symbol(self, name: str) -> list[dict]:
        """Search for *name* across the project."""
        return self._exact_ts_search(name)

    def _exact_ts_search(self, name: str) -> list[dict]:
        """Exact symbol search using tree-sitter across all .go files."""
        name_lower = name.lower()
        results: list[dict] = []
        seen: set[tuple] = set()

        go_files = self._glob_go_files()
        for go_file in go_files:
            try:
                source_bytes = self._read_file_bytes(str(go_file))
                symbols = _ts_go_extract_symbols(source_bytes, str(go_file))
            except (OSError, IOError):
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

    def _resolve(self, file: str, line: int, name: str) -> Optional[dict]:
        col = self._name_col_on_line(file, line, name)
        if col is None:
            return None

        # gopls expects 1-based columns; _name_col_on_line returns 0-based
        items = _gopls_cmd("definition", file, line, col + 1, cwd=str(self.root))
        if not items:
            return None

        normalized = _gopls_item_to_dict(items[0])
        if normalized is None:
            return None

        # Try to fetch docstring via gopls hover
        hover_items = _gopls_cmd("hover", file, line, col, cwd=str(self.root))
        if hover_items:
            hover = hover_items[0]
            content = hover.get("contents", {})
            if isinstance(content, dict):
                normalized["docstring"] = content.get("value", "")
            elif isinstance(content, str):
                normalized["docstring"] = content
            else:
                normalized["docstring"] = ""
        else:
            normalized["docstring"] = ""

        return normalized

    def _first_name_on_line(self, file: str, line: int) -> Optional[str]:
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            text = lines[line - 1]
            # Find first identifier-like token
            for m in re.finditer(r"\b[a-zA-Z_]\w*\b", text):
                return m.group()
        except Exception:
            pass
        return None

    def _def_name_col(self, file: str, line: int) -> Optional[int]:
        """Column of the name token on a func/method/type line."""
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            raw = lines[line - 1]
            stripped = raw.lstrip()
            indent = len(raw) - len(stripped)
            for kw in ("func ", "type "):
                if stripped.startswith(kw):
                    # Find the name after the keyword
                    rest = stripped[len(kw):]
                    # For methods, skip receiver: func (r *Type) Name(...)
                    if rest.startswith("("):
                        close = rest.find(")")
                        if close != -1:
                            rest = rest[close + 1:].lstrip()
                    name_match = re.match(r"(\w+)", rest)
                    if name_match:
                        return indent + len(kw) + (len(rest) - len(rest.lstrip())) + name_match.start()
        except Exception:
            pass
        return None

    def _name_col_on_line(self, file: str, line: int, name: str) -> Optional[int]:
        """First column where *name* appears as a whole word on *line* of *file*."""
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            text = lines[line - 1]
            for m in re.finditer(rf"\b{re.escape(name)}\b", text):
                return m.start()
        except Exception:
            pass
        return None

    def _find_call_positions(
        self, source: str, start_line: int, end_line: int
    ) -> list[tuple[int, int]]:
        """Return (line, col) of every call expression in [start_line, end_line]."""
        positions: list[tuple[int, int]] = []
        try:
            source_bytes = source.encode("utf-8", errors="ignore")
            tree = _TS_GO_PARSER.parse(source_bytes)
        except Exception:
            return positions

        def visit(node):
            if node.type == "call_expression":
                func_node = node.child_by_field_name("function")
                if func_node:
                    line = func_node.start_point[0] + 1
                    col = func_node.start_point[1]
                    if start_line <= line <= end_line:
                        positions.append((line, col))
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return positions
