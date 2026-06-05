"""Core resolution engine using parso, jedi, and tree-sitter."""

import concurrent.futures
import io
import logging
import os
import re
import tokenize
from pathlib import Path
from typing import Optional

import jedi
import parso
import tree_sitter_python as tspython
from deepagents.backends.protocol import SandboxBackendProtocol
from tree_sitter import Language, Parser

from metalgate_code.context.cache import _CACHE_MISS, CodeCache

_FUNC_TYPES = {"funcdef", "async_funcdef"}
_CLASS_TYPE = "classdef"
_SCOPE_TYPES = _FUNC_TYPES | {_CLASS_TYPE}

_PYCACHE = "__pycache__"
_VENV_DIR = ".venv"
_MAX_CALLERS = 50
_CALLERS_TIMEOUT = 15.0
_CALLERS_WORKERS = os.environ.get("CALLERS_WORKERS", 4)

logger = logging.getLogger("metalgate_code")

# Tree-sitter parser for fast exact symbol search
_TS_LANGUAGE = Language(tspython.language())
_TS_PARSER = Parser(_TS_LANGUAGE)


def _ts_extract_symbols(source_bytes: bytes, file: str) -> list[dict]:
    """Extract function/class names from source using tree-sitter."""
    tree = _TS_PARSER.parse(source_bytes)
    root = tree.root_node
    results: list[dict] = []

    def walk(node):
        if node.type == "function_definition":
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
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                results.append(
                    {
                        "name": name_node.text.decode("utf-8", errors="replace"),
                        "kind": "class",
                        "file": file,
                        "line": name_node.start_point[0] + 1,
                    }
                )
        for child in node.children:
            walk(child)

    walk(root)
    return results


def _iter_children(node):
    if hasattr(node, "children"):
        yield from node.children


def _collect_outline(node, result: list, parent_class: Optional[str] = None) -> None:
    """Recursively walk parso tree, appending dicts for every class/function."""
    node_type = getattr(node, "type", None)

    if node_type in _FUNC_TYPES:
        inner = node
        if node_type == "async_funcdef":
            for child in _iter_children(node):
                if getattr(child, "type", None) == "funcdef":
                    inner = child
                    break

        name_node = getattr(inner, "name", None)
        if name_node is None:
            return

        try:
            params = inner.get_params()
            param_str = ", ".join(p.name.value for p in params if hasattr(p, "name"))
        except Exception:
            param_str = "..."

        prefix = "async def " if node_type == "async_funcdef" else "def "
        result.append(
            {
                "name": name_node.value,
                "kind": "method" if parent_class else "function",
                "class": parent_class,
                "line": inner.start_pos[0],
                "end_line": inner.end_pos[0],
                "signature": f"{prefix}{name_node.value}({param_str})",
            }
        )
        for child in _iter_children(inner):
            _collect_outline(child, result, parent_class)

    elif node_type == _CLASS_TYPE:
        name_node = getattr(node, "name", None)
        if name_node is None:
            return

        bases = ""
        try:
            in_parens = False
            for child in _iter_children(node):
                if getattr(child, "value", None) == "(":
                    in_parens = True
                elif getattr(child, "value", None) == ")":
                    break
                elif in_parens:
                    bases = child.get_code().strip()
                    break
        except Exception:
            pass

        result.append(
            {
                "name": name_node.value,
                "kind": "class",
                "class": None,
                "line": node.start_pos[0],
                "end_line": node.end_pos[0],
                "signature": (
                    f"class {name_node.value}({bases})"
                    if bases
                    else f"class {name_node.value}"
                ),
            }
        )
        for child in _iter_children(node):
            _collect_outline(child, result, name_node.value)

    else:
        for child in _iter_children(node):
            _collect_outline(child, result, parent_class)


def _find_function_at(module, line: int):
    """Return the innermost funcdef node whose body contains `line`."""
    best = None
    best_size = None

    def visit(node):
        nonlocal best, best_size
        if getattr(node, "type", None) in _FUNC_TYPES:
            inner = node
            if node.type == "async_funcdef":
                for child in _iter_children(node):
                    if getattr(child, "type", None) == "funcdef":
                        inner = child
                        break
            start, end = inner.start_pos[0], inner.end_pos[0]
            if start <= line <= end:
                size = end - start
                if best is None or size < best_size:
                    best = inner
                    best_size = size
        for child in _iter_children(node):
            visit(child)

    visit(module)
    return best


def _find_scope_at_line(module, line: int):
    """Return the tightest function or class node whose definition line == `line`."""
    result = None
    best_size = None

    def visit(node):
        nonlocal result, best_size
        if getattr(node, "type", None) in _SCOPE_TYPES:
            inner = node
            if node.type == "async_funcdef":
                for child in _iter_children(node):
                    if getattr(child, "type", None) == "funcdef":
                        inner = child
                        break
            start, end = inner.start_pos[0], inner.end_pos[0]
            if start == line:
                size = end - start
                if result is None or size < best_size:
                    result = inner
                    best_size = size
        for child in _iter_children(node):
            visit(child)

    visit(module)
    return result


def _call_positions(
    source: str, start_line: int, end_line: int, func_name: str | None = None
) -> list[tuple[int, int]]:
    """Return (line, col) of every NAME token followed by '(' in [start_line, end_line].

    If `func_name` is given, skip the token when it is `func_name` on the
    function's own definition line (avoids treating ``def foo(...):`` as a
    call to ``foo``).
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

    for i, tok in enumerate(tokens):
        if tok.type != tokenize.NAME:
            continue
        tok_line = tok.start[0]
        if not (start_line <= tok_line <= end_line):
            continue
        # Skip ``def func_name(...):`` false positive
        if func_name is not None and tok.string == func_name and tok_line == start_line:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].type in _skip:
            j += 1
        if j < len(tokens) and tokens[j].string == "(":
            positions.append((tok.start[0], tok.start[1]))

    return positions


def _name_col_on_line(line_text: str, name: str) -> Optional[int]:
    """First column where `name` appears as a whole word."""
    for m in re.finditer(rf"\b{re.escape(name)}\b", line_text):
        return m.start()
    return None


class Tracer:
    def __init__(
        self,
        root: str,
        backend: SandboxBackendProtocol,
        cache: CodeCache,
    ) -> None:
        self.root = Path(root).resolve()
        self.cache = cache
        self.backend = backend

        result = backend.execute("uv run which python")
        if result.exit_code is not None and result.exit_code == 0:
            python = result.output.strip()
        else:
            result = backend.execute("which python")
            if result.exit_code is not None and result.exit_code == 0:
                python = result.output.strip()
            else:
                python = "python3"

        if python:
            logger.info(f"python: {python}")
        else:
            logger.info("Warning: python detection failed")

        self.project = jedi.Project(
            path=str(self.root),
            environment_path=python,
        )

    def _read_file(self, file: str, limit: int = 10000) -> str:
        """Read file content using backend if available, otherwise use local filesystem."""
        if self.backend is not None:
            result = self.backend.read(file, offset=0, limit=limit)
            if result.error is None and result.file_data is not None:
                return result.file_data["content"]
        return Path(file).read_text(encoding="utf-8", errors="ignore")

    def _read_file_bytes(self, file: str, limit: int = 10000) -> bytes:
        """Read file content as bytes using backend if available."""
        return self._read_file(file, limit=limit).encode("utf-8", errors="ignore")

    def _glob_py_files(self) -> list[Path]:
        """Find all .py files under root using backend if available."""
        if self.backend is not None:
            result = self.backend.glob("**/*.py", path=str(self.root))
            if result.error is None and result.matches is not None:
                return [Path(m["path"]) for m in result.matches]
        return list(self.root.rglob("*.py"))

    def get_file_outline(self, file: str) -> list[dict]:
        """Parse `file` and return every class/function/method with name, kind, line, end_line, signature."""
        cached = self.cache.get_outline(file)
        if cached is not None:
            return cached

        try:
            source = self._read_file(file)
            module = parso.parse(source)
        except Exception:
            return []

        result: list[dict] = []
        _collect_outline(module, result)

        for sym in result:
            sym["file"] = file

        self.cache.set_outline(file, result)
        return result

    def goto_definition(
        self, file: str, line: int, name: Optional[str] = None
    ) -> Optional[dict]:
        """Resolve the symbol `name` on `line` of `file` to its definition."""
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
        """Return the full source of the function/class starting on `line`."""
        try:
            source = self._read_file(file)
            all_lines = source.splitlines()

            module = parso.parse(source)
            node = _find_scope_at_line(module, line)

            if node:
                # parso line numbers are 1-based; slice indices are 0-based
                start = node.start_pos[0] - 1
                end = node.end_pos[0]
            else:
                # Fallback: return `context` lines centred on `line` (1-based).
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

    def get_callers(
        self, file: str, line: int, timeout: float = _CALLERS_TIMEOUT
    ) -> list[dict]:
        """Find every place in the project that references the symbol on `line` of `file`."""
        col = self._def_name_col(file, line)
        if col is None:
            return []

        script = self._script(file)

        def _run():
            return script.get_references(line, col, include_builtins=False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                refs = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                return [
                    {
                        "file": "",
                        "line": 0,
                        "name": "",
                        "context": (
                            f"get_callers timed out after {timeout}s. "
                            "Try narrowing the search scope."
                        ),
                    }
                ]
            except Exception:
                return []

        results = []
        for r in refs:
            if r.module_path is None:
                continue
            ref_file = str(r.module_path)
            if ref_file == file and r.line == line:
                continue

            # Find the innermost enclosing scope (function/class) at the reference line
            caller_name = ""
            try:
                ref_outline = self.get_file_outline(ref_file)
                best = None
                best_size = float("inf")
                for sym in ref_outline:
                    if sym["line"] <= r.line <= sym["end_line"]:
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
                    "line": r.line,
                    "name": r.name,
                    "caller": caller_name,
                    "context": r.description,
                }
            )
            if len(results) >= _MAX_CALLERS:
                break

        return results

    def get_callees(self, file: str, line: int) -> list[dict]:
        """Find every symbol called by the function on `line` of `file`, resolved to definitions."""
        try:
            source = self._read_file(file)
        except OSError:
            return []

        module = parso.parse(source)
        func_node = _find_function_at(module, line)
        if func_node is None:
            return []

        start_line = func_node.start_pos[0]
        end_line = func_node.end_pos[0]
        func_name = getattr(func_node, "name", None)
        func_name_str = func_name.value if func_name else None
        positions = _call_positions(
            source, start_line, end_line, func_name=func_name_str
        )

        script = self._script(file, source=source)
        results: list[dict] = []
        seen: set[tuple] = set()

        for call_line, call_col in positions:
            try:
                defs = script.goto(call_line, call_col, follow_imports=True)
                if not defs:
                    continue
                d = defs[0]
                if d.module_path is None:
                    continue
                key = (str(d.module_path), d.line)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "name": d.name,
                        "kind": str(d.type),
                        "file": str(d.module_path),
                        "line": d.line,
                        "signature": d.description,
                    }
                )
            except Exception:
                continue

        return results

    def find_symbol(self, name: str) -> list[dict]:
        """Search for `name` across the project and installed packages."""
        return self._exact_ts_search(name)

    def _exact_ts_search(self, name: str) -> list[dict]:
        """Exact symbol search using tree-sitter — covers project and site-packages.

        Project files are parsed exhaustively.  Venv files are pre-filtered by
        a fast text scan (read first 50 KB, look for ``def name`` or
        ``class name``) so we only tree-sitter-parse candidates.
        """
        name_lower = name.lower()
        results: list[dict] = []
        seen: set[tuple] = set()

        all_py_files = self._glob_py_files()
        project_files = [
            f
            for f in all_py_files
            if _PYCACHE not in f.parts and _VENV_DIR not in f.parts
        ]
        venv_files = [
            f for f in all_py_files if _PYCACHE not in f.parts and _VENV_DIR in f.parts
        ]

        # --- project files: parse everything --------------------------------
        for py_file in project_files:
            try:
                source_bytes = self._read_file_bytes(str(py_file))
                symbols = _ts_extract_symbols(source_bytes, str(py_file))
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

        # --- venv files: pre-filter with fast text scan ---------------------
        # Build byte patterns for ``def <name>`` and ``class <name>``
        name_bytes = name.encode("utf-8")
        patterns = (b"def " + name_bytes, b"class " + name_bytes)
        venv_candidates: list[Path] = []
        for py_file in venv_files:
            try:
                chunk = self._read_file_bytes(str(py_file), limit=50_000)
                if any(p in chunk for p in patterns):
                    venv_candidates.append(py_file)
            except (OSError, IOError):
                continue

        for py_file in venv_candidates:
            try:
                source_bytes = self._read_file_bytes(str(py_file))
                symbols = _ts_extract_symbols(source_bytes, str(py_file))
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

    def _script(self, file: str, source: Optional[str] = None) -> jedi.Script:
        return jedi.Script(code=source, path=file, project=self.project)

    def _resolve(self, file: str, line: int, name: str) -> Optional[dict]:
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            col = _name_col_on_line(lines[line - 1], name)
            if col is None:
                return None

            script = self._script(file, source=source)
            defs = script.goto(line, col, follow_imports=True)
            if not defs:
                return None

            d = defs[0]
            return {
                "name": d.name,
                "kind": str(d.type),
                "file": str(d.module_path) if d.module_path else None,
                "line": d.line,
                "col": d.column,
                "signature": d.description,
                "docstring": d.docstring(fast=False),
            }
        except Exception:
            return None

    def _first_name_on_line(self, file: str, line: int) -> Optional[str]:
        try:
            source = self._read_file(file)
            script = self._script(file, source=source)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None

            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
            for tok in tokens:
                if tok.type == tokenize.NAME and tok.start[0] == line:
                    defs = script.goto(tok.start[0], tok.start[1], follow_imports=True)
                    if defs and defs[0].module_path:
                        return tok.string
        except Exception:
            pass
        return None

    def _def_name_col(self, file: str, line: int) -> Optional[int]:
        """Column of the name token on a def/class line."""
        try:
            source = self._read_file(file)
            lines = source.splitlines()
            if line < 1 or line > len(lines):
                return None
            raw = lines[line - 1]
            stripped = raw.lstrip()
            indent = len(raw) - len(stripped)
            for kw in ("async def ", "def ", "class "):
                if stripped.startswith(kw):
                    return indent + len(kw)
        except Exception:
            pass
        return None
