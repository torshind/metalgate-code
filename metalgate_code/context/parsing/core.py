"""Core file parsing - hybrid Jedi + Tree-sitter approach."""

import ast
import logging
from pathlib import Path

import jedi

from metalgate_code.context.data import (
    _ClassData,
    _DecoratorApp,
    _FuncData,
    _ModuleData,
)
from metalgate_code.context.parsing.jedi import (
    _extract_class_jedi,
    _extract_function_jedi,
)
from metalgate_code.context.parsing.module import _file_to_module
from metalgate_code.context.parsing.treesitter import (
    _detect_decorator_apps_ts,
    _walk_tree_for_forwarding,
    ts_parser,
)

logger = logging.getLogger("metalgate_code")


def _extract_module_docstring(code: str) -> str | None:
    """Extract module-level docstring from source code."""
    try:
        tree = ast.parse(code)
        return ast.get_docstring(tree)
    except SyntaxError:
        return None


def _extract_module_exports(code: str) -> list[str]:
    """Extract __all__ exports from source code."""
    exports: list[str] = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    exports.append(elt.value)
    except Exception:
        pass
    return exports


def parse_file(
    file: Path,
    site_roots: list[Path],
) -> tuple[_ModuleData, list[_FuncData], list[_ClassData], list[_DecoratorApp]]:
    """Parse a Python file using jedi and extract symbol information."""
    module = _file_to_module(file, site_roots)
    package = module.split(".")[0]

    # Single file read for all operations
    try:
        code = file.read_text(encoding="utf-8")
        src_bytes = code.encode("utf-8")
    except Exception as e:
        logger.error(f"Failed to read file {file}: {e}")
        md = _ModuleData()
        md.module = module
        md.package = package
        md.file = str(file.resolve())
        return md, [], [], []

    # Create Jedi script from code (avoids second file read)
    try:
        script = jedi.Script(code=code, path=str(file))
    except Exception as e:
        logger.error(f"Failed to create Jedi script for {file}: {e}")
        md = _ModuleData()
        md.module = module
        md.package = package
        md.file = str(file.resolve())
        return md, [], [], []

    funcs: list[_FuncData] = []
    classes: list[_ClassData] = []

    for name in script.get_names(all_scopes=True):
        try:
            if name.type == "function":
                fd = _extract_function_jedi(name, module, package)
                if fd:
                    funcs.append(fd)
            elif name.type == "class":
                cd = _extract_class_jedi(name, module, package)
                if cd:
                    classes.append(cd)
        except Exception as e:
            logger.debug(
                f"Failed to extract symbol {name.name if hasattr(name, 'name') else name}: {e}"
            )
            continue

    md = _ModuleData()
    md.module = module
    md.package = package
    md.file = str(file.resolve())
    md.docstring = _extract_module_docstring(code)
    md.exports = _extract_module_exports(code)

    # === TREE-SITTER: Structural analysis for forwarding & decorators ===
    # Also fixup *args/**kwargs detection that jedi may miss
    func_by_line: dict[int, _FuncData] = {f.line: f for f in funcs}

    try:
        tree = ts_parser.parse(src_bytes)
        root = tree.root_node

        _walk_tree_for_forwarding(root, [], func_by_line)

        # Detect decorator applications
        apps = _detect_decorator_apps_ts(root, module, [], func_by_line)

    except Exception as e:
        logger.debug(f"Failed tree-sitter analysis for {file}: {e}")
        apps = []

    return md, funcs, classes, apps


__all__ = ["parse_file", "_extract_module_docstring", "_extract_module_exports"]
