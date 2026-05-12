"""Tests for context.model - symbol extraction and analysis."""

from pathlib import Path

from metalgate_code.context.parsing import collect_files, find_site_packages, parse_file
from metalgate_code.context.parsing.module import _file_to_module

# =============================================================================
# Parameter Extraction Tests
# =============================================================================


def test_extract_param_identifier(tmp_path: Path):
    """Test extracting simple identifier parameters."""
    src = b"def f(a, b): pass"
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    assert len(funcs) == 1
    assert [p.name for p in funcs[0].parameters] == ["a", "b"]


def test_extract_param_typed(tmp_path: Path):
    """Test extracting typed parameters with defaults."""
    src = b'def f(a: int, b: str = "x") -> bool: pass'
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    params = funcs[0].parameters
    assert params[0].annotation == "int"
    assert params[1].annotation == "str"
    assert params[1].default == '"x"'


def test_extract_param_star(tmp_path: Path):
    """Test extracting *args and **kwargs parameters."""
    src = b"def f(*args, **kwargs): pass"
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    params = funcs[0].parameters
    assert params[0].kind == "var_positional"
    assert params[1].kind == "var_keyword"


def test_extract_param_positional_only(tmp_path: Path):
    """Test extracting positional-only parameters."""
    src = b"def f(a, /, b): pass"
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    kinds = [p.kind for p in funcs[0].parameters]
    assert len(kinds) == 2
    assert kinds[1] == "positional"


# =============================================================================
# Docstring Extraction Tests
# =============================================================================


def test_extract_docstring(tmp_path: Path):
    """Test extracting function docstrings."""
    _, funcs, _, _ = parse_file(
        _file_from_src(b'\ndef f():\n    """hello"""\n    pass\n', "mod", tmp_path),
        [],
    )
    assert funcs[0].docstring == "hello"


def test_extract_module_docstring(tmp_path: Path):
    """Test extracting module-level docstrings."""
    md, _, _, _ = parse_file(
        _file_from_src(b'"""mod doc."""\nx = 1\n', "mod", tmp_path), []
    )
    assert md.docstring == "mod doc."


# =============================================================================
# __all__ Extraction Tests
# =============================================================================


def test_extract_exports(tmp_path: Path):
    """Test extracting __all__ exports from a module."""
    md, _, _, _ = parse_file(
        _file_from_src(b'__all__ = ["a", "b"]\n', "mod", tmp_path), []
    )
    assert md.exports == ["a", "b"]


# =============================================================================
# Class Extraction Tests
# =============================================================================


def test_extract_class_attributes(tmp_path: Path):
    """Test extracting class attributes and methods."""
    src = (
        b"class C:\n"
        b'    """A class."""\n'
        b"    x: int\n"
        b"    y = 10\n"
        b"    def __init__(self, z='hi'):\n"
        b"        self.z = z\n"
    )
    _, _, classes, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    assert len(classes) == 1
    attrs = {(a.name, a.annotation, a.default) for a in classes[0].attributes}
    assert any(a[0] == "x" and a[1] in ("int", "builtins.int") for a in attrs)
    assert any(a[0] == "y" for a in attrs)


# =============================================================================
# Forwarding Detection Tests
# =============================================================================


def test_has_args_kwargs(tmp_path: Path):
    """Test detection of functions with *args/**kwargs."""
    src = b"def deco(fn):\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs)\n    return wrapper\n"
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    wrapper = [f for f in funcs if f.name == "wrapper"][0]
    assert wrapper._is_args_kwargs is True


def test_detect_forwarding(tmp_path: Path):
    """Test detection of forwarding calls to super().__init__."""
    src = (
        b"class A:\n"
        b"    def __init__(self, a):\n"
        b"        pass\n"
        b"class B(A):\n"
        b"    def __init__(self, *args, **kwargs):\n"
        b"        super().__init__(*args, **kwargs)\n"
    )
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    b_init = [f for f in funcs if f.name == "__init__" and f.parent_class == "B"][0]
    assert b_init._forwarding_target == "super().__init__"


def test_call_forwards(tmp_path: Path):
    """Test detection of forwarding calls to other functions."""
    src = b"def f(*args, **kwargs):\n    g(*args, **kwargs)\n"
    _, funcs, _, _ = parse_file(_file_from_src(src, "mod", tmp_path), [])
    assert funcs[0]._forwarding_target == "g"


# =============================================================================
# Decorator Application Tests
# =============================================================================


def test_detect_decorator_apps(tmp_path: Path):
    """Test detection of @overload decorator applications."""
    src = "from typing import overload\n@overload\ndef foo(x: int) -> int: ...\n"
    _, _, _, apps = parse_file(_file_from_src(src.encode(), "mod", tmp_path), [])
    assert len(apps) == 1
    assert apps[0].decorator_name == "overload"


def test_detect_decorator_apps_multiple(tmp_path: Path):
    """Test detection of multiple decorators on same function."""
    src = """\n@decorator1\n@decorator2\ndef foo(x: int) -> int: ...\n"""
    _, _, _, apps = parse_file(_file_from_src(src.encode(), "mod", tmp_path), [])
    assert len(apps) == 2
    assert apps[0].decorator_name == "decorator1"
    assert apps[1].decorator_name == "decorator2"
    assert apps[0].wrapped_function == apps[1].wrapped_function


def test_detect_decorator_apps_nested_class(tmp_path: Path):
    """Test detection of decorators on methods in nested classes."""
    src = """\nclass Outer:\n    class Inner:\n        @staticmethod\n        def method(x: int) -> int: ...\n"""
    _, _, _, apps = parse_file(_file_from_src(src.encode(), "mod", tmp_path), [])
    assert len(apps) == 1
    assert apps[0].decorator_name == "staticmethod"


def test_detect_decorator_apps_with_args(tmp_path: Path):
    """Test detection of decorators with arguments."""
    src = """\n@decorator(arg1, arg2)\ndef foo(x: int) -> int: ...\n"""
    _, _, _, apps = parse_file(_file_from_src(src.encode(), "mod", tmp_path), [])
    assert len(apps) == 1
    assert apps[0].decorator_name == "decorator"


def test_detect_decorator_apps_classmethod(tmp_path: Path):
    """Test detection of @classmethod decorator."""
    src = """\nclass C:\n    @classmethod\n    def create(cls) -> 'C':\n        return cls()\n"""
    _, _, _, apps = parse_file(_file_from_src(src.encode(), "mod", tmp_path), [])
    assert len(apps) == 1
    assert apps[0].decorator_name == "classmethod"


def test_detect_decorator_apps_property(tmp_path: Path):
    """Test detection of @property decorator."""
    src = """\nclass C:\n    @property\n    def value(self) -> int:\n        return 42\n"""
    _, _, _, apps = parse_file(_file_from_src(src.encode(), "mod", tmp_path), [])
    assert len(apps) == 1
    assert apps[0].decorator_name == "property"


# =============================================================================
# File Path to Module Tests
# =============================================================================


def test_file_to_module():
    """Test conversion of file path to module name."""
    root = Path("/env/site-packages")
    f = root / "pkg" / "mod.py"
    assert _file_to_module(f, [root]) == "pkg.mod"


def test_file_to_module_init():
    """Test conversion of __init__.py to package name."""
    root = Path("/env/site-packages")
    f = root / "pkg" / "__init__.py"
    assert _file_to_module(f, [root]) == "pkg"


def test_file_to_module_stubs():
    """Test conversion of stub files to module names."""
    root = Path("/env/site-packages")
    f = root / "pkg-stubs" / "mod.pyi"
    assert _file_to_module(f, [root]) == "pkg.mod"


# =============================================================================
# File Collection Tests
# =============================================================================


def test_collect_files(tmp_pkg: Path):
    """Test collecting Python files from site-packages."""
    files = collect_files([tmp_pkg])
    stems = {f.stem for f in files}
    assert "__init__" in stems
    assert "core" in stems


# =============================================================================
# Site Packages Discovery Tests
# =============================================================================


def test_find_site_packages():
    """Test discovery of site-packages directories."""
    roots = find_site_packages()
    assert len(roots) > 0
    assert any("site-packages" in str(r) for r in roots)


# =============================================================================
# File Parsing Tests
# =============================================================================


def test_parse_file(fake_site: list[Path]):
    """Test parsing of a module file."""
    f = fake_site[0] / "fakelib" / "__init__.py"
    md, funcs, classes, _ = parse_file(f, fake_site)
    assert md.package == "fakelib"
    assert md.module == "fakelib"
    assert md.docstring == "Entry doc."
    assert "foo" in md.exports
    assert not funcs
    assert not classes


def test_parse_file_core_class(fake_site: list[Path]):
    """Test parsing of a module with classes."""
    f = fake_site[0] / "fakelib" / "core.py"
    md, funcs, classes, _ = parse_file(f, fake_site)
    assert md.module == "fakelib.core"
    assert md.docstring == "Core module."

    cls = [c for c in classes if c.name == "Worker"][0]
    assert cls.bases == []
    assert cls.method_names == ["__init__", "do"]

    init = [f for f in funcs if f.qualified_name == "fakelib.core.Worker.__init__"][0]
    assert init.is_method is True


def _file_from_src(src: bytes, name: str, tmp_path: Path) -> Path:
    """Write source to a temp file and return its path."""
    p = tmp_path / f"{name}.py"
    p.write_bytes(src)
    return p
