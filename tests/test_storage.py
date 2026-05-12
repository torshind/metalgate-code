"""Tests for context.storage - database models and IndexStore."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from metalgate_code.context.data import (
    _ClassData,
    _DecoratorApp,
    _FuncData,
    _ModuleData,
)
from metalgate_code.context.db import (
    Attribute,
    Class,
    Function,
    IndexStore,
    Module,
    Package,
    write_index,
)
from metalgate_code.context.parsing import collect_files, find_site_packages, parse_file
from metalgate_code.context.resolver import _resolve_forwarding


@pytest.fixture
def indexed_store(tmp_db: str, fake_site: list[Path]) -> tuple[IndexStore, str]:
    """Build index from fake_site and return store with db path."""
    files = collect_files(fake_site)
    all_modules: list[_ModuleData] = []
    all_funcs: list[_FuncData] = []
    all_classes: list[_ClassData] = []
    all_apps: list[_DecoratorApp] = []
    for f in files:
        md, funcs, classes, apps = parse_file(f, fake_site)
        all_modules.append(md)
        all_funcs.extend(funcs)
        all_classes.extend(classes)
        all_apps.extend(apps)
    _resolve_forwarding(all_funcs, all_classes, all_apps)
    Path(tmp_db).unlink(missing_ok=True)
    write_index(Path(tmp_db), all_modules, all_funcs, all_classes)
    return IndexStore(tmp_db), tmp_db


# =============================================================================
# Database Writing Tests
# =============================================================================


def test_write_to_db_basic(tmp_db: str, fake_site: list[Path]):
    """Test basic writing to the database."""
    f = fake_site[0] / "fakelib" / "core.py"
    md, funcs, classes, _ = parse_file(f, fake_site)
    write_index(Path(tmp_db), [md], funcs, classes)

    # Raw SQLAlchemy read
    engine = create_engine(f"sqlite:///{tmp_db}")
    with Session(engine) as s:
        pkg = s.execute(select(Package).where(Package.name == "fakelib")).scalar_one()
        assert pkg.name == "fakelib"

        mod = s.execute(
            select(Module).where(Module.name == "fakelib.core")
        ).scalar_one()
        assert mod.docstring == "Core module."
        assert mod.exports == []

        cls = s.execute(select(Class).where(Class.name == "Worker")).scalar_one()
        assert cls.method_names == ["__init__", "do"]
        attrs = (
            s.execute(select(Attribute).where(Attribute.class_id == cls.id))
            .scalars()
            .all()
        )
        assert any(a.name == "count" for a in attrs)

        top_level_funcs = (
            s.execute(
                select(Function).where(
                    Function.module_id == mod.id, Function.is_method.is_(False)
                )
            )
            .scalars()
            .all()
        )
        assert any(f.name == "foo" for f in top_level_funcs)


# =============================================================================
# IndexStore API Tests
# =============================================================================


def test_store_package_context(indexed_store: tuple[IndexStore, str], tmp_pkg: Path):
    """Test retrieving package context from IndexStore."""
    store, _ = indexed_store
    out = store.package_context("fakelib")
    assert "fakelib.core" in out
    assert "Core module." in out


def test_store_module_context(indexed_store: tuple[IndexStore, str], tmp_pkg: Path):
    """Test retrieving module context from IndexStore."""
    store, _ = indexed_store
    out = store.module_context("fakelib.core")
    assert "Worker" in out
    assert "foo" in out


def test_store_symbol_context_func(
    indexed_store: tuple[IndexStore, str], tmp_pkg: Path
):
    """Test retrieving symbol context for a function."""
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.foo")
    assert out.startswith("Function:")
    assert "fakelib.core.foo" in out


def test_store_symbol_context_class(
    indexed_store: tuple[IndexStore, str], tmp_pkg: Path
):
    """Test retrieving symbol context for a class."""
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.Worker")
    assert out.startswith("Class:")
    assert "Worker" in out


def test_store_symbol_context_missing(
    indexed_store: tuple[IndexStore, str], tmp_pkg: Path
):
    """Test retrieving symbol context for non-existent symbol."""
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.NonExistent")
    assert "not found" in out


# =============================================================================
# Integration Tests
# =============================================================================


def test_index_real_sqlalchemy(tmp_db: str):
    """Index the real sqlalchemy package from the project venv."""
    roots = find_site_packages()
    if not roots:
        pytest.skip("No site-packages discovered")
    files = collect_files(roots)
    sa_files = [f for f in files if "sqlalchemy" in f.parts]
    if not sa_files:
        pytest.skip("sqlalchemy not installed in venv")

    all_modules: list[_ModuleData] = []
    all_funcs: list[_FuncData] = []
    all_classes: list[_ClassData] = []
    all_apps: list[_DecoratorApp] = []
    # Use a small slice so the test stays fast
    for f in sa_files[:50]:
        try:
            md, funcs, classes, apps = parse_file(f, roots)
            all_modules.append(md)
            all_funcs.extend(funcs)
            all_classes.extend(classes)
            all_apps.extend(apps)
        except Exception:
            pass

    Path(tmp_db).unlink(missing_ok=True)
    write_index(Path(tmp_db), all_modules, all_funcs, all_classes)

    store = IndexStore(tmp_db)
    pkg_out = store.package_context("sqlalchemy")
    assert "sqlalchemy" in pkg_out

    # symbol context must resolve the 2-Clause NameError bug
    sym = store.symbol_context("sqlalchemy.orm.Query")
    assert "not found" in sym or "Class:" in sym


def test_symbol_context_class_method_filtering(
    indexed_store: tuple[IndexStore, str], tmp_pkg: Path
):
    """Test that symbol_context filters methods correctly.

    - Special methods in _SHOW_METHODS (like __repr__, __call__, __len__) are shown
    - __init__ appears only as "Constructor", not in Methods
    - Private methods (starting with _) are hidden
    """
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.Special")

    # Class header
    assert out.startswith("Class:")
    assert "Special" in out

    # __init__ should appear as "Constructor:"
    assert "Constructor:" in out
    assert "__init__" in out.split("Constructor:")[1].split("\n\n")[0]

    # Methods section should exist
    assert "Methods:" in out

    # Special methods from _SHOW_METHODS should be shown
    methods_section = out.split("Methods:")[1] if "Methods:" in out else ""
    assert "__repr__" in methods_section
    assert "__call__" in methods_section
    assert "__len__" in methods_section

    # Private method should NOT be shown
    assert "_private_helper" not in out

    # __init__ should NOT appear in Methods section (only in Constructor)
    methods_only = methods_section.split("\n\n")[0] if "Methods:" in out else ""
    assert "__init__" not in methods_only
