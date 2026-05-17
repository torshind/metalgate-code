"""Tests for context.storage - database models and IndexStore."""

from pathlib import Path

import pytest
import pytest_asyncio
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
    Module,
    Package,
    StreamingWriter,
    _IndexStore,
)
from metalgate_code.context.parsing import acollect_files, aparse_file
from metalgate_code.context.resolver import _resolve_forwarding


async def run_writer(tmp_path: Path, fake_site: list[Path]) -> str:
    # async indexing
    writer = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(fake_site[0])],
    )
    db_path = str(writer.db_path)
    Path(db_path).unlink(missing_ok=True)
    await writer.start()
    await writer.wait_for_completion()

    return db_path


@pytest_asyncio.fixture
async def indexed_store(
    tmp_path: Path, fake_site: list[Path]
) -> tuple[_IndexStore, str]:
    """Build index from fake_site and return store with db path."""
    files = await acollect_files(None, [str(fake_site[0])])
    all_modules: list[_ModuleData] = []
    all_funcs: list[_FuncData] = []
    all_classes: list[_ClassData] = []
    all_apps: list[_DecoratorApp] = []
    for f in files:
        md, funcs, classes, apps = await aparse_file(f, [str(fake_site[0])], None)
        all_modules.append(md)
        all_funcs.extend(funcs)
        all_classes.extend(classes)
        all_apps.extend(apps)
    _resolve_forwarding(all_funcs, all_classes, all_apps)

    db_path = await run_writer(tmp_path, fake_site)
    return _IndexStore(db_path), db_path


# =============================================================================
# Database Writing Tests
# =============================================================================


@pytest.mark.asyncio
async def test_write_to_db_basic(tmp_path: Path, fake_site: list[Path]):
    """Test basic writing to the database."""
    f = fake_site[0] / "fakelib" / "core.py"
    md, funcs, classes, _ = await aparse_file(str(f), [str(fake_site[0])], None)

    db_path = await run_writer(tmp_path, fake_site)

    # Raw SQLAlchemy read
    engine = create_engine(f"sqlite:///{db_path}")
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


@pytest.mark.asyncio
async def test_store_package_context(
    indexed_store: tuple[_IndexStore, str], tmp_pkg: Path
):
    """Test retrieving package context from IndexStore."""
    store, _ = indexed_store
    out = store.package_context("fakelib")
    assert "fakelib.core" in out
    assert "Core module." in out


@pytest.mark.asyncio
async def test_store_module_context(
    indexed_store: tuple[_IndexStore, str], tmp_pkg: Path
):
    """Test retrieving module context from IndexStore."""
    store, _ = indexed_store
    out = store.module_context("fakelib.core")
    assert "Worker" in out
    assert "foo" in out


@pytest.mark.asyncio
async def test_store_symbol_context_func(
    indexed_store: tuple[_IndexStore, str], tmp_pkg: Path
):
    """Test retrieving symbol context for a function."""
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.foo")
    assert out.startswith("Function:")
    assert "fakelib.core.foo" in out


@pytest.mark.asyncio
async def test_store_symbol_context_class(
    indexed_store: tuple[_IndexStore, str], tmp_pkg: Path
):
    """Test retrieving symbol context for a class."""
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.Worker")
    assert out.startswith("Class:")
    assert "Worker" in out


@pytest.mark.asyncio
async def test_store_symbol_context_missing(
    indexed_store: tuple[_IndexStore, str], tmp_pkg: Path
):
    """Test retrieving symbol context for non-existent symbol."""
    store, _ = indexed_store
    out = store.symbol_context("fakelib.core.NonExistent")
    assert "not found" in out


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_index_real_sqlalchemy(tmp_path: Path, fake_site: list[Path]):
    """Index a real package using StreamingWriter."""
    db_path = await run_writer(tmp_path, fake_site)
    store = _IndexStore(db_path)
    pkg_out = store.package_context("fakelib")
    assert "fakelib" in pkg_out


@pytest.mark.asyncio
async def test_symbol_context_class_method_filtering(
    indexed_store: tuple[_IndexStore, str], tmp_pkg: Path
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

    # Private methods should NOT be shown
    assert "_private_method" not in methods_section

    # __init__ should NOT appear in Methods section
    assert "__init__" not in methods_section.split("\n\n")[0]
