"""Tests for context.indexer - StreamingWriter async indexer."""

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from metalgate_code.context.db import Base, Package, StreamingWriter
from metalgate_code.context.indexer import (
    IndexStore,
    is_indexing,
    start_background_index,
)


@pytest.mark.asyncio
async def test_streaming_writer_returns_summary(tmp_path: Path, fake_site: list[Path]):
    """Test that StreamingWriter creates a database and returns summary."""
    writer = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    Path(writer.db_path).unlink(missing_ok=True)
    await writer.start()
    await writer.wait_for_completion()

    assert writer.db_path.exists()


@pytest.mark.asyncio
async def test_streaming_writer_creates_valid_db(tmp_path: Path, fake_site: list[Path]):
    """Test that StreamingWriter creates a valid SQLite database."""
    writer = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    Path(writer.db_path).unlink(missing_ok=True)
    await writer.start()
    await writer.wait_for_completion()

    # Verify database is valid by querying it
    engine = create_engine(f"sqlite:///{writer.db_path}")
    with Session(engine) as s:
        # Check that tables exist
        packages = s.execute(select(Package)).scalars().all()
        assert len(packages) > 0


@pytest.mark.asyncio
async def test_streaming_writer_with_no_site_roots(tmp_path: Path):
    """Test StreamingWriter with empty site_roots."""
    writer = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[],
    )
    Path(writer.db_path).unlink(missing_ok=True)
    await writer.start()
    await writer.wait_for_completion()

    # Should complete without error, but no DB created
    assert not writer.db_path.exists()


@pytest.mark.asyncio
async def test_streaming_writer_populates_existing(
    tmp_path: Path, fake_site: list[Path]
):
    """Test that StreamingWriter populates an existing database file."""
    # Create initial empty database
    writer = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    Path(writer.db_path).unlink(missing_ok=True)
    engine = create_engine(f"sqlite:///{writer.db_path}")
    Base.metadata.create_all(engine)

    # # Clear any existing data from previous runs
    # with Session(engine) as s:
    #     s.execute(delete(Package))
    #     s.commit()

    # Verify database is empty (no packages)
    with Session(engine) as s:
        initial_count = len(s.execute(select(Package)).scalars().all())
        assert initial_count == 0, "Database should start empty"

    # Run StreamingWriter
    await writer.start()
    await writer.wait_for_completion()

    # Database should now contain packages
    with Session(engine) as s:
        packages = s.execute(select(Package)).scalars().all()
        assert len(packages) > 0, "Database should be populated with packages"


@pytest.mark.asyncio
async def test_start_background_index_creates_store(
    tmp_path: Path, fake_site: list[Path]
):
    """Test that start_background_index creates index store and starts indexing."""
    index_store = IndexStore(str(tmp_path))
    Path(index_store.db_path).unlink(missing_ok=True)

    result = await start_background_index(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )

    assert "Background indexing started" in result

    # Wait for indexing to complete
    while is_indexing():
        await asyncio.sleep(0.1)

    assert index_store.db_path.exists()


@pytest.mark.asyncio
async def test_start_background_index_already_running(
    tmp_path: Path, fake_site: list[Path]
):
    """Test that start_background_index returns message if already running."""
    index_store = IndexStore(str(tmp_path))
    Path(index_store.db_path).unlink(missing_ok=True)

    # Start first indexing
    await start_background_index(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )

    # Try to start again
    result = await start_background_index(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )

    assert "already in progress" in result


@pytest.mark.asyncio
async def test_streaming_writer_skips_same_version(
    tmp_path: Path, fake_site: list[Path]
):
    """Test that StreamingWriter skips packages with same version."""
    # Create fake_site with version
    init_file = fake_site[0] / "fakelib" / "__init__.py"
    init_file.write_text('__version__ = "1.0.0"\n')

    # First indexing
    writer1 = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    Path(writer1.db_path).unlink(missing_ok=True)
    await writer1.start()
    await writer1.wait_for_completion()

    # Get initial file stats
    initial_mtime = writer1.db_path.stat().st_mtime

    # Second indexing with same version
    writer2 = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    await writer2.start()
    await writer2.wait_for_completion()

    # Database should not be modified (same version = skipped)
    final_mtime = writer2.db_path.stat().st_mtime
    assert final_mtime == initial_mtime, "Package with same version should be skipped"


@pytest.mark.asyncio
async def test_streaming_writer_updates_different_version(
    tmp_path: Path, fake_site: list[Path]
):
    """Test that StreamingWriter clears and re-indexes packages with different version."""
    # Create fake_site with initial version
    init_file = fake_site[0] / "fakelib" / "__init__.py"
    init_file.write_text('__version__ = "1.0.0"\n')

    # First indexing
    writer1 = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    Path(writer1.db_path).unlink(missing_ok=True)
    await writer1.start()
    await writer1.wait_for_completion()

    # Verify first version
    store1 = IndexStore(str(tmp_path))
    out1 = store1.package_context.invoke("fakelib")
    assert "fakelib" in out1

    # Update version
    init_file.write_text('__version__ = "2.0.0"\n')

    # Second indexing with different version
    writer2 = StreamingWriter(
        cwd=str(tmp_path),
        site_roots=[str(s) for s in fake_site],
    )
    await writer2.start()
    await writer2.wait_for_completion()

    # Verify second version (should be updated, not skipped)
    store2 = IndexStore(str(tmp_path))
    out2 = store2.package_context.invoke("fakelib")
    assert "fakelib" in out2
    assert "@2.0.0" in out2, "Package should be re-indexed with new version"
