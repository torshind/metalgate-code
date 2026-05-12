"""Tests for context.indexer - build_index tool."""

from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from metalgate_code.context.db import Base, Package
from metalgate_code.context.indexer import build_index


class TestBuildIndex:
    """Tests for the build_index LangChain tool."""

    def test_build_index_returns_summary(self, tmp_path: Path, fake_site: list[Path]):
        """Test that build_index returns a summary string."""
        db_path = tmp_path / "test_index.db"

        result = build_index.invoke(
            {
                "db_path": str(db_path),
                "python": None,
                "site_roots": [str(s) for s in fake_site],
            }
        )

        assert isinstance(result, str)
        assert "Indexed" in result or "files" in result.lower()
        assert db_path.exists()

    def test_build_index_creates_valid_db(self, tmp_path: Path, fake_site: list[Path]):
        """Test that build_index creates a valid SQLite database."""
        db_path = tmp_path / "test_index.db"

        build_index.invoke(
            {
                "db_path": str(db_path),
                "python": None,
                "site_roots": [str(s) for s in fake_site],
            }
        )

        # Verify database is valid by querying it
        engine = create_engine(f"sqlite:///{db_path}")
        with Session(engine) as s:
            # Check that tables exist
            packages = s.execute(select(Package)).scalars().all()
            assert len(packages) > 0

    def test_build_index_with_no_site_roots(self, tmp_path: Path):
        """Test build_index with empty site_roots."""
        db_path = tmp_path / "test_index.db"

        result = build_index.invoke(
            {"db_path": str(db_path), "python": None, "site_roots": []}
        )

        assert "No site-packages found" in result
        assert not db_path.exists()

    def test_build_index_overwrites_existing(
        self, tmp_path: Path, fake_site: list[Path]
    ):
        """Test that build_index overwrites an existing database."""
        db_path = tmp_path / "test_index.db"

        # Create initial database
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)

        # Get initial modification time
        initial_mtime = db_path.stat().st_mtime

        # Run build_index
        build_index.invoke(
            {
                "db_path": str(db_path),
                "python": None,
                "site_roots": [str(s) for s in fake_site],
            }
        )

        # Database should be recreated
        assert db_path.exists()
        new_mtime = db_path.stat().st_mtime
        assert new_mtime > initial_mtime


class TestBuildIndexToolDecorator:
    """Tests to verify build_index has the LangChain tool decorator."""

    def test_build_index_has_tool_attributes(self):
        """Test that build_index has the expected tool attributes."""

        # Check if build_index is a StructuredTool
        assert hasattr(build_index, "name")
        assert hasattr(build_index, "description")
        assert build_index.name == "build_index"
        assert "symbol index" in build_index.description.lower()

    def test_build_index_schema(self):
        """Test that build_index has the correct input schema."""
        from pydantic import BaseModel

        schema = build_index.args_schema
        assert schema is not None
        assert isinstance(schema, type) and issubclass(schema, BaseModel)
        assert "db_path" in schema.model_fields
        assert "python" in schema.model_fields
        assert "site_roots" in schema.model_fields
