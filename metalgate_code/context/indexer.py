"""High-level orchestration for building the symbol index.

This module provides the main entry point for indexing Python packages
and writing results to the database.
"""

import logging
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from metalgate_code.context.db import _IndexStore
from metalgate_code.context.db.streaming_writer import StreamingWriter
from metalgate_code.helpers.paths import get_index_data_dir

logger = logging.getLogger("metalgate_code")

# Global state
_writer: StreamingWriter | None = None


class PackageContextInput(BaseModel):
    package_name: str = Field(
        description="The name of the package to look up (e.g., 'httpx')."
    )


class ModuleContextInput(BaseModel):
    module_name: str = Field(
        description="Fully qualified module name (e.g., 'httpx._client')."
    )


class SymbolContextInput(BaseModel):
    qualified_name: str = Field(
        description="Fully qualified symbol name (e.g., 'httpx.Client.get')."
    )


class PackageContextTool(BaseTool):
    """Get overview of a package."""

    name: Literal["package_context"] = "package_context"
    description: str = "Get overview of a package."
    args_schema: type[BaseModel] = PackageContextInput

    def __init__(self, index_store: "IndexStore", **kwargs):
        super().__init__(**kwargs)
        self._index_store = index_store

    def _run(self, package_name: str) -> str:
        return self._index_store._package_context_impl(package_name)


class ModuleContextTool(BaseTool):
    """Get all public objects in a module."""

    name: Literal["module_context"] = "module_context"
    description: str = "Get all public objects in a module."
    args_schema: type[BaseModel] = ModuleContextInput

    def __init__(self, index_store: "IndexStore", **kwargs):
        super().__init__(**kwargs)
        self._index_store = index_store

    def _run(self, module_name: str) -> str:
        return self._index_store._module_context_impl(module_name)


class SymbolContextTool(BaseTool):
    """Get full detail on a function or class."""

    name: Literal["symbol_context"] = "symbol_context"
    description: str = "Get full detail on a function or class."
    args_schema: type[BaseModel] = SymbolContextInput

    def __init__(self, index_store: "IndexStore", **kwargs):
        super().__init__(**kwargs)
        self._index_store = index_store

    def _run(self, qualified_name: str) -> str:
        return self._index_store._symbol_context_impl(qualified_name)


class IndexStore:
    """Agent tool interface to the live symbol index.

    Works immediately - queries return partial results while indexing continues.
    """

    def __init__(self, cwd: str):
        self._store: _IndexStore | None = None
        self.db_path = get_index_data_dir(cwd)
        self.package_context = PackageContextTool(index_store=self)
        self.module_context = ModuleContextTool(index_store=self)
        self.symbol_context = SymbolContextTool(index_store=self)

    def _ensure_store(self) -> _IndexStore | None:
        if self._store is None:
            if not Path(self.db_path).exists():
                return None
            self._store = _IndexStore(self.db_path)
        return self._store

    def _package_context_impl(self, package_name: str) -> str:
        """Get overview of a package."""
        store = self._ensure_store()
        if store is None:
            return "Index not initialized yet."
        return store.package_context(package_name)

    def _module_context_impl(self, module_name: str) -> str:
        """Get all public objects in a module."""
        store = self._ensure_store()
        if store is None:
            return "Index not initialized yet."
        return store.module_context(module_name)

    def _symbol_context_impl(self, qualified_name: str) -> str:
        """Get full detail on a function or class."""
        store = self._ensure_store()
        if store is None:
            return "Index not initialized yet."
        return store.symbol_context(qualified_name)


async def start_indexing(
    cwd: str, python: str | None = None, site_roots: list[str] | None = None
):
    """Start async background indexing of site-packages.

    Args:
        db_path: Path to the SQLite database file.
        python: Optional Python interpreter path. Uses current environment if None.
        site_roots: Optional list of site-packages paths to index.
    """
    global _writer

    if _writer and _writer.is_running():
        logger.info("Indexing already in progress.")
        return

    _writer = StreamingWriter(
        cwd=str(cwd),
        python=python,
        site_roots=site_roots,
        on_package_done=lambda pkg: logger.info(f"Indexed: {pkg}"),
    )
    await _writer.start()

    logger.info(f"Background indexing started. Database: {_writer.db_path}")


def is_indexing() -> bool:
    """Check if background indexing is running."""
    return _writer is not None and _writer.is_running()


async def wait_for_indexing() -> None:
    """Wait for background indexing to complete.

    Does nothing if indexing is not running.
    """
    if _writer is not None:
        await _writer.wait_for_completion()


async def stop_indexing() -> None:
    """Stop background indexing.

    Does nothing if indexing is not running.
    """
    global _writer
    if _writer is not None:
        await _writer.stop()
        _writer = None


__all__ = [
    "IndexStore",
    "start_indexing",
    "is_indexing",
    "wait_for_indexing",
    "stop_indexing",
]
