"""High-level orchestration for building the symbol index.

This module provides the main entry point for indexing Python packages
and writing results to the database.
"""

import logging
import time
from pathlib import Path

from langchain_core.tools import tool

from metalgate_code.context.data import (
    _ClassData,
    _DecoratorApp,
    _FuncData,
    _ModuleData,
)
from metalgate_code.context.db import IndexStore as _IndexStore, write_index
from metalgate_code.context.parsing import collect_files, find_site_packages, parse_file
from metalgate_code.context.resolver import _resolve_forwarding

logger = logging.getLogger("metalgate_code")


@tool
def build_index(
    db_path: str, python: str | None = None, site_roots: list[str] | None = None
) -> str:
    """Build the symbol index from site-packages.

    This function:
    1. Finds site-packages directories (or uses provided ones)
    2. Collects and parses Python files
    3. Resolves *args/**kwargs forwarding
    4. Writes results to SQLite database

    Args:
        db_path: Path to the SQLite database file (as string).
        python: Optional Python interpreter path. If None, uses current environment.
        site_roots: Optional list of site-packages paths. If provided, skips discovery.

    Returns:
        A summary string of the indexing results.
    """
    path = Path(db_path)
    if site_roots is None:
        roots = find_site_packages(python)
    else:
        roots = [Path(s) for s in site_roots]
    if not roots:
        return "No site-packages found."

    def _should_report(i: int, total: int, interval: int) -> bool:
        """Check if progress should be reported at this iteration."""
        return (i + 1) % interval == 0 or (i + 1) == total

    files = collect_files(roots)

    all_modules: list[_ModuleData] = []
    all_funcs: list[_FuncData] = []
    all_classes: list[_ClassData] = []
    all_apps: list[_DecoratorApp] = []

    t0 = time.time()
    report_every = max(1, len(files) // 20)

    for i, f in enumerate(files):
        if _should_report(i, len(files), report_every):
            elapsed = time.time() - t0
            pct = 100 * (i + 1) / len(files)
            logger.info(
                f"  [{pct:5.1f}%] {i + 1}/{len(files)} ({elapsed:.1f}s) {f.name}"
            )
        try:
            md, funcs, classes, apps = parse_file(f, roots)
            all_modules.append(md)
            all_funcs.extend(funcs)
            all_classes.extend(classes)
            all_apps.extend(apps)
        except Exception as e:
            logger.warning(f"  [warn] {f}: {e}")

    elapsed = time.time() - t0

    # Resolve forwarding
    t_resolve = time.time()
    stats = _resolve_forwarding(all_funcs, all_classes, all_apps)
    resolve_time = time.time() - t_resolve

    # Write to DB
    if path.exists():
        path.unlink()
    t_write = time.time()

    write_index(path, all_modules, all_funcs, all_classes)

    write_time = time.time() - t_write
    size = path.stat().st_size
    label = f"{size / 1_000_000:.1f}MB" if size > 1_000_000 else f"{size / 1_000:.1f}KB"

    return (
        f"Indexed {len(files)} files from {len(roots)} site-packages in {elapsed:.1f}s.\n"
        f"Found {len(all_modules)} modules, {len(all_funcs)} functions, {len(all_classes)} classes.\n"
        f"Resolved forwarding: {stats['resolved']} resolved, {stats['unresolvable']} unresolvable, "
        f"{stats['opaque']} opaque ({resolve_time:.1f}s).\n"
        f"Database written to {db_path} ({label}) in {write_time:.1f}s."
    )


class IndexStore:
    """Agent tool interface to the precomputed symbol index.

    Three methods, three zoom levels:
        package_context("httpx")                       → table of contents
        module_context("httpx._client")                → importable API surface
        symbol_context("httpx._client.Client.get")     → full signature + docs
    """

    def __init__(self, db_path: str):
        self._store = _IndexStore(db_path)

    @tool
    def package_context(self, package_name: str) -> str:
        """Get overview of a package: one line per module with a docstring summary.

        Args:
            package_name: The name of the package to look up (e.g., "httpx", "requests").

        Returns:
            A formatted string with package overview, or "Package 'X' not found in index."
        """
        return self._store.package_context(package_name)

    @tool
    def module_context(self, module_name: str) -> str:
        """Get all public objects in a module with short signatures.

        Overloaded functions are shown as a grouped block of typed signatures.

        Args:
            module_name: The fully qualified module name (e.g., "httpx._client").

        Returns:
            A formatted string with module contents, or "Module 'X' not found in index."
        """
        return self._store.module_context(module_name)

    @tool
    def symbol_context(self, qualified_name: str) -> str:
        """Get full detail on a function or class.

        Overloaded functions are shown with all typed signatures listed.

        Args:
            qualified_name: The fully qualified symbol name
                (e.g., "httpx._client.Client.get", "httpx.Client").

        Returns:
            A formatted string with full symbol details, or "Symbol 'X' not found."
        """
        return self._store.symbol_context(qualified_name)


__all__ = ["build_index", "IndexStore"]
