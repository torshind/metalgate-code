"""Async streaming writer - writes packages incrementally to the database."""

import asyncio
import logging
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from metalgate_code.context.data import (
    _ClassData,
    _DecoratorApp,
    _FuncData,
    _ModuleData,
)
from metalgate_code.context.db.models import (
    Attribute,
    Base,
    Class,
    Function,
    Module,
    Package,
    Parameter,
)
from metalgate_code.context.parsing import collect_files, find_site_packages, parse_file
from metalgate_code.context.resolver import _resolve_forwarding
from metalgate_code.helpers.paths import get_index_data_dir

logger = logging.getLogger("metalgate_code")


class StreamingWriter:
    """Async background writer that indexes packages incrementally.

    Writes each package to the database as soon as it's processed,
    making the index queryable immediately without waiting for
    all packages to complete.
    """

    def __init__(
        self,
        cwd: str,
        python: str | None = None,
        site_roots: list[str] | None = None,
        on_package_done: Callable[[str], None] | None = None,
    ):
        self.db_path = get_index_data_dir(cwd)
        self.python = python
        self.site_roots = site_roots
        self.on_package_done = on_package_done
        self._task: asyncio.Task | None = None

        async_url = f"sqlite+aiosqlite:///{self.db_path}"
        self._engine = create_async_engine(async_url, echo=False)
        self._async_session = async_sessionmaker(self._engine, expire_on_commit=False)

    async def start(self) -> None:
        """Start background indexing."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop indexing."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._engine.dispose()

    def is_running(self) -> bool:
        """Check if indexing is active."""
        return self._task is not None and not self._task.done()

    async def wait_for_completion(self) -> None:
        """Wait for indexing to complete."""
        if self._task:
            await self._task

    async def _run(self) -> None:
        """Main indexing loop."""
        if self.site_roots is None:
            roots = find_site_packages(self.python)
        else:
            roots = [Path(s) for s in self.site_roots]
        if not roots:
            logger.warning("No site-packages found")
            return

        files = collect_files(roots)

        # Group by package
        by_package: dict[str, list[Path]] = {}
        for f in files:
            pkg = self._guess_package(f, roots)
            by_package.setdefault(pkg, []).append(f)

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        packages = sorted(by_package.keys())
        logger.info(f"Starting async indexing of {len(packages)} packages")

        for pkg_name in packages:
            pkg_files = by_package[pkg_name]
            await self._write_package(pkg_name, pkg_files, roots)

            if self.on_package_done:
                try:
                    self.on_package_done(pkg_name)
                except Exception:
                    pass

        logger.info("Async indexing complete")

    def _guess_package(self, file: Path, roots: list[Path]) -> str:
        """Guess package name from file path."""
        for root in roots:
            try:
                rel = file.relative_to(root)
                parts = rel.parts
                if len(parts) > 0:
                    if len(parts) > 1 and parts[1] in ("__init__.py",):
                        return parts[0]
                    return parts[0]
            except ValueError:
                continue
        return "unknown"

    def _get_package_version(self, pkg_name: str, files: list[Path]) -> str | None:
        """Extract package version from __init__.py or metadata files."""
        # Look for __init__.py with __version__
        for f in files:
            if f.name == "__init__.py":
                try:
                    content = f.read_text()
                    for line in content.split("\n"):
                        if "__version__" in line and "=" in line:
                            # Extract version string
                            parts = line.split("=")
                            if len(parts) >= 2:
                                version = parts[1].strip().strip("\"'")
                                return version if version else None
                except Exception:
                    pass
        return None

    async def _clear_package_data(self, session: AsyncSession, pkg_name: str) -> None:
        """Clear all existing data for a package (used when version changes)."""
        from sqlalchemy import delete

        # Get all package IDs with this name (may be multiple versions)
        result = await session.execute(
            select(Package.id).where(Package.name == pkg_name)
        )
        pkg_ids = [r for r in result.scalars().all()]
        if not pkg_ids:
            return

        # Delete in order: parameters -> functions -> attributes -> classes -> modules
        # Get function IDs for all package versions
        func_result = await session.execute(
            select(Function.id).where(
                Function.module_id.in_(
                    select(Module.id).where(Module.package_id.in_(pkg_ids))
                )
            )
        )
        func_ids = [r for r in func_result.scalars().all()]

        if func_ids:
            await session.execute(
                delete(Parameter).where(Parameter.function_id.in_(func_ids))
            )
            await session.execute(delete(Function).where(Function.id.in_(func_ids)))

        # Get class IDs for all package versions
        class_result = await session.execute(
            select(Class.id).where(
                Class.module_id.in_(
                    select(Module.id).where(Module.package_id.in_(pkg_ids))
                )
            )
        )
        class_ids = [r for r in class_result.scalars().all()]

        if class_ids:
            await session.execute(
                delete(Attribute).where(Attribute.class_id.in_(class_ids))
            )
            await session.execute(delete(Class).where(Class.id.in_(class_ids)))

        await session.execute(delete(Module).where(Module.package_id.in_(pkg_ids)))
        await session.execute(delete(Package).where(Package.id.in_(pkg_ids)))

    async def _write_package(
        self,
        pkg_name: str,
        files: list[Path],
        roots: list[Path],
    ) -> None:
        """Index a single package and write to DB."""
        modules: list[_ModuleData] = []
        funcs: list[_FuncData] = []
        classes: list[_ClassData] = []
        apps: list[_DecoratorApp] = []

        # Parse all files concurrently
        async def parse_one(f: Path):
            try:
                return parse_file(f, roots)
            except Exception as e:
                logger.warning(f"Parse error {f}: {e}")
                return None

        results = await asyncio.gather(*[parse_one(f) for f in files])

        for result in results:
            if result:
                md, fns, cls, ap = result
                modules.append(md)
                funcs.extend(fns)
                classes.extend(cls)
                apps.extend(ap)

        if not modules:
            return

        # Resolve forwarding
        _resolve_forwarding(funcs, classes, apps)

        # Get package version
        version = self._get_package_version(pkg_name, files)

        # Write to DB
        async with self._async_session() as session:
            async with session.begin():
                # Check if package exists with same version
                result = await session.execute(
                    select(Package).where(
                        Package.name == pkg_name, Package.version == version
                    )
                )
                if result.scalar_one_or_none():
                    # Same package + same version: skip
                    logger.debug(f"Skipping {pkg_name}@{version}: already indexed")
                    return

                # Check if package exists with different version
                result = await session.execute(
                    select(Package).where(Package.name == pkg_name)
                )
                if result.scalar_one_or_none():
                    # Same package + different version: clear old data first
                    logger.info(f"Clearing old data for {pkg_name}")
                    await self._clear_package_data(session, pkg_name)

                # Create new package with version
                pkg = Package(name=pkg_name, version=version)
                session.add(pkg)
                await session.flush()

                # Modules
                mod_map: dict[str, Module] = {}
                for md in modules:
                    mod = Module(
                        name=md.module,
                        package_id=pkg.id,
                        file=md.file,
                        docstring=md.docstring,
                    )
                    mod.exports = md.exports
                    session.add(mod)
                    await session.flush()
                    mod_map[md.module] = mod

                # Classes
                cls_map: dict[str, Class] = {}
                for cd in classes:
                    mod = mod_map.get(cd.module)
                    if not mod:
                        logger.warning(
                            f"Module not found for class {cd.qualified_name}: {cd.module}"
                        )
                        continue
                    cls = Class(
                        qualified_name=cd.qualified_name,
                        qualified_name_rev=".".join(
                            reversed(cd.qualified_name.split("."))
                        ),
                        name=cd.name,
                        module_id=mod.id,
                        line=cd.line,
                        docstring=cd.docstring,
                    )
                    cls.bases = cd.bases
                    cls.decorators = cd.decorators
                    cls.method_names = cd.method_names
                    session.add(cls)
                    await session.flush()
                    cls_map[cd.qualified_name] = cls

                # Attributes
                for cd in classes:
                    cls = cls_map.get(cd.qualified_name)
                    if not cls:
                        continue
                    for ad in cd.attributes:
                        session.add(
                            Attribute(
                                class_id=cls.id,
                                name=ad.name,
                                annotation=ad.annotation,
                                default=ad.default,
                            )
                        )

                # Functions
                for fd in funcs:
                    mod = mod_map.get(fd.module)
                    if not mod:
                        logger.warning(
                            f"Module not found for function {fd.qualified_name}: {fd.module}"
                        )
                        continue

                    class_id = None
                    if fd.parent_class:
                        parent_qname = fd.qualified_name.rsplit(".", 1)[0]
                        cls = cls_map.get(parent_qname)
                        if cls:
                            class_id = cls.id

                    func = Function(
                        qualified_name=fd.qualified_name,
                        qualified_name_rev=".".join(
                            reversed(fd.qualified_name.split("."))
                        ),
                        name=fd.name,
                        module_id=mod.id,
                        class_id=class_id,
                        line=fd.line,
                        return_annotation=fd.return_annotation,
                        docstring=fd.docstring,
                        is_method=fd.is_method,
                        parent_class_name=fd.parent_class,
                        forwards_to=fd.forwards_to,
                        resolved_from=fd.resolved_from,
                    )
                    func.decorators = fd.decorators
                    session.add(func)
                    await session.flush()

                    for i, pd in enumerate(fd.parameters):
                        session.add(
                            Parameter(
                                function_id=func.id,
                                position=i,
                                name=pd.name,
                                annotation=pd.annotation,
                                default=pd.default,
                                kind=pd.kind,
                            )
                        )

        logger.debug(
            f"Indexed: {pkg_name} ({len(modules)} mods, {len(funcs)} funcs, {len(classes)} cls)"
        )


__all__ = ["StreamingWriter"]
