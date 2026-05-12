"""Database writer for the symbol index."""

import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from metalgate_code.context.data import _ClassData, _FuncData, _ModuleData
from metalgate_code.context.db.models import (
    Attribute,
    Base,
    Class,
    Function,
    Module,
    Package,
    Parameter,
)

logger = logging.getLogger("metalgate_code")


def write_index(
    db_path: Path,
    all_modules: list[_ModuleData],
    all_funcs: list[_FuncData],
    all_classes: list[_ClassData],
):
    """Write symbol data to the database."""

    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_conn, connection_record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Packages
        pkg_map: dict[str, Package] = {}
        for md in all_modules:
            if md.package not in pkg_map:
                pkg = Package(name=md.package)
                session.add(pkg)
                pkg_map[md.package] = pkg
        session.flush()

        # Modules
        mod_map: dict[str, Module] = {}
        for md in all_modules:
            mod = Module(
                name=md.module,
                package_id=pkg_map[md.package].id,
                file=md.file,
                docstring=md.docstring,
            )
            mod.exports = md.exports
            session.add(mod)
            mod_map[md.module] = mod
        session.flush()

        # Classes
        cls_map: dict[str, Class] = {}
        for cd in all_classes:
            mod = mod_map.get(cd.module)
            if not mod:
                logger.warning(
                    f"Module not found for class {cd.qualified_name}: {cd.module}"
                )
                continue
            cls = Class(
                qualified_name=cd.qualified_name,
                qualified_name_rev=".".join(reversed(cd.qualified_name.split("."))),
                name=cd.name,
                module_id=mod.id,
                line=cd.line,
                docstring=cd.docstring,
            )
            cls.bases = cd.bases
            cls.decorators = cd.decorators
            cls.method_names = cd.method_names
            session.add(cls)
            cls_map[cd.qualified_name] = cls
        session.flush()

        # Attributes
        for cd in all_classes:
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

        # Functions + Parameters
        for fd in all_funcs:
            mod = mod_map.get(fd.module)
            if not mod:
                logger.warning(
                    f"Module not found for function {fd.qualified_name}: {fd.module}"
                )
                continue
            # Find parent class DB id
            class_id = None
            if fd.parent_class:
                parent_qname = fd.qualified_name.rsplit(".", 1)[0]
                cls = cls_map.get(parent_qname)
                if cls:
                    class_id = cls.id

            func = Function(
                qualified_name=fd.qualified_name,
                qualified_name_rev=".".join(reversed(fd.qualified_name.split("."))),
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
            session.flush()

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

        session.commit()


__all__ = ["write_index"]
