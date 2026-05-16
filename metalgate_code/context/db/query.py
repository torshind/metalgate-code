"""Query layer for the symbol index - IndexStore API."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from metalgate_code.context.db.models import (
    Attribute,
    Class,
    Function,
    Module,
    Package,
    Parameter,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger("metalgate_code")


# helpers


def _group_functions(funcs: list) -> list[list]:
    """Group a flat list of Function rows by name, preserving first-seen order."""
    groups: dict[str, list] = defaultdict(list)
    for f in funcs:
        groups[f.name].append(f)
    return list(groups.values())


def _overload_variants(group: list) -> list:
    """Return the @overload-decorated members of a group."""
    return [f for f in group if "overload" in f.decorators]


def _implementation(group: list):
    """Return the non-@overload member (the implementation), or None."""
    impls = [f for f in group if "overload" not in f.decorators]
    return impls[0] if impls else None


# agent-facing API


class IndexStore:
    """
    Agent tool interface to the precomputed symbol index.

    Three methods, three zoom levels:
        package_context("httpx")                       → table of contents
        module_context("httpx._client")                → importable API surface
        symbol_context("httpx._client.Client.get")     → full signature + docs
    """

    def __init__(self, db_path: str | Path):
        self._engine: Engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Enable foreign key support
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    def _session(self) -> Session:
        return Session(self._engine)

    def package_context(self, package_name: str) -> str:
        """
        Overview of a package: one line per module with a docstring summary.
        """
        with self._session() as s:
            pkg = s.execute(
                select(Package).where(Package.name == package_name)
            ).scalar_one_or_none()
            if not pkg:
                return f"Package '{package_name}' not found in index."

            modules = list(
                s.execute(
                    select(Module)
                    .where(Module.package_id == pkg.id)
                    .order_by(Module.name)
                )
                .scalars()
                .all()
            )

            version_str = f"@{pkg.version}" if pkg.version else ""
            lines = [
                f"Package: {package_name}{version_str}",
                f"Modules: {len(modules)}",
                "",
            ]
            for mod in modules:
                line = f"  {mod.name}"
                if mod.docstring:
                    first = mod.docstring.split("\n")[0].strip()
                    if first:
                        line += f"  — {first}"
                lines.append(line)

            return "\n".join(lines)

    def module_context(self, module_name: str) -> str:
        """
        All public objects in a module with short signatures.
        Overloaded functions are shown as a grouped block of typed signatures.
        """
        with self._session() as s:
            mod = s.execute(
                select(Module).where(Module.name == module_name)
            ).scalar_one_or_none()
            if not mod:
                return f"Module '{module_name}' not found in index."

            classes = list(
                s.execute(
                    select(Class).where(Class.module_id == mod.id).order_by(Class.line)
                )
                .scalars()
                .all()
            )
            top_funcs_raw = list(
                s.execute(
                    select(Function)
                    .where(
                        Function.module_id == mod.id,
                        Function.is_method == False,  # noqa: E712
                    )
                    .order_by(Function.line)
                )
                .scalars()
                .all()
            )

            lines = [f"Module: {module_name}"]
            if mod.docstring:
                first = mod.docstring.split("\n")[0].strip()
                if first:
                    lines.append(first)
            lines.append("")

            public_classes = [c for c in classes if not c.name.startswith("_")]
            if public_classes:
                lines.append("Classes:")
                for cls in public_classes:
                    bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
                    cls_line = f"  class {cls.name}{bases_str}"
                    if cls.docstring:
                        first = cls.docstring.split("\n")[0].strip()
                        if first:
                            cls_line += f"  — {first}"
                    lines.append(cls_line)

                    attrs = list(
                        s.execute(select(Attribute).where(Attribute.class_id == cls.id))
                        .scalars()
                        .all()
                    )
                    for a in attrs:
                        if a.name.startswith("_"):
                            continue
                        a_line = f"      {a.name}"
                        if a.annotation:
                            a_line += f": {a.annotation}"
                        if a.default:
                            a_line += f" = {a.default}"
                        lines.append(a_line)

                    show_names = [m for m in cls.method_names if not m.startswith("_")]
                    if "__init__" in cls.method_names:
                        show_names = ["__init__"] + show_names
                    if "__call__" in cls.method_names:
                        show_names.append("__call__")

                    all_methods = list(
                        s.execute(select(Function).where(Function.class_id == cls.id))
                        .scalars()
                        .all()
                    )
                    method_groups: dict[str, list] = defaultdict(list)
                    for mf in all_methods:
                        method_groups[mf.name].append(mf)

                    rendered: set[str] = set()
                    for mname in show_names:
                        if mname in rendered:
                            continue
                        rendered.add(mname)
                        group = method_groups.get(mname)
                        if not group:
                            lines.append(f"    def {mname}(...)")
                            continue
                        self._render_func_group(group, s, lines, indent="    ")
                    lines.append("")

            func_groups = _group_functions(top_funcs_raw)
            public_groups = [g for g in func_groups if not g[0].name.startswith("_")]
            if public_groups:
                lines.append("Functions:")
                for group in public_groups:
                    self._render_func_group(group, s, lines, indent="  ")

            return "\n".join(lines)

    def _render_func_group(
        self, group: list[Function], s: Session, lines: list[str], indent: str
    ) -> None:
        """
        Append one or more lines for a function group to `lines`.

        - Single entry   → normal `def name(...)` line.
        - True dups      → last entry wins (re-export / identical overload).
        - Has @overloads → header line with count, then each typed signature indented.
        """
        overloads = _overload_variants(group)
        impl = _implementation(group)

        if not overloads:
            f = impl or group[-1]
            lines.append(f"{indent}{self._sig_short(f, s)}")
            return

        # Pick the best source for docstring / return type
        ref = impl or overloads[-1]
        first_doc = ""
        if ref.docstring:
            first_doc = ref.docstring.split("\n")[0].strip()

        n = len(overloads)
        header = f"{indent}def {group[0].name}  — {n} overload{'s' if n > 1 else ''}"
        if first_doc:
            header += f"  — {first_doc}"
        lines.append(header)

        for ov in overloads:
            lines.append(f"{indent}  {self._sig_short(ov, s)}")

    def symbol_context(self, qualified_name: str) -> str:
        """
        Full detail on a function or class.
        Overloaded functions are shown with all typed signatures listed.
        """
        with self._session() as s:
            # exact qualified-name match
            func_exact = list(
                s.execute(
                    select(Function).where(Function.qualified_name == qualified_name)
                )
                .scalars()
                .all()
            )
            if func_exact:
                return self._format_overload_group(func_exact, s)

            cls = s.execute(
                select(Class).where(Class.qualified_name == qualified_name)
            ).scalar_one_or_none()
            if cls:
                return self._format_class_full(cls, s)

            # suffix match (using reversed column for index efficiency)
            # "a.b.c".endswith("b.c") == "c.b.a".startswith("c.b")
            rev_suffix = ".".join(reversed(qualified_name.split(".")))
            func_matches = list(
                s.execute(
                    select(Function).where(
                        Function.qualified_name_rev.startswith(rev_suffix)
                    )
                )
                .scalars()
                .all()
            )
            cls_matches = list(
                s.execute(
                    select(Class).where(Class.qualified_name_rev.startswith(rev_suffix))
                )
                .scalars()
                .all()
            )

            if func_matches:
                # Group by qualified_name — overloads share the same qname
                qname_groups: dict[str, list] = defaultdict(list)
                for f in func_matches:
                    qname_groups[f.qualified_name].append(f)

                if len(qname_groups) == 1:
                    # All hits resolve to one symbol (possibly with overloads)
                    group = list(qname_groups.values())[0]
                    return self._format_overload_group(group, s)

                if not cls_matches:
                    lines = [f"Ambiguous '{qualified_name}'. Matches:"]
                    for qn, grp in list(qname_groups.items())[:10]:
                        n = len(_overload_variants(grp))
                        suffix = f"  ({n} overloads)" if n else ""
                        lines.append(f"  {qn} (function){suffix}")
                    return "\n".join(lines)

            if len(cls_matches) == 1 and not func_matches:
                return self._format_class_full(cls_matches[0], s)

            if func_matches or cls_matches:
                lines = [f"Ambiguous '{qualified_name}'. Matches:"]
                for f in func_matches[:10]:
                    lines.append(f"  {f.qualified_name} (function)")
                for c in cls_matches[:10]:
                    lines.append(f"  {c.qualified_name} (class)")
                return "\n".join(lines)

            # short-name fallback
            func_matches = list(
                s.execute(select(Function).where(Function.name == qualified_name))
                .scalars()
                .all()
            )
            cls_matches = list(
                s.execute(select(Class).where(Class.name == qualified_name))
                .scalars()
                .all()
            )
            if func_matches or cls_matches:
                lines = [f"No exact match for '{qualified_name}'. Did you mean:"]
                for f in func_matches[:10]:
                    lines.append(f"  {f.qualified_name} (function)")
                for c in cls_matches[:10]:
                    lines.append(f"  {c.qualified_name} (class)")
                return "\n".join(lines)

            return f"Symbol '{qualified_name}' not found."

    # formatting helpers

    def _get_params(self, func: Function, s: Session) -> list[Parameter]:
        return list(
            s.execute(
                select(Parameter)
                .where(Parameter.function_id == func.id)
                .order_by(Parameter.position)
            )
            .scalars()
            .all()
        )

    def _sig_short(self, func: Function, s: Session) -> str:
        params = self._get_params(func, s)
        parts = []
        for p in params:
            prefix = {"var_positional": "*", "var_keyword": "**"}.get(p.kind, "")
            part = prefix + p.name
            if p.annotation:
                part += f": {p.annotation}"
            if p.default:
                part += f"={p.default}"
            parts.append(part)
        ret = f" -> {func.return_annotation}" if func.return_annotation else ""
        sig = f"def {func.name}({', '.join(parts)}){ret}"
        if func.docstring:
            first = func.docstring.split("\n")[0].strip()
            if first:
                sig += f"  — {first}"
        return sig

    def _format_func_full(self, func: Function, s: Session) -> str:
        params = self._get_params(func, s)
        mod = s.execute(
            select(Module).where(Module.id == func.module_id)
        ).scalar_one_or_none()
        mod_name = mod.name if mod else "?"

        lines = [f"Function: {func.qualified_name}"]
        lines.append(f"Defined in: {mod_name} ({func.line})")
        if func.decorators:
            lines.append(f"Decorators: {', '.join('@' + d for d in func.decorators)}")
        if func.forwards_to:
            lines.append(f"Forwards to: {func.forwards_to}")
        lines.append("")

        ret = f" -> {func.return_annotation}" if func.return_annotation else ""
        if not params:
            lines.append(f"def {func.name}(){ret}")
        else:
            lines.append(f"def {func.name}(")
            for p in params:
                prefix = {"var_positional": "*", "var_keyword": "**"}.get(p.kind, "")
                pline = f"    {prefix}{p.name}"
                if p.annotation:
                    pline += f": {p.annotation}"
                if p.default:
                    pline += f" = {p.default}"
                pline += ","
                kind_note = {
                    "positional_only": "  # positional-only",
                    "keyword_only": "  # keyword-only",
                }.get(p.kind, "")
                pline += kind_note
                lines.append(pline)
            lines.append(f"){ret}")

        if func.docstring:
            lines.append("")
            lines.append(func.docstring)

        return "\n".join(lines)

    def _format_class_full(self, cls: Class, s: Session) -> str:
        lines = [f"Class: {cls.qualified_name}"]
        mod = s.execute(
            select(Module).where(Module.id == cls.module_id)
        ).scalar_one_or_none()
        lines.append(f"Defined in: {mod.name if mod else '?'} ({cls.line})")
        if cls.bases:
            lines.append(f"Bases: {', '.join(cls.bases)}")
        if cls.decorators:
            lines.append(f"Decorators: {', '.join('@' + d for d in cls.decorators)}")
        lines.append("")

        if cls.docstring:
            lines.append(cls.docstring)
            lines.append("")

        attrs = list(
            s.execute(select(Attribute).where(Attribute.class_id == cls.id))
            .scalars()
            .all()
        )
        public_attrs = [a for a in attrs if not a.name.startswith("_")]
        if public_attrs:
            lines.append("Attributes:")
            for a in public_attrs:
                aline = f"  {a.name}"
                if a.annotation:
                    aline += f": {a.annotation}"
                if a.default:
                    aline += f" = {a.default}"
                lines.append(aline)
            lines.append("")

        # constructor
        init = s.execute(
            select(Function).where(
                Function.class_id == cls.id, Function.name == "__init__"
            )
        ).scalar_one_or_none()
        if init:
            lines.append("Constructor:")
            lines.append(self._format_func_full(init, s))
            lines.append("")

        # Methods
        methods = list(
            s.execute(
                select(Function)
                .where(Function.class_id == cls.id)
                .order_by(Function.line)
            )
            .scalars()
            .all()
        )
        _SHOW_METHODS = frozenset(
            {
                "__call__",
                "__enter__",
                "__exit__",
                "__aenter__",
                "__aexit__",
                "__getitem__",
                "__setitem__",
                "__len__",
                "__iter__",
                "__next__",
                "__contains__",
                "__eq__",
                "__hash__",
                "__repr__",
                "__str__",
            }
        )
        show = [
            m for m in methods if not m.name.startswith("_") or m.name in _SHOW_METHODS
        ]
        if show:
            lines.append("Methods:")
            for mf in show:
                lines.append(f"  {self._sig_short(mf, s)}")

        return "\n".join(lines)

    def _format_overload_group(self, group: list[Function], s: Session) -> str:
        """
        Format a group of same-qualified-name functions for symbol_context.
        Falls through to the standard formatter when there are no @overload variants.
        """
        overloads = _overload_variants(group)
        impl = _implementation(group)

        if not overloads:
            return self._format_func_full(impl or group[-1], s)

        ref = impl or overloads[-1]
        mod = s.execute(
            select(Module).where(Module.id == ref.module_id)
        ).scalar_one_or_none()
        mod_name = mod.name if mod else "?"

        lines = [f"Function: {ref.qualified_name}"]
        lines.append(f"Defined in: {mod_name} ({ref.line})")
        non_overload_decs = [d for d in ref.decorators if d != "overload"]
        if non_overload_decs:
            lines.append(f"Decorators: {', '.join('@' + d for d in non_overload_decs)}")
        if ref.forwards_to:
            lines.append(f"Forwards to: {ref.forwards_to}")
        lines.append("")

        lines.append(f"Overloads ({len(overloads)}):")
        for ov in overloads:
            params = self._get_params(ov, s)
            ret = f" -> {ov.return_annotation}" if ov.return_annotation else ""
            if not params:
                lines.append(f"  def {ov.name}(){ret}")
            else:
                parts = []
                for p in params:
                    prefix = {"var_positional": "*", "var_keyword": "**"}.get(
                        p.kind, ""
                    )
                    part = prefix + p.name
                    if p.annotation:
                        part += f": {p.annotation}"
                    if p.default:
                        part += f" = {p.default}"
                    parts.append(part)
                lines.append(f"  def {ov.name}({', '.join(parts)}){ret}")

        if ref.docstring:
            lines.append("")
            lines.append(ref.docstring)

        return "\n".join(lines)


__all__ = ["IndexStore"]
