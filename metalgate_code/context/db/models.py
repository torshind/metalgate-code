"""Database models for the symbol index."""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class Package(Base):
    __tablename__ = "package"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, index=True, unique=True)

    modules: Mapped[list[Module]] = relationship("Module", back_populates="package_rel")


class Module(Base):
    __tablename__ = "module"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, index=True)
    package_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("package.id"), index=True
    )
    file: Mapped[str] = mapped_column(String)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exports_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    package_rel: Mapped[Optional[Package]] = relationship(
        "Package", back_populates="modules"
    )
    functions: Mapped[list[Function]] = relationship(
        "Function", back_populates="module_rel"
    )
    classes: Mapped[list[Class]] = relationship("Class", back_populates="module_rel")

    @property
    def exports(self) -> list[str]:
        return json.loads(self.exports_json) if self.exports_json else []

    @exports.setter
    def exports(self, val: list[str]) -> None:
        self.exports_json = json.dumps(val) if val else None


class Function(Base):
    __tablename__ = "function"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    qualified_name: Mapped[str] = mapped_column(String, index=True)
    # Reversed qualified name for efficient suffix matching (e.g., "a.b.c" -> "c.b.a")
    qualified_name_rev: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[int] = mapped_column(Integer, ForeignKey("module.id"), index=True)
    class_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("class.id"), nullable=True, index=True
    )
    line: Mapped[int] = mapped_column(Integer)
    return_annotation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decorators_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_method: Mapped[bool] = mapped_column(Boolean, default=False)
    parent_class_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    forwards_to: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    resolved_from: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    module_rel: Mapped[Optional[Module]] = relationship(
        "Module", back_populates="functions"
    )
    class_rel: Mapped[Optional[Class]] = relationship(
        "Class", back_populates="methods_rel"
    )
    parameters: Mapped[list[Parameter]] = relationship(
        "Parameter", back_populates="function_rel"
    )

    @property
    def decorators(self) -> list[str]:
        return json.loads(self.decorators_json) if self.decorators_json else []

    @decorators.setter
    def decorators(self, val: list[str]) -> None:
        self.decorators_json = json.dumps(val) if val else None


class Parameter(Base):
    __tablename__ = "parameter"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    function_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("function.id"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String)
    annotation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    default: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    kind: Mapped[str] = mapped_column(String, default="positional")

    function_rel: Mapped[Optional[Function]] = relationship(
        "Function", back_populates="parameters"
    )


class Class(Base):
    __tablename__ = "class"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    qualified_name: Mapped[str] = mapped_column(String, index=True)
    # Reversed qualified name for efficient suffix matching (e.g., "a.b.c" -> "c.b.a")
    qualified_name_rev: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[int] = mapped_column(Integer, ForeignKey("module.id"), index=True)
    line: Mapped[int] = mapped_column(Integer)
    bases_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    docstring: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decorators_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    method_names_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    module_rel: Mapped[Optional[Module]] = relationship(
        "Module", back_populates="classes"
    )
    attributes: Mapped[list[Attribute]] = relationship(
        "Attribute", back_populates="class_rel"
    )
    methods_rel: Mapped[list[Function]] = relationship(
        "Function", back_populates="class_rel"
    )

    @property
    def bases(self) -> list[str]:
        return json.loads(self.bases_json) if self.bases_json else []

    @bases.setter
    def bases(self, val: list[str]) -> None:
        self.bases_json = json.dumps(val) if val else None

    @property
    def decorators(self) -> list[str]:
        return json.loads(self.decorators_json) if self.decorators_json else []

    @decorators.setter
    def decorators(self, val: list[str]) -> None:
        self.decorators_json = json.dumps(val) if val else None

    @property
    def method_names(self) -> list[str]:
        return json.loads(self.method_names_json) if self.method_names_json else []

    @method_names.setter
    def method_names(self, val: list[str]) -> None:
        self.method_names_json = json.dumps(val) if val else None


class Attribute(Base):
    __tablename__ = "attribute"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    class_id: Mapped[int] = mapped_column(Integer, ForeignKey("class.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    annotation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    default: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    class_rel: Mapped[Optional[Class]] = relationship(
        "Class", back_populates="attributes"
    )


__all__ = [
    "Base",
    "Package",
    "Module",
    "Function",
    "Parameter",
    "Class",
    "Attribute",
]
