"""Transient data structures for symbol extraction (not persisted directly)."""


class _ParamData:
    __slots__ = ("name", "annotation", "default", "kind")

    def __init__(
        self,
        name: str,
        annotation: str | None = None,
        default: str | None = None,
        kind: str = "positional",
    ):
        self.name = name
        self.annotation = annotation
        self.default = default
        self.kind = kind


class _AttrData:
    __slots__ = ("name", "annotation", "default")

    def __init__(
        self, name: str, annotation: str | None = None, default: str | None = None
    ):
        self.name = name
        self.annotation = annotation
        self.default = default


class _FuncData:
    __slots__ = (
        "qualified_name",
        "name",
        "module",
        "package",
        "line",
        "parameters",
        "return_annotation",
        "docstring",
        "decorators",
        "is_method",
        "parent_class",
        "forwards_to",
        "resolved_from",
        "_forwarding_target",
        "_forwarding_extra",
        "_is_args_kwargs",
    )

    def __init__(self):
        self.qualified_name: str = ""
        self.name: str = ""
        self.module: str = ""
        self.package: str = ""
        self.line: int = 0
        self.parameters: list[_ParamData] = []
        self.return_annotation: str | None = None
        self.docstring: str | None = None
        self.decorators: list[str] = []
        self.is_method: bool = False
        self.parent_class: str | None = None
        self.forwards_to: str | None = None
        self.resolved_from: str | None = None
        self._forwarding_target: str | None = None
        self._forwarding_extra: list[str] = []
        self._is_args_kwargs: bool = False


class _ClassData:
    __slots__ = (
        "qualified_name",
        "name",
        "module",
        "package",
        "line",
        "bases",
        "docstring",
        "decorators",
        "attributes",
        "method_names",
    )

    def __init__(self):
        self.qualified_name: str = ""
        self.name: str = ""
        self.module: str = ""
        self.package: str = ""
        self.line: int = 0
        self.bases: list[str] = []
        self.docstring: str | None = None
        self.decorators: list[str] = []
        self.attributes: list[_AttrData] = []
        self.method_names: list[str] = []


class _ModuleData:
    __slots__ = ("module", "package", "file", "docstring", "exports")

    def __init__(self):
        self.module: str = ""
        self.package: str = ""
        self.file: str = ""
        self.docstring: str | None = None
        self.exports: list[str] = []


class _DecoratorApp:
    __slots__ = ("decorator_name", "wrapped_function")

    def __init__(self, decorator_name: str, wrapped_function: str):
        self.decorator_name = decorator_name
        self.wrapped_function = wrapped_function


__all__ = [
    "_ParamData",
    "_AttrData",
    "_FuncData",
    "_ClassData",
    "_ModuleData",
    "_DecoratorApp",
]
