"""Jedi-based symbol extraction."""

import logging
from inspect import Parameter

import jedi

from metalgate_code.context.data import _AttrData, _ClassData, _FuncData, _ParamData

logger = logging.getLogger("metalgate_code")


def _jedi_param_to_data(param: jedi.api.classes.ParamName, pos: int) -> _ParamData:
    """Extract parameter info from jedi ParamName."""

    # Map inspect.Parameter.kind to our string representation
    kind_map = {
        Parameter.POSITIONAL_ONLY: "positional_only",
        Parameter.POSITIONAL_OR_KEYWORD: "positional",
        Parameter.VAR_POSITIONAL: "var_positional",
        Parameter.KEYWORD_ONLY: "keyword_only",
        Parameter.VAR_KEYWORD: "var_keyword",
    }
    kind = kind_map.get(param.kind, "positional")

    # Try to get annotation and default from Jedi's inferred signature
    annotation: str | None = None
    default: str | None = None

    try:
        # Get the signature and extract from description if available
        # Format: "param: annotation = default" or "param = default"
        desc = param.description
        if desc and ":" in desc:
            # Extract annotation from description
            # Example: "x: int" or "x: int = 5"
            parts = desc.split(":", 1)
            if len(parts) == 2:
                ann_part = parts[1].strip()
                if "=" in ann_part:
                    ann_part = ann_part.split("=", 1)[0].strip()
                annotation = ann_part if ann_part else None

        if desc and "=" in desc:
            # Extract default from description
            parts = desc.rsplit("=", 1)
            if len(parts) == 2:
                default = parts[1].strip()
    except Exception:
        pass

    return _ParamData(
        name=param.name, annotation=annotation, default=default, kind=kind
    )


def _extract_function_jedi(
    name: jedi.api.classes.Name, module: str, package: str
) -> _FuncData | None:
    """Extract function data from jedi Name."""
    if name.type != "function":
        return None

    fd = _FuncData()
    fd.name = name.name
    fd.module = module
    fd.package = package
    fd.decorators = []
    fd._forwarding_target = None
    fd._forwarding_extra = []

    if name.full_name:
        fd.qualified_name = name.full_name
    else:
        fd.qualified_name = f"{module}.{name.name}"

    pos = name.get_definition_start_position()
    fd.line = pos[0] if pos else 1

    fd.docstring = name.docstring(raw=True) or None

    signatures = name.get_signatures()
    if signatures:
        sig = signatures[0]
        sig_str = sig.to_string()
        if " -> " in sig_str:
            fd.return_annotation = sig_str.split(" -> ")[-1]

        params = sig.params
        is_method = bool(name.parent() and name.parent().type == "class")
        fd.is_method = is_method

        if is_method and params:
            first_param = params[0]
            if first_param.name in ("self", "cls"):
                params = params[1:]

        fd.parent_class = None
        if is_method and name.parent():
            for parent in name.parent().infer():
                if hasattr(parent, "name"):
                    fd.parent_class = parent.name
                    break

        for i, p in enumerate(params):
            fd.parameters.append(_jedi_param_to_data(p, i))

        has_varargs = any(hasattr(p, "kind") and p.kind in (2, 4) for p in sig.params)
        fd._is_args_kwargs = has_varargs

    return fd


def _extract_class_jedi(
    name: jedi.api.classes.Name, module: str, package: str
) -> _ClassData | None:
    """Extract class data from jedi Name."""
    if name.type != "class":
        return None

    cd = _ClassData()
    cd.name = name.name
    cd.module = module
    cd.package = package
    cd.bases = []
    cd.decorators = []
    cd.attributes = []
    cd.method_names = []
    cd.docstring = None

    if name.full_name:
        cd.qualified_name = name.full_name
    else:
        cd.qualified_name = f"{module}.{name.name}"

    pos = name.get_definition_start_position()
    cd.line = pos[0] if pos else 1

    cd.docstring = name.docstring(raw=True) or None

    desc = name.description
    if "(" in desc and ")" in desc:
        base_part = desc[desc.find("(") + 1 : desc.find(")")]
        if base_part:
            cd.bases = [b.strip() for b in base_part.split(",")]

    try:
        members = name.defined_names()
        for member in members:
            if member.type == "function":
                cd.method_names.append(member.name)
            elif member.type == "statement":
                annotation = None
                for inf in member.infer():
                    if hasattr(inf, "full_name") and inf.full_name:
                        annotation = inf.full_name
                    break
                cd.attributes.append(_AttrData(name=member.name, annotation=annotation))
    except Exception as e:
        logger.debug(f"Failed to extract class members for {cd.name}: {e}")

    return cd


__all__ = [
    "_jedi_param_to_data",
    "_extract_function_jedi",
    "_extract_class_jedi",
]
