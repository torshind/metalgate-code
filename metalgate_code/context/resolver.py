"""Forwarding resolution for *args/**kwargs functions."""

import logging
from collections import defaultdict

from metalgate_code.context.data import _ClassData, _DecoratorApp, _FuncData, _ParamData

logger = logging.getLogger("metalgate_code")


def _find_forwarding_target(
    func: _FuncData,
    all_funcs: list[_FuncData],
    all_classes: list[_ClassData],
    func_by_qname: dict[str, _FuncData],
    class_by_short: dict[str, list[_ClassData]],
) -> _FuncData | None:
    """Find the target function that func forwards to."""
    target_name = func._forwarding_target
    if not target_name:
        return None

    if target_name.startswith("self.") and func.parent_class:
        method = target_name.split(".")[-1]
        suffix = f".{func.parent_class}.{method}"
        cands = [
            f
            for f in all_funcs
            if f.qualified_name.endswith(suffix) and f.module == func.module
        ]
        return cands[0] if cands else None

    if target_name.startswith("super().") and func.parent_class:
        method = target_name.split(".")[-1]
        parent_classes = [
            c
            for c in all_classes
            if c.name == func.parent_class and c.module == func.module
        ]
        if parent_classes:
            for base_name in parent_classes[0].bases:
                for bc in class_by_short.get(base_name, []):
                    target_qn = f"{bc.qualified_name}.{method}"
                    if target_qn in func_by_qname:
                        return func_by_qname[target_qn]
        return None

    if "." in target_name and "(" not in target_name:
        cands = [
            f
            for f in all_funcs
            if f.qualified_name.endswith(f".{target_name}") and f.module == func.module
        ]
        if cands:
            return cands[0]
        cands = [f for f in all_funcs if f.qualified_name.endswith(f".{target_name}")]
        return cands[0] if cands else None

    cands = [
        f
        for f in all_funcs
        if f.qualified_name.endswith(f".{target_name}")
        and f.module == func.module
        and not f.is_method
    ]
    return cands[0] if cands else None


def _strip_extra_params(params: list[_ParamData], extra: list[str]) -> list[_ParamData]:
    """Strip extra positional args from parameter list."""
    if not extra:
        return list(params)
    n = len(extra)
    result, skipped = [], 0
    for p in params:
        if skipped < n and p.kind in ("positional", "positional_only"):
            skipped += 1
            continue
        result.append(p)
    return result


def _resolve_decorator_wrappers(
    all_funcs: list[_FuncData],
    all_apps: list[_DecoratorApp],
    func_by_qname: dict[str, _FuncData],
    resolved: set[str],
) -> int:
    """Resolve decorator wrapper functions. Returns count resolved."""
    decorator_wrappers: dict[str, tuple[_FuncData, str]] = {}
    for func in all_funcs:
        if not func._is_args_kwargs or not func._forwarding_target:
            continue
        t = func._forwarding_target
        if "." in t or "(" in t:
            continue
        parts = func.qualified_name.split(".")
        if len(parts) < 3:
            continue
        outer_qname = ".".join(parts[:-1])
        outer = func_by_qname.get(outer_qname)
        if outer and t in [p.name for p in outer.parameters]:
            decorator_wrappers[outer_qname] = (func, t)

    dec_resolved = 0
    for app in all_apps:
        matches = [
            q
            for q in decorator_wrappers
            if q.endswith(f".{app.decorator_name}") or q == app.decorator_name
        ]
        if not matches:
            continue
        wrapper, _ = decorator_wrappers[matches[0]]
        wrapped = func_by_qname.get(app.wrapped_function)
        if not wrapped:
            continue
        if wrapped._is_args_kwargs and wrapped.qualified_name not in resolved:
            continue
        wrapper.parameters = list(wrapped.parameters)
        wrapper.return_annotation = (
            wrapper.return_annotation or wrapped.return_annotation
        )
        wrapper.docstring = wrapper.docstring or wrapped.docstring
        wrapper.forwards_to = wrapped.qualified_name
        wrapper.resolved_from = f"decorator:{matches[0]} -> {wrapped.qualified_name}"
        resolved.add(wrapper.qualified_name)
        dec_resolved += 1

    return dec_resolved


def _forwarding_convergence_pass(
    all_funcs: list[_FuncData],
    func_by_qname: dict[str, _FuncData],
    class_by_short: dict[str, list[_ClassData]],
    all_classes: list[_ClassData],
    resolved: set[str],
) -> int:
    """Single pass of forwarding resolution. Returns progress count."""
    progress = 0
    for func in all_funcs:
        if func.qualified_name in resolved or not func._is_args_kwargs:
            continue
        if not func._forwarding_target:
            continue
        target = _find_forwarding_target(
            func, all_funcs, all_classes, func_by_qname, class_by_short
        )
        if not target:
            continue
        if target._is_args_kwargs and target.qualified_name not in resolved:
            continue
        func.parameters = _strip_extra_params(target.parameters, func._forwarding_extra)
        func.return_annotation = func.return_annotation or target.return_annotation
        func.docstring = func.docstring or target.docstring
        func.forwards_to = target.qualified_name
        func.resolved_from = target.qualified_name
        resolved.add(func.qualified_name)
        progress += 1
    return progress


def _resolve_forwarding(
    all_funcs: list[_FuncData],
    all_classes: list[_ClassData],
    all_apps: list[_DecoratorApp],
    max_depth: int = 10,
) -> dict:
    """Resolve *args/**kwargs forwarding to their actual targets."""
    func_by_qname = {f.qualified_name: f for f in all_funcs}
    class_by_short: dict[str, list[_ClassData]] = defaultdict(list)
    for c in all_classes:
        class_by_short[c.name].append(c)

    resolved: set[str] = set()
    stats = {"passes": 0, "resolved": 0, "unresolvable": 0, "opaque": 0}

    # Iterative convergence
    for pass_num in range(max_depth):
        progress = _forwarding_convergence_pass(
            all_funcs, func_by_qname, class_by_short, all_classes, resolved
        )
        stats["passes"] = pass_num + 1
        if progress == 0:
            break

    # Decorator tracing
    dec_resolved = _resolve_decorator_wrappers(
        all_funcs, all_apps, func_by_qname, resolved
    )

    # Post-decorator convergence
    if dec_resolved > 0:
        for pass_num in range(stats["passes"], stats["passes"] + max_depth):
            progress = _forwarding_convergence_pass(
                all_funcs, func_by_qname, class_by_short, all_classes, resolved
            )
            stats["passes"] = pass_num + 1
            if progress == 0:
                break

    stats["resolved"] = len(resolved)
    for func in all_funcs:
        if not func._is_args_kwargs or func.qualified_name in resolved:
            continue
        if func._forwarding_target:
            stats["unresolvable"] += 1
        else:
            stats["opaque"] += 1
    return stats


__all__ = ["_resolve_forwarding"]
