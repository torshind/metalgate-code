"""Tree-sitter based parsing for decorator detection and forwarding analysis."""

import logging

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from metalgate_code.context.data import _DecoratorApp, _FuncData

logger = logging.getLogger("metalgate_code")

MAX_DEPTH = 500
PY_LANGUAGE = Language(tspython.language())
ts_parser = Parser(PY_LANGUAGE)


def _safe_text_decode(node: Node | None) -> str | None:
    """Safely decode Node.text, returning None if node or text is missing."""
    return node.text.decode() if node and node.text else None


def _safe_text_decode_default(node: Node | None, default: str) -> str:
    """Safely decode Node.text, returning default if node or text is missing."""
    return node.text.decode() if node and node.text else default


def _has_args_kwargs(func_node: Node) -> bool:
    """Check if a function has *args or **kwargs parameters."""
    params_node = func_node.child_by_field_name("parameters")
    if not params_node:
        return False
    for child in params_node.children:
        if child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
            return True
        if child.type == "typed_parameter":
            for sub in child.children:
                if sub.type in ("list_splat_pattern", "dictionary_splat_pattern"):
                    return True
    return False


def _extract_varargs_kwargs(params_node: Node) -> tuple[str | None, str | None]:
    """Extract *args and **kwargs names from parameters node."""
    vararg_name = kwarg_name = None
    for child in params_node.children:
        if child.type == "list_splat_pattern":
            target = child.children[-1] if child.children else child
            vararg_name = _safe_text_decode(target) or vararg_name
        elif child.type == "dictionary_splat_pattern":
            target = child.children[-1] if child.children else child
            kwarg_name = _safe_text_decode(target) or kwarg_name
        elif child.type == "typed_parameter":
            for sub in child.children:
                if sub.type == "list_splat_pattern":
                    target = sub.children[-1] if sub.children else sub
                    vararg_name = _safe_text_decode(target) or vararg_name
                elif sub.type == "dictionary_splat_pattern":
                    target = sub.children[-1] if sub.children else sub
                    kwarg_name = _safe_text_decode(target) or kwarg_name
    return vararg_name, kwarg_name


def _extract_extra_args(call_node: Node) -> list[str]:
    """Extract extra args from a call (before *args/**kwargs)."""
    args_node = call_node.child_by_field_name("arguments")
    if not args_node:
        return []

    extra: list[str] = []
    for arg in args_node.children:
        if arg.type in ("(", ")", ","):
            continue
        if arg.type in ("list_splat", "dictionary_splat"):
            break
        if arg.type == "keyword_argument":
            continue
        text = _safe_text_decode(arg)
        if text:
            extra.append(text)
    return extra


def _call_forwards(args_node: Node, vn: str | None, kn: str | None) -> bool:
    """Check if a call forwards *args and **kwargs."""
    found_var = vn is None
    found_kw = kn is None
    for child in args_node.children:
        if child.type == "list_splat" and vn:
            for sub in child.children:
                if (
                    sub.type == "identifier"
                    and _safe_text_decode_default(sub, "") == vn
                ):
                    found_var = True
        if child.type == "dictionary_splat" and kn:
            for sub in child.children:
                if (
                    sub.type == "identifier"
                    and _safe_text_decode_default(sub, "") == kn
                ):
                    found_kw = True
    return found_var and found_kw


def _search_fwd(
    node: Node, vn: str | None, kn: str | None, depth: int = 0
) -> Node | None:
    """Recursively search for forwarding call."""
    if depth > MAX_DEPTH:
        logger.debug("Max depth reached in _search_fwd")
        return None
    if node.type in ("return_statement", "expression_statement", "yield", "yield_from"):
        for child in node.children:
            r = _search_fwd(child, vn, kn, depth + 1)
            if r:
                return r
    if node.type == "assignment":
        right = node.child_by_field_name("right")
        return _search_fwd(right, vn, kn, depth + 1) if right else None
    if node.type == "await":
        for child in node.children:
            r = _search_fwd(child, vn, kn, depth + 1)
            if r:
                return r
    if node.type == "call":
        args_node = node.child_by_field_name("arguments")
        if args_node and _call_forwards(args_node, vn, kn):
            return node
    return None


def _find_forwarding_call(body: Node, vn: str | None, kn: str | None) -> Node | None:
    """Search for a call that forwards *args/**kwargs."""
    for stmt in body.children:
        r = _search_fwd(stmt, vn, kn)
        if r:
            return r
    return None


def _detect_forwarding(node: Node) -> tuple[str, list[str]] | None:
    """Detect if a function forwards *args/**kwargs to another call.

    Returns (target_name, extra_args) or None.
    """
    body = node.child_by_field_name("body")
    params_node = node.child_by_field_name("parameters")
    if not body or not params_node:
        return None

    vararg_name, kwarg_name = _extract_varargs_kwargs(params_node)
    if not vararg_name and not kwarg_name:
        return None

    call_node = _find_forwarding_call(body, vararg_name, kwarg_name)
    if not call_node:
        return None

    target = _safe_text_decode(call_node.child_by_field_name("function"))
    if not target:
        return None

    return target, _extract_extra_args(call_node)


def _detect_decorator_apps_walk(
    node: Node,
    module: str,
    scope: list[str],
    func_by_line: dict[int, _FuncData],
    apps: list[_DecoratorApp],
    depth: int = 0,
):
    """Recursively find decorator applications."""
    if depth > MAX_DEPTH:
        logger.debug("Max depth reached in _detect_decorator_apps_walk")
        return
    if node.type == "decorated_definition":
        decs: list[str] = []
        inner = None
        for child in node.children:
            text = _safe_text_decode(child)
            if text and child.type == "decorator":
                decs.append(text.lstrip("@").strip().split("(")[0].strip())
            if child.type in ("function_definition", "class_definition"):
                inner = child
        if inner and inner.type == "function_definition":
            nn = inner.child_by_field_name("name")
            if nn:
                line = nn.start_point[0] + 1
                if line in func_by_line:
                    qname = func_by_line[line].qualified_name
                    for d in decs:
                        apps.append(_DecoratorApp(d, qname))

    for child in node.children:
        new_scope = scope
        if node.type in ("class_definition", "function_definition"):
            nn = node.child_by_field_name("name")
            nn_name = _safe_text_decode(nn)
            if nn_name:
                new_scope = scope + [nn_name]
        _detect_decorator_apps_walk(
            child, module, new_scope, func_by_line, apps, depth + 1
        )


def _detect_decorator_apps_ts(
    root: Node, module: str, scope: list[str], func_by_line: dict[int, _FuncData]
) -> list[_DecoratorApp]:
    """Tree-sitter based decorator detection using line-based func lookup."""
    apps: list[_DecoratorApp] = []
    _detect_decorator_apps_walk(root, module, scope, func_by_line, apps)
    return apps


def _walk_tree_for_forwarding(
    node: Node,
    scope: list[str],
    func_by_line: dict[int, _FuncData],
    depth: int = 0,
):
    """Walk tree to find forwarding patterns and fixup *args/**kwargs."""
    if depth > MAX_DEPTH:
        logger.debug("Max depth reached in _walk_tree_for_forwarding")
        return
    if node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            line = name_node.start_point[0] + 1  # 1-based
            if line in func_by_line:
                fd = func_by_line[line]

                # Fixup: Check if jedi missed *args/**kwargs
                params_node = node.child_by_field_name("parameters")
                if params_node and not fd._is_args_kwargs:
                    if _has_args_kwargs(node):
                        fd._is_args_kwargs = True

                if fd._is_args_kwargs:
                    fwd = _detect_forwarding(node)
                    if fwd:
                        fd._forwarding_target, fd._forwarding_extra = fwd

        # Recurse into function body
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                _walk_tree_for_forwarding(
                    child,
                    scope + [_safe_text_decode_default(name_node, "")],
                    func_by_line,
                    depth + 1,
                )

    elif node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                _walk_tree_for_forwarding(
                    child,
                    scope + [_safe_text_decode_default(name_node, "")],
                    func_by_line,
                    depth + 1,
                )

    else:
        for child in node.children:
            _walk_tree_for_forwarding(child, scope, func_by_line, depth + 1)


__all__ = [
    "ts_parser",
    "_safe_text_decode",
    "_safe_text_decode_default",
    "_has_args_kwargs",
    "_extract_varargs_kwargs",
    "_extract_extra_args",
    "_detect_forwarding",
    "_detect_decorator_apps_ts",
    "_walk_tree_for_forwarding",
]
