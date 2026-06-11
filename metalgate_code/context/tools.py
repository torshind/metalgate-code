"""Tool functions exposed to the agent — closures over a Tracer instance.

Each is a plain Python function with type hints and a docstring.
They are created as closures over a Tracer instance so the agent
never needs to know about the underlying engine.
"""

from __future__ import annotations

from typing import Optional

from metalgate_code.context.tracer_base import Tracer


def make_tools(tracer: Tracer) -> list:
    """Return the six code-navigation tool functions bound to `tracer`."""

    def goto_definition(
        file: str,
        line: int,
        name: Optional[str] = None,
    ) -> dict:
        """
        Resolve a symbol to its definition, crossing file and package
        boundaries — including symbols inside third-party site-packages.

        Start here whenever you encounter a call or import you want to
        understand. Cheap: result is cached after the first call.

        Args:
            file: Absolute or project-relative path to the source file.
            line: 1-indexed line number where the symbol appears.
            name: Symbol name to resolve (e.g. "validate_address").
                  If omitted, resolves the first resolvable name on the line.

        Returns a dict with keys:
            name, kind, file, line, col, signature, docstring
        Returns {} if the symbol cannot be resolved.

        Example:
            # Line 88 of orders.py contains: validate_address(order)
            goto_definition("src/orders.py", 88, "validate_address")
            # → {"file": "src/validation.py", "line": 34,
            #    "signature": "def validate_address(addr) -> bool",
            #    "docstring": "Check address has all required keys...", ...}
        """
        result = tracer.goto_definition(file, line, name)
        return result or {}

    def get_file_outline(file: str) -> list[dict]:
        """
        Return every class, function and method defined in `file` with
        its line number and signature — without any bodies.

        Use this to map a file before deciding which symbol to drill into.
        Extremely fast after the first call (cached by file mtime).

        Args:
            file: Absolute or project-relative path to the source file.

        Returns a list of dicts with keys:
            name, kind ("class"|"function"|"method"), class (parent or null),
            line, end_line, signature, file

        Example:
            get_file_outline("src/orders.py")
            # → [
            #     {"name": "Order", "kind": "class", "line": 12, ...},
            #     {"name": "process", "kind": "method", "class": "Order",
            #      "line": 45, "end_line": 61, ...},
            #   ]
        """
        return tracer.get_file_outline(file)

    def get_source(file: str, line: int, context: int = 60) -> dict:
        """
        Return the full source of the function or class whose definition
        starts on `line`.  If no scope node is found at `line`, returns
        `context` lines centred around it instead.

        Use after goto_definition or get_file_outline to read the actual
        implementation of a symbol — including library code in site-packages.

        Args:
            file:    Path to the file.
            line:    1-indexed line of the `def` or `class` statement.
            context: Fallback window size in lines (default 60).

        Returns a dict with keys:
            file, start_line, end_line, source (the code as a string)

        Example:
            # validate_address lives at src/validation.py:34
            get_source("src/validation.py", 34)
            # → {"start_line": 34, "end_line": 41,
            #    "source": "def validate_address(addr: dict) -> bool:\n ..."}
        """
        return tracer.get_source(file, line, context)

    def get_callers(file: str, line: int) -> list[dict]:
        """
        Find every place in the project that calls or references the symbol
        defined on `line` of `file`.

        Useful for understanding the blast radius of a change or tracing
        where a function is invoked from.  Capped at 50 results; times out
        gracefully on very large repos.

        Args:
            file: Path to the file containing the definition.
            line: 1-indexed line of the `def` or `class` statement.

        Returns a list of dicts with keys:
            file, line, name, caller, context

        ``name`` is the symbol being referenced; ``caller`` is the
        innermost function or class that contains the reference.

        Example:
            # validate_address is defined at src/validation.py:34
            get_callers("src/validation.py", 34)
            # → [
            #     {"file": "src/orders.py",   "line": 88,
            #      "name": "validate_address", "caller": "place_order", ...},
            #     {"file": "tests/test_val.py","line": 12,
            #      "name": "validate_address", "caller": "test_address", ...},
            #   ]
        """
        return tracer.get_callers(file, line)

    def get_callees(file: str, line: int) -> list[dict]:
        """
        List every symbol called by the function defined on `line` of `file`,
        resolved to their own definition locations — including site-packages.

        This is the primary tool for following a call trail deeper into the
        codebase.  Pair with get_source to read the body of each callee.

        Args:
            file: Path to the file containing the function definition.
            line: 1-indexed line of the `def` statement.

        Returns a list of dicts with keys:
            name, kind, file, line, signature

        Typical workflow:
            callees = get_callees("src/orders.py", 45)
            for c in callees:
                src = get_source(c["file"], c["line"])  # read its body
        """
        return tracer.get_callees(file, line)

    def find_symbol(name: str) -> list[dict]:
        """
        Search for a symbol by exact name across the project and installed
        packages.

        Use when you know a function or class name but not which file it lives
        in.  If you need the signature, call ``get_file_outline`` on the
        returned file afterwards.

        Args:
            name: Exact symbol name (e.g. "validate_address", "DataFrame").

        Returns a list of dicts with keys:
            name, kind, file, line

        Example:
            find_symbol("DataFrame")
            # → [{"name": "DataFrame", "kind": "class",
            #      "file": ".venv/lib/.../pandas/core/frame.py", "line": 517}]
        """
        return tracer.find_symbol(name)

    return [
        goto_definition,
        get_file_outline,
        get_source,
        get_callers,
        get_callees,
        find_symbol,
    ]
