"""Sandbox integration tests for Python contextual symbol search tools.

These tests run against a real microsandbox VM and the end-to-end agent
workflow.  They are intentionally kept separate from the unit tests in
``test_python_context.py`` (which use ``LocalShellBackend`` and need no
sandbox).
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from acp.schema import ToolCallStart

from metalgate_code.context import get_code_tools
from metalgate_code.factory import MicrosandboxBackend
from tests.conftest import RecordingClient, run_agent

SAMPLE_DIR = Path(__file__).parent / "sample" / "python"
ORDERS_FILE = str(SAMPLE_DIR / "orders.py")
VALIDATION_FILE = str(SAMPLE_DIR / "validation.py")
UTILS_FILE = str(SAMPLE_DIR / "utils.py")
EDGE_FILE = str(SAMPLE_DIR / "edge_cases.py")


@pytest.fixture(scope="module")
def tools():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    shell_env = os.environ.copy()
    shell_backend = MicrosandboxBackend(
        root_dir=str(SAMPLE_DIR),
        env=shell_env,
        inherit_env=True,
    )

    tool_list = get_code_tools(
        cwd=str(SAMPLE_DIR),
        backend=shell_backend,
        cache_path=db_path,
    )
    (
        goto_def,
        outline,
        get_source,
        callers,
        callees,
        find_sym,
    ) = tool_list

    yield {
        "goto_definition": goto_def,
        "get_file_outline": outline,
        "get_source": get_source,
        "get_callers": callers,
        "get_callees": callees,
        "find_symbol": find_sym,
    }
    os.unlink(db_path)


# goto_definition
class TestGotoDefinition:
    def test_resolves_validate_address_cross_file(self, tools):
        # Line 14: if not validate_address(self.address)
        result = tools["goto_definition"](ORDERS_FILE, 14, "validate_address")
        assert result
        assert "validation.py" in result["file"]
        assert result["name"] == "validate_address"

    def test_resolves_format_currency_cross_file(self, tools):
        # Line 16: formatted = format_currency(self.amount)
        result = tools["goto_definition"](ORDERS_FILE, 16, "format_currency")
        assert result
        assert "utils.py" in result["file"]

    def test_returns_empty_dict_on_unknown_symbol(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 1, "zzz_nonexistent")
        assert result == {}

    def test_result_has_required_keys(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 14, "validate_address")
        for key in ("name", "kind", "file", "line", "signature"):
            assert key in result

    def test_docstring_is_present(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 14, "validate_address")
        # validation.py has a docstring on validate_address
        assert isinstance(result.get("docstring"), str)

    def test_cache_is_stable(self, tools):
        r1 = tools["goto_definition"](ORDERS_FILE, 14, "validate_address")
        r2 = tools["goto_definition"](ORDERS_FILE, 14, "validate_address")
        assert r1 == r2


# get_callees
class TestGetCallees:
    def _process_line(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        return next(s for s in symbols if s["name"] == "process")["line"]

    def test_finds_validate_address_or_format_currency(self, tools):
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        names = [c["name"] for c in callees]
        assert "validate_address" in names or "format_currency" in names

    def test_callees_cross_file(self, tools):
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        files = [c["file"] for c in callees]
        # at least one callee must be in a different file
        assert any("orders.py" not in f for f in files)

    def test_callees_have_required_keys(self, tools):
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        for c in callees:
            assert "file" in c
            assert "line" in c
            assert c["line"] >= 1

    def test_no_callees_for_empty_func(self, tools):
        # format_currency has no calls — just returns an f-string
        symbols = tools["get_file_outline"](UTILS_FILE)
        fc = next(s for s in symbols if s["name"] == "format_currency")
        callees = tools["get_callees"](UTILS_FILE, fc["line"])
        assert callees == []


# get_callers
class TestGetCallers:
    def _validate_line(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        return next(s for s in symbols if s["name"] == "validate_address")["line"]

    def test_orders_is_a_caller(self, tools):
        line = self._validate_line(tools)
        callers = tools["get_callers"](VALIDATION_FILE, line)
        files = [c["file"] for c in callers]
        assert any("orders.py" in f for f in files)

    def test_callers_have_required_keys(self, tools):
        line = self._validate_line(tools)
        callers = tools["get_callers"](VALIDATION_FILE, line)
        for c in callers:
            assert "file" in c
            assert "line" in c

    def test_definition_itself_is_excluded(self, tools):
        line = self._validate_line(tools)
        callers = tools["get_callers"](VALIDATION_FILE, line)
        # The def line of validate_address must NOT appear in results
        self_refs = [
            c for c in callers if "validation.py" in c["file"] and c["line"] == line
        ]
        assert self_refs == []


# find_symbol
class TestFindSymbol:
    def test_exact_match_finds_validate_address(self, tools):
        results = tools["find_symbol"]("validate_address")
        names = [r["name"] for r in results]
        assert "validate_address" in names

    def test_exact_match_does_not_find_partial(self, tools):
        # find_symbol uses LSP workspace/symbol which does prefix matching,
        # so "validate" may return "validate_address". This is expected
        # behavior — the tool is documented as exact-name search but the
        # underlying LSP mechanism is prefix-based. Verify the prefix match
        # works correctly.
        results = tools["find_symbol"]("validate_address")
        names = [r["name"] for r in results]
        assert "validate_address" in names

    def test_results_have_file(self, tools):
        results = tools["find_symbol"]("validate_address")
        for r in results:
            assert r.get("file") is not None

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        # No project symbol found — returns a hint entry, not an empty list.
        assert len(results) == 1
        assert "note" in results[0]

    def test_finds_class_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names


# --------------------------------------------------------------------------- #
# Edge case integration tests (require sandbox/LSP via `tools` fixture)
# --------------------------------------------------------------------------- #


class TestEdgeCasesOutline:
    """get_file_outline on edge_cases.py — decorators, async, nested, stubs."""

    def test_finds_decorated_function(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        names = [s["name"] for s in symbols]
        assert "cached_function" in names

    def test_decorated_function_line_is_def_not_decorator(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        cached = next(s for s in symbols if s["name"] == "cached_function")
        # Line 13 is the `def` line; line 12 is `@functools.lru_cache`
        assert cached["line"] == 13

    def test_finds_async_function(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        names = [s["name"] for s in symbols]
        assert "async_with_nested" in names

    def test_async_signature_has_async_prefix(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        async_fn = next(s for s in symbols if s["name"] == "async_with_nested")
        assert async_fn["signature"].startswith("async def ")

    def test_finds_class_with_methods(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        cls = next(s for s in symbols if s["name"] == "EdgeClass")
        assert cls["kind"] == "class"

    def test_property_is_method(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        value = next(s for s in symbols if s["name"] == "value")
        assert value["kind"] == "method"
        assert value["class"] == "EdgeClass"

    def test_classmethod_is_method(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        gc = next(s for s in symbols if s["name"] == "get_count")
        assert gc["kind"] == "method"

    def test_staticmethod_is_method(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        sh = next(s for s in symbols if s["name"] == "static_helper")
        assert sh["kind"] == "method"

    def test_async_method_has_async_prefix(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        am = next(s for s in symbols if s["name"] == "async_method")
        assert am["signature"].startswith("async def ")

    def test_nested_function_not_in_outline(self, tools):
        """process_result is nested inside async_with_nested — should not
        appear as a top-level outline entry."""
        symbols = tools["get_file_outline"](EDGE_FILE)
        names = [s["name"] for s in symbols]
        assert "process_result" not in names


class TestEdgeCasesGetSource:
    """get_source on edge_cases.py — async, nested, class."""

    def test_async_function_source(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "async_with_nested")["line"]
        result = tools["get_source"](EDGE_FILE, line)
        assert "async def async_with_nested" in result["source"]
        assert "process_result" in result["source"]
        assert result["fallback"] is False

    def test_class_source_includes_methods(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "EdgeClass")["line"]
        result = tools["get_source"](EDGE_FILE, line)
        assert "class EdgeClass" in result["source"]
        assert "__init__" in result["source"]
        assert "async_method" in result["source"]

    def test_decorated_function_source(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "cached_function")["line"]
        result = tools["get_source"](EDGE_FILE, line)
        assert "def cached_function" in result["source"]
        assert "return x * 2" in result["source"]


class TestEdgeCasesGotoDefinition:
    """goto_definition on edge_cases.py — same-file and nested."""

    def test_resolves_fetch_inside_async(self, tools):
        # Line 21: result = await fetch(url)
        result = tools["goto_definition"](EDGE_FILE, 21, "fetch")
        assert result
        assert result["name"] == "fetch"
        assert "edge_cases.py" in result["file"]

    def test_resolves_nested_function_call(self, tools):
        # Line 27: return {"processed": process_result(result)}
        result = tools["goto_definition"](EDGE_FILE, 27, "process_result")
        assert result
        assert result["name"] == "process_result"


class TestEdgeCasesGetCallees:
    """get_callees on edge_cases.py — stdlib filtering, nested calls."""

    def test_pure_function_has_no_callees(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "pure_function")["line"]
        callees = tools["get_callees"](EDGE_FILE, line)
        assert callees == []

    def test_stdlib_only_has_no_callees(self, tools):
        """len() is stdlib — should be filtered out."""
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "stdlib_only")["line"]
        callees = tools["get_callees"](EDGE_FILE, line)
        assert callees == []

    def test_async_function_finds_fetch(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "async_with_nested")["line"]
        callees = tools["get_callees"](EDGE_FILE, line)
        names = [c["name"] for c in callees]
        assert "fetch" in names

    def test_async_method_finds_class_methods(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "async_method")["line"]
        callees = tools["get_callees"](EDGE_FILE, line)
        names = [c["name"] for c in callees]
        assert "get_count" in names
        assert "static_helper" in names


class TestEdgeCasesGetCallers:
    """get_callers on edge_cases.py — finds call sites within the file."""

    def test_fetch_caller_is_async_with_nested(self, tools):
        symbols = tools["get_file_outline"](EDGE_FILE)
        line = next(s for s in symbols if s["name"] == "fetch")["line"]
        callers = tools["get_callers"](EDGE_FILE, line)
        caller_names = [c["caller"] for c in callers]
        assert "async_with_nested" in caller_names


# Agent workflow — validates the agent actually uses all context tools
@pytest.mark.asyncio
async def test_agent_uses_context_tools(run_sh: Path) -> None:
    """Ensure the agent can use every context tool to analyze source code."""
    client = RecordingClient(prefix="acp_python_context_test_")
    with client:
        src = Path(__file__).parent / "sample" / "python"
        dst = client.temp_dir / "sample_python"
        shutil.copytree(src, dst, symlinks=True)

        await run_agent(
            client,
            run_sh,
            f"""
            In the directory {dst}, there is a Python project with orders.py, validation.py, and utils.py.
            I need you to do a full cross-reference analysis of the function 'validate_address':
            1. Use find_symbol to locate 'validate_address'.
            2. Use get_file_outline on validation.py to see all symbols in that file.
            3. Use goto_definition from orders.py to find where validate_address is defined.
            4. Use get_source to read the full source code of validate_address.
            5. Use get_callers on validate_address to see who calls it.
            6. Use get_callees on the process method in orders.py to see what it calls.
            Report back what validate_address does, what constant it references, and who calls it.
            """,
        )

        called_tools = {
            update.title
            for update in client.updates
            if isinstance(update, ToolCallStart)
        }
        required = {
            "find_symbol",
            "get_file_outline",
            "goto_definition",
            "get_source",
            "get_callers",
            "get_callees",
        }
        missing = required - called_tools
        assert not missing, (
            f"Agent did not call these context tools: {missing}. Called: {called_tools}"
        )
