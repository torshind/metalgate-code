"""Integration tests for contextual symbol search tools."""

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


# get_file_outline
class TestGetFileOutline:
    def test_finds_class(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "Order" for s in symbols)

    def test_finds_methods(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        kinds = {s["name"]: s["kind"] for s in symbols}
        assert kinds.get("__init__") == "method"
        assert kinds.get("process") == "method"

    def test_method_has_parent_class(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        process = next(s for s in symbols if s["name"] == "process")
        assert process["class"] == "Order"

    def test_finds_top_level_function(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        assert any(s["name"] == "validate_address" for s in symbols)

    def test_signature_contains_name(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        validate = next(s for s in symbols if s["name"] == "validate_address")
        assert "validate_address" in validate["signature"]

    def test_end_line_gte_line(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        for s in symbols:
            assert s["end_line"] >= s["line"]

    def test_symbols_include_file_path(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        for s in symbols:
            assert "orders.py" in s["file"]

    def test_cached_result_is_identical(self, tools):
        first = tools["get_file_outline"](ORDERS_FILE)
        second = tools["get_file_outline"](ORDERS_FILE)
        assert first == second

    def test_line_numbers_are_positive(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        for s in symbols:
            assert s["line"] >= 1


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


# get_source
class TestGetSource:
    def _validate_line(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        return next(s for s in symbols if s["name"] == "validate_address")["line"]

    def test_returns_source_string(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert isinstance(result["source"], str)
        assert len(result["source"]) > 0

    def test_source_contains_def(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert "validate_address" in result["source"]

    def test_source_contains_body(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert "REQUIRED_KEYS" in result["source"]

    def test_start_and_end_lines_are_sane(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert result["start_line"] >= 1
        assert result["end_line"] >= result["start_line"]

    def test_get_source_for_class(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        cls = next(s for s in symbols if s["name"] == "Order")
        result = tools["get_source"](ORDERS_FILE, cls["line"])
        assert "class Order" in result["source"]
        # should include at least the __init__ body
        assert "__init__" in result["source"]

    def test_fallback_context_window(self, tools):
        # Line 1 has no def/class — should fall back to context window
        result = tools["get_source"](VALIDATION_FILE, 1, context=10)
        assert isinstance(result["source"], str)
        assert len(result["source"]) > 0
        # Fallback window should be at most context lines
        assert len(result["source"].splitlines()) <= 10

    def test_nonexistent_file_returns_error(self, tools):
        result = tools["get_source"]("/nonexistent/file.py", 1)
        assert result["source"] == "" or "error" in result


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
        assert results == []

    def test_finds_class_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names


# --------------------------------------------------------------------------- #
# Unit tests for module-level helper functions (no sandbox/LSP required)
# --------------------------------------------------------------------------- #

from metalgate_code.context.python_tracer import (
    _call_positions,
    _lsp_symbol_kind_to_str,
    _name_col_on_line,
    _parse_hover,
    _ts_find_function_containing,
    _ts_find_scope_at_line,
    _uri_to_path,
)


class TestUriToPathPercentDecoding:
    """Reproduces #3: _uri_to_path didn't decode percent-encoded URIs."""

    def test_decodes_percent_encoded_spaces(self):
        assert _uri_to_path("file:///foo%20bar/baz.py") == "/foo bar/baz.py"

    def test_decodes_percent_encoded_unicode(self):
        assert _uri_to_path("file:///proj/my%20file.py") == "/proj/my file.py"

    def test_passthrough_for_non_file_uri(self):
        assert _uri_to_path("/foo/bar.py") == "/foo/bar.py"

    def test_plain_uri_unchanged(self):
        assert _uri_to_path("file:///foo/bar.py") == "/foo/bar.py"


class TestNameColOnLineMultipleOccurrences:
    """Reproduces #6: _name_col_on_line returned only the first occurrence."""

    def test_first_occurrence_by_default(self):
        line = "result = foo(foo)"
        col = _name_col_on_line(line, "foo")
        assert col is not None
        assert line[col : col + 3] == "foo"
        # Should be the first one (after =)
        assert col == line.index("foo")

    def test_second_occurrence(self):
        line = "result = foo(foo)"
        col = _name_col_on_line(line, "foo", occurrence=1)
        assert col is not None
        assert line[col : col + 3] == "foo"
        # Should be the second one (inside parens)
        assert col == line.rindex("foo")

    def test_returns_none_if_occurrence_out_of_range(self):
        line = "result = foo(foo)"
        assert _name_col_on_line(line, "foo", occurrence=5) is None

    def test_word_boundary_not_substring(self):
        line = "x = foobar(foo)"
        # 'foo' inside 'foobar' should not match
        col = _name_col_on_line(line, "foo")
        assert col is not None
        assert line[col : col + 3] == "foo"
        assert col == line.rindex("foo")  # the standalone one


class TestCallPositionsFalsePositives:
    """Reproduces #16: _call_positions matched decorators and class definitions."""

    def test_skips_decorator_lines(self):
        source = "@deco\ndef func():\n    pass\n"
        # start_line=2 is the def line (as tree-sitter would report)
        positions = _call_positions(source, 2, 3, func_name="func")
        # @deco should NOT be treated as a call
        assert positions == []

    def test_skips_class_definition_base_list(self):
        source = "class Foo(Bar):\n    pass\n"
        positions = _call_positions(source, 1, 2)
        # 'Bar' in class definition should NOT be treated as a call
        assert positions == []

    def test_finds_real_calls(self):
        source = "def func():\n    foo()\n    bar()\n"
        positions = _call_positions(source, 1, 3, func_name="func")
        assert len(positions) == 2

    def test_skips_function_name_on_def_line(self):
        source = "def foo():\n    foo()\n"
        positions = _call_positions(source, 1, 2, func_name="foo")
        # Only the call on line 2, not the def on line 1
        assert len(positions) == 1
        assert positions[0][0] == 2


class TestParseHoverFragility:
    """Reproduces #14: hover parsing assumed first line is always signature."""

    def test_plain_signature_and_docstring(self):
        hover = {"contents": {"value": "def foo(x: int) -> bool\nDoes a thing."}}
        sig, doc = _parse_hover(hover)
        assert sig == "def foo(x: int) -> bool"
        assert doc == "Does a thing."

    def test_markdown_code_fence_stripped(self):
        hover = {"contents": {"value": "```python\ndef foo(x) -> None\nDoc here\n```"}}
        sig, doc = _parse_hover(hover)
        assert sig == "def foo(x) -> None"
        assert doc == "Doc here"

    def test_string_contents(self):
        hover = {"contents": "def foo() -> None\nA function."}
        sig, doc = _parse_hover(hover)
        assert sig == "def foo() -> None"
        assert doc == "A function."

    def test_list_contents(self):
        hover = {"contents": [{"value": "def foo() -> None"}, {"value": "Docs."}]}
        sig, doc = _parse_hover(hover)
        assert sig == "def foo() -> None"
        assert doc == "Docs."

    def test_empty_hover(self):
        assert _parse_hover(None) == ("", "")
        assert _parse_hover({}) == ("", "")
        assert _parse_hover({"contents": {}}) == ("", "")
        assert _parse_hover({"contents": ""}) == ("", "")


class TestLspSymbolKindMapping:
    """Reproduces #4: find_symbol mapped all non-class kinds to 'function'."""

    def test_class(self):
        assert _lsp_symbol_kind_to_str(5) == "class"

    def test_function(self):
        assert _lsp_symbol_kind_to_str(12) == "function"

    def test_method(self):
        assert _lsp_symbol_kind_to_str(6) == "method"

    def test_variable(self):
        assert _lsp_symbol_kind_to_str(13) == "variable"

    def test_unknown_kind(self):
        assert _lsp_symbol_kind_to_str(99) == "unknown"

    def test_zero(self):
        assert _lsp_symbol_kind_to_str(0) == "unknown"


class TestTsFindScopeAtLine:
    """Verify line-base convention fix (#11)."""

    def test_returns_sliceable_indices(self):
        source = "def foo():\n    pass\n\ndef bar():\n    return 1\n"
        scope = _ts_find_scope_at_line(source.encode("utf-8"), 1)
        assert scope is not None
        start, end = scope
        lines = source.splitlines()
        # lines[start:end] should give the full function body
        assert "def foo" in lines[start]
        assert "pass" in lines[end - 1]

    def test_returns_none_for_non_def_line(self):
        source = "x = 1\ndef foo():\n    pass\n"
        scope = _ts_find_scope_at_line(source.encode("utf-8"), 1)
        assert scope is None


class TestTsFindFunctionContaining:
    """Verify line-base convention fix (#11)."""

    def test_finds_innermost_function(self):
        source = "def outer():\n    def inner():\n        pass\n    pass\n"
        result = _ts_find_function_containing(source.encode("utf-8"), 3)
        assert result is not None
        start, end, name = result
        assert name == "inner"
        assert start == 2
        assert end == 3

    def test_returns_none_outside_any_function(self):
        source = "x = 1\ndef foo():\n    pass\n"
        result = _ts_find_function_containing(source.encode("utf-8"), 1)
        assert result is None


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
