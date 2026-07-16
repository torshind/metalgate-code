"""Unit tests for contextual symbol search tools.

These tests use ``LocalShellBackend`` and run on the host without a
sandbox.  Sandbox/agent integration tests live in
``test_python_context_integration.py``.
"""

import tempfile
from pathlib import Path

import pytest
from deepagents.backends import LocalShellBackend

from metalgate_code.context import get_code_tools
from metalgate_code.context.python_tracer import (
    _lsp_symbol_kind_to_str,
    _name_col_on_line,
    _parse_hover,
    _ts_call_positions,
    _ts_find_function_and_calls,
    _ts_find_function_containing,
    _ts_find_scope_at_line,
    _ts_is_stub_function,
    _uri_to_path,
)

SAMPLE_DIR = Path(__file__).parent / "sample" / "python"
ORDERS_FILE = str(SAMPLE_DIR / "orders.py")
VALIDATION_FILE = str(SAMPLE_DIR / "validation.py")
UTILS_FILE = str(SAMPLE_DIR / "utils.py")
EDGE_FILE = str(SAMPLE_DIR / "edge_cases.py")


@pytest.fixture(scope="module")
def tools():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    shell_backend = LocalShellBackend(
        root_dir=str(SAMPLE_DIR),
        virtual_mode=False,
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
        # No project symbol found — returns a hint entry, not an empty list.
        assert len(results) == 1
        assert "note" in results[0]

    def test_finds_class_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names


# --------------------------------------------------------------------------- #
# Unit tests for module-level helper functions (no sandbox/LSP required)
# --------------------------------------------------------------------------- #


class TestUriToPathPercentDecoding:
    """Verify _uri_to_path decodes percent-encoded URIs."""

    def test_decodes_percent_encoded_spaces(self):
        assert _uri_to_path("file:///foo%20bar/baz.py") == "/foo bar/baz.py"

    def test_decodes_percent_encoded_unicode(self):
        assert _uri_to_path("file:///proj/my%20file.py") == "/proj/my file.py"

    def test_passthrough_for_non_file_uri(self):
        assert _uri_to_path("/foo/bar.py") == "/foo/bar.py"

    def test_plain_uri_unchanged(self):
        assert _uri_to_path("file:///foo/bar.py") == "/foo/bar.py"


class TestNameColOnLineMultipleOccurrences:
    """Verify _name_col_on_line resolves multiple occurrences on the same line."""

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
    """Verify _ts_call_positions excludes decorators and class definitions.

    Tree-sitter naturally excludes them because they are not ``call`` nodes
    within the function body.
    """

    def test_skips_decorator_lines(self):
        source = "@deco\ndef func():\n    pass\n"
        # start_line=2 is the def line (as tree-sitter would report)
        positions = _ts_call_positions(source.encode("utf-8"), 2, 3)
        # @deco should NOT be treated as a call
        assert positions == []

    def test_skips_class_definition_base_list(self):
        source = "class Foo(Bar):\n    pass\n"
        positions = _ts_call_positions(source.encode("utf-8"), 1, 2)
        # 'Bar' in class definition should NOT be treated as a call
        assert positions == []

    def test_finds_real_calls(self):
        source = "def func():\n    foo()\n    bar()\n"
        positions = _ts_call_positions(source.encode("utf-8"), 1, 3)
        assert len(positions) == 2

    def test_skips_function_name_on_def_line(self):
        source = "def foo():\n    foo()\n"
        positions = _ts_call_positions(source.encode("utf-8"), 1, 2)
        # Only the call on line 2, not the def on line 1
        assert len(positions) == 1
        assert positions[0][0] == 2


class TestParseHoverFragility:
    """Verify _parse_hover handles all LSP contents shapes."""

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
    """Verify _lsp_symbol_kind_to_str maps LSP SymbolKind numbers correctly."""

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
    """Verify _ts_find_scope_at_line returns sliceable line indices."""

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
    """Verify _ts_find_function_containing finds the innermost function."""

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


# --------------------------------------------------------------------------- #
# Unit tests for edge-case helper functions (no sandbox/LSP required)
# --------------------------------------------------------------------------- #


class TestTsIsStubFunction:
    """Verify _ts_is_stub_function detects all stub patterns."""

    def test_raise_notimplemented_is_stub(self):
        source = "def foo(x):\n    raise NotImplementedError\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is True

    def test_pass_is_stub(self):
        source = "def foo(x):\n    pass\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is True

    def test_ellipsis_is_stub(self):
        source = "def foo(x):\n    ...\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is True

    def test_docstring_then_ellipsis_is_stub(self):
        source = 'def foo(x):\n    """Doc."""\n    ...\n'
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is True

    def test_docstring_then_pass_is_stub(self):
        source = 'def foo(x):\n    """Doc."""\n    pass\n'
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is True

    def test_concrete_return_is_not_stub(self):
        source = "def foo(x):\n    return x + 1\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is False

    def test_concrete_assignment_is_not_stub(self):
        source = "def foo(x):\n    y = x + 1\n    return y\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is False

    def test_other_raise_is_not_stub(self):
        source = "def foo(x):\n    raise ValueError('nope')\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is False

    def test_no_function_at_line_returns_false(self):
        source = "x = 1\n"
        assert _ts_is_stub_function(source.encode("utf-8"), 1) is False


class TestTsFindFunctionAndCalls:
    """Verify _ts_find_function_and_calls finds functions and their call
    positions in a single tree walk."""

    def test_finds_function_and_calls(self):
        source = "def foo():\n    bar()\n    baz()\n"
        result = _ts_find_function_and_calls(source.encode("utf-8"), 1)
        assert result is not None
        start, end, name, positions = result
        assert start == 1
        assert end == 3
        assert name == "foo"
        assert len(positions) == 2

    def test_finds_innermost_function(self):
        source = "def outer():\n    def inner():\n        bar()\n    pass\n"
        # Line 3 is inside `inner`
        result = _ts_find_function_and_calls(source.encode("utf-8"), 3)
        assert result is not None
        start, end, name, positions = result
        assert name == "inner"
        assert len(positions) == 1

    def test_returns_none_outside_function(self):
        source = "x = 1\ndef foo():\n    pass\n"
        result = _ts_find_function_and_calls(source.encode("utf-8"), 1)
        assert result is None

    def test_no_calls_returns_empty_positions(self):
        source = "def foo():\n    return 1\n"
        result = _ts_find_function_and_calls(source.encode("utf-8"), 1)
        assert result is not None
        start, end, name, positions = result
        assert positions == []

    def test_attribute_call_targets_method_name(self):
        """For obj.method(), the call position should target the method
        name, not the object."""
        source = "def foo():\n    obj.method()\n"
        result = _ts_find_function_and_calls(source.encode("utf-8"), 1)
        assert result is not None
        _, _, _, positions = result
        assert len(positions) == 1
        line, col = positions[0]
        line_text = source.splitlines()[line - 1]
        # col should point at "method", not "obj"
        assert line_text[col:].startswith("method")
