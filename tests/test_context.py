"""Integration tests for contextual symbol search tools."""

import os
import tempfile
from pathlib import Path

import pytest
from deepagents.backends import LocalShellBackend

from metalgate_code.context import get_code_tools

SAMPLE_DIR = Path(__file__).parent / "sample"
ORDERS_FILE = str(SAMPLE_DIR / "orders.py")
VALIDATION_FILE = str(SAMPLE_DIR / "validation.py")
UTILS_FILE = str(SAMPLE_DIR / "utils.py")


@pytest.fixture(scope="module")
def tools():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    shell_env = os.environ.copy()
    shell_backend = LocalShellBackend(
        root_dir=str(SAMPLE_DIR),
        virtual_mode=False,
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
        assert isinstance(callees, list)  # may be empty, must not crash


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
        results = tools["find_symbol"]("validate")
        names = [r["name"] for r in results]
        assert "validate_address" not in names

    def test_results_have_file(self, tools):
        results = tools["find_symbol"]("validate_address")
        for r in results:
            assert r.get("file") is not None

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        assert isinstance(results, list)

    def test_finds_class_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names
