"""Integration tests for Go contextual symbol search tools."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from acp.schema import ToolCallStart

from metalgate_code.context import get_code_tools
from metalgate_code.factory import MicrosandboxBackend
from tests.conftest import RecordingClient, run_agent

SAMPLE_DIR = Path(__file__).parent / "sample" / "go"
ORDERS_FILE = str(SAMPLE_DIR / "orders.go")
VALIDATION_FILE = str(SAMPLE_DIR / "validation.go")
UTILS_FILE = str(SAMPLE_DIR / "utils.go")

_HAS_GOPLS = shutil.which("gopls") is not None


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
        language="go",
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
    def test_finds_struct(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "Order" and s["kind"] == "struct" for s in symbols)

    def test_finds_interface(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(
            s["name"] == "Processor" and s["kind"] == "interface" for s in symbols
        )

    def test_finds_function(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        assert any(s["name"] == "NewOrder" and s["kind"] == "function" for s in symbols)

    def test_finds_method(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        method = next(
            (s for s in symbols if s["name"] == "Process" and s["kind"] == "method"),
            None,
        )
        assert method is not None

    def test_method_has_receiver(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        process = next(s for s in symbols if s["name"] == "Process")
        assert "Order" in (process.get("class") or "")

    def test_signature_contains_name(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        validate = next(s for s in symbols if s["name"] == "ValidateAddress")
        assert "ValidateAddress" in validate["signature"]

    def test_cached_result_is_identical(self, tools):
        first = tools["get_file_outline"](ORDERS_FILE)
        second = tools["get_file_outline"](ORDERS_FILE)
        assert first == second


# get_source
class TestGetSource:
    def _validate_line(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        return next(s for s in symbols if s["name"] == "ValidateAddress")["line"]

    def test_source_contains_func(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert "ValidateAddress" in result["source"]
        assert "return false" in result["source"]

    def test_start_and_end_lines_are_sane(self, tools):
        line = self._validate_line(tools)
        result = tools["get_source"](VALIDATION_FILE, line)
        assert result["start_line"] >= 1
        assert result["end_line"] >= result["start_line"]

    def test_get_source_for_struct(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        st = next(s for s in symbols if s["name"] == "Order")
        result = tools["get_source"](ORDERS_FILE, st["line"])
        assert "type Order struct" in result["source"]

    def test_get_source_from_body_line(self, tools):
        """get_source should work when given any line inside the function body."""
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        va = next(s for s in symbols if s["name"] == "ValidateAddress")
        body_line = va["line"] + 2  # inside the body
        result = tools["get_source"](VALIDATION_FILE, body_line)
        assert "ValidateAddress" in result["source"]

    def test_fallback_context_window(self, tools):
        result = tools["get_source"](VALIDATION_FILE, 1, context=10)
        assert isinstance(result["source"], str)

    def test_nonexistent_file_returns_error(self, tools):
        result = tools["get_source"]("/nonexistent/file.go", 1)
        assert result["source"] == "" or "error" in result


# find_symbol
class TestFindSymbol:
    def test_exact_match_finds_validate_address(self, tools):
        results = tools["find_symbol"]("ValidateAddress")
        names = [r["name"] for r in results]
        assert "ValidateAddress" in names

    def test_exact_match_does_not_find_partial(self, tools):
        results = tools["find_symbol"]("Validate")
        names = [r["name"] for r in results]
        assert "ValidateAddress" not in names

    def test_case_insensitive(self, tools):
        results = tools["find_symbol"]("validateaddress")
        names = [r["name"] for r in results]
        assert "ValidateAddress" in names

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        assert results == []

    def test_finds_struct_by_name(self, tools):
        results = tools["find_symbol"]("Order")
        names = [r["name"] for r in results]
        assert "Order" in names

    def test_cached_result_is_identical(self, tools):
        first = tools["find_symbol"]("ValidateAddress")
        second = tools["find_symbol"]("ValidateAddress")
        assert first == second


# goto_definition — requires gopls
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinition:
    def _find_call_line(self, file, name):
        source = Path(file).read_text()
        for i, line in enumerate(source.splitlines(), 1):
            if name in line and "(" in line:
                return i
        return 1

    def test_resolves_validate_address_cross_file(self, tools):
        line = self._find_call_line(ORDERS_FILE, "ValidateAddress")
        result = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        assert result
        assert "validation.go" in result["file"]
        assert result["name"] == "ValidateAddress"

    def test_resolves_format_currency_cross_file(self, tools):
        line = self._find_call_line(ORDERS_FILE, "FormatCurrency")
        result = tools["goto_definition"](ORDERS_FILE, line, "FormatCurrency")
        assert result
        assert "utils.go" in result["file"]
        assert result["name"] == "FormatCurrency"

    def test_returns_empty_dict_on_unknown_symbol(self, tools):
        result = tools["goto_definition"](ORDERS_FILE, 1, "zzz_nonexistent")
        assert result == {}

    def test_result_has_required_keys(self, tools):
        line = self._find_call_line(ORDERS_FILE, "ValidateAddress")
        result = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        for key in ("name", "kind", "file", "line", "signature"):
            assert key in result

    def test_cache_is_stable(self, tools):
        line = self._find_call_line(ORDERS_FILE, "ValidateAddress")
        r1 = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        r2 = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        assert r1 == r2

    def test_docstring_from_description(self, tools):
        line = self._find_call_line(ORDERS_FILE, "ValidateAddress")
        result = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        assert "docstring" in result
        assert "required keys" in result["docstring"]

    def test_docstring_is_empty_when_no_comment(self, tools):
        result = tools["goto_definition"](UTILS_FILE, 10, "NoDocFunc")
        assert "docstring" in result
        assert result["docstring"] == ""


# get_callees — requires gopls
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGetCallees:
    def _process_line(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        return next(s for s in symbols if s["name"] == "Process")["line"]

    def test_finds_validate_address_or_format_currency(self, tools):
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        names = [c["name"] for c in callees]
        assert "ValidateAddress" in names or "FormatCurrency" in names

    def test_callees_cross_file(self, tools):
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        files = [c["file"] for c in callees]
        assert any("orders.go" not in f for f in files)

    def test_callees_have_required_keys(self, tools):
        line = self._process_line(tools)
        callees = tools["get_callees"](ORDERS_FILE, line)
        for c in callees:
            assert "file" in c
            assert "line" in c
            assert c["line"] >= 1

    def test_no_callees_for_empty_func(self, tools):
        symbols = tools["get_file_outline"](UTILS_FILE)
        fc = next(s for s in symbols if s["name"] == "FormatCurrency")
        callees = tools["get_callees"](UTILS_FILE, fc["line"])
        assert isinstance(callees, list)


# get_callers — requires gopls
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGetCallers:
    def _validate_line(self, tools):
        symbols = tools["get_file_outline"](VALIDATION_FILE)
        return next(s for s in symbols if s["name"] == "ValidateAddress")["line"]

    def test_orders_is_a_caller(self, tools):
        line = self._validate_line(tools)
        callers = tools["get_callers"](VALIDATION_FILE, line)
        files = [c["file"] for c in callers]
        assert any("orders.go" in f for f in files)

    def test_callers_have_required_keys(self, tools):
        line = self._validate_line(tools)
        callers = tools["get_callers"](VALIDATION_FILE, line)
        for c in callers:
            assert "file" in c
            assert "line" in c

    def test_no_callers_for_unused_func(self, tools):
        symbols = tools["get_file_outline"](ORDERS_FILE)
        line = next(s for s in symbols if s["name"] == "UnusedFunc")["line"]
        callers = tools["get_callers"](ORDERS_FILE, line)
        assert callers == []


# Agent workflow — validates the agent actually uses all context tools
@pytest.mark.asyncio
async def test_agent_uses_context_tools(run_sh: Path) -> None:
    """Ensure the agent can use every context tool to analyze Go source code."""
    client = RecordingClient(prefix="acp_go_context_test_")
    with client:
        src = Path(__file__).parent / "sample" / "go"
        dst = client.temp_dir / "sample_go"
        shutil.copytree(src, dst, symlinks=True)

        await run_agent(
            client,
            run_sh,
            f"""
            In the directory {dst}, there is a Go project with orders.go, validation.go, and utils.go.
            I need you to do a full cross-reference analysis of the function 'ValidateAddress':
            1. Use find_symbol to locate 'ValidateAddress'.
            2. Use get_file_outline on validation.go to see all symbols in that file.
            3. Use goto_definition from orders.go to find where ValidateAddress is defined.
            4. Use get_source to read the full source code of ValidateAddress.
            5. Use get_callers on ValidateAddress to see who calls it.
            6. Use get_callees on the Process method in orders.go to see what it calls.
            Report back what ValidateAddress does, what constant it references, and who calls it.
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
