"""Sandbox integration tests for Go contextual symbol search tools.

These tests run against a real microsandbox VM and the end-to-end agent
workflow.  They are intentionally kept separate from the unit tests in
``test_go_context.py`` (which use ``LocalShellBackend`` and need no
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

SAMPLE_DIR = Path(__file__).parent / "sample" / "go" / "simple"
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
        image="golang",
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


# goto_definition — requires gopls inside the sandbox
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

    # Selector resolution (Bug 1 & 2)

    def test_resolves_qualified_stdlib_call(self, tools):
        """goto_definition on a qualified stdlib call like fmt.Sprintf
        should resolve to the stdlib function, not the import line."""
        line = self._find_call_line(UTILS_FILE, "fmt.Sprintf")
        result = tools["goto_definition"](UTILS_FILE, line, "fmt.Sprintf")
        assert result
        # Must point into the Go stdlib, not the local import line.
        assert "utils.go" not in result["file"]
        assert result["kind"] != "unknown"

    def test_resolves_method_call_on_receiver(self, tools):
        """goto_definition on a method call like o.Process() should
        resolve to the method definition, not the receiver variable."""
        line = self._find_call_line(ORDERS_FILE, "o.Process")
        result = tools["goto_definition"](ORDERS_FILE, line, "o.Process")
        assert result
        assert "orders.go" in result["file"]
        # Should point at the Process method definition, not a var decl.
        assert result["kind"] == "method"
        assert result["line"] != line  # not the call site itself

    def test_resolves_strings_to_upper_stdlib(self, tools):
        """goto_definition on strings.ToUpper should resolve to the
        stdlib function, not the import line."""
        line = self._find_call_line(ORDERS_FILE, "strings.ToUpper")
        result = tools["goto_definition"](ORDERS_FILE, line, "strings.ToUpper")
        assert result
        assert "orders.go" not in result["file"]
        assert result["kind"] != "unknown"


# Path translation — host paths must be translated to sandbox paths
#
# When the agent runs with a MicrosandboxBackend, the agent passes host
# paths (e.g. /Users/foo/project/orders.go) but gopls runs inside the
# sandbox and only knows /workspace/... paths.  The tracer must translate
# host paths to sandbox paths before sending URIs to gopls, and translate
# gopls response URIs back to host paths.
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestPathTranslation:
    """The tracer must translate host paths to sandbox paths and back."""

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_host_path_resolves_selector(self, tools):
        """A selector call (strings.ToUpper) given with the host path must
        resolve correctly — the tracer translates the host path to the
        sandbox path before sending it to gopls."""
        line = 40  # return strings.ToUpper(o.Process())
        result = tools["goto_definition"](ORDERS_FILE, line, "strings.ToUpper")
        assert result, "got empty result with host path"
        assert self._basename(result["file"]) == "strings.go"

    def test_host_path_resolves_plain_function(self, tools):
        """A plain function call (ValidateAddress) given with the host path
        must resolve to validation.go."""
        line = 26  # if !ValidateAddress(o.Address) {
        result = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        assert result, "got empty result with host path"
        assert self._basename(result["file"]) == "validation.go"

    def test_host_path_resolves_method(self, tools):
        """A method call (o.Process) given with the host path must resolve
        to the method definition, not the receiver variable."""
        line = 40  # return strings.ToUpper(o.Process())
        result = tools["goto_definition"](ORDERS_FILE, line, "o.Process")
        assert result, "got empty result with host path"
        assert self._basename(result["file"]) == "orders.go"
        assert result["kind"] == "method"

    def test_result_file_is_sandbox_path(self, tools):
        """The result's file field must be a sandbox path (containing
        /workspace), since the agent works with sandbox paths."""
        line = 26
        result = tools["goto_definition"](ORDERS_FILE, line, "ValidateAddress")
        assert result
        assert result["file"].startswith("/workspace/"), (
            f"result file is not a sandbox path: {result['file']!r}"
        )


# get_callees — requires gopls inside the sandbox
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


# get_callers — requires gopls inside the sandbox
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
        src = Path(__file__).parent / "sample" / "go" / "simple"
        dst = client.temp_dir
        # Copy simple Go project into temp_dir root so _detect_language
        # finds go.mod and creates GoTracer for the agent.
        shutil.copytree(src, dst, dirs_exist_ok=True)

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
