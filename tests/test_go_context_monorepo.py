"""Integration tests for Go contextual symbol search tools in a monorepo layout.

This mirrors the structure of go.evroc.dev: a single module with nested packages
under private/, public/, and e2e-tests/.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from deepagents.backends import LocalShellBackend

from metalgate_code.context import get_code_tools

MONOREPO_DIR = Path(__file__).parent / "sample" / "go" / "monorepo"
SHARED_FILE = str(MONOREPO_DIR / "private" / "service" / "internal" / "shared" / "context.go")
CONTROLLER_FILE = str(MONOREPO_DIR / "private" / "service" / "api" / "controller.go")
CLIENT_FILE = str(MONOREPO_DIR / "public" / "client" / "client.go")
E2E_FILE = str(MONOREPO_DIR / "e2e-tests" / "suite" / "test.go")

_HAS_GOPLS = shutil.which("gopls") is not None


@pytest.fixture(scope="module")
def tools():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    shell_env = os.environ.copy()
    shell_backend = LocalShellBackend(
        root_dir=str(MONOREPO_DIR),
        virtual_mode=False,
        env=shell_env,
        inherit_env=True,
    )

    tool_list = get_code_tools(
        cwd=str(MONOREPO_DIR),
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
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        assert any(s["name"] == "Controller" and s["kind"] == "struct" for s in symbols)

    def test_finds_function(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        assert any(s["name"] == "NewController" and s["kind"] == "function" for s in symbols)

    def test_finds_method(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        method = next(
            (s for s in symbols if s["name"] == "Publish" and s["kind"] == "method"), None
        )
        assert method is not None

    def test_method_has_receiver(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        publish = next(s for s in symbols if s["name"] == "Publish")
        assert "Controller" in (publish.get("class") or "")

    def test_finds_function_in_shared(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        assert any(s["name"] == "ToContext" and s["kind"] == "function" for s in symbols)
        assert any(s["name"] == "FromContext" and s["kind"] == "function" for s in symbols)

    def test_cached_result_is_identical(self, tools):
        first = tools["get_file_outline"](CONTROLLER_FILE)
        second = tools["get_file_outline"](CONTROLLER_FILE)
        assert first == second


# get_source
class TestGetSource:
    def _publish_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        return next(s for s in symbols if s["name"] == "Publish")["line"]

    def test_source_contains_func(self, tools):
        line = self._publish_line(tools)
        result = tools["get_source"](CONTROLLER_FILE, line)
        assert "Publish" in result["source"]
        assert "shared.ToContext" in result["source"]

    def test_start_and_end_lines_are_sane(self, tools):
        line = self._publish_line(tools)
        result = tools["get_source"](CONTROLLER_FILE, line)
        assert result["start_line"] >= 1
        assert result["end_line"] >= result["start_line"]

    def test_get_source_from_body_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        publish = next(s for s in symbols if s["name"] == "Publish")
        body_line = publish["line"] + 2
        result = tools["get_source"](CONTROLLER_FILE, body_line)
        assert "Publish" in result["source"]

    def test_get_source_cross_package(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        tc = next(s for s in symbols if s["name"] == "ToContext")
        result = tools["get_source"](SHARED_FILE, tc["line"])
        assert "ToContext" in result["source"]


# find_symbol
class TestFindSymbol:
    def test_exact_match_finds_to_context(self, tools):
        results = tools["find_symbol"]("ToContext")
        names = [r["name"] for r in results]
        assert "ToContext" in names

    def test_exact_match_finds_from_context(self, tools):
        results = tools["find_symbol"]("FromContext")
        names = [r["name"] for r in results]
        assert "FromContext" in names

    def test_finds_struct_by_name(self, tools):
        results = tools["find_symbol"]("Controller")
        names = [r["name"] for r in results]
        assert "Controller" in names

    def test_finds_function_cross_package(self, tools):
        results = tools["find_symbol"]("NewController")
        names = [r["name"] for r in results]
        assert "NewController" in names

    def test_unknown_symbol_returns_empty_list(self, tools):
        results = tools["find_symbol"]("zzz_does_not_exist_xyz")
        assert results == []

    def test_cached_result_is_identical(self, tools):
        first = tools["find_symbol"]("ToContext")
        second = tools["find_symbol"]("ToContext")
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

    def test_resolves_to_context_cross_file(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "ToContext")
        result = tools["goto_definition"](CONTROLLER_FILE, line, "ToContext")
        assert result
        assert "shared" in result["file"]
        assert "context.go" in result["file"]
        assert result["name"] == "ToContext"

    def test_resolves_from_context_cross_file(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "FromContext")
        result = tools["goto_definition"](CONTROLLER_FILE, line, "FromContext")
        assert result
        assert "shared" in result["file"]
        assert "context.go" in result["file"]
        assert result["name"] == "FromContext"

    def test_resolves_new_controller_from_client(self, tools):
        line = self._find_call_line(CLIENT_FILE, "NewController")
        result = tools["goto_definition"](CLIENT_FILE, line, "NewController")
        assert result
        assert "controller.go" in result["file"]
        assert result["name"] == "NewController"

    def test_returns_empty_dict_on_unknown_symbol(self, tools):
        result = tools["goto_definition"](CONTROLLER_FILE, 1, "zzz_nonexistent")
        assert result == {}

    def test_result_has_required_keys(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "ToContext")
        result = tools["goto_definition"](CONTROLLER_FILE, line, "ToContext")
        for key in ("name", "kind", "file", "line", "signature"):
            assert key in result

    def test_cache_is_stable(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "ToContext")
        r1 = tools["goto_definition"](CONTROLLER_FILE, line, "ToContext")
        r2 = tools["goto_definition"](CONTROLLER_FILE, line, "ToContext")
        assert r1 == r2


# get_callees — requires gopls
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGetCallees:
    def _publish_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        return next(s for s in symbols if s["name"] == "Publish")["line"]

    def test_finds_to_context_callee(self, tools):
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        names = [c["name"] for c in callees]
        assert "ToContext" in names

    def test_callees_cross_file(self, tools):
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        files = [c["file"] for c in callees]
        assert any("shared" in f for f in files)

    def test_callees_have_required_keys(self, tools):
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        for c in callees:
            assert "file" in c
            assert "line" in c
            assert c["line"] >= 1

    def test_no_callees_for_empty_func(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        fc = next(s for s in symbols if s["name"] == "FromContext")
        callees = tools["get_callees"](SHARED_FILE, fc["line"])
        assert isinstance(callees, list)


# get_callers — requires gopls
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGetCallers:
    def _to_context_line(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        return next(s for s in symbols if s["name"] == "ToContext")["line"]

    def test_controller_is_a_caller(self, tools):
        line = self._to_context_line(tools)
        callers = tools["get_callers"](SHARED_FILE, line)
        files = [c["file"] for c in callers]
        assert any("controller.go" in f for f in files)

    def test_client_is_a_caller(self, tools):
        line = self._to_context_line(tools)
        callers = tools["get_callers"](SHARED_FILE, line)
        files = [c["file"] for c in callers]
        assert any("client.go" in f for f in files)

    def test_e2e_is_a_caller(self, tools):
        line = self._to_context_line(tools)
        callers = tools["get_callers"](SHARED_FILE, line)
        files = [c["file"] for c in callers]
        assert any("test.go" in f for f in files)

    def test_callers_have_required_keys(self, tools):
        line = self._to_context_line(tools)
        callers = tools["get_callers"](SHARED_FILE, line)
        for c in callers:
            assert "file" in c
            assert "line" in c

    def test_no_callers_for_unused_func(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        line = next(s for s in symbols if s["name"] == "UnusedFunc")["line"]
        callers = tools["get_callers"](SHARED_FILE, line)
        assert callers == []
