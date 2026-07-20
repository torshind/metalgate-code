"""Sandbox integration tests for Go contextual symbol search tools in a
monorepo layout.

These tests run against a real microsandbox VM and the end-to-end agent
workflow.  They are intentionally kept separate from the unit tests in
``test_go_context_monorepo.py`` (which use ``LocalShellBackend`` and need
no sandbox).
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

MONOREPO_DIR = Path(__file__).parent / "sample" / "go" / "monorepo"
SHARED_FILE = str(
    MONOREPO_DIR / "private" / "service" / "internal" / "shared" / "context.go"
)
RENDERER_FILE = str(
    MONOREPO_DIR / "private" / "service" / "internal" / "shared" / "renderer.go"
)
CONTROLLER_FILE = str(MONOREPO_DIR / "private" / "service" / "api" / "controller.go")
MIDDLEWARE_FILE = str(MONOREPO_DIR / "private" / "service" / "api" / "middleware.go")
RENDER_CALL_FILE = str(MONOREPO_DIR / "private" / "service" / "api" / "render_call.go")
GIN_CALL_FILE = str(MONOREPO_DIR / "private" / "service" / "api" / "gin_call.go")
CLOSURE_FILE = str(MONOREPO_DIR / "private" / "service" / "api" / "closure.go")
CLIENT_FILE = str(MONOREPO_DIR / "public" / "client" / "client.go")
E2E_FILE = str(MONOREPO_DIR / "e2e-tests" / "suite" / "test.go")

_HAS_GOPLS = shutil.which("gopls") is not None


@pytest.fixture(scope="module")
def tools():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    shell_env = os.environ.copy()
    shell_backend = MicrosandboxBackend(
        root_dir=str(MONOREPO_DIR),
        image="golang",
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


# goto_definition — requires gopls inside the sandbox
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinition:
    def _find_call_line(self, file, name):
        source = Path(file).read_text()
        for i, line in enumerate(source.splitlines(), 1):
            if name in line and "(" in line and not line.lstrip().startswith("//"):
                return i
        return 1

    def test_resolves_cross_package_function(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "shared.ToContext")
        result = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        assert result
        assert "context.go" in result["file"]
        assert result["name"] == "ToContext"

    def test_resolves_cross_package_function_from_client(self, tools):
        line = self._find_call_line(CLIENT_FILE, "api.NewController")
        result = tools["goto_definition"](CLIENT_FILE, line, "api.NewController")
        assert result
        assert "controller.go" in result["file"]
        assert result["name"] == "NewController"

    def test_resolves_cross_package_method_from_client(self, tools):
        line = self._find_call_line(CLIENT_FILE, "c.ctrl.Publish")
        result = tools["goto_definition"](CLIENT_FILE, line, "c.ctrl.Publish")
        assert result
        assert "controller.go" in result["file"]
        assert result["name"] == "Publish"

    def test_returns_empty_dict_on_unknown_symbol(self, tools):
        result = tools["goto_definition"](CONTROLLER_FILE, 1, "zzz_nonexistent")
        assert result == {}

    def test_result_has_required_keys(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "shared.ToContext")
        result = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        for key in ("name", "kind", "file", "line", "signature"):
            assert key in result

    def test_cache_is_stable(self, tools):
        line = self._find_call_line(CONTROLLER_FILE, "shared.ToContext")
        r1 = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        r2 = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        assert r1 == r2

    # Selector resolution (Bug 1 & 2)

    def test_resolves_qualified_stdlib_call(self, tools):
        """goto_definition on a qualified stdlib call like fmt.Sprintf
        should resolve to the stdlib function, not the import line."""
        line = self._find_call_line(SHARED_FILE, "fmt.Sprintf")
        result = tools["goto_definition"](SHARED_FILE, line, "fmt.Sprintf")
        assert result
        # Must point into the Go stdlib, not the local import line.
        assert "context.go" not in result["file"]
        assert result["kind"] != "unknown"

    def test_resolves_method_call_on_receiver(self, tools):
        """goto_definition on a method call like c.Publish should
        resolve to the method definition, not the receiver variable."""
        line = self._find_call_line(MIDDLEWARE_FILE, "c.Publish")
        result = tools["goto_definition"](MIDDLEWARE_FILE, line, "c.Publish")
        assert result
        assert "controller.go" in result["file"]
        # Should point at the Publish method definition, not a var decl.
        assert result["kind"] == "method"
        assert result["line"] != line  # not the call site itself

    def test_resolves_shared_to_context_cross_package(self, tools):
        """goto_definition on shared.ToContext should resolve to the
        shared package function, not the import line."""
        line = self._find_call_line(CONTROLLER_FILE, "shared.ToContext")
        result = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        assert result
        assert "controller.go" not in result["file"]
        assert result["kind"] != "unknown"


# Path translation — host paths must be translated to sandbox paths
#
# When the agent runs with a MicrosandboxBackend, the agent passes host
# paths (e.g. /Users/foo/project/controller.go) but gopls runs inside the
# sandbox and only knows /workspace/... paths.  The tracer must translate
# host paths to sandbox paths before sending URIs to gopls, and translate
# gopls response URIs back to host paths.
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestPathTranslation:
    """The tracer must translate host paths to sandbox paths and back."""

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    def test_host_path_resolves_cross_package_selector(self, tools):
        """A selector call (shared.ToContext) given with the host path must
        resolve correctly — the tracer translates the host path to the
        sandbox path before sending it to gopls."""
        line = 17  # return shared.ToContext(key, val)
        result = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        assert result, "got empty result with host path"
        assert self._basename(result["file"]) == "context.go"

    def test_host_path_resolves_cross_package_function(self, tools):
        """A cross-package function call (api.NewController) given with the
        host path must resolve to controller.go."""
        line = 14  # return &Client{ctrl: api.NewController()}
        result = tools["goto_definition"](CLIENT_FILE, line, "api.NewController")
        assert result, "got empty result with host path"
        assert self._basename(result["file"]) == "controller.go"

    def test_host_path_resolves_same_package_method(self, tools):
        """A same-package method call (c.Publish) given with the host path
        must resolve to the method definition, not the parameter
        declaration."""
        line = 8  # return c.Publish("middleware", 1)
        result = tools["goto_definition"](MIDDLEWARE_FILE, line, "c.Publish")
        assert result, "got empty result with host path"
        assert self._basename(result["file"]) == "controller.go"
        assert result["kind"] == "method"

    def test_result_file_is_sandbox_path(self, tools):
        """The result's file field must be a sandbox path (containing
        /workspace), since the agent works with sandbox paths."""
        line = 17
        result = tools["goto_definition"](CONTROLLER_FILE, line, "shared.ToContext")
        assert result
        assert result["file"].startswith("/workspace/"), (
            f"result file is not a sandbox path: {result['file']!r}"
        )


# goto_definition — cross-package resolution
#
# These tests verify that goto_definition can resolve symbols defined in
# DIFFERENT packages/directories of the monorepo — not just symbols in the
# same file or stdlib.
#
# Cross-package call sites used in these tests:
#
#     controller.go:17  shared.ToContext  -> context.go:6   (api -> shared)
#     client.go:14      api.NewController -> controller.go:11 (client -> api)
#     client.go:19      c.ctrl.Publish    -> controller.go:16 (client -> api, method)
#     test.go:14        api.NewController -> controller.go:11 (suite -> api)
#     test.go:19        r.ctrl.Publish    -> controller.go:16 (suite -> api, method)
#
# All ground-truth locations were captured from gopls v0.23.0 at the correct
# member column of each call site.
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinitionCrossPackage:
    """Resolves symbols defined in a different package/directory of the
    monorepo.

    Each call site is in one package directory and the target symbol is
    defined in a different package directory.  If gopls hasn't indexed the
    target package's files, ``goto_definition`` returns an empty ``{}``.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # cross-package function: shared.ToContext
    #
    # controller.go (package api) calls shared.ToContext, which is defined
    # in context.go (package shared) in a different directory.

    def test_resolves_cross_package_function(self, tools):
        """shared.ToContext (controller.go:17) must resolve to context.go:6
        in the shared package, not return empty.

        gopls ground truth at the member column (col 16):
            context.go:6  func shared.ToContext(key string, value int) map[string]string
        """
        result = tools["goto_definition"](CONTROLLER_FILE, 17, "shared.ToContext")
        assert result, (
            "got empty result — gopls has not indexed the shared package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "context.go", (
            f"expected context.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 6, f"expected line 6, got {result.get('line')}"
        assert "ToContext" in result["signature"], (
            f"expected signature containing 'ToContext', "
            f"got {result.get('signature', '')!r}"
        )

    def test_cross_package_function_kind(self, tools):
        """shared.ToContext must be classified as a function."""
        result = tools["goto_definition"](CONTROLLER_FILE, 17, "shared.ToContext")
        assert result, "got empty result"
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    def test_cross_package_function_not_import_line(self, tools):
        """shared.ToContext must NOT resolve to the import line in
        controller.go (line 4, signature 'package shared')."""
        result = tools["goto_definition"](CONTROLLER_FILE, 17, "shared.ToContext")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "controller.go" and result["line"] == 4
        ), (
            "resolved to the import line — column points at the qualifier, "
            "not the member"
        )

    # cross-package function: api.NewController
    #
    # client.go (package client) calls api.NewController, which is defined
    # in controller.go (package api) in a different directory.

    def test_resolves_cross_package_function_from_client(self, tools):
        """api.NewController (client.go:14) must resolve to controller.go:11
        in the api package, not return empty.

        gopls ground truth at the member column (col 27):
            controller.go:11  func api.NewController() *api.Controller
        """
        result = tools["goto_definition"](CLIENT_FILE, 14, "api.NewController")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 11, f"expected line 11, got {result.get('line')}"
        assert "NewController" in result["signature"], (
            f"expected signature containing 'NewController', "
            f"got {result.get('signature', '')!r}"
        )

    # cross-package method: c.ctrl.Publish
    #
    # client.go (package client) calls c.ctrl.Publish, where Publish is a
    # method on *Controller defined in controller.go (package api).
    # This is a chained selector: c.ctrl is a field, .Publish is the method.

    def test_resolves_cross_package_method_from_client(self, tools):
        """c.ctrl.Publish (client.go:19) must resolve to controller.go:16
        (the Publish method), not return empty and not resolve to the
        field declaration.

        gopls ground truth at the member column (col 16, the 'P' of Publish):
            controller.go:16  func (c *api.Controller) Publish(key string, val int) ...
        """
        result = tools["goto_definition"](CLIENT_FILE, 19, "c.ctrl.Publish")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_cross_package_method_kind(self, tools):
        """c.ctrl.Publish must be classified as a method."""
        result = tools["goto_definition"](CLIENT_FILE, 19, "c.ctrl.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_cross_package_method_not_field_declaration(self, tools):
        """c.ctrl.Publish must NOT resolve to the field declaration
        'field ctrl *api.Controller' in client.go:9."""
        result = tools["goto_definition"](CLIENT_FILE, 19, "c.ctrl.Publish")
        assert result, "got empty result"
        assert "field " not in result["signature"], (
            f"resolved to a field declaration: {result['signature']!r} "
            "(column points at the receiver/field, not the method)"
        )

    # cross-package from e2e-tests
    #
    # test.go (package suite) calls api.NewController and r.ctrl.Publish,
    # both defined in controller.go (package api).  The e2e-tests directory
    # is a separate package that imports the api package.

    def test_resolves_cross_package_function_from_e2e(self, tools):
        """api.NewController (test.go:14) must resolve to controller.go:11
        from the e2e-tests package, not return empty.

        gopls ground truth at the member column (col 27):
            controller.go:11  func api.NewController() *api.Controller
        """
        result = tools["goto_definition"](E2E_FILE, 14, "api.NewController")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 11, f"expected line 11, got {result.get('line')}"

    def test_resolves_cross_package_method_from_e2e(self, tools):
        """r.ctrl.Publish (test.go:19) must resolve to controller.go:16
        from the e2e-tests package, not return empty.

        gopls ground truth at the member column (col 16, the 'P' of Publish):
            controller.go:16  func (c *api.Controller) Publish(key string, val int) ...
        """
        result = tools["goto_definition"](E2E_FILE, 19, "r.ctrl.Publish")
        assert result, (
            "got empty result — gopls has not indexed the api package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )


# goto_definition — same-package, different file
#
# This reproduces the gin bug where a method call like `c.Next` in
# recovery.go (package gin) must resolve to context.go (also package gin,
# but a different file).  The original cross-package tests only exercised
# calls that cross package boundaries (different import paths).  The
# same-package-different-file case is a distinct failure mode: gopls
# must have the sibling file indexed to resolve the target.
#
# middleware.go (package api) calls c.Publish and c.Lookup, both defined
# in controller.go (also package api, different file).
#
# Ground truth (controller.go):
#     line 16: func (c *Controller) Publish(key string, val int) map[string]string
#     line 21: func (c *Controller) Lookup(ctx map[string]string, key string) (string, error)
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinitionSamePackage:
    """Resolves method calls within the same package but a different file.

    middleware.go (package api) calls c.Publish and c.Lookup, both defined
    in controller.go (also package api).  If gopls hasn't indexed
    controller.go, ``goto_definition`` returns an empty ``{}`` or resolves
    to the parameter declaration instead of the method.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # c.Publish: middleware.go:8 -> controller.go:16

    def test_resolves_same_package_method_publish(self, tools):
        """c.Publish (middleware.go:8) must resolve to controller.go:16
        (the Publish method in the same package, different file), not
        return empty and not resolve to the parameter declaration.

        gopls ground truth at the member column (col 11, the 'P' of Publish):
            controller.go:16  func (c *Controller) Publish(key string, val int) ...
        """
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, (
            "got empty result — gopls has not indexed controller.go; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"

    def test_same_package_method_publish_signature(self, tools):
        """c.Publish must return a signature containing 'Publish'."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_same_package_method_publish_kind(self, tools):
        """c.Publish must be classified as a method."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_same_package_method_publish_not_param(self, tools):
        """c.Publish must NOT resolve to the parameter declaration in
        middleware.go (line 7, 'c *Controller')."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "middleware.go" and result["line"] == 7
        ), (
            "resolved to the parameter declaration — column points at the "
            "receiver, not the member"
        )
        assert "var " not in result.get("signature", ""), (
            f"resolved to a variable declaration: {result['signature']!r}"
        )

    # c.Lookup: middleware.go:14 -> controller.go:21

    def test_resolves_same_package_method_lookup(self, tools):
        """c.Lookup (middleware.go:14) must resolve to controller.go:21
        (the Lookup method in the same package, different file).

        gopls ground truth at the member column (col 11, the 'L' of Lookup):
            controller.go:21  func (c *Controller) Lookup(ctx map[string]string, ...) ...
        """
        result = tools["goto_definition"](MIDDLEWARE_FILE, 14, "c.Lookup")
        assert result, (
            "got empty result — gopls has not indexed controller.go; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 21, f"expected line 21, got {result.get('line')}"

    def test_same_package_method_lookup_signature(self, tools):
        """c.Lookup must return a signature containing 'Lookup'."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 14, "c.Lookup")
        assert result, "got empty result"
        assert "Lookup" in result["signature"], (
            f"expected signature containing 'Lookup', "
            f"got {result.get('signature', '')!r}"
        )

    def test_same_package_method_lookup_kind(self, tools):
        """c.Lookup must be classified as a method."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 14, "c.Lookup")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_same_package_method_col_points_at_member(self, tools):
        """For c.Publish on middleware.go:8, the returned col must point
        at the 'Publish' member (col 11), not the receiver 'c' or the dot."""
        result = tools["goto_definition"](MIDDLEWARE_FILE, 8, "c.Publish")
        assert result, "got empty result"
        assert 11 <= result["col"] <= 18, (
            f"expected col in [11, 18] (the 'Publish' member), got col={result['col']}"
        )


# goto_definition — interface method called across packages
#
# This reproduces the gin bug where `r.Render` in context.go (package gin)
# must resolve to the interface method in render.go (package render).
# gopls resolves interface-method calls to the method line INSIDE the
# interface, not the `type X interface` line.
#
# render_call.go (package api) calls r.Render and r.WriteContentType on
# a shared.Renderer interface, defined in renderer.go (package shared).
#
# Ground truth (renderer.go):
#     line 10: 	Render(dest string) error          (interface method)
#     line 12: 	WriteContentType() string          (interface method)
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinitionInterfaceMethod:
    """Resolves interface method calls across packages.

    render_call.go (package api) calls r.Render and r.WriteContentType
    on a variable of type shared.Renderer (an interface).  The methods
    are defined inside the interface in renderer.go (package shared).

    gopls resolves to the method line inside the interface, NOT the
    `type Renderer interface` line.  The hover signature starts with
    `func (shared.Renderer)` so kind inference must classify it as a method.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # r.WriteContentType: render_call.go:12 -> renderer.go:12

    def test_resolves_interface_method_write_content_type(self, tools):
        """r.WriteContentType (render_call.go:12) must resolve to
        renderer.go:12 (the method line inside the interface), not the
        `type Renderer interface` line and not the parameter declaration.

        gopls ground truth at the member column (col 10, the 'W'):
            renderer.go:12  WriteContentType() string
        """
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, (
            "got empty result — gopls has not indexed the shared package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "renderer.go", (
            f"expected renderer.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 12, f"expected line 12, got {result.get('line')}"

    def test_interface_method_write_content_type_signature(self, tools):
        """r.WriteContentType must return a signature containing
        'WriteContentType'."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert "WriteContentType" in result["signature"], (
            f"expected signature containing 'WriteContentType', "
            f"got {result.get('signature', '')!r}"
        )

    def test_interface_method_write_content_type_kind(self, tools):
        """r.WriteContentType must be classified as a method (the hover
        signature starts with 'func (shared.Renderer)')."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_interface_method_write_content_type_not_type_line(self, tools):
        """r.WriteContentType must NOT resolve to the `type Renderer
        interface` line (renderer.go:8).  gopls resolves to the method
        line inside the interface, not the type declaration."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "renderer.go" and result["line"] == 8
        ), (
            "resolved to the 'type Renderer interface' line — should resolve "
            "to the method line inside the interface"
        )

    def test_interface_method_write_content_type_not_param(self, tools):
        """r.WriteContentType must NOT resolve to the parameter declaration
        in render_call.go (line 11, 'r shared.Renderer')."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 12, "r.WriteContentType")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "render_call.go" and result["line"] == 11
        ), (
            "resolved to the parameter declaration — column points at the "
            "receiver, not the member"
        )

    # r.Render: render_call.go:14 -> renderer.go:10

    def test_resolves_interface_method_render(self, tools):
        """r.Render (render_call.go:14) must resolve to renderer.go:10
        (the method line inside the interface), not the `type Renderer
        interface` line and not the parameter declaration.

        gopls ground truth at the member column (col 11, the 'R'):
            renderer.go:10  Render(dest string) error
        """
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, (
            "got empty result — gopls has not indexed the shared package; "
            "the tool must open all .go files in the package directory before querying"
        )
        assert self._basename(result["file"]) == "renderer.go", (
            f"expected renderer.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 10, f"expected line 10, got {result.get('line')}"

    def test_interface_method_render_signature(self, tools):
        """r.Render must return a signature containing 'Render'."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert "Render" in result["signature"], (
            f"expected signature containing 'Render', "
            f"got {result.get('signature', '')!r}"
        )

    def test_interface_method_render_kind(self, tools):
        """r.Render must be classified as a method."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_interface_method_render_not_type_line(self, tools):
        """r.Render must NOT resolve to the `type Renderer interface` line
        (renderer.go:8)."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "renderer.go" and result["line"] == 8
        ), (
            "resolved to the 'type Renderer interface' line — should resolve "
            "to the method line inside the interface"
        )

    def test_interface_method_col_points_at_member(self, tools):
        """For r.Render on render_call.go:14, the returned col must point
        at the 'Render' member (col 11), not the receiver 'r' or the dot."""
        result = tools["goto_definition"](RENDER_CALL_FILE, 14, "r.Render")
        assert result, "got empty result"
        assert 11 <= result["col"] <= 17, (
            f"expected col in [11, 17] (the 'Render' member), got col={result['col']}"
        )


# goto_definition — 3rd-party package qualified call
#
# This reproduces the gin bug where `gin.New()` in external.go must resolve
# to the function definition in the 3rd-party gin package, not the
# `import "github.com/gin-gonic/gin"` line in the caller.
#
# gin_call.go (package api) imports gin and calls gin.New() and gin.Default().
#
# Ground truth (gin v1.12.0):
#     gin.go:202  func New(opts ...OptionFunc) *Engine
#     gin.go:236  func Default(opts ...OptionFunc) *Engine
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinitionThirdParty:
    """Resolves qualified calls to 3rd-party package functions.

    gin_call.go (package api) imports gin and calls gin.New() and
    gin.Default().  The tool must resolve to the function definition in
    the gin package source, not the `import "github.com/gin-gonic/gin"`
    line in gin_call.go.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # gin.New: gin_call.go:11 -> gin.go:202

    def test_resolves_third_party_gin_new(self, tools):
        """gin.New (gin_call.go:11) must resolve to gin.go (the 3rd-party
        package source), not gin_call.go:4 (the import line in the caller).

        gopls ground truth at the member column (col 13, the 'N'):
            gin.go:202  func New(opts ...OptionFunc) *Engine
        """
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "gin.go", (
            f"expected gin.go (the function definition), "
            f"got {result.get('file', 'EMPTY')} "
            "(resolving the package import instead of the function)"
        )
        assert result["line"] == 202, f"expected line 202, got {result.get('line')}"

    def test_third_party_gin_new_signature(self, tools):
        """gin.New must return a signature containing 'New'."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert "New" in result["signature"], (
            f"expected signature containing 'New', got {result.get('signature', '')!r}"
        )

    def test_third_party_gin_new_kind(self, tools):
        """gin.New must be classified as a function, not 'unknown'."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r} "
            "(kind inference fell back to 'unknown' because the hover "
            "signature is the package docstring, not the function signature)"
        )

    def test_third_party_gin_new_not_import_line(self, tools):
        """gin.New must NOT resolve to the import line in gin_call.go
        (line 4, 'github.com/gin-gonic/gin')."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "gin_call.go" and result["line"] == 4
        ), (
            "resolved to the import line — the column points at the "
            "qualifier/dot, not the member 'New'"
        )

    def test_third_party_gin_new_col_points_at_member(self, tools):
        """For gin.New on gin_call.go:11, the returned col must point at
        the 'New' member (col 13), not the qualifier 'gin' or the dot."""
        result = tools["goto_definition"](GIN_CALL_FILE, 11, "gin.New")
        assert result, "got empty result"
        assert 13 <= result["col"] <= 16, (
            f"expected col in [13, 16] (the 'New' member), got col={result['col']}"
        )

    # gin.Default: gin_call.go:16 -> gin.go:236

    def test_resolves_third_party_gin_default(self, tools):
        """gin.Default (gin_call.go:16) must resolve to gin.go:236.

        gopls ground truth at the member column (col 13, the 'D'):
            gin.go:236  func Default(opts ...OptionFunc) *Engine
        """
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "gin.go", (
            f"expected gin.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 236, f"expected line 236, got {result.get('line')}"

    def test_third_party_gin_default_signature(self, tools):
        """gin.Default must return a signature containing 'Default'."""
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert "Default" in result["signature"], (
            f"expected signature containing 'Default', "
            f"got {result.get('signature', '')!r}"
        )

    def test_third_party_gin_default_kind(self, tools):
        """gin.Default must be classified as a function."""
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert result["kind"] == "function", (
            f"expected kind 'function', got {result['kind']!r}"
        )

    def test_third_party_gin_default_not_import_line(self, tools):
        """gin.Default must NOT resolve to the import line in gin_call.go."""
        result = tools["goto_definition"](GIN_CALL_FILE, 16, "gin.Default")
        assert result, "got empty result"
        assert not (
            self._basename(result["file"]) == "gin_call.go" and result["line"] == 4
        ), (
            "resolved to the import line — the column points at the "
            "qualifier/dot, not the member 'Default'"
        )


# goto_definition — method calls inside function literals (closures)
#
# This reproduces the core gin bug.  In gin's recovery.go,
# CustomRecoveryWithWriter returns a func(c *Context) (a HandlerFunc).
# Inside that returned closure, calls like c.Next(), c.Error(), c.Abort()
# all return empty {} from goto_definition, while the same calls in a
# non-closure function resolve correctly.
#
# The bug: tree-sitter fails to locate the selector_expression node when
# it is inside a function literal body that is returned, assigned, or
# passed as an argument.  The column computation returns a position that
# doesn't correspond to any identifier, so gopls returns empty {}.
#
# closure.go (package api) has five patterns:
#
#     line 21:  return c.Publish("returned", 1)    inside a RETURNED closure
#     line 29:  return c.Publish("assigned", 1)    inside an ASSIGNED closure
#     line 38:  return c.Publish("passed", 1)       inside a PASSED closure
#     line 51:  _ = c.Publish("immediate", 1)       inside an immediately-invoked closure
#     line 58:  return c.Publish("direct", 1)       direct call, no closure (control)
#
# Ground truth: all five must resolve to controller.go:16
#     func (c *Controller) Publish(key string, val int) map[string]string
#
# The first three (returned/assigned/passed) reproduce the bug.
# The last two (immediate-invoke, direct) are control cases that already work.
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGotoDefinitionInClosure:
    """Resolves method calls inside function literals (closures).

    closure.go has five call sites for c.Publish, all targeting
    controller.go:16.  The first three are inside function literals
    that are returned, assigned, or passed as an argument.  The last
    two are control cases (immediately-invoked closure and direct call).

    The bug: goto_definition returns empty {} for calls inside
    non-immediately-invoked function literals because tree-sitter
    fails to locate the selector_expression node, causing the column
    computation to return a position that doesn't correspond to any
    identifier.
    """

    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rstrip("/").split("/")[-1]

    # returned closure: closure.go:21

    def test_resolves_method_in_returned_closure(self, tools):
        """c.Publish inside a returned func literal (closure.go:21) must
        resolve to controller.go:16, not return empty.

        This reproduces gin's recovery.go pattern where
        CustomRecoveryWithWriter returns a func(c *Context) and calls
        like c.Next() inside that closure return {}.
        """
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, (
            "got empty result — tree-sitter failed to locate the "
            "selector_expression node inside a returned function literal; "
            "the column computation returned a non-identifier position"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_returned_closure_kind_is_method(self, tools):
        """c.Publish inside a returned closure must be classified as a method."""
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    def test_returned_closure_col_points_at_member(self, tools):
        """For c.Publish inside a returned closure, the returned col must
        point at the 'Publish' member, not the receiver or dot."""
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, "got empty result"
        assert 12 <= result["col"] <= 19, (
            f"expected col in [12, 19] (the 'Publish' member), got col={result['col']}"
        )

    # returned closure: path/name variants
    #
    # The returned-closure bug is path-dependent: it only manifests when
    # ALL THREE conditions hold:
    #   1. absolute file path
    #   2. returned function literal
    #   3. qualified name (c.Publish, not just Publish)
    # Removing any one condition makes the call resolve correctly.
    # These tests pin down each condition individually.

    def test_returned_closure_qualified_name_absolute_path(self, tools):
        """c.Publish with an absolute path inside a returned closure —
        the exact combination that triggers the bug.  Must NOT return empty."""
        result = tools["goto_definition"](CLOSURE_FILE, 21, "c.Publish")
        assert result, (
            "got empty result for absolute path + returned closure + "
            "qualified name — this is the reported bug"
        )
        assert self._basename(result["file"]) == "controller.go"
        assert result["line"] == 16

    def test_returned_closure_simple_name_absolute_path(self, tools):
        """Publish (simple name) with an absolute path inside a returned
        closure must resolve — removing the qualified-name condition
        fixes it, proving the bug requires the qualified name."""
        result = tools["goto_definition"](CLOSURE_FILE, 21, "Publish")
        assert result, (
            "got empty result for simple name — the bug requires a "
            "qualified name; a simple identifier should resolve"
        )
        assert self._basename(result["file"]) == "controller.go"
        assert result["line"] == 16

    # assigned closure: closure.go:29

    def test_resolves_method_in_assigned_closure(self, tools):
        """c.Publish inside an assigned func literal (closure.go:29) must
        resolve to controller.go:16, not return empty."""
        result = tools["goto_definition"](CLOSURE_FILE, 29, "c.Publish")
        assert result, (
            "got empty result — tree-sitter failed to locate the "
            "selector_expression node inside an assigned function literal"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_assigned_closure_kind_is_method(self, tools):
        """c.Publish inside an assigned closure must be classified as a method."""
        result = tools["goto_definition"](CLOSURE_FILE, 29, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    # passed closure: closure.go:38

    def test_resolves_method_in_passed_closure(self, tools):
        """c.Publish inside a func literal passed as an argument
        (closure.go:38) must resolve to controller.go:16, not return empty."""
        result = tools["goto_definition"](CLOSURE_FILE, 38, "c.Publish")
        assert result, (
            "got empty result — tree-sitter failed to locate the "
            "selector_expression node inside a passed function literal"
        )
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"
        assert "Publish" in result["signature"], (
            f"expected signature containing 'Publish', "
            f"got {result.get('signature', '')!r}"
        )

    def test_passed_closure_kind_is_method(self, tools):
        """c.Publish inside a passed closure must be classified as a method."""
        result = tools["goto_definition"](CLOSURE_FILE, 38, "c.Publish")
        assert result, "got empty result"
        assert result["kind"] == "method", (
            f"expected kind 'method', got {result['kind']!r}"
        )

    # control: immediately-invoked closure: closure.go:51

    def test_resolves_method_in_immediate_invoke_closure(self, tools):
        """c.Publish inside an immediately-invoked func literal
        (closure.go:51) must resolve to controller.go:16.  This is a
        control case that already works."""
        result = tools["goto_definition"](CLOSURE_FILE, 51, "c.Publish")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"

    # control: direct call: closure.go:58

    def test_resolves_method_direct_call(self, tools):
        """c.Publish as a direct call with no closure (closure.go:58) must
        resolve to controller.go:16.  This is a control case that already
        works."""
        result = tools["goto_definition"](CLOSURE_FILE, 58, "c.Publish")
        assert result, "got empty result"
        assert self._basename(result["file"]) == "controller.go", (
            f"expected controller.go, got {result.get('file', 'EMPTY')}"
        )
        assert result["line"] == 16, f"expected line 16, got {result.get('line')}"

    # consistency: all five patterns resolve to the same target

    def test_all_closure_patterns_resolve_same_target(self, tools):
        """All five c.Publish call sites in closure.go must resolve to the
        same target (controller.go:16).  If any returns empty or a different
        target, the column computation is inconsistent for function literals."""
        for line, label in [
            (21, "returned closure"),
            (29, "assigned closure"),
            (38, "passed closure"),
            (51, "immediate-invoke closure"),
            (58, "direct call"),
        ]:
            result = tools["goto_definition"](CLOSURE_FILE, line, "c.Publish")
            assert result, (
                f"got empty result for {label} at line {line} — "
                f"tree-sitter failed to locate the selector_expression"
            )
            assert self._basename(result["file"]) == "controller.go", (
                f"{label} at line {line}: expected controller.go, "
                f"got {result.get('file', 'EMPTY')}"
            )
            assert result["line"] == 16, (
                f"{label} at line {line}: expected line 16, got {result.get('line')}"
            )


# get_callees — requires gopls inside the sandbox
@pytest.mark.skipif(not _HAS_GOPLS, reason="gopls not installed")
class TestGetCallees:
    def _publish_line(self, tools):
        symbols = tools["get_file_outline"](CONTROLLER_FILE)
        return next(s for s in symbols if s["name"] == "Publish")["line"]

    def test_finds_to_context_or_from_context(self, tools):
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        names = [c["name"] for c in callees]
        assert "ToContext" in names or "FromContext" in names

    def test_callees_cross_package(self, tools):
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        files = [c["file"] for c in callees]
        assert any("controller.go" not in f for f in files)

    def test_callees_have_required_keys(self, tools):
        line = self._publish_line(tools)
        callees = tools["get_callees"](CONTROLLER_FILE, line)
        for c in callees:
            assert "file" in c
            assert "line" in c
            assert c["line"] >= 1

    def test_no_callees_for_empty_func(self, tools):
        symbols = tools["get_file_outline"](SHARED_FILE)
        fc = next(s for s in symbols if s["name"] == "UnusedFunc")
        callees = tools["get_callees"](SHARED_FILE, fc["line"])
        assert isinstance(callees, list)


# get_callers — requires gopls inside the sandbox
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


# Agent workflow — validates the agent actually uses all context tools
@pytest.mark.asyncio
async def test_agent_uses_context_tools(run_sh: Path) -> None:
    """Ensure the agent can use every context tool to analyze Go source code
    in a monorepo layout."""
    client = RecordingClient(prefix="acp_go_context_monorepo_test_")
    with client:
        src = Path(__file__).parent / "sample" / "go" / "monorepo"
        dst = client.temp_dir
        # Copy monorepo Go project into temp_dir root so _detect_language
        # finds go.mod and creates GoTracer for the agent.
        shutil.copytree(src, dst, dirs_exist_ok=True)

        await run_agent(
            client,
            run_sh,
            f"""
            In the directory {dst}, there is a Go monorepo project with packages
            under private/service/api, private/service/internal/shared, public/client,
            and e2e-tests/suite.
            You run in a sandbox, so to make the path translation work, you need to pass the full path to tools.
            I need you to do a full cross-reference analysis of the function 'ToContext':
            1. Use find_symbol to locate 'ToContext'.
            2. Use get_file_outline on context.go to see all symbols in that file.
            3. Use goto_definition from controller.go to find where shared.ToContext is defined.
            4. Use get_source to read the full source code of ToContext.
            5. Use get_callers on ToContext to see who calls it.
            6. Use get_callees on the Publish method in controller.go to see what it calls.
            Report back what ToContext does, what calls it, and what Publish calls.
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
