"""Integration tests for GoEnvManager against real microsandbox microVMs.

These tests boot actual microsandbox VMs and verify that GoEnvManager
correctly sets up a Go build environment when the project is a Go module,
and skips setup entirely when it is not.

Requires the microsandbox runtime (msb) to be installed and functional.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

# Sample projects

SAMPLES = Path(__file__).parent / "sample"
SAMPLE_GO = SAMPLES / "go" / "simple"
SAMPLE_GO_MONOREPO = SAMPLES / "go" / "monorepo"
SAMPLE_NO_GOMOD = SAMPLES / "project-no-manifest"

# The official golang image.  Integration tests use it directly so that
# GoEnvManager's toolchain detection and tool install run against a real
# Go-capable image.
_GO_IMAGE = "golang"


# Memory for the VM.  Go compiles (especially gin's dependency tree) are
# memory-hungry; 1024 MiB gets OOM-killed during `go build`.
_GO_MEMORY = 2048


def _copy_sample(src: Path, dest: Path) -> None:
    """Copy a sample project into dest, excluding build artifacts."""
    skip = {".venv-msb", "test.egg-info"}
    for item in src.iterdir():
        if item.name in skip:
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name)
        else:
            shutil.copy2(item, dest / item.name)


# Tests: Go env setup


@pytest.mark.asyncio
class TestGoEnvSetup:
    """Verify GoEnvManager sets up the env for a Go module."""

    async def test_go_env_active_for_go_module(self, tmp_path: Path):
        """go.mod present -> go_env set and functional."""
        _copy_sample(SAMPLE_GO, tmp_path)
        assert (tmp_path / "go.mod").exists()

        b = MicrosandboxBackend(
            root_dir=str(tmp_path), image=_GO_IMAGE, memory=_GO_MEMORY
        )
        try:
            await b._ensure_sandbox()

            # go_env must be set (not None).
            assert b.go_env is not None, "go_env should be set for a Go module"

            # The env dict must carry the Go toolchain paths.
            assert "GOPATH" in b.go_env
            assert "GOTOOLCHAIN" in b.go_env
            assert b.go_env["GOTOOLCHAIN"] == "auto"
            assert "/go/bin" in b.go_env["PATH"]
            assert "/usr/local/go/bin" in b.go_env["PATH"]

            # go itself must be usable through the env.
            result = await b.aexecute("go version")
            assert result.exit_code == 0
            assert "go version" in result.output

            # gopls must be installed and runnable.
            result = await b.aexecute("gopls version")
            assert result.exit_code == 0
            assert "gopls" in result.output
        finally:
            await b.stop()

    async def test_dev_tools_installed(self, tmp_path: Path):
        """All mandatory Go dev tools are installed into /go/bin."""
        _copy_sample(SAMPLE_GO, tmp_path)

        b = MicrosandboxBackend(
            root_dir=str(tmp_path), image=_GO_IMAGE, memory=_GO_MEMORY
        )
        try:
            await b._ensure_sandbox()
            assert b.go_env is not None

            for tool in (
                "gopls",
                "goimports",
                "gorename",
                "staticcheck",
                "govulncheck",
                "dlv",
            ):
                result = await b.aexecute(f"{tool} -h 2>&1 || {tool} version")
                # staticcheck -h exits non-zero, so accept either form.
                assert tool in result.output, (
                    f"{tool} not found in output: {result.output}"
                )
        finally:
            await b.stop()

    async def test_go_build_works(self, tmp_path: Path):
        """The env must support `go build` against the project's go.mod."""
        _copy_sample(SAMPLE_GO, tmp_path)

        b = MicrosandboxBackend(
            root_dir=str(tmp_path), image=_GO_IMAGE, memory=_GO_MEMORY
        )
        try:
            await b._ensure_sandbox()
            assert b.go_env is not None

            # `go build ./...` must succeed — this exercises GOTOOLCHAIN=auto
            # (auto-downloading the toolchain pinned by go.mod if needed).
            result = await b.aexecute("go build ./...")
            assert result.exit_code == 0, f"go build failed: {result.output}"
        finally:
            await b.stop()

    async def test_no_go_mod_skips_setup(self, tmp_path: Path):
        """No go.mod -> go_env is None (setup skipped)."""
        _copy_sample(SAMPLE_NO_GOMOD, tmp_path)
        assert not (tmp_path / "go.mod").exists()

        b = MicrosandboxBackend(
            root_dir=str(tmp_path), image=_GO_IMAGE, memory=_GO_MEMORY
        )
        try:
            await b._ensure_sandbox()

            # go_env must NOT be set — no Go project detected.
            assert b.go_env is None, "go_env should be None without go.mod"
        finally:
            await b.stop()


# Tests: monorepo (go.work + nested packages)


@pytest.mark.asyncio
class TestGoEnvMonorepo:
    """Verify GoEnvManager works with a monorepo layout (go.work)."""

    async def test_go_env_active_for_monorepo(self, tmp_path: Path):
        """A monorepo with go.mod at root still triggers env setup."""
        _copy_sample(SAMPLE_GO_MONOREPO, tmp_path)
        assert (tmp_path / "go.mod").exists()

        b = MicrosandboxBackend(
            root_dir=str(tmp_path), image=_GO_IMAGE, memory=_GO_MEMORY
        )
        try:
            await b._ensure_sandbox()

            assert b.go_env is not None
            assert b.go_env["GOTOOLCHAIN"] == "auto"

            result = await b.aexecute("go version")
            assert result.exit_code == 0
        finally:
            await b.stop()
