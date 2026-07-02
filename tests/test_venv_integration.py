"""Integration tests for VenvManager against real microsandbox microVMs.

These tests boot actual microsandbox VMs and verify that VenvManager
correctly builds a guest-compatible Python venv for each dependency
manifest type (uv.lock, pyproject.toml, requirements.txt, none).

Requires the microsandbox runtime (msb) to be installed and functional.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

# --------------------------------------------------------------------------- #
# Sample projects
# --------------------------------------------------------------------------- #

SAMPLES = Path(__file__).parent / "sample"
SAMPLE_PYTHON = SAMPLES / "python"  # uv.lock (default)
SAMPLE_PYPYPROJECT = SAMPLES / "python-pyproject"
SAMPLE_REQUIREMENTS = SAMPLES / "python-requirements"
SAMPLE_NO_MANIFEST = SAMPLES / "python-no-manifest"


def _copy_sample(src: Path, dest: Path) -> None:
    """Copy a sample project into dest, excluding .venv-msb and egg-info."""
    for item in src.iterdir():
        if item.name in (".venv-msb", "test.egg-info"):
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name)
        else:
            shutil.copy2(item, dest / item.name)


# --------------------------------------------------------------------------- #
# Tests: venv creation by manifest type
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestVenvCreation:
    """Verify venv is built and functional for each manifest type."""

    async def test_venv_with_uv_lock(self, tmp_path: Path):
        """uv.lock present -> uv sync."""
        _copy_sample(SAMPLE_PYTHON, tmp_path)

        assert (tmp_path / "uv.lock").exists(), "uv.lock should exist in sample"

        b = MicrosandboxBackend(root_dir=str(tmp_path), memory=1024)
        try:
            await b._ensure_sandbox()

            assert b._venv_bin is not None
            assert ".venv-msb" in b._venv_bin

            # The venv python must work.
            result = await b.aexecute(f"{b._venv_bin}/python --version")
            assert result.exit_code == 0
            assert "Python" in result.output

            # The project must be importable.
            result = await b.aexecute(
                f"{b._venv_bin}/python -c 'import orders; print(orders.__name__)'"
            )
            assert result.exit_code == 0
            assert "orders" in result.output

            # uv must be available in the venv (installed by VenvManager).
            result = await b.aexecute(
                f"{b._venv_bin}/python -c 'import uv; print(uv.__name__)'"
            )
            assert result.exit_code == 0
        finally:
            await b.stop()

    async def test_venv_with_pyproject_toml(self, tmp_path: Path):
        """pyproject.toml present (no uv.lock) -> pip install -e ."""
        _copy_sample(SAMPLE_PYPYPROJECT, tmp_path)
        (tmp_path / "uv.lock").unlink(missing_ok=True)

        b = MicrosandboxBackend(root_dir=str(tmp_path), memory=1024)
        try:
            await b._ensure_sandbox()

            assert b._venv_bin is not None
            assert b._venv_env is not None
            assert ".venv-msb" in b._venv_bin

            # The venv python must work.
            result = await b.aexecute(f"{b._venv_bin}/python --version")
            assert result.exit_code == 0
            assert "Python" in result.output

            # The project must be importable (pip install -e . installed it).
            result = await b.aexecute(
                f"{b._venv_bin}/python -c 'import orders; print(orders.__name__)'"
            )
            assert result.exit_code == 0
            assert "orders" in result.output

            # Project dependencies (requests) must be installed.
            result = await b.aexecute(
                f"{b._venv_bin}/python -c 'import requests; print(requests.__name__)'"
            )
            assert result.exit_code == 0
        finally:
            await b.stop()

    async def test_venv_with_requirements_txt(self, tmp_path: Path):
        """requirements.txt present (no uv.lock, no pyproject.toml) -> pip install -r."""
        _copy_sample(SAMPLE_REQUIREMENTS, tmp_path)

        b = MicrosandboxBackend(root_dir=str(tmp_path), memory=1024)
        try:
            await b._ensure_sandbox()

            assert b._venv_bin is not None
            assert ".venv-msb" in b._venv_bin

            # The venv python must work.
            result = await b.aexecute(f"{b._venv_bin}/python --version")
            assert result.exit_code == 0

            # requests must be importable.
            result = await b.aexecute(
                f"{b._venv_bin}/python -c 'import requests; print(requests.__version__)'"
            )
            assert result.exit_code == 0
        finally:
            await b.stop()

    async def test_no_manifest_skips_venv(self, tmp_path: Path):
        """No dependency manifest -> no venv created."""
        _copy_sample(SAMPLE_NO_MANIFEST, tmp_path)

        b = MicrosandboxBackend(root_dir=str(tmp_path), memory=1024)
        try:
            await b._ensure_sandbox()

            assert b._venv_bin is None
            assert b._venv_env is None
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# Tests: venv reuse and marker
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestVenvReuse:
    """Verify the marker file skips rebuilds across sessions."""

    async def test_marker_skips_rebuild(self, tmp_path: Path):
        """Second _ensure_sandbox reuses the venv without rebuilding."""
        _copy_sample(SAMPLE_PYPYPROJECT, tmp_path)
        (tmp_path / "uv.lock").unlink(missing_ok=True)

        b = MicrosandboxBackend(root_dir=str(tmp_path), memory=1024)
        try:
            await b._ensure_sandbox()
            assert b._venv_bin is not None

            # Marker must exist.
            marker = tmp_path / ".venv-msb" / ".msb_built"
            assert marker.exists(), "marker file should exist after first build"

            # Stop and re-create the backend — venv should be reused.
            await b.stop()

            b2 = MicrosandboxBackend(root_dir=str(tmp_path), memory=1024)
            await b2._ensure_sandbox()
            assert b2._venv_bin is not None
            assert b2._venv_bin == b._venv_bin
            await b2.stop()
        finally:
            await b.stop()
