"""Guest-compatible Python venv management for microsandbox.

When the host venv's architecture doesn't match the VM (e.g. macOS arm64
→ Linux aarch64), the host ``.venv`` won't run inside the VM.  This module
builds and maintains a separate ``.venv-msb`` inside the bind mount, with a
marker file to skip rebuilds when nothing changed.

The user's own ``.venv`` is never modified.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from microsandbox import Sandbox

logger = logging.getLogger("metalgate_code")

SANDBOX_WORKDIR = "/workspace"

_MS_VENV_DIR = ".venv-msb"
"""Name of the guest-compatible venv created inside the bind mount."""

_MS_VENV_MARKER = ".msb_built"
"""Marker file recording the build's VM arch and dep hash."""

_VENV_SETUP_TIMEOUT_SEC = 300
"""Timeout for the one-shot venv creation / dependency install step."""

_VENV_REBUILD_TIMEOUT_SEC = 600
"""Timeout for full dependency reinstall when the venv is stale."""

_PYTHON_IMAGE_MARKERS = ("python", "uv:python", "uv")
"""Substrings that identify a Python-capable image (for venv setup)."""


def is_python_image(image: str) -> bool:
    """Whether the image is Python-capable (triggers venv setup)."""
    img = image.lower()
    return any(marker in img for marker in _PYTHON_IMAGE_MARKERS)


def build_venv_env(venv_bin: str, base_path: str) -> dict[str, str]:
    """Build the env dict that activates a venv for ``sb.shell(env=...)``.

    ``base_path`` is the VM's real ``$PATH`` (captured during venv setup),
    since env dict values are not shell-expanded.
    """
    venv_dir = str(Path(venv_bin).parent)
    return {
        "VIRTUAL_ENV": venv_dir,
        "PATH": f"{venv_bin}:{base_path}",
        "UV_PROJECT_ENVIRONMENT": venv_dir,
        "UV_NO_SYNC": "1",
    }


class VenvManager:
    """Manages a guest-compatible Python venv inside a microsandbox VM.

    Depends on a small set of VM primitives (run command, check path
    exists, read/write file) provided by the backend, so it stays
    decoupled from the sandbox lifecycle.
    """

    def __init__(
        self,
        sb: Sandbox,
        *,
        run_in_vm,
        path_exists,
        read_file_in_vm,
        write_file_in_vm,
    ) -> None:
        self._sb = sb
        self._run_in_vm = run_in_vm
        self._path_exists = path_exists
        self._read_file_in_vm = read_file_in_vm
        self._write_file_in_vm = write_file_in_vm

    async def ensure(self) -> tuple[str | None, dict[str, str] | None]:
        """Ensure a guest-compatible venv exists; return ``(venv_bin, venv_env)``.

        Returns ``(None, None)`` if no venv could be established.
        """
        workdir = SANDBOX_WORKDIR
        host_venv_bin = f"{workdir}/.venv/bin"

        # Capture the VM's real PATH (env dict values aren't shell-expanded).
        path_res = await self._run_in_vm(self._sb, "echo $PATH", timeout=10, env=None)
        vm_path = path_res.output.strip() if path_res.exit_code == 0 else ""

        # 1. Try the user's .venv — empirical arch check (runs inside VM).
        if await self._path_exists(self._sb, f"{workdir}/.venv"):
            check = await self._run_in_vm(
                self._sb,
                f"{host_venv_bin}/python --version",
                timeout=30,
                env=None,
            )
            if check.exit_code == 0:
                logger.info("Reusing host .venv (arch-compatible) at %s", host_venv_bin)
                return host_venv_bin, build_venv_env(host_venv_bin, vm_path)
            logger.info(
                "Host .venv is not guest-compatible (exit_code=%s); "
                "building .venv-msb instead",
                check.exit_code,
            )

        # 2. Build / refresh .venv-msb
        ms_venv = f"{workdir}/{_MS_VENV_DIR}"
        ms_venv_bin = f"{ms_venv}/bin"
        marker = f"{ms_venv}/{_MS_VENV_MARKER}"

        # Detect VM arch for the marker.
        arch_res = await self._run_in_vm(self._sb, "uname -m", timeout=10, env=None)
        vm_arch = arch_res.output.strip() if arch_res.exit_code == 0 else "unknown"

        # Compute a hash of the dependency manifest so we rebuild on changes.
        dep_hash = await self._dep_manifest_hash(workdir)

        # Check the marker: skip rebuild if arch + dep_hash match.
        if await self._path_exists(self._sb, marker):
            marker_content = await self._read_file_in_vm(self._sb, marker)
            if marker_content and _marker_matches(marker_content, vm_arch, dep_hash):
                logger.info(
                    ".venv-msb up to date (arch=%s, hash=%s); reusing",
                    vm_arch,
                    dep_hash,
                )
                return ms_venv_bin, build_venv_env(ms_venv_bin, vm_path)
            logger.info(".venv-msb stale (arch/deps changed); rebuilding")

        # Remove stale venv.
        await self._run_in_vm(
            self._sb,
            f"rm -rf {shlex.quote(ms_venv)}",
            timeout=30,
            env=None,
        )

        # Detect which dependency manifest exists.
        manifest = await self._detect_manifest(workdir)

        if manifest is None:
            logger.warning(
                "No pyproject.toml, uv.lock, or requirements.txt found in %s; "
                "skipping venv creation.",
                workdir,
            )
            return None, None

        if manifest == "uv.lock":
            built = await self._build_with_uv(ms_venv, ms_venv_bin, workdir)
        else:
            built = await self._build_with_pip(ms_venv, ms_venv_bin, workdir, manifest)

        if not built:
            return ms_venv_bin, build_venv_env(ms_venv_bin, vm_path)

        # Write the marker.
        marker_body = f"arch={vm_arch}\ndeps={dep_hash}\n"
        await self._write_file_in_vm(self._sb, marker, marker_body)

        return ms_venv_bin, build_venv_env(ms_venv_bin, vm_path)

    async def _build_with_uv(
        self, ms_venv: str, ms_venv_bin: str, workdir: str
    ) -> bool:
        """Build venv using uv (uv.lock present).  Returns True on success."""
        # Install uv into system Python, then use it for everything.
        uv_install_res = await self._run_in_vm(
            self._sb,
            "python3 -m pip install uv",
            timeout=_VENV_SETUP_TIMEOUT_SEC,
            env=None,
        )
        if uv_install_res.exit_code != 0:
            logger.warning(
                "uv install failed (exit %s): %s",
                uv_install_res.exit_code,
                uv_install_res.output[-2000:],
            )
            return False

        # Create venv with uv (respects .python-version).
        venv_res = await self._run_in_vm(
            self._sb,
            f"uv venv {shlex.quote(ms_venv)}",
            timeout=_VENV_SETUP_TIMEOUT_SEC,
            env=None,
        )
        if venv_res.exit_code != 0:
            logger.warning(
                "venv creation failed (exit %s): %s",
                venv_res.exit_code,
                venv_res.output[-2000:],
            )
            return False

        # Sync dependencies from uv.lock (including dev group for ty, etc).
        py = shlex.quote(f"{ms_venv}/bin/python")
        sync_cmd = (
            f"cd {shlex.quote(workdir)} && "
            f"VIRTUAL_ENV={shlex.quote(ms_venv)} "
            f"PATH={shlex.quote(ms_venv_bin)}:$PATH "
            f"UV_PROJECT_ENVIRONMENT={shlex.quote(ms_venv)} "
            f"uv sync --all-groups --python {py}"
        )
        sync_res = await self._run_in_vm(
            self._sb,
            sync_cmd,
            timeout=_VENV_REBUILD_TIMEOUT_SEC,
            env=None,
        )
        if sync_res.exit_code != 0:
            logger.warning(
                "uv sync failed (exit %s): %s",
                sync_res.exit_code,
                sync_res.output[-2000:],
            )
            return False
        logger.info("Dependencies installed into .venv-msb via uv sync")

        # Install uv into the venv so skills can find it via PATH.
        # uv venv creates venvs without pip, so use uv pip install
        # (uv's own installer) instead of python -m pip.
        uv_venv_res = await self._run_in_vm(
            self._sb,
            f"VIRTUAL_ENV={shlex.quote(ms_venv)} "
            f"UV_PROJECT_ENVIRONMENT={shlex.quote(ms_venv)} "
            f"uv pip install uv --python {py}",
            timeout=_VENV_SETUP_TIMEOUT_SEC,
            env=None,
        )
        if uv_venv_res.exit_code != 0:
            logger.warning(
                "uv install into .venv-msb failed (exit %s): %s",
                uv_venv_res.exit_code,
                uv_venv_res.output[-2000:],
            )
        else:
            logger.info("uv installed into .venv-msb")

        return True

    async def _build_with_pip(
        self, ms_venv: str, ms_venv_bin: str, workdir: str, manifest: str
    ) -> bool:
        """Build venv using pip (no uv.lock).  Returns True on success."""
        venv_res = await self._run_in_vm(
            self._sb,
            f"python3 -m venv {shlex.quote(ms_venv)}",
            timeout=_VENV_SETUP_TIMEOUT_SEC,
            env=None,
        )
        if venv_res.exit_code != 0:
            logger.warning(
                "venv creation failed (exit %s): %s",
                venv_res.exit_code,
                venv_res.output[-2000:],
            )
            return False

        install_cmd = _build_pip_install_cmd(workdir, ms_venv, manifest)
        res = await self._run_in_vm(
            self._sb,
            install_cmd,
            timeout=_VENV_REBUILD_TIMEOUT_SEC,
            env=None,
        )
        if res.exit_code != 0:
            logger.warning(
                "Dependency install into .venv-msb failed (exit %s): %s",
                res.exit_code,
                res.output[-2000:],
            )
            return False
        logger.info("Dependencies installed into .venv-msb via pip")
        return True

    async def _detect_manifest(self, workdir: str) -> str | None:
        """Return the name of the dependency manifest that exists, or None.

        Preference order: ``uv.lock`` → ``pyproject.toml`` →
        ``requirements.txt``.  Only the first existing one is returned.
        """
        for manifest in ("uv.lock", "pyproject.toml", "requirements.txt"):
            if await self._path_exists(self._sb, f"{workdir}/{manifest}"):
                return manifest
        return None

    async def _dep_manifest_hash(self, workdir: str) -> str:
        """Hash of the project's dependency manifest (uv.lock or pyproject.toml)."""
        for manifest in ("uv.lock", "pyproject.toml", "requirements.txt"):
            path = f"{workdir}/{manifest}"
            if await self._path_exists(self._sb, path):
                res = await self._run_in_vm(
                    self._sb,
                    f"sha256sum {shlex.quote(path)} 2>/dev/null || true",
                    timeout=15,
                    env=None,
                )
                h = res.output.strip().split()[0] if res.output else ""
                if h:
                    return f"{manifest}:{h[:16]}"
        return "none"


def _build_pip_install_cmd(workdir: str, venv: str, manifest: str) -> str:
    """Build a ``pip install`` command for the given manifest.

    Used when ``uv.lock`` is not present.  ``manifest`` is either
    ``pyproject.toml`` or ``requirements.txt``.
    """
    py = shlex.quote(f"{venv}/bin/python")
    activate = (
        f"VIRTUAL_ENV={shlex.quote(venv)} PATH={shlex.quote(venv + '/bin')}:$PATH"
    )
    cd = f"cd {shlex.quote(workdir)} && "
    if manifest == "pyproject.toml":
        return f"{cd}{activate} {py} -m pip install -e ."
    return (
        f"{cd}{activate} {py} -m pip install "
        f"-r {shlex.quote(workdir + '/requirements.txt')}"
    )


def _marker_matches(marker: str, arch: str, dep_hash: str) -> bool:
    """Check whether a marker file matches the current arch + dep hash."""
    arch_match = False
    hash_match = False
    for line in marker.splitlines():
        if line.startswith("arch=") and line[5:].strip() == arch:
            arch_match = True
        elif line.startswith("deps=") and line[5:].strip() == dep_hash:
            hash_match = True
    return arch_match and hash_match
