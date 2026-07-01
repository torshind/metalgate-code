"""microsandbox backend for isolated shell command execution and file operations.

Implements ``SandboxBackendProtocol`` by delegating to a microsandbox
microVM.  Commands run inside the VM, not on the host, providing process
isolation and resource limits.

The backend is async-first (microsandbox's SDK is async-only).  Sync
variants (``execute``, ``read``, ``write``, …) bridge to the async
implementations via ``_run_async`` so that callers using the sync
``BackendProtocol`` interface continue to work.
"""

import asyncio
import base64
import logging
import os
import shlex
import uuid
from pathlib import Path
from typing import TypeVar

from deepagents.backends.filesystem import _map_exception_to_standard_error
from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    IS_DIRECTORY,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from microsandbox import Sandbox, Volume

logger = logging.getLogger("metalgate_code")

T = TypeVar("T")

SANDBOX_WORKDIR = "/workspace"

DEFAULT_COMMAND_TIMEOUT_SEC = 300
"""Default per-command timeout (5 minutes) to prevent stuck command hangs."""

SANDBOX_CREATE_TIMEOUT_SEC = 5
SANDBOX_CREATE_RETRY_ATTEMPTS = 3
"""Timeout for booting a new microsandbox VM.

The msb runtime occasionally hangs during VM boot (SDK-level race).
When this happens, the msb process is already running but the Python
SDK never receives the ready signal.  A shorter timeout lets the retry
logic kick in quickly.
"""

SANDBOX_STOP_TIMEOUT_SEC = 30
"""Timeout for stopping/cleaning up a microsandbox VM."""

FS_OP_TIMEOUT_SEC = 60
"""Timeout for individual filesystem operations (read/write/exists/mkdir/list)."""

HEALTH_CHECK_TIMEOUT_SEC = 5
"""Timeout for the health-check commands."""

SYNC_BRIDGE_TIMEOUT_SEC = 600
"""Timeout for sync wrappers bridging to async via _run_async."""

_MAX_OUTPUT_BYTES = 100_000
"""Maximum number of bytes to capture from command output before truncation."""

_PIPE_FIELD_COUNT = 2
_GREP_FIELD_COUNT = 3

# ------------------------------------------------------------------ #
# Python venv management (architecture-mismatch handling)
# ------------------------------------------------------------------ #

_MS_VENV_DIR = ".venv-msb"
"""Name of the guest-compatible venv created inside the bind mount.

The user's own ``.venv`` is never modified.  When the host venv's
architecture doesn't match the VM (e.g. macOS arm64 → Linux aarch64),
we build a fresh venv here instead.
"""

_MS_VENV_MARKER = ".msb_built"
"""Marker file inside ``_MS_VENV_DIR`` recording the build's VM arch
and a hash of the project's dependency manifest, so we can skip
rebuilds across sessions when nothing has changed.
"""

_VENV_SETUP_TIMEOUT_SEC = 300
"""Timeout for the one-shot venv creation / dependency install step."""

_VENV_REBUILD_TIMEOUT_SEC = 600
"""Timeout for full dependency reinstall when the venv is stale."""

_PYTHON_IMAGE_MARKERS = ("python", "uv:python", "uv")
"""Substrings that identify a Python-capable image (for venv setup)."""


def _run_async(coro):
    """Run an async coroutine, handling both sync and async contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=SYNC_BRIDGE_TIMEOUT_SEC)


class MicrosandboxBackend(SandboxBackendProtocol):
    """Sandbox backend backed by a microsandbox microVM.

    Each backend instance creates a microVM sandbox on first use (or at
    construction if ``eager=True``).  The sandbox is configured with:

    - A bind-mount of the host ``root_dir`` at ``/workspace`` inside the VM.
    - Environment variables from ``env`` (plus host env if ``inherit_env``).
    - Optional secrets injected via microsandbox's TLS-proxy secret mechanism.

    All file and shell operations execute inside the VM.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        image: str = "python",
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
        cpus: int = 1,
        memory: int = 1024,
        secrets: list | None = None,
        eager: bool = False,
    ) -> None:
        """Initialize the microsandbox backend.

        Args:
            root_dir: Host directory to bind-mount into the sandbox at
                ``/workspace``.  This becomes the working directory for all
                commands.
            image: OCI image to boot (e.g. ``"python"``, ``"ubuntu:24.04"``).
            env: Environment variables for the sandbox.  Merged on top of the
                host environment when ``inherit_env=True``.
            inherit_env: Whether to inherit the parent process's environment.
            cpus: Virtual CPUs allocated to the sandbox.
            memory: Guest memory in MiB.
            secrets: List of microsandbox ``SecretEntry`` objects for
                credential injection.
            eager: If ``True``, boot the sandbox immediately in the
                constructor.  Otherwise it boots on first use.
        """
        self._root_dir = str(Path(root_dir))
        self._image = image
        self._cpus = cpus
        self._memory = memory
        self._secrets = secrets or []

        # Build environment
        if inherit_env:
            self._env = os.environ.copy()
            if env is not None:
                self._env.update(env)
        else:
            self._env = env if env is not None else {}

        self._sandbox_id = f"msb-{uuid.uuid4().hex[:8]}"
        self._sandbox: Sandbox | None = None
        self._lock = asyncio.Lock()

        # Guest-compatible Python venv path (set after _ensure_python_venv).
        # When non-None, commands are run with this venv activated.
        self._venv_bin: str | None = None
        self._venv_env: dict[str, str] | None = None

        if eager:
            _run_async(self._ensure_sandbox())

    # ------------------------------------------------------------------ #
    # Sandbox lifecycle
    # ------------------------------------------------------------------ #

    async def _cleanup_stale_sandboxes(self) -> None:
        """Stop and remove any sandboxes left over from previous runs.

        When a sandbox is created but not properly stopped (e.g. the
        event loop closed before cleanup ran), the msb runtime kills the
        VM process but leaves a stale DB record and stale agent sockets.
        These accumulate and eventually cause ``Sandbox.create`` to hang.
        """
        try:
            sandboxes = await asyncio.wait_for(
                Sandbox.list(), timeout=HEALTH_CHECK_TIMEOUT_SEC
            )
        except Exception:
            return
        for sb in sandboxes:
            try:
                handle = await asyncio.wait_for(
                    Sandbox.get(sb.name), timeout=HEALTH_CHECK_TIMEOUT_SEC
                )
                if handle.status == "running":
                    await handle.request_drain()
                    await asyncio.wait_for(
                        handle.wait_until_stopped(),
                        timeout=SANDBOX_STOP_TIMEOUT_SEC,
                    )
                await asyncio.wait_for(
                    Sandbox.remove(sb.name), timeout=SANDBOX_STOP_TIMEOUT_SEC
                )
            except Exception:
                try:
                    handle = await asyncio.wait_for(
                        Sandbox.get(sb.name), timeout=HEALTH_CHECK_TIMEOUT_SEC
                    )
                    await handle.kill()
                    await asyncio.wait_for(
                        handle.wait_until_stopped(),
                        timeout=SANDBOX_STOP_TIMEOUT_SEC,
                    )
                    await asyncio.wait_for(
                        Sandbox.remove(sb.name), timeout=SANDBOX_STOP_TIMEOUT_SEC
                    )
                except Exception:
                    pass

    async def _create_sandbox(self) -> Sandbox:
        """Create the microsandbox VM, retrying once on timeout.

        ``Sandbox.create`` intermittently hangs — the msb process starts
        and enters the VM, but the SDK never receives the ready signal.
        When this happens, the timeout fires, we clean up the half-created
        sandbox, and retry.
        """
        for attempt in range(SANDBOX_CREATE_RETRY_ATTEMPTS):
            try:
                sb = await asyncio.wait_for(
                    Sandbox.create(
                        self._sandbox_id,
                        image=self._image,
                        cpus=self._cpus,
                        memory=self._memory,
                        workdir=SANDBOX_WORKDIR,
                        env=self._env,
                        volumes={
                            SANDBOX_WORKDIR: Volume.bind(self._root_dir),
                        },
                        secrets=self._secrets,
                        replace=True,
                    ),
                    timeout=SANDBOX_CREATE_TIMEOUT_SEC,
                )
                return sb
            except asyncio.TimeoutError:
                if attempt == 0:
                    logger.warning(
                        "Sandbox.create timed out after %ds, cleaning up and retrying.",
                        SANDBOX_CREATE_TIMEOUT_SEC,
                    )
                    await self._cleanup_stale_sandboxes()
                    continue
                raise
        raise RuntimeError("unreachable")

    async def _ensure_sandbox(self) -> Sandbox:
        """Lazily create and boot the microsandbox VM."""
        if self._sandbox is not None:
            if await self._is_sandbox_alive():
                return self._sandbox
            self._sandbox = None

        async with self._lock:
            if self._sandbox is not None:
                return self._sandbox

            logger.info(
                "Creating microsandbox VM (id=%s, image=%s, cpus=%d, memory=%dMiB, root_dir=%s)",
                self._sandbox_id,
                self._image,
                self._cpus,
                self._memory,
                self._root_dir,
            )

            sb = await self._create_sandbox()
            self._sandbox = sb
            logger.info("microsandbox VM ready: %s", self._sandbox_id)

            # For Python images, ensure a guest-compatible venv exists.
            # The host .venv may be built for a different OS/arch (e.g.
            # macOS arm64) and won't run inside the Linux VM.
            if self._is_python_image():
                await self._ensure_python_venv(sb)

            return sb

    # ------------------------------------------------------------------ #
    # Python venv management
    # ------------------------------------------------------------------ #

    def _is_python_image(self) -> bool:
        """Whether the image is Python-capable (triggers venv setup)."""
        img = self._image.lower()
        return any(marker in img for marker in _PYTHON_IMAGE_MARKERS)

    async def _ensure_python_venv(self, sb: Sandbox) -> None:
        """Ensure a guest-compatible Python venv exists and activate it.

        Never modifies the user's ``.venv``.  If the host venv is
        arch-compatible, reuse it.  Otherwise build ``.venv-msb``:

        - uv.lock present → install uv, create venv with ``uv venv``,
          sync with ``uv sync``.
        - otherwise → create venv with ``python3 -m venv``, install deps
          with ``pip``.

        A marker file records the VM arch + dep hash so we skip rebuilds
        when nothing changed.
        """
        workdir = SANDBOX_WORKDIR
        host_venv_bin = f"{workdir}/.venv/bin"

        # Capture the VM's real PATH (env dict values aren't shell-expanded).
        path_res = await self._run_in_vm(sb, "echo $PATH", timeout=10, env=None)
        vm_path = path_res.output.strip() if path_res.exit_code == 0 else ""

        # 1. Try the user's .venv — empirical arch check (runs inside VM).
        if await self._path_exists(sb, f"{workdir}/.venv"):
            check = await self._run_in_vm(
                sb, f"{host_venv_bin}/python --version", timeout=30, env=None
            )
            if check.exit_code == 0:
                logger.info("Reusing host .venv (arch-compatible) at %s", host_venv_bin)
                self._venv_bin = host_venv_bin
                self._venv_env = self._build_venv_env(host_venv_bin, vm_path)
                return
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
        arch_res = await self._run_in_vm(sb, "uname -m", timeout=10, env=None)
        vm_arch = arch_res.output.strip() if arch_res.exit_code == 0 else "unknown"

        # Compute a hash of the dependency manifest so we rebuild on changes.
        dep_hash = await self._dep_manifest_hash(sb, workdir)

        # Check the marker: skip rebuild if arch + dep_hash match.
        if await self._path_exists(sb, marker):
            marker_content = await self._read_file_in_vm(sb, marker)
            if marker_content and self._marker_matches(
                marker_content, vm_arch, dep_hash
            ):
                logger.info(
                    ".venv-msb up to date (arch=%s, hash=%s); reusing",
                    vm_arch,
                    dep_hash,
                )
                self._venv_bin = ms_venv_bin
                self._venv_env = self._build_venv_env(ms_venv_bin, vm_path)
                return
            logger.info(".venv-msb stale (arch/deps changed); rebuilding")

        # Remove stale venv.
        await self._run_in_vm(
            sb, f"rm -rf {shlex.quote(ms_venv)}", timeout=30, env=None
        )

        # Detect which dependency manifest exists.
        manifest = await self._detect_manifest(sb, workdir)

        if manifest is None:
            logger.warning(
                "No pyproject.toml, uv.lock, or requirements.txt found in %s; "
                "skipping dependency install (empty venv).",
                workdir,
            )
            return

        if manifest == "uv.lock":
            # Install uv into system Python, then use it for everything.
            uv_install_res = await self._run_in_vm(
                sb,
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
                return

            # Create venv with uv (respects .python-version).
            venv_res = await self._run_in_vm(
                sb,
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
                return

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
                sb,
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
                return
            logger.info("Dependencies installed into .venv-msb via uv sync")

            # Install uv into the venv so skills can find it via PATH.
            # uv venv creates venvs without pip, so use uv pip install
            # (uv's own installer) instead of python -m pip.
            uv_venv_res = await self._run_in_vm(
                sb,
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

        else:
            # No uv.lock — create venv with python3 -m venv, install with pip.
            venv_res = await self._run_in_vm(
                sb,
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
                return
            else:
                install_cmd = self._build_pip_install_cmd(workdir, ms_venv, manifest)
                res = await self._run_in_vm(
                    sb,
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
                    return
                logger.info("Dependencies installed into .venv-msb via pip")

        # Write the marker.
        marker_body = f"arch={vm_arch}\ndeps={dep_hash}\n"
        await self._write_file_in_vm(sb, marker, marker_body)

        self._venv_bin = ms_venv_bin
        self._venv_env = self._build_venv_env(ms_venv_bin, vm_path)

    def _build_venv_env(self, venv_bin: str, base_path: str) -> dict[str, str]:
        """Build the env dict that activates a venv for ``sb.shell(env=...)``.

        ``base_path`` is the VM's real ``$PATH`` (captured during venv
        setup), since env dict values are not shell-expanded.
        """
        venv_dir = str(Path(venv_bin).parent)
        return {
            "VIRTUAL_ENV": venv_dir,
            "PATH": f"{venv_bin}:{base_path}",
            "UV_PROJECT_ENVIRONMENT": venv_dir,
            "UV_NO_SYNC": "1",
        }

    async def _detect_manifest(self, sb: Sandbox, workdir: str) -> str | None:
        """Return the name of the dependency manifest that exists, or None.

        Preference order: ``uv.lock`` → ``pyproject.toml`` →
        ``requirements.txt``.  Only the first existing one is returned.
        """
        for manifest in ("uv.lock", "pyproject.toml", "requirements.txt"):
            if await self._path_exists(sb, f"{workdir}/{manifest}"):
                return manifest
        return None

    def _build_pip_install_cmd(self, workdir: str, venv: str, manifest: str) -> str:
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

    async def _dep_manifest_hash(self, sb: Sandbox, workdir: str) -> str:
        """Hash of the project's dependency manifest (uv.lock or pyproject.toml)."""
        # Hash uv.lock if present (most representative), else pyproject.toml.
        for manifest in ("uv.lock", "pyproject.toml", "requirements.txt"):
            path = f"{workdir}/{manifest}"
            if await self._path_exists(sb, path):
                res = await self._run_in_vm(
                    sb,
                    f"sha256sum {shlex.quote(path)} 2>/dev/null || true",
                    timeout=15,
                    env=None,
                )
                h = res.output.strip().split()[0] if res.output else ""
                if h:
                    return f"{manifest}:{h[:16]}"
        return "none"

    @staticmethod
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

    # -- low-level VM helpers (thin wrappers over sb.shell) ------------ #

    async def _run_in_vm(
        self,
        sb: Sandbox,
        command: str,
        *,
        timeout: int,
        env: dict[str, str] | None,
    ) -> ExecuteResponse:
        """Run a command in the VM and return an ExecuteResponse.

        ``env`` controls the environment passed to ``sb.shell``:
        - ``"default"`` (default): use ``self._venv_env`` if set, else None.
        - ``None``: pass None explicitly (inherit VM's default env).
        - a dict: use that dict directly.
        """
        if env == "default":
            env = self._venv_env
        try:
            result = await asyncio.wait_for(
                sb.shell(
                    command,
                    cwd=SANDBOX_WORKDIR,
                    env=env,
                    timeout=float(timeout),
                ),
                timeout=timeout + 10,
            )
        except asyncio.TimeoutError:
            return ExecuteResponse(
                output=f"Error: timed out after {timeout}s",
                exit_code=124,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error: {type(e).__name__}: {e}", exit_code=1, truncated=False
            )

        output_parts = []
        if result.stdout_text:
            output_parts.append(result.stdout_text)
        if result.stderr_text:
            for line in result.stderr_text.strip().split("\n"):
                output_parts.append(f"[stderr] {line}")
        output = "\n".join(output_parts) if output_parts else ""
        return ExecuteResponse(
            output=output, exit_code=result.exit_code, truncated=False
        )

    async def _path_exists(self, sb: Sandbox, guest_path: str) -> bool:
        """Check if a path exists inside the VM via the filesystem API."""
        try:
            return await asyncio.wait_for(
                sb.fs.exists(guest_path), timeout=FS_OP_TIMEOUT_SEC
            )
        except Exception:
            return False

    async def _read_file_in_vm(self, sb: Sandbox, guest_path: str) -> str | None:
        """Read a small text file from the VM."""
        try:
            raw = await asyncio.wait_for(
                sb.fs.read(guest_path), timeout=FS_OP_TIMEOUT_SEC
            )
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return None

    async def _write_file_in_vm(
        self, sb: Sandbox, guest_path: str, content: str
    ) -> None:
        """Write a small text file into the VM, creating parent dirs."""
        parent = str(Path(guest_path).parent)
        if parent and parent != "/":
            try:
                await asyncio.wait_for(sb.fs.mkdir(parent), timeout=FS_OP_TIMEOUT_SEC)
            except Exception:
                pass
        try:
            await asyncio.wait_for(
                sb.fs.write(guest_path, content.encode("utf-8")),
                timeout=FS_OP_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning("Failed to write %s in VM: %s", guest_path, e)

    def _ensure_sandbox_sync(self) -> Sandbox:
        """Sync wrapper for :meth:`_ensure_sandbox`."""
        return _run_async(self._ensure_sandbox())

    async def _is_sandbox_alive(self) -> bool:
        """Check if the current sandbox is reachable via agentd."""
        if self._sandbox is None:
            return False
        try:
            handle = await asyncio.wait_for(
                Sandbox.get(self._sandbox_id),
                timeout=HEALTH_CHECK_TIMEOUT_SEC,
            )
            await asyncio.wait_for(
                handle.connect(timeout=HEALTH_CHECK_TIMEOUT_SEC),
                timeout=HEALTH_CHECK_TIMEOUT_SEC,
            )
            return True
        except Exception:
            return False

    async def stop(self) -> None:
        """Stop and clean up the sandbox VM."""
        if self._sandbox is not None:
            self._sandbox = None
            try:
                handle = await asyncio.wait_for(
                    Sandbox.get(self._sandbox_id),
                    timeout=HEALTH_CHECK_TIMEOUT_SEC,
                )
                await handle.request_drain()
                await asyncio.wait_for(
                    handle.wait_until_stopped(),
                    timeout=SANDBOX_STOP_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Sandbox %s stop timed out after %ds, killing.",
                    self._sandbox_id,
                    SANDBOX_STOP_TIMEOUT_SEC,
                )
                try:
                    handle = await asyncio.wait_for(
                        Sandbox.get(self._sandbox_id),
                        timeout=HEALTH_CHECK_TIMEOUT_SEC,
                    )
                    await handle.kill()
                    await asyncio.wait_for(
                        handle.wait_until_stopped(),
                        timeout=SANDBOX_STOP_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.warning("Sandbox %s stop failed: %s", self._sandbox_id, e)
            try:
                await asyncio.wait_for(
                    Sandbox.remove(self._sandbox_id),
                    timeout=SANDBOX_STOP_TIMEOUT_SEC,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def id(self) -> str:
        """Unique identifier for this backend instance."""
        return self._sandbox_id

    @property
    def cwd(self) -> str:
        """Working directory inside the sandbox."""
        return SANDBOX_WORKDIR

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #

    def _is_sandbox_dead_output(self, output: str) -> bool:
        """Detect microsandbox socket/agent death from command output."""
        msg = output.lower()
        return any(
            phrase in msg
            for phrase in [
                "no agent socket",
                "exec session ended without exit event",
                "sandbox is not running",
                "connection refused",
            ]
        )

    def _invalidate_sandbox_if_dead(self, output: str) -> None:
        """Invalidate the sandbox reference if the socket appears dead."""
        if self._is_sandbox_dead_output(output):
            logger.warning(
                "Sandbox %s socket died (detected from output). Will recreate on next call.",
                self._sandbox_id,
            )
            self._sandbox = None

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the microsandbox VM.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds.  Uses default if ``None``.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        sb = await self._ensure_sandbox()
        effective_timeout = (
            timeout if timeout is not None else DEFAULT_COMMAND_TIMEOUT_SEC
        )
        if effective_timeout <= 0:
            effective_timeout = DEFAULT_COMMAND_TIMEOUT_SEC

        sb_command = self._to_guest_path(command)
        logger.debug(f"Executing command: {sb_command}")

        try:
            result = await asyncio.wait_for(
                sb.shell(
                    sb_command,
                    cwd=SANDBOX_WORKDIR,
                    env=self._venv_env,
                    timeout=float(effective_timeout),
                ),
                timeout=effective_timeout + 10,
            )
        except asyncio.TimeoutError:
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout}s.",
                exit_code=124,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

        # Combine stdout and stderr, prefixing stderr lines with [stderr].
        output_parts = []
        if result.stdout_text:
            output_parts.append(result.stdout_text)
        if result.stderr_text:
            stderr_lines = result.stderr_text.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

        output = "\n".join(output_parts) if output_parts else "<no output>"

        # Truncation
        truncated = False
        if len(output) > _MAX_OUTPUT_BYTES:
            output = output[:_MAX_OUTPUT_BYTES]
            output += f"\n\n... Output truncated at {_MAX_OUTPUT_BYTES} bytes."
            truncated = True

        # Add exit code info if non-zero
        if result.exit_code != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.exit_code}"

        self._invalidate_sandbox_if_dead(output)

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=truncated,
        )

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command (sync wrapper)."""
        return _run_async(self.aexecute(command, timeout=timeout))

    # ------------------------------------------------------------------ #
    # File read
    # ------------------------------------------------------------------ #

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content with line-based pagination.

        Uses the microsandbox filesystem API to read the file, then applies
        offset/limit pagination locally for text files.  Binary files are
        returned base64-encoded.
        """
        sb = await self._ensure_sandbox()

        # Resolve to an absolute guest path for the filesystem API
        guest_path = self._resolve_guest_path(file_path)

        try:
            raw = await asyncio.wait_for(
                sb.fs.read(guest_path), timeout=FS_OP_TIMEOUT_SEC
            )
        except Exception as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")

        # Try to decode as UTF-8 text
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Binary file — return base64-encoded
            encoded = base64.standard_b64encode(raw).decode("ascii")
            return ReadResult(file_data=FileData(content=encoded, encoding="base64"))

        # Apply pagination
        if offset < 0:
            return ReadResult(
                error=f"Negative offset {offset} is not supported (must be >= 0)"
            )

        lines = content.splitlines(keepends=True)
        start_idx = offset
        end_idx = min(start_idx + limit, len(lines))

        if start_idx > len(lines):
            return ReadResult(
                error=f"Line offset {offset} exceeds file length ({len(lines)} lines)"
            )

        page = "".join(lines[start_idx:end_idx])
        return ReadResult(file_data=FileData(content=page, encoding="utf-8"))

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content (sync wrapper)."""
        return _run_async(self.aread(file_path, offset=offset, limit=limit))

    # ------------------------------------------------------------------ #
    # File write
    # ------------------------------------------------------------------ #

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file, failing if it already exists."""
        sb = await self._ensure_sandbox()
        guest_path = self._resolve_guest_path(file_path)

        # Check if file already exists
        try:
            exists = await asyncio.wait_for(
                sb.fs.exists(guest_path), timeout=FS_OP_TIMEOUT_SEC
            )
        except Exception:
            exists = False

        if exists:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )

        # Ensure parent directory exists
        parent = str(Path(guest_path).parent)
        if parent and parent != "/":
            try:
                await asyncio.wait_for(sb.fs.mkdir(parent), timeout=FS_OP_TIMEOUT_SEC)
            except Exception:
                pass  # May already exist

        try:
            await asyncio.wait_for(
                sb.fs.write(guest_path, content.encode("utf-8")),
                timeout=FS_OP_TIMEOUT_SEC,
            )
        except Exception as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

        return WriteResult(path=file_path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file (sync wrapper)."""
        return _run_async(self.awrite(file_path, content))

    # ------------------------------------------------------------------ #
    # File edit
    # ------------------------------------------------------------------ #

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing exact string occurrences.

        Reads the file via the microsandbox filesystem API, performs the
        replacement locally, and writes the result back.
        """
        sb = await self._ensure_sandbox()
        guest_path = self._resolve_guest_path(file_path)

        try:
            raw = await asyncio.wait_for(
                sb.fs.read(guest_path), timeout=FS_OP_TIMEOUT_SEC
            )
        except Exception as e:
            return EditResult(error=f"Error: File '{file_path}' not found: {e}")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return EditResult(error=f"Error: File '{file_path}' is not a text file")

        # Normalize line endings
        old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
        new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")

        count = text.count(old_string)
        if count == 0:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if count > 1 and not replace_all:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. "
                "Use replace_all=True to replace all occurrences."
            )

        result_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )

        try:
            await asyncio.wait_for(
                sb.fs.write(guest_path, result_text.encode("utf-8")),
                timeout=FS_OP_TIMEOUT_SEC,
            )
        except Exception as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

        return EditResult(path=file_path, occurrences=count if replace_all else 1)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file (sync wrapper)."""
        return _run_async(
            self.aedit(file_path, old_string, new_string, replace_all=replace_all)
        )

    # ------------------------------------------------------------------ #
    # Directory listing
    # ------------------------------------------------------------------ #

    async def als(self, path: str) -> LsResult:
        """List directory contents with metadata."""
        sb = await self._ensure_sandbox()
        guest_path = self._resolve_guest_path(path)

        try:
            entries = await asyncio.wait_for(
                sb.fs.list(guest_path), timeout=FS_OP_TIMEOUT_SEC
            )
        except Exception as e:
            return LsResult(error=f"Path '{path}': {e}", entries=None)

        file_infos: list[FileInfo] = []
        for entry in entries:
            file_infos.append(
                {
                    "path": self._to_host_path(entry.path),
                    "is_dir": entry.kind == "directory",
                }
            )

        return LsResult(entries=file_infos)

    def ls(self, path: str) -> LsResult:
        """List directory contents (sync wrapper)."""
        return _run_async(self.als(path))

    # ------------------------------------------------------------------ #
    # Grep
    # ------------------------------------------------------------------ #

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal string in files using ``grep -F``."""
        search_path = self._to_guest_path(path) if path else "."

        grep_opts = "-rHnFZ"
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include={shlex.quote(glob)}"

        pattern_escaped = shlex.quote(pattern)
        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {shlex.quote(search_path)} 2>/dev/null || true"

        result = await self.aexecute(cmd)

        output = result.output.rstrip("\n")
        if not output:
            return GrepResult(matches=[])

        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            parts = line.split("\0", 1)
            if len(parts) != _PIPE_FIELD_COUNT:
                continue
            line_parts = parts[1].split(":", 1)
            if len(line_parts) != _PIPE_FIELD_COUNT:
                continue
            try:
                matches.append(
                    {
                        "path": self._to_host_path(parts[0]),
                        "line": int(line_parts[0]),
                        "text": line_parts[1],
                    }
                )
            except ValueError:
                continue

        return GrepResult(matches=matches)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal string (sync wrapper)."""
        return _run_async(self.agrep(pattern, path=path, glob=glob))

    # ------------------------------------------------------------------ #
    # Glob
    # ------------------------------------------------------------------ #

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern using ``find``."""
        resolved_path = self._to_guest_path(path) if path else "."

        # Convert glob pattern to find-compatible pattern
        find_pattern = pattern
        if find_pattern.startswith("**/"):
            find_pattern = find_pattern[3:]
            find_cmd = f"find {shlex.quote(resolved_path)} -name {shlex.quote(find_pattern)} -type f 2>/dev/null"
        elif "/" in find_pattern:
            find_cmd = f"find {shlex.quote(resolved_path)} -path {shlex.quote(find_pattern)} -type f 2>/dev/null"
        else:
            find_cmd = f"find {shlex.quote(resolved_path)} -name {shlex.quote(find_pattern)} -type f 2>/dev/null"

        result = await self.aexecute(find_cmd)

        output = result.output.strip()
        if not output:
            return GlobResult(matches=[])

        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            if not line:
                continue
            file_infos.append(
                {
                    "path": self._to_host_path(line),
                    "is_dir": False,
                }
            )

        return GlobResult(matches=file_infos)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files matching a glob pattern (sync wrapper)."""
        return _run_async(self.aglob(pattern, path=path))

    # ------------------------------------------------------------------ #
    # File upload / download
    # ------------------------------------------------------------------ #

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox via the microsandbox filesystem API."""
        sb = await self._ensure_sandbox()
        results: list[FileUploadResponse] = []

        for path, content in files:
            guest_path = self._resolve_guest_path(path)

            # Ensure parent directory exists
            parent = str(Path(guest_path).parent)
            if parent and parent != "/":
                try:
                    await asyncio.wait_for(
                        sb.fs.mkdir(parent), timeout=FS_OP_TIMEOUT_SEC
                    )
                except Exception:
                    pass

            try:
                await asyncio.wait_for(
                    sb.fs.write(guest_path, content), timeout=FS_OP_TIMEOUT_SEC
                )
                results.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                error = _map_exception_to_standard_error(e)
                if error is None:
                    raise
                results.append(FileUploadResponse(path=path, error=error))

        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files (sync wrapper)."""
        return _run_async(self.aupload_files(files))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox."""
        sb = await self._ensure_sandbox()
        results: list[FileDownloadResponse] = []

        for path in paths:
            guest_path = self._resolve_guest_path(path)
            try:
                content = await asyncio.wait_for(
                    sb.fs.read(guest_path), timeout=FS_OP_TIMEOUT_SEC
                )
                results.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except Exception as e:
                error = _map_exception_to_standard_error(e)
                if error is None:
                    # Map microsandbox FilesystemError with "not found" messages
                    msg = str(e).lower()
                    if "no such file" in msg or "not found" in msg:
                        error = FILE_NOT_FOUND
                    elif "is a directory" in msg:
                        error = IS_DIRECTORY
                    else:
                        raise
                results.append(
                    FileDownloadResponse(path=path, content=None, error=error)
                )

        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files (sync wrapper)."""
        return _run_async(self.adownload_files(paths))

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def _to_guest_path(self, input: str) -> str:
        """Replace root_dir with /workspace."""
        return input.replace(self._root_dir, SANDBOX_WORKDIR)

    def _resolve_guest_path(self, input: str) -> str:
        """Convert a path to an absolute guest path for the filesystem API.

        Handles three input forms:
        - Host absolute path (contains ``root_dir``) → replaced with ``/workspace``.
        - Already a guest path (starts with ``/workspace``) → passed through.
        - Relative or slash-prefixed project path (e.g. ``metalgate_code/...``
          or ``/metalgate_code/...`` as produced by built-in tools) →
          resolved under ``/workspace``.
        """
        result = self._to_guest_path(input)
        if not result.startswith("/"):
            result = str(Path(SANDBOX_WORKDIR) / result)
        elif not result.startswith(SANDBOX_WORKDIR):
            # Absolute path that isn't under /workspace and wasn't a host
            # path — likely a project-relative path with a leading slash
            # (e.g. from built-in read_file).  Re-anchor under /workspace.
            result = str(Path(SANDBOX_WORKDIR) / result.lstrip("/"))
        return result

    def _to_host_path(self, input: str) -> str:
        """Replace /workspace prefix with root_dir."""
        p = Path(input)
        workspace = Path(SANDBOX_WORKDIR)
        if p.is_relative_to(workspace):
            p = Path(self._root_dir) / p.relative_to(workspace)
        return str(p)
