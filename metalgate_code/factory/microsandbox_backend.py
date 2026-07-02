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
from microsandbox import Image, ImageNotFoundError, Sandbox, Volume

from metalgate_code.factory.venv_manager import is_python_image as _is_python_image

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

The image is pre-cached at init (see :meth:`_precache_image`) so this
timeout only covers VM boot, not image pulling.
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
        cpus: int = 4,
        memory: int = 4096,
        secrets: list | None = None,
        eager: bool = False,
    ) -> None:
        """Initialize the microsandbox backend.

        Args:
            root_dir: Host directory to bind-mount into the sandbox at
                ``/workspace``.  This becomes the working directory for all
                commands.  Must not be the root volume ``/``.
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
        if self._root_dir == "/":
            raise ValueError(
                "root_dir must not be '/': refusing to bind-mount the entire "
                "root volume into the sandbox."
            )
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
        else:
            _run_async(self._precache_image())

    # ------------------------------------------------------------------ #
    # Sandbox lifecycle
    # ------------------------------------------------------------------ #

    async def _precache_image(self) -> None:
        """Ensure the OCI image is cached locally before any VM boot.

        ``Sandbox.create`` pulls the image on demand, but a pull during
        creation competes with the short ``SANDBOX_CREATE_TIMEOUT_SEC``
        guard and causes spurious timeouts.  By pre-caching here we keep
        ``Sandbox.create`` fast (boot only, no network).
        """
        try:
            await asyncio.wait_for(
                Image.get(self._image), timeout=HEALTH_CHECK_TIMEOUT_SEC
            )
        except ImageNotFoundError:
            logger.info("Pre-caching image %s (not in local cache)", self._image)
            await self._pull_image()
        except Exception:
            pass  # Best-effort; let _create_sandbox surface real errors.

    async def _pull_image(self) -> None:
        """Pull the image via the ``msb image pull`` CLI command."""
        proc = await asyncio.create_subprocess_exec(
            "msb",
            "image",
            "pull",
            self._image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def _kill_sandbox(self, name: str) -> None:
        """Force-kill a sandbox that did not stop gracefully."""
        try:
            handle = await asyncio.wait_for(
                Sandbox.get(name), timeout=HEALTH_CHECK_TIMEOUT_SEC
            )
            await handle.kill()
            await asyncio.wait_for(
                handle.wait_until_stopped(),
                timeout=SANDBOX_STOP_TIMEOUT_SEC,
            )
        except Exception:
            pass

    async def _destroy_sandbox(self, name: str) -> None:
        """Drain, stop (kill on failure), and remove a sandbox by name.

        Encapsulates the full teardown sequence used by both :meth:`stop`
        (for this instance's sandbox) and :meth:`_cleanup_stale_sandboxes`
        (for leftover sandboxes from previous runs).
        """
        try:
            handle = await asyncio.wait_for(
                Sandbox.get(name), timeout=HEALTH_CHECK_TIMEOUT_SEC
            )
            if handle.status == "running":
                await handle.request_drain()
                await asyncio.wait_for(
                    handle.wait_until_stopped(),
                    timeout=SANDBOX_STOP_TIMEOUT_SEC,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Sandbox %s stop timed out after %ds, killing.",
                name,
                SANDBOX_STOP_TIMEOUT_SEC,
            )
            await self._kill_sandbox(name)
        except Exception as e:
            logger.warning("Sandbox %s stop failed: %s", name, e)
            await self._kill_sandbox(name)

        try:
            await asyncio.wait_for(
                Sandbox.remove(name), timeout=SANDBOX_STOP_TIMEOUT_SEC
            )
        except Exception:
            pass

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
            await self._destroy_sandbox(sb.name)

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
            if _is_python_image(self._image):
                from metalgate_code.factory.venv_manager import VenvManager

                venv = VenvManager(
                    sb,
                    run_in_vm=self._run_in_vm,
                    path_exists=self._path_exists,
                    read_file_in_vm=self._read_file_in_vm,
                    write_file_in_vm=self._write_file_in_vm,
                )
                self._venv_bin, self._venv_env = await venv.ensure()

            return sb

    # -- low-level VM helpers (thin wrappers over sb.shell) ------------ #

    @staticmethod
    def _format_output(result) -> str:
        """Combine stdout and stderr, prefixing stderr lines with ``[stderr]``."""
        parts = []
        if result.stdout_text:
            parts.append(result.stdout_text)
        if result.stderr_text:
            for line in result.stderr_text.strip().split("\n"):
                parts.append(f"[stderr] {line}")
        return "\n".join(parts) if parts else ""

    async def _run_in_vm(
        self,
        sb: Sandbox,
        command: str,
        *,
        timeout: int,
        env: dict[str, str] | None = None,
    ) -> ExecuteResponse:
        """Run a command in the VM and return an ExecuteResponse.

        ``env`` controls the environment passed to ``sb.shell``:
        - ``"default"`` (default): use ``self._venv_env`` if set, else None.
        - ``None``: pass None explicitly (inherit VM's default env).
        - a dict: use that dict directly.
        """
        if env is None:
            env = self._venv_env
        try:
            result = await sb.shell(
                command,
                cwd=SANDBOX_WORKDIR,
                env=env,
                timeout=float(timeout),
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error: {type(e).__name__}: {e}", exit_code=1, truncated=False
            )

        return ExecuteResponse(
            output=self._format_output(result),
            exit_code=result.exit_code,
            truncated=False,
        )

    async def _fs(self, sb: Sandbox, coro, *, timeout: int = FS_OP_TIMEOUT_SEC):
        """Run a filesystem operation on the VM with the standard FS timeout.

        Centralizes the ``asyncio.wait_for`` wrapper so call sites don't
        repeat ``timeout=FS_OP_TIMEOUT_SEC`` at every ``sb.fs.*`` call.
        Callers handle their own exceptions (returning ``False``, ``None``,
        or a typed ``Result(error=...)`` as appropriate).
        """
        return await asyncio.wait_for(coro, timeout=timeout)

    async def _path_exists(self, sb: Sandbox, guest_path: str) -> bool:
        """Check if a path exists inside the VM via the filesystem API."""
        try:
            return await self._fs(sb, sb.fs.exists(guest_path))
        except Exception:
            return False

    async def _read_file_in_vm(self, sb: Sandbox, guest_path: str) -> str | None:
        """Read a small text file from the VM."""
        try:
            raw = await self._fs(sb, sb.fs.read(guest_path))
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
                await self._fs(sb, sb.fs.mkdir(parent))
            except Exception:
                pass
        try:
            await self._fs(sb, sb.fs.write(guest_path, content.encode("utf-8")))
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
        if self._sandbox is None:
            return
        self._sandbox = None
        await self._destroy_sandbox(self._sandbox_id)

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

    @property
    def venv_bin(self) -> str | None:
        """Path to the guest-compatible venv's ``bin`` directory, or ``None``.

        Established by :class:`~metalgate_code.factory.venv_manager.VenvManager`
        during sandbox boot when the image is Python-capable.  When set,
        commands run with the venv activated (see :attr:`venv_env`).
        """
        return self._venv_bin

    @property
    def venv_dir(self) -> str | None:
        """Name of the guest-compatible venv's directory (e.g. ``.venv`` or
        ``.venv-msb``), or ``None`` if no venv was established.

        Derived from :attr:`venv_bin`` — the parent directory's basename.
        """
        if self._venv_bin is None:
            return None
        return Path(self._venv_bin).parent.name

    @property
    def venv_env(self) -> dict[str, str] | None:
        """Env dict that activates :attr:`venv_bin`, or ``None`` if no venv."""
        return self._venv_env

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

        result = await self._run_in_vm(
            sb,
            sb_command,
            timeout=effective_timeout,
        )

        output = result.output or "<no output>"

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
            raw = await self._fs(sb, sb.fs.read(guest_path))
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
            exists = await self._fs(sb, sb.fs.exists(guest_path))
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
                await self._fs(sb, sb.fs.mkdir(parent))
            except Exception:
                pass  # May already exist

        try:
            await self._fs(sb, sb.fs.write(guest_path, content.encode("utf-8")))
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
            raw = await self._fs(sb, sb.fs.read(guest_path))
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
            await self._fs(sb, sb.fs.write(guest_path, result_text.encode("utf-8")))
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
            entries = await self._fs(sb, sb.fs.list(guest_path))
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
                    await self._fs(sb, sb.fs.mkdir(parent))
                except Exception:
                    pass

            try:
                await self._fs(sb, sb.fs.write(guest_path, content))
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
                content = await self._fs(sb, sb.fs.read(guest_path))
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
