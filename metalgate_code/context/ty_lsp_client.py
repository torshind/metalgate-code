"""LSP client for ``ty server``.

Manages a persistent ``ty server`` subprocess, communicating using the
Language Server Protocol (JSON-RPC over stdio with ``Content-Length``
framing).

The transport layer (framing, reader loop, request/response correlation)
is provided by :class:`~metalgate_code.context.lsp_base.LspBaseClient`.
:class:`TyLspClient` is the abstract base: it holds all ty-specific logic
(initialization options, site-packages discovery, ty installation,
pyproject.toml creation, ``did_open``).  The transport-specific server
launch/teardown and shell/filesystem helpers are implemented by the two
concrete subclasses:

- :class:`SandboxTyLspClient` — runs ``ty server`` inside a microsandbox VM.
- :class:`LocalTyLspClient` — runs ``ty server`` as a local subprocess.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from abc import abstractmethod
from typing import Any, Optional

from metalgate_code.context.lsp_base import LspBaseClient

logger = logging.getLogger("metalgate_code")

_TY_INSTALL_TIMEOUT_SEC = 120
"""Timeout for installing ty if not present."""


class TyLspClient(LspBaseClient):
    """Abstract base for persistent LSP clients for ``ty server``.

    The server process is started lazily on first request and kept alive
    for the lifetime of the client.  All LSP requests are serialised
    through an :class:`asyncio.Lock` so that framing stays consistent.

    Subclasses implement the transport-specific server launch/teardown
    (:meth:`_start_server` / :meth:`_stop_server` / :meth:`_on_stream_closed`)
    and the shell/filesystem helpers (:meth:`_shell`, :meth:`_fs_exists`,
    :meth:`_fs_write`, :meth:`_resolve_ty_command`) used by the shared
    ty setup logic.
    """

    def __init__(
        self,
        root_uri: str,
        *,
        python_path: Optional[str] = None,
        venv_bin: Optional[str] = None,
        venv_env: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(root_uri)
        self._root_path = root_uri.replace("file://", "")
        self._python_path = python_path
        # Guest-compatible venv (sandbox).  When set, all shell commands
        # run with venv_env activated, and ty is launched from venv_bin
        # so it uses the same venv as the project.
        self._venv_bin = venv_bin
        self._venv_env = venv_env

    # ------------------------------------------------------------------ #
    # Abstract shell / filesystem helpers (transport-specific)
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def _shell(
        self,
        command: str,
        *,
        env: Optional[dict[str, str]] = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        """Run *command* and return ``(exit_code, stdout)``."""

    @abstractmethod
    async def _fs_exists(self, path: str) -> bool:
        """Return True if *path* exists."""

    @abstractmethod
    async def _fs_write(self, path: str, content: bytes) -> None:
        """Write *content* to *path*."""

    @abstractmethod
    async def _resolve_ty_command(self) -> str:
        """Return the ty command to launch (path or name)."""

    # ------------------------------------------------------------------ #
    # Shared ty setup logic
    # ------------------------------------------------------------------ #

    async def _ensure_ty_installed(self) -> None:
        """Install ty if not present and ensure a pyproject.toml exists.

        ``ty`` needs a ``pyproject.toml`` (or ``ty.toml``) at the project
        root to discover first-party modules.  If none exists, we create
        a minimal one so that relative imports resolve correctly.

        When a venv is established (``venv_bin``), ty is installed into
        that venv so the ``ty`` binary lands in ``venv_bin`` and uses the
        venv's Python.  Otherwise it falls back to system pip.
        """
        # Check if ty is already available (in the venv or on PATH).
        ty_check_cmd = (
            f"{self._venv_bin}/ty --version 2>/dev/null"
            if self._venv_bin
            else "which ty 2>/dev/null"
        )
        exit_code, stdout = await self._shell(ty_check_cmd, env=self._venv_env)
        if exit_code != 0 or not stdout.strip():
            logger.info("Installing ty…")
            if self._venv_bin:
                # Install into the venv so the ty binary lands in venv_bin.
                # uv-created venvs lack pip, so try pip first then uv pip.
                py = f"{self._venv_bin}/python"
                exit_code, stdout = await self._shell(
                    f"{py} -m pip install ty -q 2>&1",
                    env=self._venv_env,
                    timeout=_TY_INSTALL_TIMEOUT_SEC,
                )
                if exit_code != 0:
                    logger.info("pip not available in venv, trying uv pip install…")
                    exit_code, stdout = await self._shell(
                        f"uv pip install ty --python {py} -q 2>&1",
                        env=self._venv_env,
                        timeout=_TY_INSTALL_TIMEOUT_SEC,
                    )
            else:
                exit_code, stdout = await self._shell(
                    "pip install ty -q 2>&1", timeout=_TY_INSTALL_TIMEOUT_SEC
                )
            if exit_code != 0:
                raise RuntimeError(f"Failed to install ty: {stdout}")

        # Ensure pyproject.toml exists for module discovery.
        # Site-packages paths are passed via LSP initializationOptions
        # (see _start), not via ty.toml — avoids writing user files.
        root_path = self._root_path
        pyproject = f"{root_path}/pyproject.toml"
        try:
            exists = await asyncio.wait_for(self._fs_exists(pyproject), timeout=10)
        except Exception:
            exists = False

        if not exists:
            content = b'[project]\nname = "project"\nversion = "0.0.0"\n'
            try:
                await asyncio.wait_for(self._fs_write(pyproject, content), timeout=10)
            except Exception as e:
                logger.warning("Failed to create pyproject.toml for ty: %s", e)

    async def _find_site_packages(self) -> list[str]:
        """Discover site-packages directories.

        Returns paths to the venv's site-packages so ty can resolve
        third-party imports.

        Uses the venv's Python when available, avoiding ``uv run`` which
        would create a second venv.  Falls back to system Python
        discovery only when no venv was established.
        """
        paths: list[str] = []

        # Prefer the venv's Python — no uv run, no second venv.
        if self._venv_bin:
            py = f"{self._venv_bin}/python"
            try:
                exit_code, stdout = await self._shell(
                    f"{py} -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
                    env=self._venv_env,
                )
                if exit_code == 0 and stdout.strip():
                    paths = [
                        p.strip() for p in stdout.strip().splitlines() if p.strip()
                    ]
            except Exception:
                pass
            if paths:
                return paths

        # Fallback: system Python (no venv established).
        for cmd in (
            "uv run python -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
            "python -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
        ):
            try:
                exit_code, stdout = await self._shell(cmd)
                if exit_code == 0 and stdout.strip():
                    paths = [
                        p.strip() for p in stdout.strip().splitlines() if p.strip()
                    ]
                    if paths:
                        break
            except Exception:
                continue

        # Also look for a .venv site-packages under the project root
        try:
            exit_code, stdout = await self._shell(
                f"ls {self._root_path}/.venv/lib/ 2>/dev/null"
            )
            if exit_code == 0:
                for line in stdout.strip().splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    candidate = f"{self._root_path}/.venv/lib/{name}/site-packages"
                    try:
                        exists = await asyncio.wait_for(
                            self._fs_exists(candidate), timeout=5
                        )
                        if exists and candidate not in paths:
                            paths.append(candidate)
                    except Exception:
                        pass
        except Exception:
            pass

        return paths

    # ------------------------------------------------------------------ #
    # Shared LSP initialization
    # ------------------------------------------------------------------ #

    def _customize_init_params(self, params: dict[str, Any]) -> None:
        if self._python_path:
            params["initializationOptions"] = {
                "pythonPath": self._python_path,
            }

    def did_open(self, uri: str, text: str, language_id: str = "python") -> None:
        """Notify the server that a document was opened."""
        super().did_open(uri, text, language_id=language_id)


class SandboxTyLspClient(TyLspClient):
    """Persistent LSP client for ``ty server`` inside a microsandbox VM.

    The server process is started lazily on first request and kept alive
    for the lifetime of the client.  All LSP requests are serialised
    through an :class:`asyncio.Lock` so that framing stays consistent.
    """

    def __init__(
        self,
        sandbox,
        root_uri: str,
        *,
        python_path: Optional[str] = None,
        venv_bin: Optional[str] = None,
        venv_env: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            root_uri,
            python_path=python_path,
            venv_bin=venv_bin,
            venv_env=venv_env,
        )
        from microsandbox import Stdin  # deferred to keep base import-light

        self._sandbox = sandbox
        self._Stdin = Stdin
        self._handle = None
        self._stdin = None

    # ------------------------------------------------------------------ #
    # Shell / filesystem helpers (sandbox)
    # ------------------------------------------------------------------ #

    async def _shell(
        self,
        command: str,
        *,
        env: Optional[dict[str, str]] = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        kwargs: dict[str, Any] = {"env": env}
        if timeout is not None:
            kwargs["timeout"] = timeout
        result = await self._sandbox.shell(command, **kwargs)
        return result.exit_code, result.stdout_text

    async def _fs_exists(self, path: str) -> bool:
        return await self._sandbox.fs.exists(path)

    async def _fs_write(self, path: str, content: bytes) -> None:
        await self._sandbox.fs.write(path, content)

    async def _resolve_ty_command(self) -> str:
        if self._venv_bin:
            ty_bin = f"{self._venv_bin}/ty"
            try:
                exists = await asyncio.wait_for(
                    self._sandbox.fs.exists(ty_bin), timeout=10
                )
            except Exception:
                exists = False
            if exists:
                return ty_bin
        return "ty"

    # ------------------------------------------------------------------ #
    # Server lifecycle (sandbox transport)
    # ------------------------------------------------------------------ #

    async def _start_server(self) -> None:
        await self._ensure_ty_installed()

        # Discover site-packages so ty can resolve third-party imports.
        # The host .venv is bind-mounted but unusable inside the sandbox
        # (pyvenv.cfg points to a host Python home), so we pass the
        # site-packages directory via PYTHONPATH — ty reads it and adds
        # it to extra_paths.
        site_paths = await self._find_site_packages()
        env: dict[str, str] = {}
        if site_paths:
            env["PYTHONPATH"] = ":".join(site_paths)

        ty_cmd = await self._resolve_ty_command()

        self._handle = await self._sandbox.exec_stream(
            ty_cmd,
            ["server"],
            stdin=self._Stdin.pipe(),
            env=env or None,
            timeout=0,  # no timeout — long-lived process
        )
        self._stdin = self._handle.take_stdin()

    async def _stop_server(self) -> None:
        if self._stdin:
            try:
                await self._stdin.close()
            except Exception:
                pass
            self._stdin = None

        if self._handle:
            try:
                await self._handle.kill()
            except Exception:
                pass
            self._handle = None

    async def _on_stream_closed(self) -> None:
        """Close stdin and kill the handle so resources are released."""
        if self._stdin is not None:
            try:
                await self._stdin.close()
            except Exception:
                pass
            self._stdin = None
        if self._handle is not None:
            try:
                await self._handle.kill()
            except Exception:
                pass
            self._handle = None

    # ------------------------------------------------------------------ #
    # Transport-specific read/write (sandbox)
    # ------------------------------------------------------------------ #

    async def _write_raw(self, frame: bytes) -> None:
        if self._stdin is None:
            raise RuntimeError("ty server stdin is not available")
        await self._stdin.write(frame)

    async def _read_raw(self) -> Optional[bytes]:
        assert self._handle is not None
        event = await self._handle.recv()

        if event is None:
            return None

        # Only process stdout events (contain LSP responses).
        # stderr events carry ty's logging output and must be ignored.
        if getattr(event, "event_type", None) != "stdout":
            return b""  # signal "no data, keep looping"

        data = getattr(event, "data", None)
        if data is None:
            return b""
        return data


class LocalTyLspClient(TyLspClient):
    """Persistent LSP client for ``ty server`` on the host.

    The server process is started lazily on first request and kept alive
    for the lifetime of the client.  All LSP requests are serialised
    through an :class:`asyncio.Lock` so that framing stays consistent.
    """

    def __init__(
        self,
        root_uri: str,
        *,
        python_path: Optional[str] = None,
    ) -> None:
        super().__init__(root_uri, python_path=python_path)
        self._process: Optional[asyncio.subprocess.Process] = None

    # ------------------------------------------------------------------ #
    # Shell / filesystem helpers (local)
    # ------------------------------------------------------------------ #

    async def _shell(
        self,
        command: str,
        *,
        env: Optional[dict[str, str]] = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        import os

        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout if timeout and timeout > 0 else None,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 1, ""
        return proc.returncode or 0, stdout_bytes.decode("utf-8", errors="replace")

    async def _fs_exists(self, path: str) -> bool:
        import os

        return os.path.exists(path)

    async def _fs_write(self, path: str, content: bytes) -> None:
        await asyncio.to_thread(lambda: open(path, "wb").write(content))

    async def _resolve_ty_command(self) -> str:
        ty_bin = shutil.which("ty")
        if ty_bin is None:
            raise FileNotFoundError("ty not found in PATH")
        return ty_bin

    # ------------------------------------------------------------------ #
    # Server lifecycle (local transport)
    # ------------------------------------------------------------------ #

    async def _start_server(self) -> None:
        await self._ensure_ty_installed()

        # Discover site-packages so ty can resolve third-party imports.
        site_paths = await self._find_site_packages()
        env: dict[str, str] = {}
        if site_paths:
            env["PYTHONPATH"] = ":".join(site_paths)

        ty_cmd = await self._resolve_ty_command()

        import os

        full_env = dict(os.environ)
        if env:
            full_env.update(env)

        self._process = await asyncio.create_subprocess_exec(
            ty_cmd,
            "server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # discard — no pipe to fill
            env=full_env,
        )

    async def _stop_server(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
        self._process = None

    async def _on_stream_closed(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass
            self._process = None

    # ------------------------------------------------------------------ #
    # Transport-specific read/write (local)
    # ------------------------------------------------------------------ #

    async def _write_raw(self, frame: bytes) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("ty server stdin is not available")
        self._process.stdin.write(frame)
        await self._process.stdin.drain()

    async def _read_raw(self) -> Optional[bytes]:
        if self._process is None or self._process.stdout is None:
            return None
        try:
            data = await self._process.stdout.read(4096)
        except Exception:
            return None
        if not data:
            return None
        return data
