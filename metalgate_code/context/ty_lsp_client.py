"""Minimal LSP client for ``ty server`` running inside a microsandbox VM.

Manages a persistent ``ty server`` subprocess via ``Sandbox.exec_stream``,
communicating using the Language Server Protocol (JSON-RPC over stdio with
``Content-Length`` framing).

The client runs a dedicated background event loop so that the LSP reader
task and pending futures persist across sync calls from the tracer.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Optional

from microsandbox import Sandbox, Stdin

logger = logging.getLogger("metalgate_code")

_LSP_TIMEOUT_SEC = 30
"""Default timeout for individual LSP requests."""

_SERVER_BOOT_TIMEOUT_SEC = 60
"""Timeout for booting the ty server process and completing initialize."""

_TY_INSTALL_TIMEOUT_SEC = 120
"""Timeout for installing ty inside the sandbox if not present."""


class _BackgroundLoop:
    """Run a dedicated event loop in a background thread for the LSP client.

    All coroutines are submitted via :meth:`run` and executed on the
    background loop, so the reader task and futures share one loop.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._loop is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro, *, timeout: float = 300) -> Any:
        """Submit *coro* to the background loop and wait for the result.

        If the background loop has died (thread crashed), restarts it
        transparently so a single failure doesn't permanently break the
        client.
        """
        if self._loop is None or not self._loop.is_running():
            self.start()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


class TyLspClient:
    """Persistent LSP client for ``ty server`` inside a microsandbox VM.

    The server process is started lazily on first request and kept alive
    for the lifetime of the client.  All LSP requests are serialised through
    an :class:`asyncio.Lock` so that framing stays consistent.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        root_uri: str,
        *,
        python_path: Optional[str] = None,
        venv_bin: Optional[str] = None,
        venv_env: Optional[dict[str, str]] = None,
    ) -> None:
        self._sandbox = sandbox
        self._root_uri = root_uri
        self._root_path = root_uri.replace("file://", "")
        self._python_path = python_path
        # Guest-compatible venv established by VenvManager.  When set, all
        # shell commands run with venv_env activated, and ty is launched
        # from venv_bin so it uses the same venv as the project.
        self._venv_bin = venv_bin
        self._venv_env = venv_env
        self._handle = None
        self._stdin = None
        self._lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._stdout_buf = bytearray()
        self._started = False
        self._bg = _BackgroundLoop()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Boot the ty server process and complete the LSP initialize handshake."""
        self._bg.run(
            self._start(),
            timeout=_SERVER_BOOT_TIMEOUT_SEC + _TY_INSTALL_TIMEOUT_SEC + 30,
        )

    async def _start(self) -> None:
        if self._started:
            return

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

        # Launch ty from the venv's bin directory when available, so it
        # uses the same venv as the project (ty is installed there by
        # _ensure_ty_installed).  Otherwise fall back to PATH lookup.
        if self._venv_bin:
            ty_bin = f"{self._venv_bin}/ty"
            try:
                exists = await asyncio.wait_for(
                    self._sandbox.fs.exists(ty_bin), timeout=10
                )
            except Exception:
                exists = False
            if exists:
                ty_cmd = ty_bin
            else:
                ty_cmd = "ty"
        else:
            ty_cmd = "ty"

        self._handle = await self._sandbox.exec_stream(
            ty_cmd,
            ["server"],
            stdin=Stdin.pipe(),
            env=env or None,
            timeout=0,  # no timeout — long-lived process
        )
        self._stdin = self._handle.take_stdin()

        # Start background reader
        self._reader_task = asyncio.ensure_future(self._reader_loop())

        init_params: dict[str, Any] = {
            "processId": 0,
            "rootUri": self._root_uri,
            "capabilities": {},
        }
        if self._python_path:
            init_params["initializationOptions"] = {
                "pythonPath": self._python_path,
            }

        # Use _send_request / _send_notify (lock-free) instead of
        # _request / _notify (which acquire self._lock).  _start() may
        # be called from within _request() which already holds the lock,
        # so using the lock-acquiring variants would deadlock.
        await self._send_request(
            "initialize",
            init_params,
            timeout=_SERVER_BOOT_TIMEOUT_SEC,
        )
        await self._send_notify("initialized", {})

        self._started = True
        logger.info("ty server started (root_uri=%s)", self._root_uri)

    def stop(self) -> None:
        """Shut down the server process and the background loop."""
        try:
            self._bg.run(self._stop(), timeout=10)
        except Exception:
            pass
        self._bg.stop()

    async def _stop(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

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

        self._started = False

    async def _find_site_packages(self) -> list[str]:
        """Discover site-packages directories inside the sandbox.

        Returns paths to the venv's site-packages so ty can resolve
        third-party imports.

        Uses the venv's Python (from VenvManager) when available, avoiding
        ``uv run`` which would create a second venv.  Falls back to system
        Python discovery only when no venv was established.
        """
        paths: list[str] = []

        # Prefer the venv's Python — no uv run, no second venv.
        if self._venv_bin:
            py = f"{self._venv_bin}/python"
            try:
                result = await self._sandbox.shell(
                    f"{py} -c 'import site; print(\"\\n\".join(site.getsitepackages()))'",
                    env=self._venv_env,
                )
                if result.exit_code == 0 and result.stdout_text.strip():
                    paths = [
                        p.strip()
                        for p in result.stdout_text.strip().splitlines()
                        if p.strip()
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
                result = await self._sandbox.shell(cmd)
                if result.exit_code == 0 and result.stdout_text.strip():
                    paths = [
                        p.strip()
                        for p in result.stdout_text.strip().splitlines()
                        if p.strip()
                    ]
                    if paths:
                        break
            except Exception:
                continue

        # Also look for a .venv site-packages under the project root
        try:
            result = await self._sandbox.shell(
                f"ls {self._root_path}/.venv/lib/ 2>/dev/null"
            )
            if result.exit_code == 0:
                for line in result.stdout_text.strip().splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    candidate = f"{self._root_path}/.venv/lib/{name}/site-packages"
                    try:
                        exists = await asyncio.wait_for(
                            self._sandbox.fs.exists(candidate), timeout=5
                        )
                        if exists and candidate not in paths:
                            paths.append(candidate)
                    except Exception:
                        pass
        except Exception:
            pass

        return paths

    async def _ensure_ty_installed(self) -> None:
        """Install ty inside the sandbox and ensure a pyproject.toml exists.

        ``ty`` needs a ``pyproject.toml`` (or ``ty.toml``) at the project root
        to discover first-party modules.  If none exists, we create a minimal
        one with ``environment.extra-paths`` set to the workspace root so that
        relative imports resolve correctly.

        When a venv is established (``venv_bin``), ty is installed into that
        venv so the ``ty`` binary lands in ``venv_bin`` and uses the venv's
        Python.  Otherwise it falls back to system pip.
        """
        # Check if ty is already available (in the venv or on PATH).
        ty_check_cmd = (
            f"{self._venv_bin}/ty --version 2>/dev/null"
            if self._venv_bin
            else "which ty 2>/dev/null"
        )
        result = await self._sandbox.shell(ty_check_cmd, env=self._venv_env)
        if result.exit_code != 0 or not result.stdout_text.strip():
            logger.info("Installing ty inside sandbox…")
            if self._venv_bin:
                # Install into the venv so the ty binary lands in venv_bin.
                # uv-created venvs lack pip, so try pip first then uv pip.
                py = f"{self._venv_bin}/python"
                result = await self._sandbox.shell(
                    f"{py} -m pip install ty -q 2>&1",
                    env=self._venv_env,
                    timeout=_TY_INSTALL_TIMEOUT_SEC,
                )
                if result.exit_code != 0:
                    logger.info("pip not available in venv, trying uv pip install…")
                    result = await self._sandbox.shell(
                        f"uv pip install ty --python {py} -q 2>&1",
                        env=self._venv_env,
                        timeout=_TY_INSTALL_TIMEOUT_SEC,
                    )
            else:
                result = await self._sandbox.shell(
                    "pip install ty -q 2>&1", timeout=_TY_INSTALL_TIMEOUT_SEC
                )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to install ty in sandbox: {result.stdout_text}"
                )

        # Ensure pyproject.toml exists for module discovery.
        # Site-packages paths are passed via LSP initializationOptions
        # (see _start), not via ty.toml — avoids writing user files.
        root_path = self._root_path
        pyproject = f"{root_path}/pyproject.toml"
        try:
            exists = await asyncio.wait_for(
                self._sandbox.fs.exists(pyproject), timeout=10
            )
        except Exception:
            exists = False

        if not exists:
            content = '[project]\nname = "project"\nversion = "0.0.0"\n'
            try:
                await asyncio.wait_for(
                    self._sandbox.fs.write(pyproject, content.encode("utf-8")),
                    timeout=10,
                )
            except Exception as e:
                logger.warning("Failed to create pyproject.toml for ty: %s", e)

    # ------------------------------------------------------------------ #
    # LSP message framing
    # ------------------------------------------------------------------ #

    async def _send(self, message: dict) -> None:
        """Write a single LSP message to the server's stdin."""
        data = json.dumps(message).encode("utf-8")
        frame = b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data
        if self._stdin is None:
            raise RuntimeError("ty server stdin is not available")
        await self._stdin.write(frame)

    async def _reader_loop(self) -> None:
        """Background task that reads LSP messages from the server stdout."""
        assert self._handle is not None
        while True:
            try:
                event = await self._handle.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("ty server stream ended", exc_info=True)
                break

            if event is None:
                logger.debug("ty server stream closed")
                break

            # Only process stdout events (contain LSP responses).
            # stderr events carry ty's logging output and must be ignored.
            if getattr(event, "event_type", None) != "stdout":
                continue

            data = getattr(event, "data", None)
            if data is None:
                continue

            self._stdout_buf.extend(data)
            self._process_buffer()

        # Server crashed or stream closed — clean up state so _request
        # can auto-restart on the next call.
        self._started = False

        # Wake up any pending futures with an error so they don't hang.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("ty server closed"))
        self._pending.clear()

        # Close stdin and kill the handle so resources are released.
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

    def _process_buffer(self) -> None:
        """Extract complete LSP messages from the stdout buffer."""
        while True:
            header_end = self._stdout_buf.find(b"\r\n\r\n")
            if header_end == -1:
                return

            headers = self._stdout_buf[:header_end].decode("ascii", errors="replace")
            content_length = 0
            for line in headers.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":", 1)[1].strip())

            body_start = header_end + 4
            if len(self._stdout_buf) < body_start + content_length:
                return

            body = self._stdout_buf[body_start : body_start + content_length]
            del self._stdout_buf[: body_start + content_length]

            try:
                message = json.loads(body)
            except json.JSONDecodeError:
                logger.warning("ty server sent invalid JSON: %s", body[:200])
                continue

            self._dispatch(message)

    def _dispatch(self, message: dict) -> None:
        """Route a received LSP message to the appropriate pending future."""
        msg_id = message.get("id")
        if msg_id is None:
            # Notification (e.g. publishDiagnostics) — ignore
            return

        fut = self._pending.pop(msg_id, None)
        if fut is None:
            return

        if "error" in message:
            fut.set_exception(RuntimeError(f"ty server error: {message['error']}"))
        else:
            fut.set_result(message.get("result"))

    # ------------------------------------------------------------------ #
    # Public LSP request methods (sync wrappers)
    # ------------------------------------------------------------------ #

    def request(
        self,
        method: str,
        params: Optional[dict] = None,
        *,
        timeout: float = _LSP_TIMEOUT_SEC,
    ) -> Any:
        """Send an LSP request synchronously and return the response."""
        return self._bg.run(
            self._request(method, params, timeout=timeout), timeout=timeout + 10
        )

    async def _request(
        self,
        method: str,
        params: Optional[dict] = None,
        *,
        timeout: float = _LSP_TIMEOUT_SEC,
    ) -> Any:
        """Send an LSP request and await the response.

        The lock is only held during the send (to serialise message framing
        and ID assignment), then released before waiting for the response.
        This allows concurrent requests and notifications to proceed while
        a response is pending.
        """
        async with self._lock:
            if not self._started:
                # Server may have crashed and been cleaned up by
                # _reader_loop.  Restart it transparently.
                await self._start()

            msg_id = self._next_id
            self._next_id += 1

            message = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": method,
                "params": params or {},
            }

            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending[msg_id] = fut

            await self._send(message)

        # Lock released — wait for the response outside the critical section.
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send an LSP notification synchronously."""
        self._bg.run(self._notify(method, params), timeout=60)

    async def _notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send an LSP notification (no response expected)."""
        async with self._lock:
            if not self._started:
                await self._start()

            message = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
            await self._send(message)

    # ------------------------------------------------------------------ #
    # Lock-free send helpers (used by _start to avoid self-deadlock)
    # ------------------------------------------------------------------ #

    async def _send_request(
        self,
        method: str,
        params: Optional[dict] = None,
        *,
        timeout: float = _LSP_TIMEOUT_SEC,
    ) -> Any:
        """Send an LSP request without acquiring self._lock.

        Used by :meth:`_start` which may be called from within :meth:`_request`
        (which already holds the lock).  Caller must ensure no concurrent
        access to ``_next_id`` / ``_pending``.
        """
        msg_id = self._next_id
        self._next_id += 1

        message = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params or {},
        }

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut

        await self._send(message)

        # Lock released — wait for the response outside the critical section.
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise

    async def _send_notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send an LSP notification without acquiring self._lock."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        await self._send(message)

    # ------------------------------------------------------------------ #
    # High-level LSP operations (sync)
    # ------------------------------------------------------------------ #

    def did_open(self, uri: str, text: str) -> None:
        """Notify the server that a document was opened."""
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": text,
                }
            },
        )

    def document_symbol(self, uri: str) -> list[dict]:
        """Return document symbols for *uri*."""
        result = self.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
        )
        if result is None:
            return []
        return result if isinstance(result, list) else []

    def definition(self, uri: str, line: int, character: int) -> Any:
        """Return definition location(s) for the symbol at *line*/*character*."""
        return self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )

    def references(
        self, uri: str, line: int, character: int, *, include_declaration: bool = False
    ) -> list[dict]:
        """Return reference locations for the symbol at *line*/*character*."""
        result = self.request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_declaration},
            },
        )
        if result is None:
            return []
        return result if isinstance(result, list) else []

    def hover(self, uri: str, line: int, character: int) -> Optional[dict]:
        """Return hover information for the symbol at *line*/*character*."""
        return self.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )

    def workspace_symbol(self, query: str) -> list[dict]:
        """Return workspace symbols matching *query*."""
        result = self.request(
            "workspace/symbol",
            {"query": query},
        )
        if result is None:
            return []
        return result if isinstance(result, list) else []
