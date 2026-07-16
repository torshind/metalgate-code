"""Shared LSP transport layer for language-server clients.

Provides :class:`_BackgroundLoop` and :class:`LspBaseClient` — the
transport-agnostic machinery for communicating with any LSP server over
JSON-RPC with ``Content-Length`` framing.

Subclasses implement :meth:`_start_server` (launch the server process)
and :meth:`_stop_server` (tear it down).  Everything else — message
framing, the reader loop, request/response correlation, notifications,
and the public ``request``/``notify`` API — is handled here.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("metalgate_code")

_LSP_TIMEOUT_SEC = 30
"""Default timeout for individual LSP requests."""

_SERVER_BOOT_TIMEOUT_SEC = 60
"""Timeout for booting the server process and completing initialize."""


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


class LspBaseClient:
    """Transport-agnostic LSP client base.

    Manages a persistent LSP server subprocess, communicating using the
    Language Server Protocol (JSON-RPC over stdio with
    ``Content-Length`` framing).

    Subclasses must implement :meth:`_start_server` and
    :meth:`_stop_server`.  The base class handles message framing,
    the reader loop, request/response correlation, and the public
    ``request``/``notify`` API.

    All LSP requests are serialised through an :class:`asyncio.Lock` so
    that framing stays consistent.
    """

    def __init__(self, root_uri: str) -> None:
        self._root_uri = root_uri
        self._lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._stdout_buf = bytearray()
        self._started = False
        self._bg = _BackgroundLoop()

    # ------------------------------------------------------------------ #
    # Lifecycle — subclasses implement the transport-specific parts
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Boot the server process and complete the LSP initialize handshake."""
        self._bg.run(
            self._start(),
            timeout=_SERVER_BOOT_TIMEOUT_SEC,
        )

    async def _start(self) -> None:
        if self._started:
            return

        # Clear stale buffer from any previous crashed server so it
        # doesn't corrupt message framing for the new one.
        self._stdout_buf.clear()

        # Cancel any leftover reader task from a previous crashed server.
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reader_task = None

        await self._start_server()

        # Start background reader
        self._reader_task = asyncio.ensure_future(self._reader_loop())

        init_params: dict[str, Any] = {
            "processId": 0,
            "rootUri": self._root_uri,
            "capabilities": {},
        }
        self._customize_init_params(init_params)

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
        logger.info("LSP server started (root_uri=%s)", self._root_uri)

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

        await self._stop_server()

        self._started = False
        self._stdout_buf.clear()

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #

    async def _start_server(self) -> None:
        """Launch the server process and set up ``self._stdin`` / ``self._handle``.

        Subclasses must set ``self._stdin`` to a writable stream and
        ``self._handle`` to an object whose ``recv()`` coroutine yields
        stdout data (or ``None`` on EOF).
        """
        raise NotImplementedError

    async def _stop_server(self) -> None:
        """Tear down the server process and close streams."""
        raise NotImplementedError

    def _customize_init_params(self, params: dict[str, Any]) -> None:
        """Hook for subclasses to add initializationOptions or other fields."""
        pass

    # ------------------------------------------------------------------ #
    # LSP message framing
    # ------------------------------------------------------------------ #

    async def _send(self, message: dict) -> None:
        """Write a single LSP message to the server's stdin."""
        data = json.dumps(message).encode("utf-8")
        frame = b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data
        await self._write_raw(frame)

    async def _write_raw(self, frame: bytes) -> None:
        """Write raw bytes to the server's stdin (subclass-specific)."""
        raise NotImplementedError

    async def _reader_loop(self) -> None:
        """Background task that reads LSP messages from the server stdout."""
        while True:
            try:
                data = await self._read_raw()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("LSP server stream ended", exc_info=True)
                break

            if data is None:
                logger.debug("LSP server stream closed")
                break

            self._stdout_buf.extend(data)
            self._process_buffer()

        # Server crashed or stream closed — clean up state so _request
        # can auto-restart on the next call.
        self._started = False
        self._stdout_buf.clear()

        # Wake up any pending futures with an error so they don't hang.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("LSP server closed"))
        self._pending.clear()

        # Let subclass clean up its transport resources.
        await self._on_stream_closed()

    async def _read_raw(self) -> Optional[bytes]:
        """Read a chunk of bytes from the server's stdout (subclass-specific)."""
        raise NotImplementedError

    async def _on_stream_closed(self) -> None:
        """Hook called after the reader loop exits — clean up transport resources."""
        pass

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

            if content_length == 0:
                # gopls sends empty keepalive frames — skip them.
                continue

            try:
                message = json.loads(body)
            except json.JSONDecodeError:
                logger.warning("LSP server sent invalid JSON: %s", body[:200])
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
            fut.set_exception(RuntimeError(f"LSP server error: {message['error']}"))
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
            self._request(method, params, timeout=timeout), timeout=timeout
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

    def did_open(self, uri: str, text: str, language_id: str = "") -> None:
        """Notify the server that a document was opened."""
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
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

    def prepare_call_hierarchy(self, uri: str, line: int, character: int) -> list[dict]:
        """Return call hierarchy items for the symbol at *line*/*character*."""
        result = self.request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )
        if result is None:
            return []
        return result if isinstance(result, list) else []

    def incoming_calls(self, item: dict) -> list[dict]:
        """Return incoming calls for a call hierarchy item."""
        result = self.request(
            "callHierarchy/incomingCalls",
            {"item": item},
        )
        if result is None:
            return []
        return result if isinstance(result, list) else []
