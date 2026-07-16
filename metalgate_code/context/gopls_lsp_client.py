"""LSP client for ``gopls`` running on the host.

Manages a persistent ``gopls serve`` subprocess via :mod:`subprocess`,
communicating using the Language Server Protocol (JSON-RPC over stdio with
``Content-Length`` framing).

The transport layer (framing, reader loop, request/response correlation)
is provided by :class:`~metalgate_code.context.lsp_base.LspBaseClient`.
This subclass implements the host-process launch/teardown using
``asyncio``-compatible pipes.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Optional

from metalgate_code.context.lsp_base import LspBaseClient

logger = logging.getLogger("metalgate_code")


class GoplsLspClient(LspBaseClient):
    """Persistent LSP client for ``gopls serve`` on the host.

    The server process is started lazily on first request and kept alive
    for the lifetime of the client.  All LSP requests are serialised through
    an :class:`asyncio.Lock` so that framing stays consistent.
    """

    def __init__(self, root_uri: str, *, cwd: Optional[str] = None) -> None:
        super().__init__(root_uri)
        self._cwd = cwd or os.getcwd()
        self._process: Optional[asyncio.subprocess.Process] = None

    # ------------------------------------------------------------------ #
    # Server lifecycle (transport-specific)
    # ------------------------------------------------------------------ #

    async def _start_server(self) -> None:
        gopls_bin = shutil.which("gopls")
        if gopls_bin is None:
            raise FileNotFoundError("gopls not found in PATH")

        # Ensure TMPDIR points to a writable directory.  When the agent
        # runs in a sandbox the inherited TMPDIR may not exist, which
        # causes gopls to fail with "no package metadata for file".
        env = os.environ.copy()
        tmpdir = env.get("TMPDIR", "")
        if not tmpdir or not os.path.isdir(tmpdir):
            env["TMPDIR"] = "/tmp"

        self._process = await asyncio.create_subprocess_exec(
            gopls_bin,
            "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # discard — no pipe to fill
            cwd=self._cwd,
            env=env,
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
    # Transport-specific read/write
    # ------------------------------------------------------------------ #

    async def _write_raw(self, frame: bytes) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("gopls server stdin is not available")
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

    # ------------------------------------------------------------------ #
    # High-level LSP operations (go-specific overrides)
    # ------------------------------------------------------------------ #

    def did_open(self, uri: str, text: str, language_id: str = "go") -> None:
        """Notify the server that a document was opened."""
        super().did_open(uri, text, language_id=language_id)
