"""
Shared ACP client code - NOT conftest.py to avoid ACP interference.
"""

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from acp import spawn_agent_process, text_block
from acp.interfaces import Client
from acp.schema import (
    AllowedOutcome,
    CreateTerminalRequest,
    CreateTerminalResponse,
    DeniedOutcome,
    KillTerminalRequest,
    KillTerminalResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionNotification,
    TerminalOutputRequest,
    TerminalOutputResponse,
    ToolCallStart,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from acp.utils import param_model

# Logging (callers must set up basicConfig)
logger = logging.getLogger("acp_test")

# Config
RUN_SH = Path(__file__).parent.parent / "run.sh"
AGENT_TIMEOUT = 300


class RecordingClient(Client):
    """ACP client that records updates and auto-approves permissions with a temp working directory."""

    def __init__(self, prefix: str = "acp_test_") -> None:
        self.prefix = prefix
        self.temp_dir: Path = Path(tempfile.mkdtemp(prefix=self.prefix))
        self.updates: list[Any] = []
        self.written_files: list[str] = []
        self.approved_options: list[str] = []
        self.denied_requests: list[str] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temp working directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @param_model(RequestPermissionRequest)
    async def request_permission(self, options, session_id, tool_call, **kwargs):
        allow_option = next(
            (o for o in options if o.kind in ("allow_once", "allow_always")), None
        )
        if allow_option:
            self.approved_options.append(allow_option.option_id)
            return RequestPermissionResponse(
                outcome=AllowedOutcome(
                    outcome="selected", option_id=allow_option.option_id
                )
            )
        self.denied_requests.append(getattr(tool_call, "name", "<unknown>"))
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    @param_model(SessionNotification)
    async def session_update(self, session_id, update, **kwargs):
        logger.info("UPDATE %s: %s", type(update).__name__, update)
        self.updates.append(update)
        if isinstance(update, ToolCallStart) and update.kind == "edit":
            for block in update.content or []:
                if hasattr(block, "path"):
                    self.written_files.append(str(Path(block.path).resolve()))

    @param_model(ReadTextFileRequest)
    async def read_text_file(self, path, session_id, limit=None, line=None, **kwargs):
        logger.info("READ %s", path)
        try:
            content = Path(path).read_text(encoding="utf-8")
            if line is not None:
                lines = content.splitlines(keepends=True)
                start = max(0, line - 1)
                end = start + (limit or len(lines))
                content = "".join(lines[start:end])
            elif limit is not None:
                content = content[:limit]
        except (OSError, UnicodeDecodeError) as exc:
            content = f"[read error: {exc}]"
        return ReadTextFileResponse(content=content)

    @param_model(WriteTextFileRequest)
    async def write_text_file(self, content, path, session_id, **kwargs):
        logger.info("WRITE %s", path)
        Path(path).write_text(content, encoding="utf-8")
        self.written_files.append(str(Path(path).resolve()))
        return WriteTextFileResponse()

    @param_model(CreateTerminalRequest)
    async def create_terminal(
        self,
        command,
        session_id,
        args=None,
        cwd=None,
        env=None,
        output_byte_limit=None,
        **kwargs,
    ):
        return CreateTerminalResponse(terminal_id="stub-terminal")

    @param_model(TerminalOutputRequest)
    async def terminal_output(self, session_id, terminal_id, **kwargs):
        return TerminalOutputResponse(output="", truncated=False)

    @param_model(ReleaseTerminalRequest)
    async def release_terminal(self, session_id, terminal_id, **kwargs):
        return ReleaseTerminalResponse()

    @param_model(WaitForTerminalExitRequest)
    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs):
        return WaitForTerminalExitResponse()

    @param_model(KillTerminalRequest)
    async def kill_terminal(self, session_id, terminal_id, **kwargs):
        return KillTerminalResponse()

    async def ext_method(self, method, params):
        return {}

    async def ext_notification(self, method, params):
        pass

    def on_connect(self, conn):
        logger.info("CONNECTED")

    def _extract_text(self, obj: Any) -> list[str]:
        if obj is None:
            return []
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, list):
            return [t for item in obj for t in self._extract_text(item)]
        if text := getattr(obj, "text", None):
            return self._extract_text(text)
        if content := getattr(obj, "content", None):
            return self._extract_text(content)
        return []

    @property
    def all_text(self) -> str:
        return "\n".join(t for upd in self.updates for t in self._extract_text(upd))


@pytest.fixture
def run_sh() -> Generator[Path, None, None]:
    """Resolve the agent launcher and fail early if it is missing."""
    if not RUN_SH.exists():
        pytest.fail(
            f"run.sh not found at {RUN_SH}. "
            "Set RUN_SH at the top of conftest.py to the correct path."
        )
    yield RUN_SH


async def run_agent(
    client: RecordingClient, run_sh: Path, prompt: str, timeout: int = AGENT_TIMEOUT
) -> None:
    """Spawn the agent, run a single prompt, and wait for it to finish.

    Returns the session_id used for the session.
    """
    logger.info("Starting agent: %s", run_sh)
    async with spawn_agent_process(
        client,
        "bash",
        str(run_sh),
    ) as (conn, _proc):
        await conn.initialize(protocol_version=1)
        logger.info("Initializing...")
        session = await conn.new_session(
            cwd=str(client.temp_dir),
            mcp_servers=[],
        )
        logger.info("Session: %s", session.session_id)
        await conn.set_config_option(
            config_id="model",
            session_id=session.session_id,
            value="evroc:moonshotai/Kimi-K2.5",
        )
        await asyncio.wait_for(
            conn.prompt(
                session_id=session.session_id,
                prompt=[text_block(prompt)],
            ),
            timeout=timeout,
        )


@pytest.fixture
def tmp_pkg(tmp_path: Path) -> Path:
    """Create a small fake package directory with one module."""
    pkg = tmp_path / "fakelib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        '"""Entry doc."""\n__all__ = ["foo"]\n', encoding="utf-8"
    )
    (pkg / "core.py").write_text(
        '"""Core module."""\n'
        "\n"
        "class Worker:\n"
        '    """Does work."""\n'
        "    count: int\n"
        '    def __init__(self, name: str = "anon"):\n'
        "        self.name = name\n"
        "    def do(self, amount: int) -> None:\n"
        "        pass\n"
        "\n"
        "class Special:\n"
        '    """Class with special methods."""\n'
        "    def __init__(self, value: int) -> None:\n"
        "        self.value = value\n"
        "    def __repr__(self) -> str:\n"
        '        return f"Special({self.value})"\n'
        "    def __call__(self, x: int) -> int:\n"
        "        return x * self.value\n"
        "    def __len__(self) -> int:\n"
        "        return self.value\n"
        "    def _private_helper(self) -> None:\n"
        "        pass\n"
        "\n"
        "def foo(x: int, *args, **kwargs) -> int:\n"
        '    """Returns x."""\n'
        "    return x\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def fake_site(tmp_pkg: Path) -> list[Path]:
    """Return the temp path as a site-packages root so that _file_to_module works."""
    return [tmp_pkg]


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")
