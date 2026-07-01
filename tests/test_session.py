"""
Verifies that the ACP agent launched by run.sh is capable of saving and resuming sessions.
"""

import asyncio
from pathlib import Path

import aiosqlite
import pytest
from acp import spawn_agent_process, text_block
from acp.schema import ToolCallProgress, ToolCallStart

from tests.conftest import AGENT_TIMEOUT, RecordingClient, logger

MODEL = "evroc:moonshotai/Kimi-K2.6"


async def _setup_and_prompt(
    conn,
    session_id: str,
    prompt: str,
    timeout: int,
) -> None:
    """Set model config, send prompt, and close session."""
    await conn.set_config_option(
        config_id="model",
        session_id=session_id,
        value=MODEL,
    )
    await conn.set_config_option(
        config_id="mode",
        session_id=session_id,
        value="accept_everything",
    )
    await asyncio.wait_for(
        conn.prompt(
            session_id=session_id,
            prompt=[text_block(prompt)],
        ),
        timeout=timeout,
    )
    await conn.close_session(session_id)


async def run_agent_with_session(
    client: RecordingClient,
    run_sh: Path,
    prompt: str,
    session_id: str | None = None,
    timeout: int = AGENT_TIMEOUT,
) -> str:
    """Spawn the agent and run a prompt. If session_id is provided, resume that session.

    Returns the session_id used/created.
    """
    logger.info("Starting agent: %s", run_sh)
    async with spawn_agent_process(
        client,
        "bash",
        str(run_sh),
    ) as (conn, _proc):
        await conn.initialize(protocol_version=1)
        logger.info("Initialized")

        if session_id:
            logger.info("Resuming session: %s", session_id)
            await conn.resume_session(
                cwd=str(client.temp_dir),
                session_id=session_id,
                mcp_servers=[],
            )
            logger.info("Resumed session: %s", session_id)
            await _setup_and_prompt(conn, session_id, prompt, timeout)
            return session_id

        session = await conn.new_session(
            cwd=str(client.temp_dir),
            mcp_servers=[],
        )
        logger.info("New session: %s", session.session_id)
        await _setup_and_prompt(conn, session.session_id, prompt, timeout)
        return session.session_id


@pytest.mark.asyncio
async def test_session_saved_to_checkpointer(run_sh: Path) -> None:
    """
    Verify that when a session is created and used, the conversation is saved
    to the checkpointer database (SQLite file).
    """
    from metalgate_code.helpers import get_checkpoints_data_dir

    client = RecordingClient(prefix="acp_session_test_")
    with client:
        session_id = await run_agent_with_session(
            client,
            run_sh,
            "Hello! This is a test message for session persistence.",
        )
        logger.info("Session ID: %s", session_id)
        logger.info("Agent output:\n%s", client.all_text)

        assert client.updates, (
            "Agent produced no session/update notifications — "
            "it may have crashed or silently ignored the prompt."
        )

        db_path = get_checkpoints_data_dir(str(client.temp_dir))
        logger.info("Expected checkpointer DB path: %s", db_path)
        assert db_path.exists(), f"Checkpointer database not found at {db_path}"

        async with aiosqlite.connect(str(db_path)) as db:
            async with db.execute(
                "SELECT thread_id, title FROM sessions WHERE thread_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                assert row is not None, (
                    f"Session {session_id} not found in sessions table"
                )
                assert row[0] == session_id, "Session ID mismatch in database"
                logger.info(
                    "Found session in DB: thread_id=%s, title=%s", row[0], row[1]
                )

            async with db.execute(
                "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
                msg_count = row[0] if row else 0
                assert msg_count > 0, f"No messages found for session {session_id}"
                logger.info("Found %d messages for session %s", msg_count, session_id)


@pytest.mark.asyncio
async def test_list_sessions_returns_created_session(run_sh: Path) -> None:
    """
    Verify that after creating a session, it appears in the list of sessions.
    """
    client = RecordingClient(prefix="acp_session_test_")
    with client:
        session_id = await run_agent_with_session(
            client,
            run_sh,
            "Hello! This is a test session for list_sessions.",
        )
        logger.info("Session ID: %s", session_id)
        logger.info("Agent output:\n%s", client.all_text)

        assert client.updates, "Agent produced no session/update notifications"

        async with spawn_agent_process(
            client,
            "bash",
            str(run_sh),
        ) as (conn, _proc):
            await conn.initialize(protocol_version=1)
            list_response = await conn.list_sessions(cwd=str(client.temp_dir))
            logger.info("Listed %d sessions", len(list_response.sessions))

            session_ids = [s.session_id for s in list_response.sessions]
            assert session_id in session_ids, (
                f"Created session {session_id} not found in list_sessions. "
                f"Available sessions: {session_ids}"
            )


@pytest.mark.asyncio
async def test_session_resumes_with_conversation_history(run_sh: Path) -> None:
    """
    Verify that when a session is resumed, the conversation history is replayed.
    This tests the full save/resume cycle.
    """
    name = "Alice"

    client = RecordingClient(prefix="acp_session_test_")
    with client:
        session_id = await run_agent_with_session(
            client,
            run_sh,
            f"My name is {name}. Please remember this.",
        )
        logger.info("First interaction output:\n%s", client.all_text)

        assert client.updates, "First interaction produced no updates"

        client_resumed = RecordingClient(prefix="acp_session_test_")
        client_resumed.temp_dir = client.temp_dir
        with client_resumed:
            resumed_session_id = await run_agent_with_session(
                client_resumed,
                run_sh,
                "What is my name that I just told you? Answer with just the name.",
                session_id=session_id,
            )
            logger.info("Resumed session: %s", resumed_session_id)
            logger.info("Second interaction output:\n%s", client_resumed.all_text)

            assert resumed_session_id == session_id, (
                f"Session ID mismatch: expected {session_id}, got {resumed_session_id}"
            )
            assert client_resumed.updates, "Second interaction produced no updates"
            assert name in client_resumed.agent_text, (
                f"AI did not recall name '{name}' in resumed session response. "
                f"Agent text: {client_resumed.agent_text[:500]}"
            )


@pytest.mark.asyncio
async def test_session_isolation_between_cwds(run_sh: Path) -> None:
    """
    Verify that sessions in different working directories are isolated.
    """
    client1 = RecordingClient(prefix="acp_session_test_1_")
    client2 = RecordingClient(prefix="acp_session_test_2_")

    with client1, client2:
        session_id1 = await run_agent_with_session(
            client1,
            run_sh,
            "This is session in directory one.",
        )

        session_id2 = await run_agent_with_session(
            client2,
            run_sh,
            "This is session in directory two.",
        )
        logger.info("Session 1 (%s) output:\n%s", session_id1, client1.all_text)
        logger.info("Session 2 (%s) output:\n%s", session_id2, client2.all_text)

        assert client1.updates, "First session produced no updates"
        assert client2.updates, "Second session produced no updates"
        assert session_id1 != session_id2, "Session IDs should be unique"

        async with spawn_agent_process(
            client1,
            "bash",
            str(run_sh),
        ) as (conn, _proc):
            await conn.initialize(protocol_version=1)

            list_response1 = await conn.list_sessions(cwd=str(client1.temp_dir))
            session_ids1 = [s.session_id for s in list_response1.sessions]
            assert session_id1 in session_ids1, (
                f"Session 1 {session_id1} not found in its directory"
            )

            list_response2 = await conn.list_sessions(cwd=str(client2.temp_dir))
            session_ids2 = [s.session_id for s in list_response2.sessions]
            assert session_id1 not in session_ids2, (
                f"Session isolation failed: {session_id1} appeared in wrong directory"
            )


def _extract_tool_output(updates: list) -> str | None:
    """Return the text output from the first completed ToolCallProgress update."""
    for upd in updates:
        if not isinstance(upd, ToolCallProgress):
            continue
        if getattr(upd, "status", None) != "completed":
            continue
        if not getattr(upd, "content", None):
            continue
        for item in upd.content:
            content = getattr(item, "content", None)
            if content and hasattr(content, "text"):
                return content.text
    return None


@pytest.mark.asyncio
async def test_session_replays_tool_calls(run_sh: Path) -> None:
    """
    Verify that when a session is resumed, tool calls in the conversation history
    are replayed as ToolCallStart notifications, and the tool's output is perfectly
    replicated in the resumed session.
    """
    client_play = RecordingClient(prefix="acp_session_test_")

    test_skill_code = '''
import subprocess
from typing import Tuple


def _run(cmd: str) -> Tuple[int, str]:
    """Run a shell command, using the sandbox backend if available."""
    backend = get_backend()
    if backend is not None:
        result = backend.execute(cmd)
        output = result.output.strip()
        return (
            result.exit_code if result.exit_code is not None else 1,
            output or f"exited {result.exit_code}",
        )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    output = (result.stdout + result.stderr).strip()
    return (result.returncode, (output or f"exited {result.returncode}"))


@tool
def list_all_files(path: str) -> Tuple[int, str]:
    """List all files and directories at the given path.
    Returns a tuple of (returncode, output)."""
    return _run(f"ls -al {path}")
'''
    metalgate_dir = client_play.temp_dir / ".metalgate"
    metalgate_dir.mkdir(exist_ok=True)
    skills_path = metalgate_dir / "skills.py"
    skills_path.write_text(test_skill_code)

    with client_play:
        session_id = await run_agent_with_session(
            client_play,
            run_sh,
            "Use list_all_files tool skill to list files in the current directory.",
        )
        logger.info("First interaction output:\n%s", client_play.all_text)

        assert client_play.updates, "First interaction produced no updates"

        tool_calls_first = [
            u for u in client_play.updates if isinstance(u, ToolCallStart)
        ]
        assert len(tool_calls_first) > 0, "Expected tool calls in first interaction"

        tool_output = _extract_tool_output(client_play.updates)
        assert tool_output is not None, (
            "Expected to find completed tool call with output"
        )
        logger.info("Captured tool output: %s", tool_output[:200])

        client_replay = RecordingClient(prefix="acp_session_test_")
        client_replay.temp_dir = client_play.temp_dir
        with client_replay:
            resumed_session_id = await run_agent_with_session(
                client_replay,
                run_sh,
                "What did the list_all_files tool show in the previous turn? "
                "Repeat the exact output from the tool.",
                session_id=session_id,
            )
            logger.info("Resumed session: %s", resumed_session_id)
            logger.info("Second interaction output:\n%s", client_replay.all_text)

            assert resumed_session_id == session_id, "Session ID mismatch"

            tool_calls_replayed = [
                u for u in client_replay.updates if isinstance(u, ToolCallStart)
            ]
            assert len(tool_calls_replayed) > 0, (
                "Tool calls from history should be replayed when session is resumed"
            )

            agent_response = client_replay.agent_text
            assert ".metalgate" in agent_response, (
                f"Agent did not reproduce '.metalgate' from tool output. "
                f"Tool output: {tool_output[:200]}... "
                f"Second response: {agent_response[:500]}..."
            )
            assert "total" in agent_response.lower(), (
                f"Agent did not reproduce the 'total' line from ls -al output. "
                f"Second response: {agent_response[:500]}..."
            )
            assert "0" in agent_response, (
                f"Agent did not reproduce the return code from tool output. "
                f"Second response: {agent_response[:500]}..."
            )
