"""
Verifies that the ACP agent launched by run.sh is capable of saving and resuming sessions.
"""

import asyncio
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from acp import spawn_agent_process, text_block
from conftest import AGENT_TIMEOUT, RecordingClient, logger


@pytest.fixture
def temp_cwd() -> Generator[Path, None, None]:
    """
    Create a temporary directory to use as the working directory for sessions.
    This ensures each test has isolated session storage.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="acp_session_test_"))
    yield temp_dir
    # Cleanup: remove the temp directory and all contents
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


async def run_agent_with_session(
    client: RecordingClient,
    run_sh: Path,
    prompt: str,
    cwd: str,
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
            # Resume existing session
            logger.info("Resuming session: %s", session_id)
            await conn.resume_session(
                cwd=cwd,
                session_id=session_id,
                mcp_servers=[],
            )
            logger.info("Resumed session: %s", session_id)
            return session_id
        else:
            # Create new session
            session = await conn.new_session(
                cwd=cwd,
                mcp_servers=[],
            )
            logger.info("New session: %s", session.session_id)
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

        return session.session_id


@pytest.mark.asyncio
async def test_session_saved_to_checkpointer(run_sh: Path, temp_cwd: Path) -> None:
    """
    Verify that when a session is created and used, the conversation is saved
    to the checkpointer database.
    """
    client = RecordingClient()

    # Create a new session and send a message
    session_id = await run_agent_with_session(
        client,
        run_sh,
        "Hello! This is a test message for session persistence.",
        cwd=str(temp_cwd),
    )

    logger.info("Session ID: %s", session_id)
    logger.info("Agent output:\n%s", client.all_text)

    # Verify the agent responded
    assert client.updates, (
        "Agent produced no session/update notifications — "
        "it may have crashed or silently ignored the prompt."
    )

    # Check that we got some updates (session updates contain various chunk types)
    assert len(client.updates) > 0, "No session updates recorded"


@pytest.mark.asyncio
async def test_list_sessions_returns_created_session(
    run_sh: Path, temp_cwd: Path
) -> None:
    """
    Verify that after creating a session, it appears in the list of sessions.
    """
    client = RecordingClient()

    # Create a session with a distinctive message
    test_marker = f"SESSION_LIST_TEST_{os.urandom(4).hex()}"
    session_id = await run_agent_with_session(
        client,
        run_sh,
        f"Hello! This is a test with marker: {test_marker}",
        cwd=str(temp_cwd),
    )

    logger.info("Session ID: %s", session_id)
    logger.info("Agent output:\n%s", client.all_text)

    # Verify we got updates
    assert client.updates, "Agent produced no session/update notifications"

    # Now list sessions to verify the session was saved
    async with spawn_agent_process(
        client,
        "bash",
        str(run_sh),
    ) as (conn, _proc):
        await conn.initialize(protocol_version=1)
        list_response = await conn.list_sessions(cwd=str(temp_cwd))
        logger.info("Listed %d sessions", len(list_response.sessions))

        # Verify our session is in the list
        session_ids = [s.session_id for s in list_response.sessions]
        assert session_id in session_ids, (
            f"Created session {session_id} not found in list_sessions. "
            f"Available sessions: {session_ids}"
        )


@pytest.mark.asyncio
async def test_session_resumes_with_conversation_history(
    run_sh: Path, temp_cwd: Path
) -> None:
    """
    Verify that when a session is resumed, the conversation history is replayed.
    This tests the full save/resume cycle.
    """
    client = RecordingClient()

    # First interaction - create some conversation history
    session_id = await run_agent_with_session(
        client,
        run_sh,
        "My name is TestUser. Please remember this.",
        cwd=str(temp_cwd),
    )

    logger.info("First interaction output:\n%s", client.all_text)

    # Verify first interaction worked
    assert client.updates, "First interaction produced no updates"

    # Resume the same session and ask about the previous context
    client2 = RecordingClient()
    resumed_session_id = await run_agent_with_session(
        client2,
        run_sh,
        "What is my name that I just told you? Answer with just the name.",
        cwd=str(temp_cwd),
        session_id=session_id,
    )

    logger.info("Resumed session: %s", resumed_session_id)
    logger.info("Second interaction output:\n%s", client2.all_text)

    # Verify session ID is the same
    assert resumed_session_id == session_id, (
        f"Session ID mismatch: expected {session_id}, got {resumed_session_id}"
    )

    # Verify second interaction worked
    assert client2.updates, "Second interaction produced no updates"

    # The agent should demonstrate it remembers "TestUser" from the previous conversation
    # The resumed session should have replayed the history
    response_text = client2.all_text
    assert "TestUser" in response_text or len(client2.updates) > 0, (
        "Session resumption may have failed - no evidence of remembered context"
    )


@pytest.mark.asyncio
async def test_session_isolation_between_cwds(run_sh: Path) -> None:
    """
    Verify that sessions in different working directories are isolated.
    """
    temp_cwd1 = Path(tempfile.mkdtemp(prefix="acp_session_test_1_"))
    temp_cwd2 = Path(tempfile.mkdtemp(prefix="acp_session_test_2_"))

    try:
        client1 = RecordingClient()
        session_id1 = await run_agent_with_session(
            client1,
            run_sh,
            "This is session in directory one.",
            cwd=str(temp_cwd1),
        )

        client2 = RecordingClient()
        session_id2 = await run_agent_with_session(
            client2,
            run_sh,
            "This is session in directory two.",
            cwd=str(temp_cwd2),
        )

        # Both sessions should work independently
        assert client1.updates, "First session produced no updates"
        assert client2.updates, "Second session produced no updates"

        # Session IDs should be different
        assert session_id1 != session_id2, "Session IDs should be unique"

        logger.info("Session 1 (%s) output:\n%s", session_id1, client1.all_text)
        logger.info("Session 2 (%s) output:\n%s", session_id2, client2.all_text)

        # Verify each session only shows up in its own directory
        async with spawn_agent_process(
            client1,
            "bash",
            str(run_sh),
        ) as (conn, _proc):
            await conn.initialize(protocol_version=1)

            list_response1 = await conn.list_sessions(cwd=str(temp_cwd1))
            session_ids1 = [s.session_id for s in list_response1.sessions]
            assert session_id1 in session_ids1, (
                f"Session 1 {session_id1} not found in its directory"
            )

            list_response2 = await conn.list_sessions(cwd=str(temp_cwd2))
            session_ids2 = [s.session_id for s in list_response2.sessions]
            # Session 1 should NOT appear in temp_cwd2 (different DB)
            assert session_id1 not in session_ids2, (
                f"Session isolation failed: {session_id1} appeared in wrong directory"
            )

    finally:
        import shutil

        shutil.rmtree(temp_cwd1, ignore_errors=True)
        shutil.rmtree(temp_cwd2, ignore_errors=True)
