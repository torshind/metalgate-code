"""
E2E tests for the Mem0 memory system.

These tests verify that:
1. Memories are extracted and stored during sessions
2. Memories are retrieved and injected into context in new sessions
"""

import asyncio
import os
from pathlib import Path

import pytest
from acp import spawn_agent_process, text_block
from conftest import AGENT_TIMEOUT, RecordingClient, logger


async def run_agent_with_memory(
    client: RecordingClient,
    run_sh: Path,
    prompt: str,
    memory_enabled: bool = True,
    session_id: str | None = None,
    user_id: str = "test_user",
    timeout: int = AGENT_TIMEOUT,
) -> str:
    """Spawn the agent and run a prompt with optional memory enabled.

    Returns the session_id used/created.
    """
    # Set memory environment variable
    env = os.environ.copy()
    if memory_enabled:
        env["MEMORY"] = "true"
    else:
        env.pop("MEMORY", None)

    env["USER"] = user_id

    logger.info("Starting agent with memory=%s: %s", memory_enabled, run_sh)

    async with spawn_agent_process(
        client,
        "bash",
        str(run_sh),
        env=env,
    ) as (conn, _proc):
        await conn.initialize(protocol_version=1)
        logger.info("Initialized")

        if session_id:
            # Resume existing session
            logger.info("Resuming session: %s", session_id)
            await conn.resume_session(
                cwd=str(client.temp_dir),
                session_id=session_id,
                mcp_servers=[],
            )
            logger.info("Resumed session: %s", session_id)
            return session_id
        else:
            # Create new session
            session = await conn.new_session(
                cwd=str(client.temp_dir),
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
async def test_memory_session(run_sh: Path) -> None:
    """
    E2E test: Verify Mem0 extracts preferences cross-session.

    Session 1: User states a project specific statement
    Session 2: New session asks about that
    Expected: Agent should remember it
    Session 3: New session and different project, same question
    Expected: Agent shouldn't remember it
    """
    # Session 1: Share something memorable
    client_share = RecordingClient(prefix="memory_e2e_test_")
    await run_agent_with_memory(
        client_share,
        run_sh,
        "You are a coding agent with an embedded automatic memory. "
        "My name is Alice, I'm starting a new Python project using FastAPI and pytest.",
        memory_enabled=True,
    )

    logger.info("Sharing session output:\n%s", client_share.all_text)
    assert client_share.updates, "Sharing session produced no updates"

    # Session 2: New session - ask what API framework to use
    client_ask = RecordingClient(prefix="memory_e2e_test_")
    client_ask.temp_dir = client_share.temp_dir  # Share temp dir with first client
    await run_agent_with_memory(
        client_ask,
        run_sh,
        "I'm continuing this Python project. Based on my preferences, "
        'which API framework should I use? Just answer with the framework name or "I don\'t know".',
        memory_enabled=True,
    )

    logger.info("Asking session output:\n%s", client_ask.all_text)
    assert client_ask.updates, "Asking session produced no updates"

    # The agent should remember "fastapi" from session 1's memory
    response_text = "".join(
        c for c in client_ask.all_text.lower() if c.isprintable() and not c.isspace()
    )
    # Look for FastAPI in the response
    assert "fastapi" in response_text, "Agent did not remember fastapi from memory"

    # Session 3: New session, new project
    client_ask = RecordingClient(prefix="memory_e2e_test_")
    await run_agent_with_memory(
        client_ask,
        run_sh,
        "I'm starting a new golang project. Based on my preferences, "
        'which API framework should I use? Just answer with the framework name or "I don\'t know".',
        memory_enabled=True,
    )

    logger.info("Asking session output:\n%s", client_ask.all_text)
    assert client_ask.updates, "Asking session produced no updates"

    # The agent should NOT remember "FastAPI" from session 1's memory
    response_text = "".join(
        c for c in client_ask.all_text.lower() if c.isprintable() and not c.isspace()
    )
    # Look for FastAPI in the response
    assert "fastapi" not in response_text, (
        "Agent remembered fastapi from another project's memory"
    )


@pytest.mark.asyncio
async def test_memory_user(run_sh: Path) -> None:
    """
    E2E test: Verify Mem0 extracts preferences cross-project.

    Session 1: User states a preference ("My name is Alice")
    Session 2: New session asks about user's name
    Expected: Agent should remember it
    Session 3: New session and same project but different user asks about user's name
    Expected: Agent shouldn't remember it
    """
    # Session 1: Share something memorable
    client_share = RecordingClient(prefix="memory_e2e_test_")
    await run_agent_with_memory(
        client_share,
        run_sh,
        "You are a coding agent with an embedded automatic memory. "
        "My name is Alice and I want you to remember me.",
        memory_enabled=True,
    )

    logger.info("Sharing session output:\n%s", client_share.all_text)
    assert client_share.updates, "Sharing session produced no updates"

    # Session 2: New session - ask user's name
    client_ask = RecordingClient(prefix="memory_e2e_test_")
    await run_agent_with_memory(
        client_ask,
        run_sh,
        "I'm starting a new project. Who am I?"
        'Just answer with my name or "I don\'t know".',
        memory_enabled=True,
    )

    logger.info("Asking session output:\n%s", client_ask.all_text)
    assert client_ask.updates, "Asking session produced no updates"

    # The agent should remember "Alice" from session 1's memory
    response_text = "".join(
        c for c in client_ask.all_text.lower() if c.isprintable() and not c.isspace()
    )
    # Look for Alice in the response
    assert "alice" in response_text, "Agent did not remember Alice from memory"

    # Session 2: New session, same project, different user
    client_ask = RecordingClient(prefix="memory_e2e_test_")
    client_ask.temp_dir = client_share.temp_dir  # Share temp dir with first client
    await run_agent_with_memory(
        client_ask,
        run_sh,
        "I'm starting a new project. Who am I?"
        'Just answer with my name or "I don\'t know".',
        user_id="new_user",
        memory_enabled=True,
    )

    logger.info("Asking session output:\n%s", client_ask.all_text)
    assert client_ask.updates, "Asking session produced no updates"

    # The agent should not remember "Alice" from session 1's memory
    response_text = "".join(
        c for c in client_ask.all_text.lower() if c.isprintable() and not c.isspace()
    )
    # Look for Alice in the response
    assert "alice" not in response_text, "Agent did not remember Alice from memory"


@pytest.mark.asyncio
async def test_memory_disabled_when_env_not_set(run_sh: Path) -> None:
    """
    Verify that when MEMORY env var is not set, memory middleware
    is not active.
    """
    # Session 1: Share something memorable WITHOUT memory enabled
    client_share = RecordingClient()
    await run_agent_with_memory(
        client_share,
        run_sh,
        "You are a coding agent with disabled embedded memory. "
        "You are going to forget what I am going to say. "
        "My name is Alice and I prefer JavaScript over Python.",
        memory_enabled=False,  # Memory disabled
    )

    logger.info("Session 1 (memory disabled) output:\n%s", client_share.all_text)
    assert client_share.updates, "Session 1 produced no updates"

    # Session 2: Ask about preference WITHOUT memory
    client_ask = RecordingClient()
    client_ask.temp_dir = client_share.temp_dir  # Share temp dir with first client
    await run_agent_with_memory(
        client_ask,
        run_sh,
        "What programming language do I prefer?"
        'Just answer with the language name or "I don\'t know".',
        memory_enabled=False,
    )

    logger.info("Session 2 (memory disabled) output:\n%s", client_ask.all_text)
    assert client_ask.updates, "Session 2 produced no updates"

    # The agent should NOT remember "JavaScript" from session 1's memory
    response_text = "".join(
        c for c in client_ask.all_text.lower() if c.isprintable() and not c.isspace()
    )
    # Look for "JavaScript" in the response
    assert "javascript" not in response_text, (
        "Agent remembered javascript with disabled memory"
    )
