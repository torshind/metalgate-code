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
async def test_memory_extracts_and_retrieves_preferences(run_sh: Path) -> None:
    """
    E2E test: Verify Mem0 extracts preferences in session 1
    and retrieves them in session 2.

    Session 1: User states a preference ("I always use pytest")
    Session 2: New session asks what testing framework user prefers
    Expected: Agent should remember "pytest" from Mem0
    """
    # Session 1: Share something memorable
    client_share = RecordingClient(prefix="memory_e2e_test_")
    await run_agent_with_memory(
        client_share,
        run_sh,
        "You are a coding agent with an embedded automatic memory. "
        "My name is Alice and I want you to know that I always use pytest for testing.",
        memory_enabled=True,
    )

    logger.info("Session 1 output:\n%s", client_share.all_text)
    assert client_share.updates, "Session 1 produced no updates"

    # Session 2: New session - ask what testing framework to use
    client_ask = RecordingClient(prefix="memory_e2e_test_")
    client_ask.temp_dir = client_share.temp_dir  # Share temp dir with first client
    await run_agent_with_memory(
        client_ask,
        run_sh,
        "I'm starting a new Python project. Based on my preferences, "
        'what testing framework should I use? Just answer with the framework name or "I don\'t know".',
        memory_enabled=True,
    )

    logger.info("Session 2 output:\n%s", client_ask.all_text)
    assert client_ask.updates, "Session 2 produced no updates"

    # The agent should remember "pytest" from session 1's memory
    response_text = client_ask.all_text.lower()
    # Look for pytest in the response
    assert "pytest" in response_text, "Agent did not remember pytest from memory"


# @pytest.mark.asyncio
# async def test_memory_disabled_when_env_not_set(run_sh: Path) -> None:
#     """
#     Verify that when MEMORY env var is not set, memory middleware
#     is not active (no errors, just doesn't store/retrieve).
#     """
#     # Session 1: Share something memorable WITHOUT memory enabled
#     client_share = RecordingClient()
#     session_share = await run_agent_with_memory(
#         client_share,
#         run_sh,
#         "You are a coding agent with disabled embedded memory. "
#         "You are going to forget what I am going to say. "
#         "My name is Alice and I prefer unittest over pytest.",
#         memory_enabled=False,  # Memory disabled
#     )

#     logger.info("Session 1 (memory disabled) output:\n%s", client_share.all_text)
#     assert client_share.updates, "Session 1 produced no updates"

#     # Session 2: Ask about preference WITHOUT memory
#     client_ask = RecordingClient()
#     session_ask = await run_agent_with_memory(
#         client_ask,
#         run_sh,
#         "What testing framework do I prefer?",
#         memory_enabled=False,
#     )

#     logger.info("Session 2 (memory disabled) output:\n%s", client_ask.all_text)
#     assert client_ask.updates, "Session 2 produced no updates"

#     # Sessions should work even without memory
#     assert session_share != session_ask, "Sessions should be different"
#     logger.info("SUCCESS: Agent works without MEMORY env var")


# @pytest.mark.asyncio
# async def test_memory_isolated_between_projects(run_sh: Path) -> None:
#     """
#     Verify that memories are isolated between different projects/directories.
#     """
#     temp_cwd1 = Path(tempfile.mkdtemp(prefix="memory_project1_"))
#     temp_cwd2 = Path(tempfile.mkdtemp(prefix="memory_project2_"))

#     try:
#         # Project 1: Store memory
#         client_project1 = RecordingClient()
#         await run_agent_with_memory(
#             client_project1,
#             run_sh,
#             "You are a coding agent with enabled embedded memory. "
#             "My favorite color is blue.",
#             cwd=str(temp_cwd1),
#             memory_enabled=True,
#         )

#         logger.info("Project 1 output:\n%s", client_project1.all_text)
#         assert client_project1.updates, "Project 1 session produced no updates"

#         # Project 2: Should not have access to Project 1's memory
#         client_project2 = RecordingClient()
#         await run_agent_with_memory(
#             client_project2,
#             run_sh,
#             "You are a coding agent with enabled embedded memory. "
#             "What's my favorite color?",
#             cwd=str(temp_cwd2),
#             memory_enabled=True,
#         )

#         logger.info("Project 2 output:\n%s", client_project2.all_text)
#         assert client_project2.updates, "Project 2 session produced no updates"

#         # Verify both projects work independently
#         logger.info("SUCCESS: Projects have isolated memory storage")

#     finally:
#         shutil.rmtree(temp_cwd1, ignore_errors=True)
#         shutil.rmtree(temp_cwd2, ignore_errors=True)
