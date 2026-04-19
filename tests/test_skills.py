"""
Verifies that the ACP agent launched by run.sh is capable of managing tool skills.
"""

import logging
from pathlib import Path

import pytest
from acp.schema import ToolCallStart
from conftest import RecordingClient, run_agent

logger = logging.getLogger("acp_test")


# Tests
@pytest.mark.asyncio
async def test_agent_call_skills(run_sh: Path) -> None:
    """
    Ensure the agent calls one of the predefined tool skills.
    """
    client = RecordingClient(prefix="acp_skills_test_")

    test_skill_code = '''
import subprocess
from typing import Tuple

from langchain_core.tools import tool


def _run(cmd: str) -> Tuple[int, str]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    output = (result.stdout + result.stderr).strip()
    return (result.returncode, (output or f"exited {result.returncode}"))


@tool
def list_all_files(path: str) -> Tuple[int, str]:
    """List all files and directories at the given path.
    Returns a tuple of (returncode, output)."""
    return _run(f"ls -al {path}")
'''
    skills_path = client.temp_dir / "skills.py"
    skills_path.write_text(test_skill_code)

    await run_agent(
        client,
        run_sh,
        "Use tool skills to list all files in the current directory.",
    )

    success = False
    for update in client.updates:
        if isinstance(update, ToolCallStart) and update.title == "list_all_files":
            success = True
            break
    assert success, "Agent did not call the list_all_files tool skill"

    logger.info("Agent output:\n%s", client.all_text)


@pytest.mark.asyncio
async def test_agent_reload_skills(run_sh: Path, tmp_path: Path) -> None:
    """
    Ensure the agent can reload tool skills after external modification.

    Simulates a user manually adding a skill to skills.py while the agent
    is running, then the agent reloading and using that skill.
    """
    import asyncio

    from acp import spawn_agent_process, text_block

    client = RecordingClient(prefix="acp_skills_test_")
    skills_path = client.temp_dir / "skills.py"

    # Start with an empty skills.py file
    skills_path.write_text("")

    async with spawn_agent_process(
        client,
        "bash",
        str(run_sh),
    ) as (conn, _proc):
        await conn.initialize(protocol_version=1)
        session = await conn.new_session(
            cwd=str(client.temp_dir),
            mcp_servers=[],
        )
        await conn.set_config_option(
            config_id="model",
            session_id=session.session_id,
            value="evroc:moonshotai/Kimi-K2.5",
        )

        # FIRST: Write the skill file while agent is running
        test_skill_code = '''
from langchain_core.tools import tool


@tool
def hello_test_skill() -> str:
    """Say hello for testing."""
    return "Hello from the test skill!"
'''
        skills_path.write_text(test_skill_code)

        # THEN: Send the prompt asking to reload and use the skill
        await asyncio.wait_for(
            conn.prompt(
                session_id=session.session_id,
                prompt=[
                    text_block("""
                First, reload tool skills to pick up any changes.
                Then, call the hello_test_skill tool skill.
                """)
                ],
            ),
            timeout=300,
        )

    # Verify reload_tool_skills was called
    reload_called = False
    for update in client.updates:
        if isinstance(update, ToolCallStart) and update.title == "reload_tool_skills":
            reload_called = True
            break
    assert reload_called, "Agent did not call reload_tool_skills"

    # Verify hello_test_skill was called after reload
    skill_called = False
    for update in client.updates:
        if isinstance(update, ToolCallStart) and update.title == "hello_test_skill":
            skill_called = True
            break
    assert skill_called, "Agent did not call hello_test_skill after reload"

    logger.info("Agent output:\n%s", client.all_text)


@pytest.mark.asyncio
async def test_agent_create_test_delete_skills(run_sh: Path) -> None:
    """
    Ensure the agent can manage tool skills lifecycle in a single session: create, test, delete.
    """
    client = RecordingClient(prefix="acp_skills_test_")

    test_skill_code = """
from langchain_core.tools import tool


"""
    skills_path = client.temp_dir / "skills.py"
    skills_path.write_text(test_skill_code)

    await run_agent(
        client,
        run_sh,
        """
        Use tool skills to create a new tool skill called make_dataframe. It'll use pandas to create a DataFrame from a CSV file.
        Then use tool skills to create a dataframe from a test CSV file.
        Finally, use tool skills to delete the tool skill called make_dataframe.
        """,
    )

    success = False
    for update in client.updates:
        if isinstance(update, ToolCallStart) and update.title == "create_tool_skill":
            success = True
            break
    assert success, "Agent did not call the create_tool_skill skill"

    success = False
    for update in client.updates:
        if isinstance(update, ToolCallStart) and update.title == "make_dataframe":
            success = True
            break
    assert success, "Agent did not call the make_dataframe skill"

    success = False
    for update in client.updates:
        if isinstance(update, ToolCallStart) and update.title == "delete_tool_skill":
            success = True
            break
    assert success, "Agent did not call the delete_tool_skill skill"

    logger.info("Agent output:\n%s", client.all_text)
