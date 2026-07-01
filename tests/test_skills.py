"""
Verifies that the ACP agent launched by run.sh is capable of managing tool skills.
"""

import asyncio
import logging
from pathlib import Path

import pytest
from acp import spawn_agent_process, text_block
from acp.schema import ToolCallStart

from tests.conftest import RecordingClient, run_agent

logger = logging.getLogger("acp_test")


# Tests
@pytest.mark.asyncio
async def test_agent_call_skills(run_sh: Path) -> None:
    """
    Ensure the agent calls one of the predefined tool skills.
    """
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
    client = RecordingClient(prefix="acp_skills_test_")
    with client:
        metalgate_dir = client.temp_dir / ".metalgate"
        metalgate_dir.mkdir(exist_ok=True)
        skills_path = metalgate_dir / "skills.py"
        skills_path.write_text(test_skill_code)

        await run_agent(
            client,
            run_sh,
            "Use list_all_files tool skill to list all files in the current directory.",
        )
        logger.info("Agent output:\n%s", client.all_text)

        success = False
        for update in client.updates:
            if isinstance(update, ToolCallStart) and update.title == "list_all_files":
                success = True
                break
        assert success, "Agent did not call the list_all_files tool skill"


@pytest.mark.asyncio
async def test_agent_reload_skills(run_sh: Path, tmp_path: Path) -> None:
    """
    Ensure the agent can reload tool skills after external modification.

    Simulates a user manually adding a skill to skills.py while the agent
    is running, then the agent reloading and using that skill.
    """
    test_skill_code_empty = """
"""
    test_skill_code_hello = '''
@tool
def hello_test_skill() -> str:
    """Say hello for testing."""
    return "Hello from the test skill!"
'''
    client = RecordingClient(prefix="acp_skills_test_")
    with client:
        metalgate_dir = client.temp_dir / ".metalgate"
        metalgate_dir.mkdir(exist_ok=True)
        skills_path = metalgate_dir / "skills.py"

        # Start with an empty skills.py file
        skills_path.write_text(test_skill_code_empty)

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
                value="evroc:moonshotai/Kimi-K2.6",
            )
            await conn.set_config_option(
                config_id="mode",
                session_id=session.session_id,
                value="accept_everything",
            )

            # FIRST: Write the skill file while agent is running
            skills_path.write_text(test_skill_code_hello)

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
            await conn.close_session(session.session_id)
            logger.info("Agent output:\n%s", client.all_text)

        # Verify reload_tool_skills was called
        reload_called = False
        for update in client.updates:
            if (
                isinstance(update, ToolCallStart)
                and update.title == "reload_tool_skills"
            ):
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


@pytest.mark.asyncio
async def test_agent_create_test_delete_skills(run_sh: Path) -> None:
    """
    Ensure the agent can manage tool skills lifecycle in a single session: create, test, delete.
    """
    test_skill_code = """
"""
    client = RecordingClient(prefix="acp_skills_test_")
    with client:
        metalgate_dir = client.temp_dir / ".metalgate"
        metalgate_dir.mkdir(exist_ok=True)
        skills_path = metalgate_dir / "skills.py"
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
        logger.info("Agent output:\n%s", client.all_text)

        success = False
        for update in client.updates:
            if (
                isinstance(update, ToolCallStart)
                and update.title == "create_tool_skill"
            ):
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
            if (
                isinstance(update, ToolCallStart)
                and update.title == "delete_tool_skill"
            ):
                success = True
                break
        assert success, "Agent did not call the delete_tool_skill skill"
