"""
Verifies that the ACP agent launched by run.sh is capable of managing tool skills.
"""

import logging
from pathlib import Path

import pytest
from acp.schema import ToolCallStart
from conftest import RecordingClient, run_agent

logger = logging.getLogger("acp_test")


# Fixtures
# Tests
@pytest.mark.asyncio
async def test_agent_call_skills(run_sh: Path) -> None:
    """
    Ensure the agent calls one of the predefined tool skills.
    """
    client = RecordingClient()

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

    Simulates an user manually adding a skill to skills.py, then the agent
    reloading using that skill.
    """
    client = RecordingClient()

    # Pre-create a simple skill directly in skills.py (mocking user edit)
    test_skill_code = '''


@tool
def hello_test_skill() -> str:
    """Say hello for testing."""
    return "Hello from the test skill!"
'''
    skills_path = Path(__file__).parent.parent / "skills.py"
    original_content = skills_path.read_text()
    skills_path.write_text(original_content + test_skill_code)

    try:
        await run_agent(
            client,
            run_sh,
            """
            First, reload tool skills to pick up any changes.
            Then, call the hello_test_skill tool skill.
            """,
        )

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

        logger.info("Agent output:\n%s", client.all_text)
    finally:
        # Cleanup: restore original skills.py
        skills_path.write_text(original_content)


@pytest.mark.asyncio
async def test_agent_create_test_delete_skills(run_sh: Path) -> None:
    """
    Ensure the agent can manage tool skills lifecycle in a single session: create, test, delete.
    """
    client = RecordingClient()

    skills_path = Path(__file__).parent.parent / "skills.py"
    original_content = skills_path.read_text()

    try:
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

        logger.info("Agent output:\n%s", client.all_text)
    finally:
        # Cleanup: restore original skills.py
        skills_path.write_text(original_content)
