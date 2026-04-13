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
async def test_agent_create_test_delete_skills(run_sh: Path) -> None:
    """
    Ensure the agent can manage tool skills lifecycle in a single session: create, test, delete.
    """
    client = RecordingClient()

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
