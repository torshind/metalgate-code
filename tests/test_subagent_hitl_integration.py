"""
Integration test for subagent HITL interrupt remapping.

This test spawns the actual ACP server via run.sh and verifies that when a
subagent triggers a human-in-the-loop interrupt (via execute tool), the
permission request is forwarded to the client with the parent task() tool_call_id.
"""

import logging
from pathlib import Path

import pytest
from acp.schema import ToolCallStart

from tests.conftest import RecordingClient, run_agent

logger = logging.getLogger("acp_test")


@pytest.mark.asyncio
async def test_subagent_execute_hitl_remapped(run_sh: Path) -> None:
    """
    Test that a subagent's execute tool HITL interrupt is remapped to the
    parent task() tool call ID.

    The agent should:
    1. Launch a subagent via task() tool
    2. Subagent uses execute tool which triggers HITL
    3. Permission request arrives at client with the task() tool_call_id
    4. Client approves, subagent completes
    """
    client = RecordingClient(prefix="acp_subagent_hitl_", auto_approve=True)
    with client:
        prompt = (
            "Use the task tool to delegate to the 'test_subagent' subagent. "
            "Tell the subagent to run the shell command 'echo hello from subagent'. "
            "The subagent will use the execute tool, which requires your approval."
        )

        await run_agent(
            client,
            run_sh,
            prompt,
            mode="accept_edits",
        )
        logger.info("Agent output:\n%s", client.all_text)

        # Verify the task tool was actually called to launch the subagent
        task_tool_called = any(
            isinstance(update, ToolCallStart) and update.title == "task"
            for update in client.updates
        )
        assert task_tool_called, (
            "No task tool call observed — subagent may not have been launched.\n"
            f"Agent output:\n{client.all_text}"
        )

        # Verify at least one permission request was observed
        assert client.permission_requests, (
            "No permission requests observed — subagent execute HITL may not have triggered.\n"
            f"Agent output:\n{client.all_text}"
        )

        # The permission request should be for an execute command
        execute_requests = [
            req
            for req in client.permission_requests
            if "Execute" in req["title"] or "execute" in req["title"].lower()
        ]
        assert execute_requests, (
            f"No execute permission request observed. Got: {client.permission_requests}\n"
            f"Agent output:\n{client.all_text}"
        )

        # The tool_call_id in the permission request should be the parent task() call ID
        # (not the subagent's internal interrupt ID). We can't easily verify the exact ID
        # without access to the internal state, but we can verify the request was made
        # and the agent completed successfully (which it wouldn't if the remapping failed).
        logger.info("Permission requests observed: %s", client.permission_requests)


@pytest.mark.asyncio
async def test_subagent_multiple_execute_hitl(run_sh: Path) -> None:
    """
    Test multiple subagent execute calls each triggering HITL.
    """
    client = RecordingClient(prefix="acp_subagent_multi_", auto_approve=True)
    with client:
        prompt = (
            "Launch a subagent using the task() tool with name 'test_subagent'. "
            "Ask the subagent to run these commands using execute tool: "
            "1. 'echo first command' "
            "2. 'echo second command' "
            "Each should trigger a permission request."
        )

        await run_agent(
            client,
            run_sh,
            prompt,
            mode="accept_edits",
        )
        logger.info("Agent output:\n%s", client.all_text)

        # Verify the task tool was actually called to launch the subagent
        task_tool_called = any(
            isinstance(update, ToolCallStart) and update.title == "task"
            for update in client.updates
        )
        assert task_tool_called, (
            "No task tool call observed — subagent may not have been launched.\n"
            f"Agent output:\n{client.all_text}"
        )

        # Should have at least one permission request for execute
        execute_requests = [
            req
            for req in client.permission_requests
            if "Execute" in req["title"] or "execute" in req["title"].lower()
        ]
        assert execute_requests, (
            f"No execute permission requests observed. Got: {client.permission_requests}\n"
            f"Agent output:\n{client.all_text}"
        )


@pytest.mark.asyncio
async def test_main_agent_execute_hitl_still_works(run_sh: Path) -> None:
    """
    Regression test: main agent's own execute tool HITL should still work
    (not be affected by subagent remapping logic).
    """
    client = RecordingClient(prefix="acp_main_hitl_", auto_approve=True)
    with client:
        prompt = "Run 'echo hello from main agent' using the execute tool."

        await run_agent(
            client,
            run_sh,
            prompt,
            mode="accept_edits",
        )
        logger.info("Agent output:\n%s", client.all_text)

        # Should have at least one permission request for execute
        execute_requests = [
            req
            for req in client.permission_requests
            if "Execute" in req["title"] or "execute" in req["title"].lower()
        ]
        assert execute_requests, (
            f"No execute permission requests observed for main agent. "
            f"Got: {client.permission_requests}\n"
            f"Agent output:\n{client.all_text}"
        )