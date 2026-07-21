"""Tests for subagent HITL interrupt remapping in MetalGateACP."""

from __future__ import annotations

from metalgate_code.factory.acp_server import MetalGateACP


class FakeAIMessage:
    """Minimal AIMessage-like object."""

    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class FakeToolMessage:
    """Minimal ToolMessage-like object."""

    def __init__(self, tool_call_id=""):
        self.tool_call_id = tool_call_id


class FakeInterrupt:
    """Minimal Interrupt-like object."""

    def __init__(self, id, value):
        self.id = id
        self.value = value


def test_no_interrupts():
    """Empty interrupts list returns unchanged."""
    result = MetalGateACP._remap_subagent_interrupts([], [])
    assert result == []


def test_parent_interrupt_not_remapped():
    """An interrupt whose ID matches a parent tool call is left alone."""
    parent_tool_id = "call_parent_123"
    messages = [
        FakeAIMessage(tool_calls=[{"id": parent_tool_id, "name": "execute"}]),
    ]
    interrupts = [
        FakeInterrupt(parent_tool_id, {"action_requests": [{"name": "execute"}]}),
    ]
    result = MetalGateACP._remap_subagent_interrupts(interrupts, messages)
    assert result[0].id == parent_tool_id


def test_subagent_interrupt_remapped():
    """A subagent interrupt with an unknown ID is remapped to the active task()."""
    task_id = "call_task_456"
    subagent_interrupt_id = "subagent_internal_789"

    messages = [
        FakeAIMessage(tool_calls=[{"id": task_id, "name": "task"}]),
    ]
    interrupts = [
        FakeInterrupt(
            subagent_interrupt_id,
            {"action_requests": [{"name": "execute", "args": {"command": "echo hi"}}]},
        ),
    ]
    result = MetalGateACP._remap_subagent_interrupts(interrupts, messages)
    assert result[0].id == task_id


def test_task_with_response_is_used():
    """A completed task (has ToolMessage response) IS used for remapping - this is the fix.

    The old behavior incorrectly skipped tasks that had responses. But by the time a
    subagent triggers an interrupt, the parent's task() call has typically already
    completed and received its ToolMessage response. We should use the most recent
    task() call regardless of response status.
    """
    old_task_id = "call_task_old"
    new_task_id = "call_task_new"
    subagent_id = "subagent_999"

    messages = [
        FakeAIMessage(tool_calls=[{"id": old_task_id, "name": "task"}]),
        FakeToolMessage(tool_call_id=old_task_id),
        FakeAIMessage(tool_calls=[{"id": new_task_id, "name": "task"}]),
    ]
    interrupts = [
        FakeInterrupt(
            subagent_id,
            {"action_requests": [{"name": "execute"}]},
        ),
    ]
    result = MetalGateACP._remap_subagent_interrupts(interrupts, messages)
    # Should use the MOST RECENT task() call (new_task_id), even though it has no response yet
    assert result[0].id == new_task_id


def test_non_action_interrupt_not_remapped():
    """Interrupts without action_requests are not remapped."""
    messages = [
        FakeAIMessage(tool_calls=[{"id": "call_task_1", "name": "task"}]),
    ]
    interrupts = [
        FakeInterrupt("unknown_id", {"something_else": True}),
    ]
    result = MetalGateACP._remap_subagent_interrupts(interrupts, messages)
    assert result[0].id == "unknown_id"


def test_no_active_task_no_remapping():
    """If there is no active task(), orphan interrupts are left alone."""
    messages = [
        FakeAIMessage(tool_calls=[{"id": "call_exec", "name": "execute"}]),
    ]
    interrupts = [
        FakeInterrupt("orphan", {"action_requests": [{"name": "execute"}]}),
    ]
    result = MetalGateACP._remap_subagent_interrupts(interrupts, messages)
    assert result[0].id == "orphan"
