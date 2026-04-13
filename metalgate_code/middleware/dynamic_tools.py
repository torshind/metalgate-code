"""
Dynamic Skills Middleware - routes tool calls to registry skills.
"""

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from metalgate_code.skills.registry import registry


class DynamicToolsMiddleware(AgentMiddleware):
    """Intercepts tool calls and routes dynamically to registry skills."""

    def _get_tool_info(self, request):
        return (
            request.tool_call.get("name"),
            request.tool_call.get("id", ""),
            request.tool_call.get("args", {}),
        )

    def _create_tool_message(self, result, error, tool_call_id, tool_name):
        """Create success or error ToolMessage based on whether error is set."""
        if error:
            return ToolMessage(
                content=f"Error: {error}",
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
            )
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    def _exec_skill(self, request):
        """Execute skill from registry, or return None if not found."""
        tool_name, tool_call_id, args = self._get_tool_info(request)
        skill = registry.get(tool_name)
        if skill is None:
            return None
        try:
            result = skill.invoke(args)
            return self._create_tool_message(result, None, tool_call_id, tool_name)
        except Exception as e:
            return self._create_tool_message(None, e, tool_call_id, tool_name)

    async def _exec_skill_async(self, request):
        """Execute skill from registry async, or return None if not found."""
        tool_name, tool_call_id, args = self._get_tool_info(request)
        skill = registry.get(tool_name)
        if skill is None:
            return None
        try:
            result = await skill.ainvoke(args)
            return self._create_tool_message(result, None, tool_call_id, tool_name)
        except Exception as e:
            return self._create_tool_message(None, e, tool_call_id, tool_name)

    def wrap_tool_call(self, request, handler):
        """Intercept tool call and route to registry if tool not in defaults."""
        msg = self._exec_skill(request)
        return msg if msg is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        """Async version of wrap_tool_call."""
        msg = await self._exec_skill_async(request)
        return msg if msg is not None else await handler(request)
