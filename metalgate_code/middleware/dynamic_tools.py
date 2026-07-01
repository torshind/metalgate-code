"""
Dynamic Skills Middleware - routes tool calls to registry skills.

Before invoking a skill, the middleware publishes the sandbox backend
into a context variable so skills can call ``get_backend()`` to execute
shell commands and file operations inside the agent's sandbox.
"""

import logging

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from metalgate_code.context.backend_context import reset_backend, set_backend
from metalgate_code.skills.registry import registry

logger = logging.getLogger("metalgate_code")


class DynamicToolsMiddleware(AgentMiddleware):
    """Intercepts tool calls and routes dynamically to registry skills.

    Args:
        backend: Sandbox backend to expose to skills via ``get_backend()``.
            When set, skills can call ``get_backend()`` to run shell commands
            and file operations inside the sandbox.
    """

    def __init__(
        self,
        backend: SandboxBackendProtocol | None = None,
    ) -> None:
        self._backend = backend

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
        token = set_backend(self._backend)
        try:
            result = skill.invoke(args)
            return self._create_tool_message(result, None, tool_call_id, tool_name)
        except Exception as e:
            return self._create_tool_message(None, e, tool_call_id, tool_name)
        finally:
            reset_backend(token)

    async def _exec_skill_async(self, request):
        """Execute skill from registry async, or return None if not found."""
        tool_name, tool_call_id, args = self._get_tool_info(request)
        skill = registry.get(tool_name)
        if skill is None:
            return None
        token = set_backend(self._backend)
        try:
            result = await skill.ainvoke(args)
            return self._create_tool_message(result, None, tool_call_id, tool_name)
        except Exception as e:
            return self._create_tool_message(None, e, tool_call_id, tool_name)
        finally:
            reset_backend(token)

    def wrap_tool_call(self, request, handler):
        """Intercept tool call and route to registry if tool not in defaults."""
        tool_name, tool_call_id, args = self._get_tool_info(request)
        logger.info(
            f"DynamicToolsMiddleware.wrap_tool_call: name={tool_name} id={tool_call_id} args={args}"
        )
        msg = self._exec_skill(request)
        if msg is not None:
            logger.info(f"DynamicToolsMiddleware: routed {tool_name} to registry skill")
            return msg
        logger.info(
            f"DynamicToolsMiddleware: {tool_name} not in registry, passing to handler"
        )
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """Async version of wrap_tool_call."""
        tool_name, tool_call_id, args = self._get_tool_info(request)
        logger.info(
            f"DynamicToolsMiddleware.awrap_tool_call: name={tool_name} id={tool_call_id} args={args}"
        )
        msg = await self._exec_skill_async(request)
        if msg is not None:
            logger.info(f"DynamicToolsMiddleware: routed {tool_name} to registry skill")
            return msg
        logger.info(
            f"DynamicToolsMiddleware: {tool_name} not in registry, passing to handler"
        )
        return await handler(request)
