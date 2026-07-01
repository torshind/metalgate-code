"""
tool_skills_middleware.py
"""

import logging

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
)

from metalgate_code.skills.registry import registry
from metalgate_code.skills.registry_mcp import registry_mcp

logger = logging.getLogger("metalgate_code")


class ToolSkillsMiddleware(AgentMiddleware):
    """Dynamically injects current project tool skills into every model call."""

    def _inject_skills(self, request: ModelRequest) -> ModelRequest:
        current_skills = registry.all()
        if not current_skills:
            current_skills = []
        current_mcp_tools = registry_mcp.all()
        if not current_mcp_tools:
            current_mcp_tools = []

        all_tools = list(request.tools or []) + current_skills + current_mcp_tools
        request = request.override(tools=all_tools)

        tool_names = [getattr(t, "name", str(t)) for t in all_tools]
        logger.info(
            f"ToolSkillsMiddleware injecting {len(current_skills)} skills, {len(current_mcp_tools)} mcp tools. Total tools: {tool_names}"
        )

        logger.debug(f"Request tools: {request.tools}")

        return request

    def _log_result(self, result):
        # result.result is a list of messages, not a single AIMessage
        msgs = getattr(result, "result", None)
        if msgs is not None:
            for msg in msgs:
                if hasattr(msg, "tool_calls") or hasattr(msg, "content"):
                    content = getattr(msg, "content", None)
                    tool_calls = getattr(msg, "tool_calls", None)
                    add_kw = getattr(msg, "additional_kwargs", {})
                    finish = getattr(msg, "response_metadata", {}).get(
                        "finish_reason", "unknown"
                    )
                    logger.debug(
                        f"AIMessage content={content!r} tool_calls={tool_calls} finish_reason={finish!r} additional_kwargs={add_kw!r}"
                    )
        else:
            logger.debug(f"ModelResponse type={type(result).__name__} no .result attr")

    def wrap_model_call(self, request, handler):
        result = handler(self._inject_skills(request))
        if logger.isEnabledFor(logging.DEBUG):
            self._log_result(result)
        return result

    async def awrap_model_call(self, request, handler):
        result = await handler(self._inject_skills(request))
        if logger.isEnabledFor(logging.DEBUG):
            self._log_result(result)
        return result
