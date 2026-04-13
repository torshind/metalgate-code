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

        request = request.override(
            tools=list(request.tools or []) + current_skills + current_mcp_tools
        )

        logger.info(f"Request tools: {request.tools}")

        return request

    def wrap_model_call(self, request, handler):
        return handler(self._inject_skills(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._inject_skills(request))
