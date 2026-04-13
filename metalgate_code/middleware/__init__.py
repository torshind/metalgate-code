"""
Middleware for the agent.
"""

from metalgate_code.middleware.dynamic_tools import DynamicToolsMiddleware
from metalgate_code.middleware.tool_skills import ToolSkillsMiddleware

__all__ = ["DynamicToolsMiddleware", "ToolSkillsMiddleware"]
