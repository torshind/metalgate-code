"""
Middleware for the agent.
"""

from metalgate_code.middleware.collector import CollectorMiddleware
from metalgate_code.middleware.dynamic_tools import DynamicToolsMiddleware
from metalgate_code.middleware.recollector import RecollectorMiddleware
from metalgate_code.middleware.tool_skills import ToolSkillsMiddleware

__all__ = [
    "CollectorMiddleware",
    "DynamicToolsMiddleware",
    "RecollectorMiddleware",
    "ToolSkillsMiddleware",
]
