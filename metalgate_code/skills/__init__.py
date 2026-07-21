"""
Skills package for metalgate_code.
"""

from metalgate_code.skills.meta_skills import (
    create_tool_skill,
    delete_tool_skill,
    list_textual_skills,
    load_textual_skill,
    read_tool_skill,
    reload_textual_skills,
    reload_tool_skills,
)
from metalgate_code.skills.registry import SkillRegistry, registry
from metalgate_code.skills.registry_mcp import registry_mcp
from metalgate_code.skills.skills_mcp import (
    add_mcp_server,
    list_mcp_servers,
    remove_mcp_server,
)
from metalgate_code.skills.textual_skills import TextualSkillRegistry, textual_registry

__all__ = [
    "create_tool_skill",
    "delete_tool_skill",
    "read_tool_skill",
    "reload_tool_skills",
    "list_textual_skills",
    "load_textual_skill",
    "reload_textual_skills",
    "SkillRegistry",
    "registry",
    "registry_mcp",
    "list_mcp_servers",
    "add_mcp_server",
    "remove_mcp_server",
    "TextualSkillRegistry",
    "textual_registry",
]
