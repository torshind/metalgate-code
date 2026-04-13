"""
Agent factory for creating Deep Agent instances based on session context.
"""

import logging
import os
from functools import partial

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from deepagents_acp.server import AgentSessionContext
from deepagents_cli.local_context import LocalContextMiddleware
from langgraph.graph.state import Checkpointer, CompiledStateGraph

from metalgate_code.config import get_interrupt_config
from metalgate_code.middleware.dynamic_tools import DynamicToolsMiddleware
from metalgate_code.middleware.tool_skills import ToolSkillsMiddleware
from metalgate_code.models import create_chat_model
from metalgate_code.skills import (
    create_tool_skill,
    delete_tool_skill,
    read_tool_skill,
    registry,
)
from metalgate_code.skills.registry_mcp import registry_mcp

logger = logging.getLogger("metalgate_code")

META_SKILLS = [
    create_tool_skill,
    delete_tool_skill,
    read_tool_skill,
]


def _build_agent(
    context: AgentSessionContext,
    checkpointer: Checkpointer,
) -> CompiledStateGraph:
    """Agent factory based on the given root directory."""
    logger.info("Model: %s", context.model)

    _root_dir = context.cwd
    # Load project tool skills
    logger.info("Loading tool skills from %s", _root_dir)
    registry.load(_root_dir)
    # Load project MCP tools
    logger.info("Loading MCP tools from %s", _root_dir)
    registry_mcp.load(_root_dir)

    interrupt_config = get_interrupt_config(context.mode)

    # Load AGENTS.md if it exists and add to system prompt
    agents_md_path = os.path.join(_root_dir, "AGENTS.md")
    agents_md_content = ""
    if os.path.isfile(agents_md_path):
        try:
            with open(agents_md_path, "r", encoding="utf-8") as f:
                agents_md_content = f.read()
        except (OSError, IOError):
            pass  # Silently ignore file read errors

    ephemeral_backend = StateBackend()
    shell_env = os.environ.copy()

    # Use LocalShellBackend for filesystem + shell execution.
    # Provides `execute` tool via FilesystemMiddleware with per-command
    # timeout support.
    shell_backend = LocalShellBackend(
        root_dir=_root_dir,
        inherit_env=True,
        env=shell_env,
    )
    backend = CompositeBackend(
        default=shell_backend,
        routes={
            "/memories/": ephemeral_backend,
            "/conversation_history/": ephemeral_backend,
        },
    )

    model = create_chat_model(context.model)

    # Build system prompt
    system_prompt_parts = []

    if agents_md_content:
        system_prompt_parts.append(agents_md_content)

    system_prompt_parts.append(
        "---"
        "\n"
        "## Tool Usage"
        "\n"
        "CRITICAL: Before using `execute`, check if a project-specific tool skill applies. "
        "If a tool skill exists for the operation, use it instead of `execute`. "
        "Tool skills are optimized for this project and handle environment/configuration automatically."
    )

    system_prompt = "\n".join(system_prompt_parts)

    return create_deep_agent(
        # Falls back to Deep Agent default model if not provided
        model=model,
        checkpointer=checkpointer,
        backend=backend,
        interrupt_on=interrupt_config,
        middleware=[
            LocalContextMiddleware(backend=backend),
            ToolSkillsMiddleware(),
            DynamicToolsMiddleware(),
        ],
        tools=META_SKILLS,
        system_prompt=system_prompt,
    )


def create_agent(
    checkpointer: Checkpointer,
) -> partial:
    """Create a partial _build_agent function with the checkpointer pre-bound."""
    return partial(_build_agent, checkpointer=checkpointer)
