"""
Agent factory for creating Deep Agent instances based on session context.
"""

import logging
import os
from functools import partial
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import (
    BackendProtocol,
    CompositeBackend,
    LocalShellBackend,
    StateBackend,
)
from deepagents_acp.server import AgentSessionContext
from deepagents_cli.local_context import LocalContextMiddleware
from langgraph.graph.state import CompiledStateGraph

from metalgate_code.config import get_interrupt_config
from metalgate_code.context.indexer import IndexStore
from metalgate_code.memory.store import MemoryStore
from metalgate_code.middleware import (
    CollectorMiddleware,
    DynamicToolsMiddleware,
    RecollectorMiddleware,
    ToolSkillsMiddleware,
)
from metalgate_code.models import create_chat_model
from metalgate_code.skills import (
    create_tool_skill,
    delete_tool_skill,
    read_tool_skill,
    registry,
    reload_tool_skills,
)
from metalgate_code.skills.registry_mcp import registry_mcp

logger = logging.getLogger("metalgate_code")


def _is_memory_enabled() -> bool:
    """Check if memory is enabled via environment variable."""
    return os.environ.get("MEMORY", "").lower() in ("true", "1", "yes", "on", "enabled")


def _get_userid() -> str:
    """Get the user ID from environment variable.

    Returns:
        The user ID if set in the USER environment variable, otherwise None.
    """
    return os.environ.get("USER", "anonymous")


META_SKILLS = [
    create_tool_skill,
    delete_tool_skill,
    read_tool_skill,
    reload_tool_skills,
]


def _build_agent(
    context: AgentSessionContext,
    backend: BackendProtocol | None = None,
) -> CompiledStateGraph:
    """Agent factory based on the given root directory."""
    logger.info("Model: %s", context.model)

    cwd = context.cwd
    # Load project tool skills
    logger.info("Loading tool skills from %s", cwd)
    registry.load(cwd)
    # Load project MCP tools
    logger.info("Loading MCP tools from %s", cwd)
    registry_mcp.load(cwd)

    interrupt_config = get_interrupt_config(context.mode)

    # Load AGENTS.md if it exists and add to system prompt
    agents_md_path = Path(cwd) / ".metalgate" / "AGENTS.md"
    agents_md_content = ""
    if os.path.isfile(agents_md_path):
        try:
            with open(agents_md_path, "r", encoding="utf-8") as f:
                agents_md_content = f.read()
        except (OSError, IOError):
            pass  # Silently ignore file read errors

    ephemeral_backend = StateBackend()
    shell_env = os.environ.copy()

    if not backend:
        # Use LocalShellBackend for filesystem + shell execution.
        # Provides `execute` tool via FilesystemMiddleware with per-command
        # timeout support.
        shell_backend = LocalShellBackend(
            root_dir=cwd,
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
        "CRITICAL: Before using `execute`, check if an available tool skill applies. "
        "If a tool skill exists for the operation, use it instead of `execute`. "
        "Tool skills are optimized for this project and handle environment/configuration automatically."
    )

    system_prompt = "\n".join(system_prompt_parts)

    # Initialize memory if enabled
    memory = None
    if _is_memory_enabled():
        try:
            user_id = _get_userid()
            memory = MemoryStore(cwd=cwd, user_id=user_id)
            logger.info(f"Memory enabled for project: {cwd}, user_id: {user_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize memory: {e}")

    index_store = IndexStore(cwd)

    return create_deep_agent(
        # Falls back to Deep Agent default model if not provided
        model=model,
        backend=backend,
        interrupt_on=interrupt_config,
        middleware=[
            LocalContextMiddleware(backend=backend),  # type: ignore
            RecollectorMiddleware(memory=memory),
            ToolSkillsMiddleware(),
            DynamicToolsMiddleware(),
            CollectorMiddleware(memory=memory),
        ],
        tools=META_SKILLS
        + [
            index_store.package_context,
            index_store.module_context,
            index_store.symbol_context,
        ],
        system_prompt=system_prompt,
    )


def create_agent() -> partial:
    """Create a partial _build_agent function with project-specific checkpointer."""
    return partial(_build_agent)
