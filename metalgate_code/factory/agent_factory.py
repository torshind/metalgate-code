"""
Agent factory for creating Deep Agent instances based on session context.
"""

import logging
import os
from pathlib import Path
from typing import Callable

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    LocalShellBackend,
    StateBackend,
)
from deepagents.backends.protocol import SandboxBackendProtocol
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
    shell_backend: SandboxBackendProtocol | None = None,
) -> CompiledStateGraph:
    """Agent factory based on the given root directory."""
    logger.info("Model: %s", context.model)

    cwd = context.cwd
    # Load project tool skills
    logger.info("Loading tool skills from %s", cwd)
    registry.load(cwd, backend=shell_backend)
    # Load project MCP tools
    logger.info("Loading MCP tools from %s", cwd)
    registry_mcp.load(cwd, backend=shell_backend)

    interrupt_config = get_interrupt_config(context.mode)

    # Load AGENTS.md if it exists and add to system prompt
    agents_md_path = Path(cwd) / ".metalgate" / "AGENTS.md"
    agents_md_content = ""
    file_exists = False
    if shell_backend:
        try:
            result = shell_backend.execute(f"test -f {agents_md_path} && echo 'exists'")
            file_exists = "exists" in result.output
        except Exception:
            pass
    else:
        file_exists = os.path.isfile(agents_md_path)

    if file_exists:
        try:
            if shell_backend:
                result = shell_backend.execute(f"cat {agents_md_path}")
                agents_md_content = result.output
            else:
                with open(agents_md_path, "r", encoding="utf-8") as f:
                    agents_md_content = f.read()
        except (OSError, IOError):
            pass  # Silently ignore file read errors

    ephemeral_backend = StateBackend()
    shell_env = os.environ.copy()

    if not shell_backend:
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
            LocalContextMiddleware(backend=backend),
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


def create_agent() -> Callable[
    [AgentSessionContext, SandboxBackendProtocol | None], CompiledStateGraph
]:
    """Create a factory function that accepts (context, backend) and returns a compiled agent."""

    def factory(
        context: AgentSessionContext,
        shell_backend: SandboxBackendProtocol | None = None,
    ) -> CompiledStateGraph:
        return _build_agent(context, shell_backend=shell_backend)

    return factory
