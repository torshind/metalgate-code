"""
ProjectContextMiddleware - Injects relevant memories into system message at session start.
"""

import asyncio
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from metalgate_code.memory.config import (
    DEFAULT_EPISODIC_LIMIT,
    EPISODIC_AGENT_ID,
    SEMANTIC_AGENT_ID,
)
from metalgate_code.memory.store import MemoryStore

logger = logging.getLogger("metalgate_code")


class RecollectorMiddleware(AgentMiddleware):
    """
    Middleware that queries Mem0 at session start and injects relevant
    memories into the system message.
    """

    def __init__(
        self,
        memory: MemoryStore | None = None,
    ):
        """
        Initialize the middleware.

        Args:
            memory: AsyncMemory instance for searching memories.
                    If None, memory retrieval is skipped.
            project: Project name for scoping memories.
        """
        self._memory = memory
        self._injection_cache = set()

    def _is_first_message(self, request: ModelRequest) -> bool:
        """Check if this is the first message in the conversation."""
        # Check if messages only contains a single user message
        user_messages = [m for m in request.messages if m.type == "human"]
        return len(user_messages) == 1

    def _get_latest_message(self, request: ModelRequest) -> str:
        """Extract the latest user message from the request."""
        user_messages = [m for m in request.messages if m.type == "human"]
        return str(user_messages[-1].content) if user_messages else ""

    async def _collect_memories(
        self,
        request: ModelRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Search both semantic and episodic memories in parallel.

        Args:
            query: The search query (first user message).
            user_id: The user identifier.
            project: The project name.

        Returns:
            Tuple of (semantic_results, episodic_results).
        """
        if self._memory is None:
            return [], []

        semantic_task = None
        if self._is_first_message(request):
            semantic_task = self._memory.get_all(agent_id=SEMANTIC_AGENT_ID)

        episodic_task = self._memory.search(
            query=self._get_latest_message(request),
            agent_id=EPISODIC_AGENT_ID,
            limit=DEFAULT_EPISODIC_LIMIT,
        )

        if semantic_task:
            semantic, episodic = await asyncio.gather(
                semantic_task, episodic_task, return_exceptions=False
            )
        else:
            semantic = {}
            episodic = await episodic_task

        for name, result in [("semantic", semantic), ("episodic", episodic)]:
            logger.info(f"{name}: type={type(result)}, value={result!r}")

        # Extract results from response dicts
        semantic_results = semantic.get("results", [])
        episodic_results = episodic.get("results", [])

        return semantic_results, episodic_results

    def _format_memories(
        self,
        semantic_results: list[dict[str, Any]],
        episodic_results: list[dict[str, Any]],
    ) -> str:
        """
        Format memory results for injection into system message.

        Args:
            semantic_results: List of semantic memory results.
            episodic_results: List of episodic memory results.

        Returns:
            Formatted memory context string.
        """
        parts = ["## Relevant Context\n"]

        if semantic_results:
            parts.append("### Memories")
            for result in semantic_results:
                memory_text = result.get("memory", "")
                if memory_text:
                    parts.append(f"- {memory_text}")
            parts.append("")

        if episodic_results:
            parts.append("### Related Conversations")
            for result in episodic_results:
                if (id := result.get("id")) not in self._injection_cache:
                    self._injection_cache.add(id)
                    memory_text = result.get("memory", "")
                    if memory_text:
                        parts.append(f"- {memory_text}")
            parts.append("")

        logger.debug(f"Formatted memories: {parts}")

        return "\n".join(parts)

    async def awrap_model_call(self, request: ModelRequest, handler) -> Any:
        """Async override to inject memories before model call."""
        logger.debug(f"Searching memories for request: {request}")

        # Search memories
        semantic_results, episodic_results = await self._collect_memories(request)

        # If no memories found, skip injection
        if not semantic_results and not episodic_results:
            return await handler(request)

        # Format and inject context into system message
        context = self._format_memories(semantic_results, episodic_results)

        # Build new system message with context prepended
        existing_system_message = request.system_message or SystemMessage(content="")
        new_system_message_content = (
            f"{existing_system_message.content}\n\n{context}".strip()
        )

        new_system_message = SystemMessage(content=new_system_message_content)
        request = request.override(system_message=new_system_message)

        return await handler(request)

    def wrap_model_call(self, request: ModelRequest, handler) -> Any:
        """Sync override - raises since we require async."""
        raise NotImplementedError("ProjectContextMiddleware requires async execution")
