"""
ProjectContextMiddleware - Injects relevant memories into system message at session start.
"""

import asyncio
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from metalgate_code.memory.config import (
    DEFAULT_HISTORICAL_LIMIT,
    HEURISTIC_AGENT_ID,
    HISTORICAL_AGENT_ID,
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
        Search both heuristic and historical memories in parallel.

        Args:
            query: The search query (first user message).
            user_id: The user identifier.
            project: The project name.

        Returns:
            Tuple of (heuristic_results, historical_results).
        """
        if self._memory is None:
            return [], []

        heuristic_task = None
        if self._is_first_message(request):
            heuristic_task = self._memory.store.get_all(
                user_id=self._memory.user_id,
                agent_id=HEURISTIC_AGENT_ID,
                run_id=self._memory.project_id,
            )

        historical_task = self._memory.store.search(
            query=self._get_latest_message(request),
            user_id=self._memory.user_id,
            agent_id=HISTORICAL_AGENT_ID,
            run_id=self._memory.project_id,
            limit=DEFAULT_HISTORICAL_LIMIT,
        )

        if heuristic_task:
            heuristic, historical = await asyncio.gather(
                heuristic_task, historical_task, return_exceptions=False
            )
        else:
            heuristic = {}
            historical = await historical_task

        for name, result in [("heuristic", heuristic), ("historical", historical)]:
            logger.info(f"{name}: type={type(result)}, value={result!r}")

        # Extract results from response dicts
        heuristic_results = heuristic.get("results", [])
        historical_results = historical.get("results", [])

        return heuristic_results, historical_results

    def _format_memories(
        self,
        heuristic_results: list[dict[str, Any]],
        historical_results: list[dict[str, Any]],
    ) -> str:
        """
        Format memory results for injection into system message.

        Args:
            heuristic_results: List of heuristic memory results.
            historical_results: List of historical memory results.

        Returns:
            Formatted memory context string.
        """
        parts = ["## Relevant Context\n"]

        if heuristic_results:
            parts.append("### Memories")
            for result in heuristic_results:
                memory_text = result.get("memory", "")
                if memory_text:
                    parts.append(f"- {memory_text}")
            parts.append("")

        if historical_results:
            parts.append("### Related Conversations")
            for result in historical_results:
                if id := result.get("id") not in self._injection_cache:
                    self._injection_cache.add(id)
                    memory_text = result.get("memory", "")
                    if memory_text:
                        parts.append(f"- {memory_text}")
            parts.append("")

        return "\n".join(parts)

    async def awrap_model_call(self, request: ModelRequest, handler) -> Any:
        """Async override to inject memories before model call."""
        logger.debug(f"Searching memories for request: {request}")

        # Search memories
        heuristic_results, historical_results = await self._collect_memories(request)

        # If no memories found, skip injection
        if not heuristic_results and not historical_results:
            return await handler(request)

        # Format and inject context into system message
        context = self._format_memories(heuristic_results, historical_results)

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
