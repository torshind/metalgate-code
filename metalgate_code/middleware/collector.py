"""
SessionSummaryMiddleware - stores session summaries to Mem0.
"""

import asyncio
import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    StateT,
)
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime

from metalgate_code.memory import (
    EPISODIC_AGENT_ID,
    EPISODIC_INSTRUCTIONS,
    SEMANTIC_AGENT_ID,
    SEMANTIC_INSTRUCTIONS,
)
from metalgate_code.memory.store import MemoryStore

logger = logging.getLogger("metalgate_code")


class CollectorMiddleware(AgentMiddleware):
    """
    Middleware that stores session summaries to Mem0 at session end.

    Writes to two memory scopes:
    - Semantic: General facts, conventions, preferences (Mem0 extracts facts)
    - Episodic: Session summaries and specific experiences
    """

    def __init__(
        self,
        memory: MemoryStore | None = None,
    ):
        """
        Initialize the middleware.

        Args:
            memory: AsyncMemory instance or None if memory is disabled.
        """
        self._memory = memory
        self._save_task = None
        self._saved_message_count = 0

    def _detect_outcome(self, request: ModelRequest) -> str:
        """
        Detect session outcome from messages.

        Args:
            request: ModelRequest containing messages.

        Returns:
            Outcome string: "success", "error", or "incomplete".
        """
        messages = getattr(request, "messages", []) or []

        # Check for error messages
        for msg in messages:
            if isinstance(msg, dict):
                status = msg.get("status")
                content = str(msg.get("content", "")).lower()
            else:
                status = getattr(msg, "status", None)
                content = str(getattr(msg, "content", "")).lower()

            if status == "error" or "error" in content:
                return "error"

        # Check if session seems complete
        if messages and len(messages) >= 2:
            return "success"

        return "incomplete"

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert messages to a list of strings for summarization."""
        # Convert messages to dict format for Mem0
        message_dicts = []
        for msg in messages:
            if isinstance(msg, dict):
                message_dicts.append(msg)
            else:
                role = getattr(msg, "type", "unknown")
                # Map langchain roles to Mem0 roles
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"

                message_dicts.append(
                    {
                        "role": role,
                        "content": str(getattr(msg, "content", "")),
                    }
                )
        return message_dicts

    async def _store_memories(
        self,
        request: ModelRequest,
    ) -> None:
        """
        Store memories to both semantic and episodic scopes.

        Args:
            request: ModelRequest containing messages and state.
        """
        if self._memory is None:
            return

        messages = getattr(request, "messages", []) or []
        new_messages = messages[self._saved_message_count :]
        self._saved_message_count = len(messages)

        if not new_messages:
            return

        message_dicts = self._convert_messages(new_messages)
        logger.info(f"Storing {len(message_dicts)} messages")

        # Store to semantic scope with inference
        try:
            await self._memory.add(
                messages=message_dicts,
                agent_id=SEMANTIC_AGENT_ID,
                prompt=SEMANTIC_INSTRUCTIONS,
            )
        except Exception as e:
            # Log but don't fail the session
            logger.error(f"Failed to store semantic memory: {e}")

        # Store to episodic scope
        try:
            await self._memory.add(
                messages=message_dicts,
                agent_id=EPISODIC_AGENT_ID,
                prompt=EPISODIC_INSTRUCTIONS,
            )
        except Exception as e:
            # Log but don't fail the session
            logger.error(f"Failed to store episodic memory: {e}")

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        if self._save_task is not None:
            await self._save_task
            self._save_task = None

    def wrap_model_call(self, request: ModelRequest, handler) -> Any:
        """Sync override - raises since we require async."""
        raise NotImplementedError("SessionSummaryMiddleware requires async execution")

    async def awrap_model_call(self, request: ModelRequest, handler) -> Any:
        """
        Async override - stores memories after model call.

        Args:
            request: ModelRequest containing messages and state.
            handler: Handler function to call.

        Returns:
            Result from handler.
        """
        logger.debug(f"Storing memories for request: {request}")
        # Wait for previous session's save to complete before proceeding
        if self._save_task is not None:
            await self._save_task
            self._save_task = None
        # Start current session's save in background
        if self._memory is not None and getattr(request, "messages", None):
            self._save_task = asyncio.create_task(self._store_memories(request))
        # Call handler
        return await handler(request)
