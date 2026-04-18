"""
SessionSummaryMiddleware - stores session summaries to Mem0.
"""

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage

from metalgate_code.memory import (
    HEURISTIC_AGENT_ID,
    HEURISTIC_INSTRUCTIONS,
    HISTORICAL_AGENT_ID,
)
from metalgate_code.memory.config import HISTORICAL_INSTRUCTIONS
from metalgate_code.memory.store import MemoryStore

logger = logging.getLogger("metalgate_code")


class CollectorMiddleware(AgentMiddleware):
    """
    Middleware that stores session summaries to Mem0 at session end.

    Writes to two memory scopes:
    - Heuristic: Messages with infer=True (Mem0 extracts facts)
    - Historical: Summary with infer=False (stored as-is)
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
        Store memories to both heuristic and historical scopes.

        Args:
            request: ModelRequest containing messages and state.
        """
        if self._memory is None:
            return

        messages = getattr(request, "messages", []) or []

        if not messages:
            return

        message_dicts = self._convert_messages(messages)

        # Store to heuristic scope with inference
        try:
            await self._memory.store.add(
                message_dicts,
                user_id=self._memory.user_id,
                agent_id=HEURISTIC_AGENT_ID,
                run_id=self._memory.project_id,
                infer=True,
                prompt=HEURISTIC_INSTRUCTIONS,
            )
        except Exception as e:
            # Log but don't fail the session
            logger.error(f"Failed to store heuristic memory: {e}")

        # Store to historical scope
        try:
            await self._memory.store.add(
                message_dicts,
                user_id=self._memory.user_id,
                agent_id=HISTORICAL_AGENT_ID,
                run_id=self._memory.project_id,
                infer=True,
                prompt=HISTORICAL_INSTRUCTIONS,
            )
        except Exception as e:
            # Log but don't fail the session
            logger.error(f"Failed to store historical memory: {e}")

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

        # Call handler first
        result = await handler(request)

        # Store memories after the call
        if self._memory is not None and getattr(request, "messages", None):
            await self._store_memories(request)

        return result
