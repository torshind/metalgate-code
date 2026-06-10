"""
Chat history replayer that sends ACP session_update notifications.
"""

import asyncio
import logging
from typing import Any

from acp.helpers import (
    start_tool_call,
    text_block,
    update_agent_message,
    update_tool_call,
    update_user_message,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from metalgate_code.memory.session_store import _extract_text_from_content

logger = logging.getLogger("metalgate_code")


class ChatHistoryReplayer:
    """Replays chat history by sending ACP session_update notifications."""

    async def replay(self, conn, session_id: str, messages: list[Any]) -> None:
        """Replay chat history by sending session_update notifications in batches."""
        logger.info(f"Replaying {len(messages)} messages for session {session_id}")
        for i, msg in enumerate(messages):
            await self._send_message_chunk(conn, session_id, msg)
            # Yield control every 10 messages to avoid blocking the event loop
            if (i + 1) % 10 == 0:
                await asyncio.sleep(0)

    async def _send_message_chunk(self, conn, session_id: str, msg: Any) -> None:
        """Send a single message as a session update notification."""
        try:
            logger.debug(f"Sending message chunk type: {type(msg)}")
            if isinstance(msg, HumanMessage):
                await self._send_human_message(conn, session_id, msg)
            elif isinstance(msg, AIMessage):
                await self._send_ai_message(conn, session_id, msg)
            elif isinstance(msg, ToolMessage):
                await self._send_tool_message(conn, session_id, msg)
        except Exception as e:
            logger.warning(f"Error sending message chunk: {e}")

    async def _send_human_message(
        self, conn, session_id: str, msg: HumanMessage
    ) -> None:
        text = _extract_text_from_content(msg.content)
        if text:
            await conn.session_update(
                session_id=session_id,
                update=update_user_message(text_block(text)),
            )

    async def _send_ai_message(self, conn, session_id: str, msg: AIMessage) -> None:
        text = _extract_text_from_content(msg.content)

        if text:
            await conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(text)),
            )
        elif msg.content:
            # Log when non-text content is dropped so we don't silently lose data
            logger.debug(
                f"Dropping non-text AIMessage content for session {session_id}: {type(msg.content)}"
            )

        for tc in msg.tool_calls:
            call_id = tc.get("id") or ""
            await conn.session_update(
                session_id=session_id,
                update=start_tool_call(
                    title=f"Using {tc.get('name') or 'tool'}",
                    tool_call_id=call_id,
                    status="in_progress",
                    kind=None,
                    raw_input=tc.get("args") or None,
                ),
            )

    async def _send_tool_message(self, conn, session_id: str, msg: ToolMessage) -> None:
        await conn.session_update(
            session_id=session_id,
            update=update_tool_call(
                tool_call_id=msg.tool_call_id or "",
                status="completed",
                raw_output=_extract_text_from_content(msg.content) or None,
            ),
        )
