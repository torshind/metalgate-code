"""
ACP Server factory for MetalGate Code agent.
"""

import logging
import os
from typing import Any

from acp.helpers import (
    start_tool_call,
    text_block,
    update_agent_message,
    update_tool_call,
    update_user_message,
)
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    PromptCapabilities,
    PromptResponse,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SseMcpServer,
    TextContentBlock,
)
from deepagents_acp.server import AgentServerACP
from deepagents_cli.sessions import _patch_aiosqlite
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from metalgate_code.memory import get_db_path

logger = logging.getLogger("metalgate_code")


class MetalGateACP(AgentServerACP):
    """Custom ACP server with AsyncSqliteSaver checkpointer."""

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Return server capabilities including session support."""
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    image=True,
                ),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
        )

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List available sessions from the SQLite checkpointer database."""

        _patch_aiosqlite()

        # Use provided cwd or current working directory
        target_cwd = cwd or self._cwd or os.getcwd()
        db_path = get_db_path(target_cwd)

        sessions: list[SessionInfo] = []

        if db_path.exists():
            async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
                conn = checkpointer.conn
                # Query threads with metadata - deepagents stores metadata as JSON
                async with conn.execute(
                    """SELECT thread_id,
                           MAX(json_extract(metadata, '$.title')) as title,
                           MAX(json_extract(metadata, '$.updated_at')) as updated_at
                    FROM checkpoints
                    GROUP BY thread_id
                    ORDER BY updated_at DESC NULLS LAST, thread_id DESC""",
                ) as cursor_obj:
                    rows = await cursor_obj.fetchall()
                    for row in rows:
                        thread_id, title, updated_at = row
                        display_title = title or f"Session {thread_id[:8]}"
                        sessions.append(
                            SessionInfo(
                                session_id=thread_id,
                                cwd=target_cwd,
                                title=display_title,
                                updated_at=updated_at,
                            )
                        )

        logger.info(f"List sessions: {len(sessions)} found in {target_cwd}")

        return ListSessionsResponse(sessions=sessions, next_cursor=None)

    async def _session_exists(self, session_id: str, cwd: str) -> bool:
        """Check if a session exists in the checkpointer database."""

        _patch_aiosqlite()
        db_path = get_db_path(cwd)

        if not db_path.exists():
            return False

        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            conn = checkpointer.conn
            async with conn.execute(
                "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1",
                (session_id,),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def _setup_session(
        self,
        session_id: str,
        cwd: str,
    ) -> tuple[Any | None, Any | None]:
        """Common session setup logic for load and resume operations.

        Verifies the session exists, stores context, initializes state,
        and replays chat history.

        Returns:
            Tuple of (modes, config_options) to use in the response.
        """
        # Verify session exists
        exists = await self._session_exists(session_id, cwd)
        if not exists:
            logger.warning(
                "Session %s not found in database for cwd %s", session_id, cwd
            )

        # Store session context - cwd comes from client
        self._session_cwds[session_id] = cwd
        self._cwd = cwd

        # Initialize session state
        if self._modes:
            self._session_modes[session_id] = self._modes.current_mode_id
            self._session_mode_states[session_id] = self._modes

        if self._models:
            self._session_models[session_id] = self._models[0]["value"]

        config_options = None
        if self._modes or self._models:
            config_options = self._build_config_options(session_id)

        # Replay chat history after returning the response
        await self._replay_chat_history(session_id, cwd)

        return self._modes, config_options

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        """Load an existing session with the given ID.

        Verifies the session exists in the checkpointer before returning.
        The cwd is provided by the client and used as-is.
        """
        logger.info("Loading session %s for cwd %s", session_id, cwd)

        modes, config_options = await self._setup_session(session_id, cwd)

        return LoadSessionResponse(
            modes=modes,
            config_options=config_options,
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """Resume an existing session with the given ID.

        Verifies the session exists in the checkpointer before returning.
        The cwd is provided by the client and used as-is.
        """
        logger.info("Resuming session %s for cwd %s", session_id, cwd)

        modes, config_options = await self._setup_session(session_id, cwd)

        return ResumeSessionResponse(
            modes=modes,
            config_options=config_options,
        )

    async def _replay_chat_history(self, session_id: str, cwd: str) -> None:
        """Replay chat history by sending session_update notifications."""

        _patch_aiosqlite()

        db_path = get_db_path(cwd)
        if not db_path.exists():
            return

        try:
            async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
                # Get the checkpoint tuple for this thread
                config: RunnableConfig = {"configurable": {"thread_id": session_id}}  # type: ignore[dict-item]
                tuple_result = await checkpointer.aget_tuple(config)

                if tuple_result is None or tuple_result.checkpoint is None:
                    return

                # Get messages from the checkpoint state
                checkpoint = tuple_result.checkpoint
                channel_values = checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])

                logger.info(
                    "Replaying %d messages for session %s", len(messages), session_id
                )

                # Replay each message as a session update
                for msg in messages:
                    await self._send_message_chunk(session_id, msg)

        except Exception as e:
            logger.warning("Error replaying chat history: %s", e)

    def _extract_text_from_content(self, content: Any) -> str | None:
        """Extract text from content which may be a string or list of blocks."""
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts) if text_parts else None
        if content:
            return str(content)
        return None

    async def _send_message_chunk(self, session_id: str, msg: Any) -> None:
        """Send a single message as a session update notification."""
        try:
            logger.debug(f"Sending message chunk type: {type(msg)}")
            if isinstance(msg, HumanMessage):
                await self._send_human_message(session_id, msg)
            elif isinstance(msg, AIMessage):
                await self._send_ai_message(session_id, msg)
            elif isinstance(msg, ToolMessage):
                await self._send_tool_message(session_id, msg)
        except Exception as e:
            logger.warning("Error sending message chunk: %s", e)

    async def _send_human_message(self, session_id: str, msg: HumanMessage) -> None:
        text = self._extract_text_from_content(msg.content)
        if text:
            await self._conn.session_update(
                session_id=session_id,
                update=update_user_message(text_block(text)),
            )

    async def _send_ai_message(self, session_id: str, msg: AIMessage) -> None:
        text = self._extract_text_from_content(msg.content)

        if text:
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(text)),
            )

        for tc in msg.tool_calls:
            call_id = tc.get("id") or ""
            await self._conn.session_update(
                session_id=session_id,
                update=start_tool_call(
                    title=f"Using {tc.get('name') or 'tool'}",
                    tool_call_id=call_id,
                    status="in_progress",
                    kind=None,
                    raw_input=tc.get("args") or None,
                ),
            )

    async def _send_tool_message(self, session_id: str, msg: ToolMessage) -> None:
        await self._conn.session_update(
            session_id=session_id,
            update=update_tool_call(
                tool_call_id=msg.tool_call_id or "",
                status="completed",
                raw_output=self._extract_text_from_content(msg.content) or None,
            ),
        )

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """Process a user prompt with AsyncSqliteSaver checkpointer."""

        _patch_aiosqlite()

        if self._agent is None:
            self._reset_agent(session_id)

        if self._agent is None:
            msg = "Agent initialization failed"
            raise RuntimeError(msg)

        # Use the cwd associated with this session (stored at load/resume time)
        # Fall back to current cwd if session not found
        session_cwd = self._session_cwds.get(session_id, self._cwd)
        db_path = get_db_path(session_cwd)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Opening AsyncSqliteSaver checkpointer at %s for session %s",
            db_path,
            session_id,
        )

        # from_conn_string is an async context manager
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            self._agent.checkpointer = checkpointer

            result = await super().prompt(prompt, session_id, message_id, **kwargs)

        # Note: checkpointer is closed when exiting the context
        self._agent.checkpointer = None
        return result
