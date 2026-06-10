"""
ACP Server factory for MetalGate Code agent.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from acp.helpers import (
    start_tool_call,
    text_block,
    update_agent_message,
    update_tool_call,
    update_user_message,
)
from acp.schema import (
    AgentCapabilities,
    CloseSessionResponse,
    HttpMcpServer,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PromptCapabilities,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SseMcpServer,
    TextContentBlock,
)
from aiosqlite import connect
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents_acp.server import AgentServerACP, AgentSessionContext
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from metalgate_code.helpers import get_checkpoints_data_dir

logger = logging.getLogger("metalgate_code")


def _message_to_dict(msg: Any) -> dict:
    """Serialize a LangChain message to a plain dict."""
    data: dict = {"type": msg.type, "content": msg.content}
    if getattr(msg, "id", None):
        data["id"] = msg.id
    if isinstance(msg, AIMessage):
        data["tool_calls"] = msg.tool_calls
    elif isinstance(msg, ToolMessage):
        data["tool_call_id"] = msg.tool_call_id
    return data


def _messages_from_dict(data: list[dict]) -> list[Any]:
    """Reconstruct LangChain messages from plain dicts."""
    messages = []
    for item in data:
        msg_type = item.get("type")
        kwargs: dict = {"content": item.get("content", "")}
        if "id" in item:
            kwargs["id"] = item["id"]
        if msg_type == "human":
            messages.append(HumanMessage(**kwargs))
        elif msg_type == "ai":
            kwargs["tool_calls"] = item.get("tool_calls", [])
            messages.append(AIMessage(**kwargs))
        elif msg_type == "tool":
            kwargs["tool_call_id"] = item.get("tool_call_id", "")
            messages.append(ToolMessage(**kwargs))
        else:
            logger.warning(f"Unknown message type '{msg_type}' during deserialization")
    return messages


class MetalGateACP(AgentServerACP):
    """Custom ACP server with manual SQLite message persistence."""

    def __init__(
        self,
        agent_factory: Callable[
            [AgentSessionContext, SandboxBackendProtocol | None], CompiledStateGraph
        ],
        backend_factory: Callable[[str], SandboxBackendProtocol],
        modes: Any,
        models: Any,
    ) -> None:
        """Initialize with agent factory and backend factory.

        Args:
            agent_factory: Function that takes (context, backend) and returns CompiledStateGraph
            backend_factory: Function that takes cwd and returns SandboxBackendProtocol
            modes: Session modes configuration
            models: Available models list
        """
        self._user_agent_factory = agent_factory
        self._backend_factory = backend_factory
        # Pass wrapper as the agent to the base class.
        # The base class will use MemorySaver automatically when checkpointer is None.
        super().__init__(agent=self._create_agent, modes=modes, models=models)

    def _create_agent(self, context: AgentSessionContext) -> CompiledStateGraph:
        """Called by base class to create the agent for a session."""
        self._shell_backend = self._backend_factory(context.cwd)
        return self._user_agent_factory(context, self._shell_backend)

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

    async def _init_db(self, cwd: str) -> None:
        """Initialize the SQLite database with the sessions and messages tables."""
        db_path = get_checkpoints_data_dir(cwd)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with connect(str(db_path)) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT,
                    updated_at TEXT
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON messages(thread_id, created_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at)
                """
            )
            await db.commit()

    async def _save_messages(
        self, cwd: str, session_id: str, messages: list[Any]
    ) -> None:
        """Serialize and save messages to SQLite.

        Replaces any existing messages for the session to avoid duplicates.
        """
        # Pre-serialize all messages before touching the DB
        rows = [(session_id, json.dumps(_message_to_dict(msg))) for msg in messages]
        # Derive title from the first human message
        title = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = self._extract_text_from_content(msg.content)
                if text:
                    title = text[:100]
                    break

        await self._init_db(cwd)
        db_path = get_checkpoints_data_dir(cwd)
        async with connect(str(db_path)) as db:
            await db.execute("BEGIN")
            try:
                await db.execute(
                    "INSERT OR REPLACE INTO sessions (thread_id, title, updated_at) VALUES (?, ?, ?)",
                    (session_id, title, datetime.now(timezone.utc).isoformat()),
                )
                await db.execute(
                    "DELETE FROM messages WHERE thread_id = ?",
                    (session_id,),
                )
                if rows:
                    await db.executemany(
                        "INSERT INTO messages (thread_id, data) VALUES (?, ?)",
                        rows,
                    )
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def _load_messages(self, cwd: str, session_id: str) -> list[Any]:
        """Load and deserialize messages from SQLite."""
        db_path = get_checkpoints_data_dir(cwd)
        if not db_path.exists():
            return []
        async with connect(str(db_path)) as db:
            async with db.execute(
                "SELECT data FROM messages WHERE thread_id = ? ORDER BY id",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return _messages_from_dict([json.loads(row[0]) for row in rows])

    @staticmethod
    def _resolve_resource_uri(block: ResourceContentBlock, root_dir: str) -> str:
        """Convert a ResourceContentBlock to resolved text with absolute path."""
        file_prefix = "file://"
        resource_text = f"[Resource: {block.name}"
        if block.uri:
            uri = block.uri
            has_file_prefix = uri.startswith(file_prefix)
            path = uri[len(file_prefix) :] if has_file_prefix else uri

            # Resolve relative paths against root_dir
            if not path.startswith("/"):
                path = str(Path(root_dir) / path)

            # Strip root_dir prefix for cleaner display
            if path.startswith(root_dir):
                path = path[len(root_dir) :].lstrip("/")

            uri = f"file://{path}" if has_file_prefix else path
            resource_text += f"\nURI: {uri}"
        if block.description:
            resource_text += f"\nDescription: {block.description}"
        if block.mime_type:
            resource_text += f"\nMIME type: {block.mime_type}"
        resource_text += "]"
        return resource_text

    async def prompt(self, prompt, session_id, message_id=None, **kwargs):  # noqa: PLR0913
        """Process a user prompt and persist messages afterward."""
        # Pre-resolve ResourceContentBlock URIs before base class processes them
        processed = []
        for block in prompt:
            if isinstance(block, ResourceContentBlock):
                text = self._resolve_resource_uri(block, self._cwd)
                processed.append(TextContentBlock(type="text", text=text))
            else:
                processed.append(block)
        response = await super().prompt(processed, session_id, message_id, **kwargs)

        # Persist messages after the prompt completes
        if self._agent is not None:
            config: RunnableConfig = {"configurable": {"thread_id": session_id}}
            try:
                state = await self._agent.aget_state(config)
                if isinstance(state.values, dict):
                    messages = state.values.get("messages", [])
                    if messages:
                        cwd = (
                            self._session_cwds.get(session_id)
                            or self._cwd
                            or os.getcwd()
                        )
                        await self._save_messages(cwd, session_id, messages)
            except Exception as e:
                logger.warning(f"Failed to save messages for session {session_id}: {e}")

        return response

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> NewSessionResponse:
        logger.info(f"Creating new session for cwd {cwd}")
        await self._init_db(cwd)
        return await super().new_session(cwd, mcp_servers, **kwargs)

    async def list_sessions(
        self,
        additional_directories: list[str] | None = None,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List available sessions from the SQLite database."""
        target_cwd = cwd or self._cwd or os.getcwd()
        db_path = get_checkpoints_data_dir(target_cwd)

        sessions: list[SessionInfo] = []

        if db_path.exists():
            async with connect(str(db_path)) as db:
                async with db.execute(
                    """
                    SELECT thread_id, title, updated_at
                    FROM sessions
                    ORDER BY updated_at DESC NULLS LAST, thread_id DESC
                    """,
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
        """Check if a session exists in the database."""
        db_path = get_checkpoints_data_dir(cwd)
        if not db_path.exists():
            return False
        async with connect(str(db_path)) as db:
            async with db.execute(
                "SELECT 1 FROM sessions WHERE thread_id = ? LIMIT 1",
                (session_id,),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def _restore_session(
        self,
        session_id: str,
        cwd: str,
    ) -> tuple[Any | None, Any | None]:
        """Common session setup logic for load and resume operations."""
        # Verify session exists
        exists = await self._session_exists(session_id, cwd)
        if not exists:
            logger.warning(f"Session {session_id} not found in database for cwd {cwd}")

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
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        """Load an existing session with the given ID."""
        logger.info(f"Loading session {session_id} for cwd {cwd}")
        modes, config_options = await self._restore_session(session_id, cwd)
        return LoadSessionResponse(
            modes=modes,
            config_options=config_options,
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """Resume an existing session with the given ID."""
        logger.info(f"Resuming session {session_id} for cwd {cwd}")
        modes, config_options = await self._restore_session(session_id, cwd)
        return ResumeSessionResponse(
            modes=modes,
            config_options=config_options,
        )

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse | None:
        logger.info(f"Closing session {session_id}")
        return await super().close_session(session_id, **kwargs)

    async def _replay_chat_history(self, session_id: str, cwd: str) -> None:
        """Replay chat history by sending session_update notifications in batches."""
        messages = await self._load_messages(cwd, session_id)
        logger.info(f"Replaying {len(messages)} messages for session {session_id}")
        for i, msg in enumerate(messages):
            await self._send_message_chunk(session_id, msg)
            # Yield control every 10 messages to avoid blocking the event loop
            if (i + 1) % 10 == 0:
                await asyncio.sleep(0)

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
            logger.warning(f"Error sending message chunk: {e}")

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
        elif msg.content:
            # Log when non-text content is dropped so we don't silently lose data
            logger.debug(
                f"Dropping non-text AIMessage content for session {session_id}: {type(msg.content)}"
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
