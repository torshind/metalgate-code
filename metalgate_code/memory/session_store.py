"""
SQLite-backed session store for persisting chat history and session metadata.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from aiosqlite import connect
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


def _extract_text_from_content(content: Any) -> str | None:
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


class SessionStore:
    """SQLite-backed store for session metadata and message history."""

    async def init_db(self, cwd: str) -> None:
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

    async def save_messages(
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
                text = _extract_text_from_content(msg.content)
                if text:
                    title = text[:100]
                    break

        await self.init_db(cwd)
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

    async def load_messages(self, cwd: str, session_id: str) -> list[Any]:
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

    async def list_sessions(self, cwd: str) -> list[tuple[str, str | None, str | None]]:
        """List available sessions from the SQLite database.

        Returns:
            List of (session_id, title, updated_at) tuples.
        """
        db_path = get_checkpoints_data_dir(cwd)
        if not db_path.exists():
            return []

        sessions: list[tuple[str, str | None, str | None]] = []
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
                    sessions.append((row[0], row[1], row[2]))
        return sessions

    async def session_exists(self, cwd: str, session_id: str) -> bool:
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
