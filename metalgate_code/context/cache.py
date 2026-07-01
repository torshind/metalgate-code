"""SQLite-backed cache for resolved symbols and file outlines.

Keys are always (file, mtime) so stale entries are never served.
Two tables: outlines (tree-sitter-extracted symbols) and definitions (ty-resolved goto).
"""

import json
import os
import sqlite3
import threading
from typing import Any, Optional

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS outlines (
    file    TEXT    NOT NULL,
    mtime   REAL    NOT NULL,
    symbols TEXT    NOT NULL,
    PRIMARY KEY (file)
);

CREATE TABLE IF NOT EXISTS definitions (
    file      TEXT    NOT NULL,
    mtime     REAL    NOT NULL,
    line      INTEGER NOT NULL,
    name      TEXT    NOT NULL,
    result    TEXT,
    PRIMARY KEY (file, line, name)
);

CREATE INDEX IF NOT EXISTS idx_def_file ON definitions(file, mtime);
"""


def _mtime(file: str) -> float:
    try:
        return os.path.getmtime(file)
    except OSError:
        return 0.0


class CodeCache:
    """Thread-safe SQLite cache using thread-local connections."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._execute_script(_SCHEMA)

    # ------------------------------------------------------------------ #
    # connection management
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _execute_script(self, sql: str) -> None:
        conn = self._conn()
        conn.executescript(sql)
        conn.commit()

    # ------------------------------------------------------------------ #
    # outline cache
    # ------------------------------------------------------------------ #

    def get_outline(self, file: str) -> Optional[list[dict]]:
        current_mtime = _mtime(file)
        row = (
            self._conn()
            .execute("SELECT mtime, symbols FROM outlines WHERE file = ?", (file,))
            .fetchone()
        )
        if row and row["mtime"] == current_mtime:
            return json.loads(row["symbols"])
        return None

    def set_outline(self, file: str, symbols: list[dict]) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO outlines(file, mtime, symbols) VALUES (?, ?, ?)",
            (file, _mtime(file), json.dumps(symbols)),
        )
        self._conn().commit()

    # ------------------------------------------------------------------ #
    # definition cache
    # ------------------------------------------------------------------ #

    def get_definition(self, file: str, line: int, name: str) -> Optional[Any]:
        """Returns cached result (may be None if we cached a miss)."""
        current_mtime = _mtime(file)
        row = (
            self._conn()
            .execute(
                "SELECT mtime, result FROM definitions WHERE file = ? AND line = ? AND name = ?",
                (file, line, name),
            )
            .fetchone()
        )
        if row and row["mtime"] == current_mtime:
            raw = row["result"]
            return json.loads(raw) if raw else None
        return _CACHE_MISS  # sentinel: not in cache at all

    def set_definition(
        self, file: str, line: int, name: str, result: Optional[dict]
    ) -> None:
        self._conn().execute(
            """INSERT OR REPLACE INTO definitions(file, mtime, line, name, result)
               VALUES (?, ?, ?, ?, ?)""",
            (file, _mtime(file), line, name, json.dumps(result)),
        )
        self._conn().commit()


# Sentinel to distinguish "cached as None (miss)" from "not in cache"
class _CacheMiss:
    pass


_CACHE_MISS = _CacheMiss()
