"""
Mem0 memory store initialization and utilities.
"""

import atexit
from pathlib import PurePath
from typing import Any

from mem0 import AsyncMemory

from metalgate_code.memory.config import (
    DEFAULT_HISTORICAL_LIMIT,
)
from metalgate_code.memory.paths import get_memory_data_dir
from metalgate_code.models.provider import get_mem0_config

# Singleton cache: (cwd, user_id) -> MemoryStore
_store_cache: dict[tuple[str, str], "MemoryStore"] = {}


class MemoryStore:
    _instance: "MemoryStore | None" = None

    def __new__(
        cls,
        cwd: str,
        user_id: str,
    ) -> "MemoryStore":
        key = (cwd, user_id)
        if key not in _store_cache:
            _store_cache[key] = super().__new__(cls)
            _store_cache[key]._initialized = False
        return _store_cache[key]

    def __init__(
        self,
        cwd: str,
        user_id: str,
    ):
        if self._initialized:
            return
        self._initialized = True
        self.project_id = PurePath(cwd).name
        self.user_id = user_id
        self.store = self._create_memory_store(cwd)
        atexit.register(self._cleanup)

    def _create_memory_store(self, cwd: str) -> AsyncMemory:
        """
        Create and configure an AsyncMemory instance.

        Args:
            data_dir: Directory for storing memory data (Qdrant + SQLite).

        Returns:
            Configured AsyncMemory instance.
        """
        data_dir = get_memory_data_dir(cwd)

        # Get provider-specific Mem0 configuration
        provider_config = get_mem0_config()

        # Build the full configuration
        config: dict[str, Any] = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": str(data_dir / "chroma"),
                    "collection_name": self.project_id + "_" + self.user_id,
                },
            },
            "history_db_path": str(data_dir / "mem0_history.db"),
            **provider_config,
        }

        return AsyncMemory.from_config(config)

    async def search(
        self,
        query: str,
        agent_id: str,
        limit: int = DEFAULT_HISTORICAL_LIMIT,
    ) -> dict[str, Any]:
        """
        Search memories by query.

        Args:
            query: The search query.
            agent_id: The agent ID to scope the search.
            limit: Maximum number of results to return.

        Returns:
            Search results dict with 'results' key.
        """
        return await self.store.search(
            query=query,
            user_id=self.user_id,
            agent_id=agent_id,
            run_id=self.project_id,
            limit=limit,
        )

    async def get_all(self, agent_id: str) -> dict[str, Any]:
        """
        Get all memories for an agent.

        Args:
            agent_id: The agent ID to scope the query.

        Returns:
            Results dict with 'results' key.
        """
        return await self.store.get_all(
            user_id=self.user_id,
            agent_id=agent_id,
            run_id=self.project_id,
        )

    async def add(
        self,
        messages: list[dict[str, Any]],
        agent_id: str,
        infer: bool = True,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Add messages to memory.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            agent_id: The agent ID to scope the memory.
            infer: Whether to infer facts from the messages.
            prompt: Optional prompt to guide inference.

        Returns:
            Add results dict.
        """
        return await self.store.add(
            messages,
            user_id=self.user_id,
            agent_id=agent_id,
            run_id=self.project_id,
            infer=infer,
            prompt=prompt,
        )

    def _cleanup(self) -> None:
        """Close the underlying store and remove from cache."""
        self.store.close()
        _store_cache.pop((self.project_id, self.user_id), None)
