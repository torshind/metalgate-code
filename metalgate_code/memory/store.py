"""
Mem0 memory store initialization and utilities.
"""

import atexit
import os
from typing import Any

from mem0 import AsyncMemory

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
        self.project_id = cwd
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

        # Get embedding dimensions from env (matching the embedder model)
        embedding_dims = int(os.environ.get("EMBEDDING_DIMS", 4096))

        # Build the full configuration
        config: dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(data_dir / "qdrant"),
                    "embedding_model_dims": embedding_dims,
                },
            },
            "history_db_path": str(data_dir / "mem0_history.db"),
            **provider_config,
        }

        return AsyncMemory.from_config(config)

    def _cleanup(self) -> None:
        """Close the underlying store and remove from cache."""
        self.store.close()
        _store_cache.pop((self.project_id, self.user_id), None)
