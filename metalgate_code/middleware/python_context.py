"""
PythonContextMiddleware - Starts and stops the Python context indexer.
"""

import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    StateT,
)
from langgraph.runtime import Runtime

from metalgate_code.context.indexer import start_indexing, stop_indexing

logger = logging.getLogger("metalgate_code")


class PythonContextMiddleware(AgentMiddleware):
    """
    Middleware that starts the Python context indexer when the agent starts
    and stops it when the agent closes.
    """

    def __init__(self, cwd: str):
        """
        Initialize the middleware.

        Args:
            cwd: The current working directory.
        """
        self._cwd = cwd

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Start the Python context indexer when the agent starts."""
        await start_indexing(cwd=self._cwd)
        return None

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Stop the Python context indexer when the agent closes."""
        await stop_indexing()
        return None
