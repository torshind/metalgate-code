"""
PythonContextMiddleware - Starts and stops the Python context indexer.
"""

import logging
from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol
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

    def __init__(
        self,
        cwd: str,
        backend: SandboxBackendProtocol | None = None,
    ):
        """
        Initialize the middleware.

        Args:
            cwd: The current working directory.
        """
        self._cwd = cwd
        self._backend = backend

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Start the Python context indexer when the agent starts."""
        if self._backend is not None:
            result = await self._backend.aexecute("uv run which python")
            if result.exit_code is not None and result.exit_code == 0:
                python = result.output.strip()
            else:
                result = await self._backend.aexecute("which python")
                if result.exit_code is not None and result.exit_code == 0:
                    python = result.output.strip()
                else:
                    python = None
        logger.info(f"Detected Python executable: {python}")

        await start_indexing(cwd=self._cwd, python=python)
        return None

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Stop the Python context indexer when the agent closes."""
        await stop_indexing()
        return None
