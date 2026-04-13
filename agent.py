"""
Coding agent using ACP.
"""

import asyncio
import logging
import os

from acp import (
    run_agent as run_acp_agent,
)
from deepagents_acp.server import AgentServerACP
from dotenv import find_dotenv, load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import Checkpointer

from metalgate_code.config import get_available_modes
from metalgate_code.factory import create_agent
from metalgate_code.models import fetch_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/agent_debug.log", mode="w"),
    ],
)
logger = logging.getLogger("metalgate_code")


async def _serve_agent() -> None:
    """Run example agent from the root of the repository with ACP integration."""
    logger.info("Agent starting...")

    load_dotenv(find_dotenv(".env", usecwd=True), override=True)

    logger.info("Environment loaded")
    logger.info("API Key: %s", "ok" if os.environ.get("OPENAI_API_KEY") else "not set")

    checkpointer: Checkpointer = MemorySaver()

    modes = get_available_modes()

    # Fetch models from the configured provider API
    models = fetch_models()

    acp_agent = AgentServerACP(
        agent=create_agent(checkpointer),
        modes=modes,
        models=models,
    )
    await run_acp_agent(acp_agent)


def main() -> None:
    """Run the demo agent."""
    asyncio.run(_serve_agent())


if __name__ == "__main__":
    main()
