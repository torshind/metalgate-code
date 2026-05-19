"""
Coding agent using ACP.
"""

import asyncio
import logging
import os

from acp import run_agent as run_acp_agent
from dotenv import find_dotenv, load_dotenv

from metalgate_code.config import get_available_modes
from metalgate_code.factory import MetalGateACP, create_agent
from metalgate_code.factory.agent_factory import LocalShellBackend
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
    logger.info("API Key: %s", "ok" if os.environ.get("MODEL_API_KEY") else "not set")

    modes = get_available_modes()

    # Fetch models from the configured provider API
    models = fetch_models()

    def make_shell_backend(cwd: str) -> LocalShellBackend:
        """Factory that creates the shell backend when cwd is known."""
        shell_env = os.environ.copy()
        return LocalShellBackend(
            root_dir=cwd,
            inherit_env=True,
            env=shell_env,
        )

    acp_agent = MetalGateACP(
        agent_factory=create_agent(),
        backend_factory=make_shell_backend,
        modes=modes,
        models=models,
    )
    await run_acp_agent(acp_agent, use_unstable_protocol=True)


def main() -> None:
    """Run the demo agent."""
    asyncio.run(_serve_agent())


if __name__ == "__main__":
    main()
