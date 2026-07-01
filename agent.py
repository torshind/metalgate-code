"""
Coding agent using ACP.
"""

import asyncio
import logging
import os
from pathlib import Path

from acp import run_agent as run_acp_agent
from dotenv import load_dotenv

from metalgate_code.config import get_available_modes
from metalgate_code.factory import MetalGateACP, create_agent
from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend
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

    load_dotenv(os.path.expanduser("~/.metalgate/.env"), override=True)

    logger.info("Environment loaded")
    logger.info("API Key: %s", "ok" if os.environ.get("MODEL_API_KEY") else "not set")

    modes = get_available_modes()

    # Fetch models from the configured provider API
    models = fetch_models()

    def make_shell_backend(cwd: str) -> MicrosandboxBackend:
        """Factory that creates the shell backend when cwd is known.

        Image selection (lowest → highest precedence):
          1. Auto-detect from project files (go.mod → "go", else "uv:python").
          2. ``SANDBOX_IMAGE`` env var (user override).
        """
        shell_env = os.environ.copy()

        if os.environ.get("SANDBOX_IMAGE"):
            image = os.environ["SANDBOX_IMAGE"]
        elif Path(cwd, "go.mod").exists():
            image = "go"
        else:
            image = "python"

        logger.info("Using sandbox image: %s", image)

        return MicrosandboxBackend(
            root_dir=cwd,
            image=image,
            env=shell_env,
            inherit_env=True,
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
