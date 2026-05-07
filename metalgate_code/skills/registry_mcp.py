"""
mcp_registry.py
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.documents.base import Blob
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)

logger = logging.getLogger("metalgate_code")


def _build_stdio_connection(cfg: dict) -> StdioConnection:
    return {
        "transport": "stdio",
        "command": cfg["command"],
        "args": cfg.get("args", []),
    }


def _build_sse_connection(cfg: dict) -> SSEConnection:
    return {
        "transport": "sse",
        "url": cfg["url"],
    }


def _resolve_auth(cfg: dict) -> dict | None:
    """Extract auth headers from config if present."""
    auth = cfg.get("auth")
    if not auth:
        return None

    auth_type = auth.get("type")
    env_var = auth.get("env")

    if not env_var:
        return None

    value = os.environ.get(env_var)
    if not value:
        logger.warning(
            f"Auth env var '{env_var}' not set for {cfg.get('url', 'unknown')}"
        )
        return None

    match auth_type:
        case "bearer":
            return {"Authorization": f"Bearer {value}"}
        case "header":
            # header name == env var name
            return {env_var: value}
        case "basic":
            import base64

            credentials = base64.b64encode(value.encode()).decode()
            return {"Authorization": f"Basic {credentials}"}
        case _:
            logger.warning(f"Unknown auth type '{auth_type}'")
            return None


def _build_streamable_http_connection(cfg: dict) -> StreamableHttpConnection:
    from datetime import timedelta

    conn: StreamableHttpConnection = {
        "transport": "streamable_http",
        "url": cfg["url"],
        "timeout": timedelta(seconds=cfg.get("timeout", 30)),
    }

    headers = _resolve_auth(cfg)
    if headers:
        conn["headers"] = headers

    return conn


def _build_websocket_connection(cfg: dict) -> WebsocketConnection:
    return {
        "transport": "websocket",
        "url": cfg["url"],
    }


class RegistryMCP(MultiServerMCPClient):
    """Registry for MCP servers with dynamic reload support."""

    def __init__(self) -> None:
        super().__init__()
        self._tools: list[BaseTool] = []
        self._resources: list[Blob] = []
        self._config_path: Path | None = None

    async def aload(self, project_path: str | Path):
        """Initialize the registry by loading config and connecting to servers."""
        self._config_path = Path(project_path) / "mcp.yaml"
        if self._config_path.exists():
            logger.info(f"Loading MCP tools from {self._config_path}")
            await self.areload()
        else:
            logger.info(f"No mcp.yaml found at {self._config_path}")

    def load(self, project_path: str | Path):
        """Initialize the registry by loading config and connecting to servers."""
        self._config_path = Path(project_path) / ".metalgate" / "mcp.yaml"
        if self._config_path.exists():
            logger.info(f"Loading MCP tools from {self._config_path}")
            self.reload()
        else:
            logger.info(f"No mcp.yaml found at {self._config_path}")

    def _load_config(self) -> dict:
        """Load server connections from the YAML config file."""
        connections: dict = {}
        if self._config_path is None or not self._config_path.exists():
            return connections

        try:
            config = yaml.safe_load(self._config_path.read_text()) or {}
            servers = config.get("servers", {})

            for name, cfg in servers.items():
                transport: Literal["stdio", "http", "sse", "websocket"] = cfg.get(
                    "transport"
                )
                if transport == "stdio":
                    connections[name] = _build_stdio_connection(cfg)
                elif transport == "sse":
                    connections[name] = _build_sse_connection(cfg)
                elif transport == "http":
                    connections[name] = _build_streamable_http_connection(cfg)
                elif transport == "websocket":
                    connections[name] = _build_websocket_connection(cfg)
                else:
                    logger.warning(
                        f"Unknown transport '{transport}' for server '{name}'"
                    )
        except Exception as e:
            logger.error(
                f"Failed to load MCP config from {self._config_path}: {e}",
                exc_info=True,
            )

        return connections

    async def areload(self) -> None:
        """Async reload MCP servers from config and refresh tools."""
        if self._config_path is None or not self._config_path.exists():
            return
        self.connections = self._load_config()
        self._tools = await self.get_tools()
        try:
            self._resources = await self.get_resources()
        except Exception as e:
            logger.warning(f"Failed to load resources from MCP servers: {e}")
            self._resources = []
        logger.info(f"Loaded {len(self._tools)} tools from MCP servers: {self.names()}")
        logger.info(f"Loaded {len(self._resources)} resources from MCP servers")

    def reload(self) -> None:
        """Sync reload - schedules async reload in the current event loop."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(self.areload())
            else:
                loop.run_until_complete(self.areload())
        except RuntimeError:
            asyncio.run(self.areload())

    def _load_config_dict(self) -> dict:
        """Load the raw config dict from file."""
        if self._config_path is None or not self._config_path.exists():
            return {}
        return yaml.safe_load(self._config_path.read_text()) or {}

    def _save_config_dict(self, config: dict) -> None:
        """Save config dict to file."""
        if self._config_path is None:
            raise RuntimeError("No config path set")
        self._config_path.write_text(yaml.dump(config, default_flow_style=False))

    def get_servers(self) -> dict:
        """Get all configured servers from config."""
        return self._load_config_dict().get("servers", {})

    def add_server(
        self,
        name: str,
        transport: Literal["stdio", "http", "sse", "websocket"],
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        auth: dict | None = None,
    ) -> None:
        """Add a server to the config.

        auth: dict with 'type' ('header', 'bearer', 'basic') and 'env' (env var name)
        """
        config = self._load_config_dict()
        servers = config.setdefault("servers", {})
        if name in servers:
            raise ValueError(f"Server '{name}' already exists")

        entry: dict = {"transport": transport}
        if transport == "stdio":
            if not command:
                raise ValueError("command is required for stdio transport")
            entry["command"] = command
            if args:
                entry["args"] = args
        else:
            if not url:
                raise ValueError(f"url is required for {transport} transport")
            entry["url"] = url

        if auth:
            entry["auth"] = auth

        servers[name] = entry
        self._save_config_dict(config)

    def remove_server(self, name: str) -> None:
        """Remove a server from the config."""
        config = self._load_config_dict()
        servers = config.get("servers", {})
        if name not in servers:
            raise ValueError(f"Server '{name}' not found")
        del servers[name]
        self._save_config_dict(config)

    def names(self) -> list[str]:
        """Return a list of available tool names."""
        return [tool.name for tool in self._tools]

    def all(self) -> list[BaseTool]:
        """Return all loaded tools."""
        return self._tools

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None


registry_mcp = RegistryMCP()
