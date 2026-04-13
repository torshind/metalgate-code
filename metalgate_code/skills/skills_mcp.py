"""
skills_mcp.py - Tools for managing MCP servers
"""

from langchain_core.tools import tool

from metalgate_code.skills.registry_mcp import registry_mcp


@tool
def list_mcp_servers() -> dict:
    """List all configured MCP servers and their current connection status."""
    servers = registry_mcp.get_servers()
    active = registry_mcp.names()
    return {
        name: {"config": cfg, "active": any(t.startswith(name) for t in active)}
        for name, cfg in servers.items()
    }


@tool
def add_mcp_server(
    name: str,
    transport: str,
    url: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    auth_type: str | None = None,
    auth_env: str | None = None,
) -> str:
    """
    Add an MCP server to the project config and connect immediately.
    transport: 'http', 'stdio', 'sse', or 'websocket'.
    For http/sse/websocket: provide url.
    For stdio: provide command and optionally args.
    For auth: provide auth_type ('header', 'bearer', or 'basic') and auth_env (env var name containing the key/token).
    """
    try:
        config: dict = {
            "transport": transport,
        }
        if url:
            config["url"] = url
        if command:
            config["command"] = command
        if args:
            config["args"] = args
        if auth_type and auth_env:
            config["auth"] = {
                "type": auth_type,
                "env": auth_env,
            }

        registry_mcp.add_server(
            name=name,
            transport=transport,  # ty: ignore[invalid-argument-type]
            url=url,
            command=command,
            args=args,
            auth=config.get("auth"),
        )
    except ValueError as e:
        return f"Error: {e}"

    registry_mcp.reload()
    return f"MCP server '{name}' added and connected. Tools available: {registry_mcp.names()}"


@tool
def remove_mcp_server(name: str) -> str:
    """Remove an MCP server from the project config and disconnect immediately."""
    try:
        registry_mcp.remove_server(name)
    except ValueError as e:
        return f"Error: {e}"

    registry_mcp.reload()
    return f"MCP server '{name}' removed."
