# metalgate-code

An ACP (Agent Communication Protocol) code agent featuring **in-session dynamic tool skills** — create, register, and use new tools during a conversation without restarting the agent.

## Dynamic Tool Skills Lifecycle

The agent can create custom tools on-the-fly within a session:

1. **Create** — Define a new `@tool` decorated function with code and optional dependencies
2. **Register** — The tool is persisted to `skills.py` and immediately available
3. **Use** — The agent can invoke the new tool in the same session
4. **Delete** — Remove tools when no longer needed

Example workflow:

```
User: Create a tool that fetches the weather for a city

Agent: Creates `get_weather(city: str)` → tool is registered and ready to use

User: What's the weather in Paris?

Agent: Calls get_weather("Paris") → returns result
```

Meta-skills for managing tools:
- `create_tool_skill` — Create and register a new tool
- `delete_tool_skill` — Remove a tool
- `read_tool_skill` — Get details about a specific tool
- `list_tool_skills` — List all available tools

### Important Notes on Dynamic Tool Skills

- Dynamic skill creation works only by asking the agent itself (e.g., "Create a tool skill that...").
- Dynamic skill creation can potentially change the agent's virtual environment, as new dependencies may be installed.
- Users can add skills manually by editing `skills.py`, but the agent must be restarted for the new skills to be recognized.

## Configuration Files

### AGENTS.md

An `AGENTS.md` file is supported and changes only the system prompt. Place it in your workspace root for the agent to read and apply custom prompting.

### MCP Configuration

The agent supports MCP (Model Context Protocol) servers via the `mcp.yaml` configuration file. There, You can define MCP servers and tools that will be available to the agent.

Example `mcp.yaml`:

```yaml
mcpServers:
  - name: filesystem
    command: npx -y @modelcontextprotocol/server-filesystem /path/to/your/project
    type: stdio
```

## Installation (Zed)

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd metalgate-code
   ```

2. Sync dependencies with uv:
   ```bash
   uv sync
   ```

3. Configure your environment variables:
   Example `.env`:
   ```bash
   # LLM Provider: evroc (default), openai, or anthropic
   PROVIDER=evroc

   # API Keys (use the one matching your provider)
   # Get your evroc API key from: https://www.evroc.com
   OPENAI_API_KEY=your_evroc_api_key_here

   # OpenAI API key
   # OPENAI_API_KEY=your_openai_api_key_here

   # Anthropic API key
   # ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. Add to your Zed `settings.json`:

   ```json
   {
     "agent_servers": {
       "metalgate": {
         "type": "custom",
         "command": "/path/to/metalgate-code/run.sh",
         "default_config_options": {
           "model": "evroc:moonshotai/Kimi-K2.5"
         }
       }
     }
   }
   ```

5. Select the `metalgate` agent in Zed's Agent Panel.

## Tested Environment

| Component | Tested | Notes |
|-----------|--------|-------|
| **LLM Provider** | [evroc](https://www.evroc.com) | Only tested provider. `moonshotai/Kimi-K2.5` model recommended. |
| **Editor** | [Zed](https://zed.dev) | Only tested editor. |

Other providers and editors may work but are untested. Testers and PRs are welcome!

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Zed editor
- API key (set via environment or config)

## Roadmap

- [ ] **Session reload** — Persist and restore conversation state across sessions (not yet working)
- [ ] **Memory** — Long-term memory for cross-session context and learned patterns
- [ ] **`SKILLS.md`** — Text-based skills could be supported in a later release for defining skills declaratively

## License

MIT
