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
- `reload_tool_skills` — Reload skills from `skills.py` after manual edits

### Important Notes on Dynamic Tool Skills

- Dynamic skill creation works by asking the agent itself (e.g., "Create a tool skill that...").
- Dynamic skill creation can potentially change the agent's virtual environment, as new dependencies may be installed (pending confirmation)
- Users can add skills manually by editing `skills.py`, then ask for `reload_tool_skills` to pick up changes without restarting the agent.

## Persistent Memory (Optional) **EXPERIMENTAL**

The agent supports **persistent memory** powered by [Mem0](https://github.com/mem0ai/mem0), enabling it to remember facts, preferences, and past interactions across sessions.

### Opt-In Configuration

Memory is **disabled by default**. To enable it, set the `MEMORY` environment variable:

```bash
export MEMORY=true  # or "1", "yes", "on", "enabled"
```

### How Memory Works

When enabled, the agent uses a two-tier memory system:

1. **Collector** — At the end of each turn, saves conversation summaries to memory in the background
   - **Semantic memories**: Facts, architectural decisions, and conventions extracted from conversations
   - **Episodic memories**: Session summaries with concrete outcomes, problems encountered, and resolutions

2. **Recollector** — Retrieves relevant memories and injects them into the system prompt
   - **Semantic memories**: Loaded once at session start (facts, preferences, conventions)
   - **Episodic memories**: Searched on every prompt for semantically relevant past sessions

### Memory Scopes

Memories are isolated per **project** based on the working directory (`cwd`).

### Storage Location

Memory files are stored at `~/.metalgate/memory/<project-name>/`:
- `chroma/` — ChromaDB vector database for semantic search
- `mem0_history.db` — SQLite database for conversation history

### Pros and Cons

**Pros:**
- **Context continuity**: The agent remembers your preferences, coding style, and past decisions across sessions
- **Resumability**: Interrupted work can be resumed more easily with relevant episodic context
- **Project knowledge**: Learns and retains architectural decisions and conventions specific to your project
- **Privacy**: All conversation data is stored **locally** on your machine
- **Context protection**: Not all stored memories are included in the main context window, reducing token usage and context bloating compared to naive history injection

**Cons:**
- **Experimental**: Feature is still in development and may have bugs
- **Token usage**: Higher token consumption due to Mem0's LLM calls for memory extraction and retrieval
- **Storage overhead**: Requires disk space for the vector database (Chroma) and SQLite history
- **Initial latency**: First request may have slight latency while memory is queried
- **Potential context noise**: Retrieved memories may add noise if not perfectly relevant to the current query
- **Complexity**: Additional component that can fail or require troubleshooting

## Configuration Files

### AGENTS.md

An `AGENTS.md` file is supported and changes only the system prompt. Place it in your workspace root for the agent to read and apply custom prompting.

### MCP Configuration

The agent supports MCP (Model Context Protocol) servers via the `mcp.yaml` configuration file. There, you can define MCP servers and tools that will be available to the agent.

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

> **Note:** Some versions of Zed exhibit instability with history replaying. This is currently under investigation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Zed editor
- API key (set via environment or config)

## Roadmap

- [x] **Session reload** — Persist and restore conversation state across sessions
- [x] **Memory** — Long-term memory for cross-session context and learned patterns (via mem0)
- [ ] **`SKILLS.md`** — Text-based skills could be supported in a later release for defining skills declaratively

## Contributing

Only pull requests developed by humans or with **metalgate-code** will be accepted.

This policy ensures:
- **Consistency** — Contributions follow the project's established patterns and conventions
- **Quality control** — Changes align with the agent's principles
- **Scope discipline** — PRs remain focused without feature creep or unnecessary refactoring

## License

MIT
