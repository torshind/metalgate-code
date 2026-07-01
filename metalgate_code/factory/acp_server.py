"""
ACP Server factory for MetalGate Code agent.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable

from acp.schema import (
    AgentCapabilities,
    CloseSessionResponse,
    HttpMcpServer,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PromptCapabilities,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SseMcpServer,
    TextContentBlock,
)
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents_acp.server import AgentServerACP, AgentSessionContext
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from metalgate_code.memory.replayer import ChatHistoryReplayer
from metalgate_code.memory.session_store import SessionStore

logger = logging.getLogger("metalgate_code")


class MetalGateACP(AgentServerACP):
    """Custom ACP server with manual SQLite message persistence."""

    def __init__(
        self,
        agent_factory: Callable[
            [AgentSessionContext, SandboxBackendProtocol | None], CompiledStateGraph
        ],
        backend_factory: Callable[[str], SandboxBackendProtocol],
        modes: Any,
        models: Any,
    ) -> None:
        """Initialize with agent factory and backend factory.

        Args:
            agent_factory: Function that takes (context, backend) and returns CompiledStateGraph
            backend_factory: Function that takes cwd and returns SandboxBackendProtocol
            modes: Session modes configuration
            models: Available models list
        """
        self._user_agent_factory = agent_factory
        self._backend_factory = backend_factory
        # Pass wrapper as the agent to the base class.
        # The base class will use MemorySaver automatically when checkpointer is None.
        super().__init__(agent=self._create_agent, modes=modes, models=models)
        self._store = SessionStore()
        self._replayer = ChatHistoryReplayer()
        self._pending_session_messages: dict[str, list[Any]] = {}

    def _create_agent(self, context: AgentSessionContext) -> CompiledStateGraph:
        """Called by base class to create the agent for a session."""
        self._shell_backend = self._backend_factory(context.cwd)
        return self._user_agent_factory(context, self._shell_backend)

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Return server capabilities including session support."""
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    image=True,
                ),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
        )

    @staticmethod
    def _resolve_resource_uri(block: ResourceContentBlock, root_dir: str) -> str:
        """Convert a ResourceContentBlock to resolved text with absolute path."""
        file_prefix = "file://"
        resource_text = f"[Resource: {block.name}"
        if block.uri:
            uri = block.uri
            has_file_prefix = uri.startswith(file_prefix)
            path = uri[len(file_prefix) :] if has_file_prefix else uri

            # Resolve relative paths against root_dir
            if not path.startswith("/"):
                path = str(Path(root_dir) / path)

            # Strip root_dir prefix for cleaner display
            if path.startswith(root_dir):
                path = path[len(root_dir) :].lstrip("/")

            uri = f"file://{path}" if has_file_prefix else path
            resource_text += f"\nURI: {uri}"
        if block.description:
            resource_text += f"\nDescription: {block.description}"
        if block.mime_type:
            resource_text += f"\nMIME type: {block.mime_type}"
        resource_text += "]"
        return resource_text

    async def prompt(self, prompt, session_id, message_id=None, **kwargs):  # noqa: PLR0913
        """Process a user prompt and persist messages afterward."""
        # Pre-resolve ResourceContentBlock URIs before base class processes them
        processed = []
        for block in prompt:
            if isinstance(block, ResourceContentBlock):
                text = self._resolve_resource_uri(block, self._cwd)
                processed.append(TextContentBlock(type="text", text=text))
            else:
                processed.append(block)

        # Inject pending historical messages into the agent's state if this is the
        # first prompt after a load/resume.
        pending = self._pending_session_messages.pop(session_id, None)
        if pending:
            if self._agent is None:
                self._reset_agent(session_id)
            if self._agent is not None:
                config: RunnableConfig = {"configurable": {"thread_id": session_id}}
                try:
                    # aupdate_state needs an existing checkpoint to branch from.
                    # On a fresh process the in-memory checkpointer has none for
                    # this thread, so seed it directly.
                    checkpointer = self._agent.checkpointer
                    if isinstance(checkpointer, MemorySaver):
                        seed_config: RunnableConfig = {
                            "configurable": {
                                "thread_id": session_id,
                                "checkpoint_ns": "",
                            }
                        }
                        if not await checkpointer.aget_tuple(seed_config):
                            metadata: CheckpointMetadata = {
                                "source": "input",
                                "step": -1,
                                "parents": {},
                            }
                            await checkpointer.aput(
                                seed_config,
                                empty_checkpoint(),
                                metadata,
                                {},
                            )
                    await self._agent.aupdate_state(
                        config, {"messages": pending}, as_node="__start__"
                    )
                    logger.info(
                        f"Injected {len(pending)} pending messages into session {session_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to inject pending messages for session {session_id}: {e}"
                    )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"ACP prompt: session={session_id} blocks={[getattr(b, 'type', type(b).__name__) for b in processed]}"
            )
            for i, block in enumerate(processed):
                text = getattr(block, "text", None)
                if text:
                    logger.debug(f"ACP prompt block {i}: {text[:200]!r}")
        try:
            response = await super().prompt(processed, session_id, message_id, **kwargs)
            logger.debug(
                f"ACP prompt response: {type(response).__name__} {getattr(response, 'updates', 'no updates')}"
            )
        except Exception as e:
            logger.exception(f"ACP prompt failed for session {session_id}: {e}")
            raise

        # Persist messages after the prompt completes
        if self._agent is not None:
            config: RunnableConfig = {"configurable": {"thread_id": session_id}}
            try:
                state = await self._agent.aget_state(config)
                if isinstance(state.values, dict):
                    messages = state.values.get("messages", [])
                    if messages:
                        cwd = (
                            self._session_cwds.get(session_id)
                            or self._cwd
                            or os.getcwd()
                        )
                        await self._store.save_messages(cwd, session_id, messages)
            except Exception as e:
                logger.warning(f"Failed to save messages for session {session_id}: {e}")

        return response

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,  # noqa: ARG002  # ACP protocol interface parameter
    ) -> NewSessionResponse:
        logger.info(f"Creating new session for cwd {cwd}")
        await self._store.init_db(cwd)
        return await super().new_session(cwd, mcp_servers, **kwargs)

    async def list_sessions(
        self,
        additional_directories: list[str] | None = None,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        """List available sessions from the SQLite database."""
        target_cwd = cwd or self._cwd or os.getcwd()
        session_rows = await self._store.list_sessions(target_cwd)

        sessions: list[SessionInfo] = []
        for thread_id, title, updated_at in session_rows:
            display_title = title or f"Session {thread_id[:8]}"
            sessions.append(
                SessionInfo(
                    session_id=thread_id,
                    cwd=target_cwd,
                    title=display_title,
                    updated_at=updated_at,
                )
            )

        logger.info(f"List sessions: {len(sessions)} found in {target_cwd}")
        return ListSessionsResponse(sessions=sessions, next_cursor=None)

    async def _restore_session(
        self,
        session_id: str,
        cwd: str,
    ) -> tuple[Any | None, Any | None]:
        """Common session setup logic for load and resume operations."""
        # Verify session exists
        exists = await self._store.session_exists(cwd, session_id)
        if not exists:
            logger.warning(f"Session {session_id} not found in database for cwd {cwd}")

        # Store session context - cwd comes from client
        self._session_cwds[session_id] = cwd
        self._cwd = cwd

        # Initialize session state
        if self._modes:
            self._session_modes[session_id] = self._modes.current_mode_id
            self._session_mode_states[session_id] = self._modes

        if self._models:
            self._session_models[session_id] = self._models[0]["value"]

        config_options = None
        if self._modes or self._models:
            config_options = self._build_config_options(session_id)

        # Replay chat history after returning the response
        messages = await self._store.load_messages(cwd, session_id)
        if messages:
            self._pending_session_messages[session_id] = messages
        await self._replayer.replay(self._conn, session_id, messages)

        return self._modes, config_options

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        """Load an existing session with the given ID."""
        logger.info(f"Loading session {session_id} for cwd {cwd}")
        modes, config_options = await self._restore_session(session_id, cwd)
        return LoadSessionResponse(
            modes=modes,
            config_options=config_options,
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        """Resume an existing session with the given ID."""
        logger.info(f"Resuming session {session_id} for cwd {cwd}")
        modes, config_options = await self._restore_session(session_id, cwd)
        return ResumeSessionResponse(
            modes=modes,
            config_options=config_options,
        )

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse | None:
        logger.info(f"Closing session {session_id}")
        return await super().close_session(session_id, **kwargs)
