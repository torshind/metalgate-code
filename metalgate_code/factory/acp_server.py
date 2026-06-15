"""
ACP Server factory for MetalGate Code agent.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from acp import (
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    text_block,
    tool_content,
    update_tool_call,
)
from acp.exceptions import RequestError
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    CloseSessionResponse,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    PermissionOption,
    PromptCapabilities,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SseMcpServer,
    TextContentBlock,
    ToolCallUpdate,
)
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents_acp.server import (
    AgentServerACP,
    AgentSessionContext,
)
from deepagents_acp.utils import (
    convert_audio_block_to_content_blocks,
    convert_embedded_resource_block_to_content_blocks,
    convert_image_block_to_content_blocks,
    convert_resource_block_to_content_blocks,
    convert_text_block_to_content_blocks,
    extract_command_types,
    format_execute_result,
    truncate_execute_command_for_display,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from metalgate_code.memory.replayer import ChatHistoryReplayer
from metalgate_code.memory.session_store import SessionStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import RunnableConfig

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

    async def prompt(self, prompt, session_id, message_id=None, **kwargs):  # noqa: PLR0913, PLR0915
        """Process a user prompt with subagent HITL interrupt remapping support."""
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

        # Set up agent and config
        if self._agent is None:
            self._reset_agent(session_id)

        if self._agent is None:
            msg = "Agent initialization failed"
            raise RuntimeError(msg)

        if getattr(self._agent, "checkpointer", None) is None:
            self._agent.checkpointer = MemorySaver()
        agent = self._agent

        # Reset cancellation flag for new prompt
        self._cancelled = False

        # Convert ACP content blocks to LangChain multimodal content format
        content_blocks = []

        for block in processed:
            if isinstance(block, TextContentBlock):
                content_blocks.extend(convert_text_block_to_content_blocks(block))
            elif isinstance(block, ImageContentBlock):
                content_blocks.extend(convert_image_block_to_content_blocks(block))
            elif isinstance(block, AudioContentBlock):
                content_blocks.extend(convert_audio_block_to_content_blocks(block))
            elif isinstance(block, ResourceContentBlock):
                content_blocks.extend(
                    convert_resource_block_to_content_blocks(block, root_dir=self._cwd)
                )
            elif isinstance(block, EmbeddedResourceContentBlock):
                content_blocks.extend(
                    convert_embedded_resource_block_to_content_blocks(block)
                )

        # Stream the deep agent response with multimodal content
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        # Track active tool calls and accumulate chunks by index
        active_tool_calls = {}
        tool_call_accumulator = {}  # index -> {id, name, args_str}

        # Map subagent namespace to parent task tool_call_id for remapping
        namespace_to_task_id: dict[str, str] = {}
        latest_task_id: str | None = None

        current_state = None
        user_decisions = []

        while current_state is None or current_state.interrupts:
            # Check for cancellation
            if self._cancelled:
                self._cancelled = False  # Reset for next prompt
                return PromptResponse(stop_reason="cancelled")

            async for stream_chunk in agent.astream(
                Command(resume={"decisions": user_decisions})
                if user_decisions
                else {"messages": [{"role": "user", "content": content_blocks}]},
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                _expected_len = 3  # (namespace, stream_mode, data)
                if (
                    not isinstance(stream_chunk, tuple)
                    or len(stream_chunk) != _expected_len
                ):
                    logger.info(f"SKIPPED CHUNK (not tuple or wrong len): {stream_chunk}")
                    continue

                _namespace, stream_mode, data = stream_chunk
                logger.info(f"STREAM CHUNK: namespace={_namespace}, mode={stream_mode}, data_keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                # Check for cancellation during streaming
                if self._cancelled:
                    self._cancelled = False  # Reset for next prompt
                    return PromptResponse(stop_reason="cancelled")

                # Track task tool calls for subagent remapping
                if stream_mode == "messages":
                    message_chunk, _metadata = data
                    logger.info(f"MESSAGE CHUNK: namespace={_namespace}, type={type(message_chunk).__name__}, has_tool_calls={hasattr(message_chunk, 'tool_calls')}")
                    if hasattr(message_chunk, "tool_calls"):
                        tool_calls = message_chunk.tool_calls or []
                        logger.info(f"  tool_calls={tool_calls}")
                        for tc in tool_calls:
                            if tc.get("name") == "task":
                                task_id = tc.get("id")
                                if task_id:
                                    # Store the most recent task_id for subagent remapping
                                    # We'll map it to the subagent namespace when we see it
                                    latest_task_id = task_id
                                    logger.info(f"SAVED TASK_ID: {task_id} for future subagent mapping")
                
                # Track subagent namespace for remapping
                if stream_mode == "updates":
                    updates = data
                    logger.info(f"UPDATES: namespace={_namespace}, updates_keys={list(updates.keys()) if isinstance(updates, dict) else type(updates)}")
                    # Check if this is the start of a subagent (PatchToolCallsMiddleware.before_agent)
                    if isinstance(updates, dict) and "PatchToolCallsMiddleware.before_agent" in updates:
                        if _namespace and isinstance(_namespace, tuple) and len(_namespace) > 0:
                            namespace_key = _namespace[0]
                            if namespace_key and latest_task_id:
                                namespace_to_task_id[namespace_key] = latest_task_id
                                logger.info(f"MAP: namespace_key={namespace_key} -> task_id={latest_task_id}")

                if stream_mode == "updates":
                    updates = data
                    logger.info(f"UPDATES: namespace={_namespace}, updates_keys={list(updates.keys()) if isinstance(updates, dict) else type(updates)}")
                    if isinstance(updates, dict) and "__interrupt__" in updates:
                        logger.info(f"INTERRUPT FOUND: namespace={_namespace}, interrupt={updates.get('__interrupt__')}")
                        interrupt_objs = updates.get("__interrupt__")
                        if interrupt_objs:
                            for interrupt_obj in interrupt_objs:
                                interrupt_value = interrupt_obj.value
                                if not isinstance(interrupt_value, dict):
                                    raise RequestError(
                                        -32600,
                                        (
                                            "ACP limitation: this agent raised a free-form "
                                            "LangGraph interrupt(), which ACP cannot display.\n\n"
                                            "ACP only supports human-in-the-loop permission "
                                            "prompts with a fixed set of decisions "
                                            "(approve/reject/edit).\n"
                                            "Spec: https://agentclientprotocol.com/protocol/overview\n\n"
                                            "Fix: use LangChain HumanInTheLoopMiddleware-style "
                                            "interrupts (action_requests/review_configs).\n"
                                            "Docs: https://docs.langchain.com/oss/python/langchain/"
                                            "human-in-the-loop\n\n"
                                            "This is a protocol limitation, not a bug in the agent."
                                        ),
                                        {"interrupt_value": interrupt_value},
                                    )

                            current_state = await agent.aget_state(config)

                            # Get parent task tool_call_id for remapping
                            parent_tool_call_id = None
                            if _namespace:
                                if isinstance(_namespace, tuple):
                                    namespace_key = (
                                        _namespace[0] if _namespace else None
                                    )
                                else:
                                    namespace_key = str(_namespace)
                                parent_tool_call_id = namespace_to_task_id.get(
                                    namespace_key
                                )
                                logger.info(f"INTERRUPT: namespace={_namespace}, namespace_key={namespace_key}, parent_tool_call_id={parent_tool_call_id}")

                            user_decisions = (
                                await self._handle_interrupts_with_remapping(
                                    current_state=current_state,
                                    session_id=session_id,
                                    interrupt_objs=interrupt_objs,
                                    parent_tool_call_id=parent_tool_call_id,
                                )
                            )
                            break

                    for node_name, update in updates.items():
                        if (
                            node_name == "tools"
                            and isinstance(update, dict)
                            and "todos" in update
                        ):
                            todos = update.get("todos", [])
                            if todos:
                                await self._handle_todo_update(
                                    session_id, todos, log_plan=False
                                )

                    continue

                message_chunk, _metadata = data

                # Process tool call chunks
                await self._process_tool_call_chunks(
                    session_id,
                    message_chunk,
                    active_tool_calls,
                    tool_call_accumulator,
                )

                if isinstance(message_chunk, str):
                    if not _namespace:
                        await self._log_text(text=message_chunk, session_id=session_id)
                # Check for tool results (ToolMessage responses)
                elif hasattr(message_chunk, "type") and message_chunk.type == "tool":
                    # This is a tool result message
                    tool_call_id = getattr(message_chunk, "tool_call_id", None)
                    if (
                        tool_call_id
                        and tool_call_id in active_tool_calls
                        and active_tool_calls[tool_call_id].get("name") != "edit_file"
                    ):
                        # Update the tool call with completion status and result
                        content = getattr(message_chunk, "content", "")
                        tool_info = active_tool_calls[tool_call_id]
                        tool_name = tool_info.get("name")

                        # Format execute tool results specially
                        if tool_name == "execute":
                            tool_args = tool_info.get("args", {})
                            command = tool_args.get("command", "")
                            formatted_content = format_execute_result(
                                command=command, result=str(content)
                            )
                        else:
                            formatted_content = str(content)
                        update = update_tool_call(
                            tool_call_id=tool_call_id,
                            status="completed",
                            content=[tool_content(text_block(formatted_content))],
                        )
                        await self._conn.session_update(
                            session_id=session_id, update=update, source="DeepAgent"
                        )

                elif message_chunk.content:
                    # content can be a string or a list of content blocks
                    if isinstance(message_chunk.content, str):
                        text = message_chunk.content
                    elif isinstance(message_chunk.content, list):
                        # Extract text from content blocks
                        text = ""
                        for block in message_chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text += block.get("text", "")
                            elif isinstance(block, str):
                                text += block
                    else:
                        text = str(message_chunk.content)

                    if text and not _namespace:
                        await self._log_text(text=text, session_id=session_id)

            # After streaming completes, check if we need to exit the loop
            # The loop continues while there are interrupts (line 467)
            # We get the current state to check the loop condition
            current_state = await agent.aget_state(config)
            # Note: Interrupts are handled during streaming via __interrupt__ updates
            # This state check is only for the while loop condition

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

        return PromptResponse(stop_reason="end_turn")

    async def _handle_interrupts_with_remapping(
        self,
        *,
        current_state: Any,
        session_id: str,
        interrupt_objs: list,
        parent_tool_call_id: str | None,
    ) -> list[dict[str, Any]]:
        """Handle interrupts with tool_call_id remapping for subagents."""
        user_decisions: list[dict[str, Any]] = []
        logger.info(f"_handle_interrupts_with_remapping: parent_tool_call_id={parent_tool_call_id}, num_interrupts={len(interrupt_objs)}")

        if current_state.next and interrupt_objs:
            for interrupt in interrupt_objs:
                # Use parent tool_call_id if available, otherwise use interrupt.id
                tool_call_id = parent_tool_call_id or interrupt.id
                logger.info(f"Handling interrupt: tool_call_id={tool_call_id} (parent={parent_tool_call_id}, interrupt.id={interrupt.id})")
                interrupt_value = interrupt.value

                # Extract action requests from interrupt_value
                action_requests = []
                if isinstance(interrupt_value, dict):
                    action_requests = interrupt_value.get("action_requests", [])

                # Process each action request
                for action in action_requests:
                    tool_name = action.get("name", "tool")
                    tool_args = action.get("args", {})

                    # Create a title for the permission request
                    if tool_name == "write_todos":
                        title = "Review Plan"
                        # Log the plan text when requesting approval
                        todos = tool_args.get("todos", [])
                        plan_text = "## Plan\n\n"
                        for i, todo in enumerate(todos, 1):
                            content = todo.get("content", "")
                            plan_text += f"{i}. {content}\n"
                        await self._log_text(session_id=session_id, text=plan_text)
                    elif tool_name == "edit_file" and isinstance(tool_args, dict):
                        file_path = tool_args.get("file_path", "file")
                        title = f"Edit `{file_path}`"
                    elif tool_name == "write_file" and isinstance(tool_args, dict):
                        file_path = tool_args.get("file_path", "file")
                        title = f"Write `{file_path}`"
                    elif tool_name == "execute" and isinstance(tool_args, dict):
                        command = tool_args.get("command", "")
                        # Truncate long commands for display
                        display_command = truncate_execute_command_for_display(
                            command=command
                        )
                        title = (
                            f"Execute: `{display_command}`"
                            if command
                            else "Execute command"
                        )
                    else:
                        title = tool_name

                    desc = tool_name
                    if tool_name == "execute" and isinstance(tool_args, dict):
                        command = tool_args.get("command", "")
                        command_types = extract_command_types(command)
                        if command_types:
                            # Create a descriptive name based on the command types
                            if len(command_types) == 1:
                                desc = f"`{command_types[0]}`"
                            else:
                                # Show all unique command types
                                unique_types = list(
                                    dict.fromkeys(command_types)
                                )  # Preserve order, remove duplicates
                                desc = ", ".join(f"`{ct}`" for ct in unique_types)

                    # Create permission options
                    options = [
                        PermissionOption(
                            option_id="approve",
                            name="Approve",
                            kind="allow_once",
                        ),
                        PermissionOption(
                            option_id="reject",
                            name="Reject",
                            kind="reject_once",
                        ),
                        PermissionOption(
                            option_id="approve_always",
                            name=f"Always allow {desc} commands",
                            kind="allow_always",
                        ),
                    ]

                    # Request permission from the client with remapped tool_call_id
                    tool_call_update = ToolCallUpdate(
                        tool_call_id=tool_call_id, title=title, raw_input=tool_args
                    )
                    response = await self._conn.request_permission(
                        session_id=session_id,
                        tool_call=tool_call_update,
                        options=options,
                    )
                    # Handle the user's decision
                    if response.outcome.outcome == "selected":
                        decision_type = response.outcome.option_id
                        user_decisions.append({"type": decision_type})

        return user_decisions

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
