"""Context variable for sharing the sandbox backend with skills.

Skills loaded from ``.metalgate/skills.py`` run on the host process, but
shell commands and file operations should execute inside the agent's
sandbox.  ``DynamicToolsMiddleware`` sets this context variable before
invoking a skill so the skill can call :func:`get_backend` to access the
sandbox backend.
"""

import contextvars

from deepagents.backends.protocol import SandboxBackendProtocol

_backend: contextvars.ContextVar["SandboxBackendProtocol | None"] = (
    contextvars.ContextVar("metalgate_backend", default=None)
)


def get_backend() -> SandboxBackendProtocol | None:
    """Return the sandbox backend for the current skill execution, or ``None``."""
    return _backend.get()


def set_backend(
    backend: SandboxBackendProtocol | None,
) -> contextvars.Token[SandboxBackendProtocol | None]:
    """Set the backend for the current context. Returns a token for reset."""
    return _backend.set(backend)


def reset_backend(
    token: contextvars.Token[SandboxBackendProtocol | None],
) -> None:
    """Reset the backend to its previous value."""
    _backend.reset(token)
