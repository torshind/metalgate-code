"""Real-time contextual symbol search tools."""

from deepagents.backends.protocol import SandboxBackendProtocol

from metalgate_code.context.cache import CodeCache
from metalgate_code.context.tools import make_tools
from metalgate_code.context.tracer import Tracer
from metalgate_code.helpers.paths import get_context_cache_dir


def get_code_tools(
    cwd: str,
    backend: SandboxBackendProtocol,
    cache_path: str | None = None,
) -> list:
    """Build and return the six code-navigation tool functions."""
    if cache_path is None:
        cache_path = str(get_context_cache_dir(cwd))

    cache = CodeCache(cache_path)
    tracer = Tracer(root=cwd, backend=backend, cache=cache)
    return make_tools(tracer)


__all__ = ["get_code_tools"]
