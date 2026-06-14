"""Real-time contextual symbol search tools."""

from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol

from metalgate_code.context.cache import CodeCache
from metalgate_code.context.go_tracer import GoTracer
from metalgate_code.context.python_tracer import PythonTracer
from metalgate_code.context.tools import make_tools
from metalgate_code.context.tracer_base import Tracer
from metalgate_code.helpers.paths import get_context_cache_dir


def _detect_language(root: str) -> str:
    """Detect the dominant language of the project."""
    root_path = Path(root).resolve()
    go_mod = root_path / "go.mod"
    if go_mod.exists():
        return "go"
    # Default to Python if no go.mod found
    return "python"


def _create_tracer(
    root: str,
    backend: SandboxBackendProtocol,
    cache: CodeCache,
    language: str | None = None,
) -> Tracer:
    """Create the appropriate tracer for the detected language."""
    if language is None:
        language = _detect_language(root)

    if language == "go":
        return GoTracer(root=root, backend=backend, cache=cache)
    return PythonTracer(root=root, backend=backend, cache=cache)


def get_code_tools(
    cwd: str,
    backend: SandboxBackendProtocol,
    cache_path: str | None = None,
    language: str | None = None,
) -> list:
    """Build and return the six code-navigation tool functions."""
    if cache_path is None:
        cache_path = str(get_context_cache_dir(cwd))

    cache = CodeCache(cache_path)
    tracer = _create_tracer(root=cwd, backend=backend, cache=cache, language=language)
    return make_tools(tracer)


__all__ = ["get_code_tools", "Tracer", "PythonTracer", "GoTracer"]
