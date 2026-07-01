"""
registry.py
"""

import logging
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.tools import BaseTool, tool

from metalgate_code.context.backend_context import get_backend

logger = logging.getLogger("metalgate_code")


class SkillRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._skills_path: Path | None = None
        self._backend: SandboxBackendProtocol | None = None

    def load(
        self, project_path: str | Path, backend: SandboxBackendProtocol | None = None
    ):
        """Load project skills if a .metalgate/skills.py exists. No-op otherwise."""
        self._skills_path = Path(project_path) / ".metalgate" / "skills.py"
        self._backend = backend
        if self._path_exists(self._skills_path):
            logger.info(f"Loading skills from {self._skills_path}")
            self.reload()
        else:
            logger.info(f"No skills.py found at {self._skills_path}")

    def _path_exists(self, path: Path) -> bool:
        """Check if path exists, using backend if available."""
        if self._backend is not None:
            result = self._backend.execute(f"test -f {path} && echo exists")
            return "exists" in result.output
        return path.exists()

    def _read_text(self, path: Path) -> str:
        """Read file text, using backend if available."""
        if self._backend is not None:
            result = self._backend.read(str(path))
            if result.error:
                raise FileNotFoundError(f"Cannot read {path}: {result.error}")
            return result.file_data["content"]
        return path.read_text()

    def reload(self):
        if self._skills_path is None or not self._path_exists(self._skills_path):
            logger.info("Path does not exist, skipping reload")
            return
        try:
            source = self._read_text(self._skills_path)
            logger.info(f"Compiling {len(source)} bytes from {self._skills_path}")
            logger.info(f"Source: {source!r}")
            module_globals = {"get_backend": get_backend, "tool": tool}
            exec(compile(source, str(self._skills_path), "exec"), module_globals)
            logger.info(
                f"Loaded {len(module_globals)} objects from {self._skills_path}"
            )
            self._tools = {
                obj.name: obj
                for obj in module_globals.values()
                if isinstance(obj, BaseTool)
            }
        except Exception as e:
            logger.error(
                f"Failed to load skills from {self._skills_path}: {e}", exc_info=True
            )
            raise
        logger.info(f"Loaded {len(self._tools)} skills from {self._skills_path}")

    @property
    def skills_path(self) -> Path:
        if self._skills_path is None:
            raise RuntimeError(
                "Registry not loaded. Call registry.load(project_path) first."
            )
        return self._skills_path

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)


registry = SkillRegistry()
