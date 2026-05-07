"""
registry.py
"""

import logging
from pathlib import Path

from langchain_core.tools import BaseTool

logger = logging.getLogger("metalgate_code")


class SkillRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._skills_path: Path | None = None

    def load(self, project_path: str | Path):
        """Load project skills if a .metalgate/skills.py exists. No-op otherwise."""
        self._skills_path = Path(project_path) / ".metalgate" / "skills.py"
        if self._skills_path.exists():
            logger.info(f"Loading skills from {self._skills_path}")
            self.reload()
        else:
            logger.info(f"No skills.py found at {self._skills_path}")

    def reload(self):
        if self._skills_path is None or not self._skills_path.exists():
            return
        try:
            source = self._skills_path.read_text()
            logger.info(f"Compiling {len(source)} bytes from {self._skills_path}")
            module_globals = {}
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
