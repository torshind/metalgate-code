"""
Registry for textual skill documents.

Scans .metalgate/skills/ for subdirectories containing SKILL.md files.
Each subdirectory name becomes the skill name (e.g. deploy, migrations, code-review).
"""

import logging
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol

logger = logging.getLogger("metalgate_code")

SKILLS_DIR = Path(".metalgate") / "skills"
SKILL_FILENAME = "SKILL.md"


class TextualSkillRegistry:
    """Registry that loads and caches textual skill documents."""

    def __init__(self):
        self._skills: dict[str, str] = {}
        self._project_path: Path | None = None
        self._skills_dir: Path | None = None
        self._backend: SandboxBackendProtocol | None = None

    def load(
        self,
        project_path: str | Path,
        backend: SandboxBackendProtocol | None = None,
    ) -> None:
        """Scan .metalgate/skills/ and load all SKILL.md files."""
        self._project_path = Path(project_path)
        self._skills_dir = self._project_path / SKILLS_DIR
        self._backend = backend
        self._skills = {}

        if not self._path_exists(self._skills_dir):
            logger.info(f"No textual skills directory at {self._skills_dir}")
            return

        for entry in self._list_dir(self._skills_dir):
            skill_dir = self._skills_dir / entry
            skill_file = skill_dir / SKILL_FILENAME
            if self._path_exists(skill_file):
                try:
                    content = self._read_text(skill_file)
                    self._skills[entry] = content
                    logger.info(
                        f"Loaded textual skill '{entry}' ({len(content)} bytes)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load textual skill '{entry}': {e}")

        logger.info(
            f"Loaded {len(self._skills)} textual skills: {list(self._skills.keys())}"
        )

    def _path_exists(self, path: Path) -> bool:
        """Check if path exists, using backend if available."""
        if self._backend is not None:
            result = self._backend.execute(f"test -e {path} && echo 'exists'")
            return "exists" in result.output
        return path.exists()

    def _list_dir(self, path: Path) -> list[str]:
        """List directory entries, using backend if available."""
        if self._backend is not None:
            result = self._backend.execute(f"ls -1 {path}")
            return [line.strip() for line in result.output.splitlines() if line.strip()]
        return [entry.name for entry in path.iterdir() if entry.is_dir()]

    def _read_text(self, path: Path) -> str:
        """Read file text, using backend if available."""
        if self._backend is not None:
            result = self._backend.read(str(path))
            if result.error:
                raise FileNotFoundError(f"Cannot read {path}: {result.error}")
            return result.file_data["content"] if result.file_data else ""
        return path.read_text(encoding="utf-8")

    def list(self) -> list[str]:
        """Return names of all loaded textual skills."""
        return list(self._skills.keys())

    def get(self, name: str) -> str | None:
        """Return the content of a textual skill by name, or None if not found."""
        return self._skills.get(name)

    def get_all(self) -> dict[str, str]:
        """Return a copy of all loaded textual skills."""
        return dict(self._skills)

    def reload(self) -> None:
        """Reload textual skills from disk."""
        if self._project_path is not None:
            self.load(self._project_path, backend=self._backend)


textual_registry = TextualSkillRegistry()
