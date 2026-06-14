"""Abstract base class and shared helpers for language-specific Tracers."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from deepagents.backends.protocol import SandboxBackendProtocol

from metalgate_code.context.cache import CodeCache

logger = logging.getLogger("metalgate_code")

_MAX_CALLERS = 50
_CALLERS_TIMEOUT = 15.0
_CALLERS_WORKERS = os.environ.get("CALLERS_WORKERS", 4)


class Tracer(ABC):
    """Abstract base for language-specific code navigation."""

    def __init__(
        self,
        root: str,
        backend: SandboxBackendProtocol,
        cache: CodeCache,
    ) -> None:
        self.root = Path(root).resolve()
        self.cache = cache
        self.backend = backend

    def _read_file(self, file: str, limit: int = 10000) -> str:
        """Read file content using backend if available, otherwise local filesystem."""
        if self.backend is not None:
            result = self.backend.read(file, offset=0, limit=limit)
            if result.error is None and result.file_data is not None:
                return result.file_data["content"]
        return Path(file).read_text(encoding="utf-8", errors="ignore")

    def _read_file_bytes(self, file: str, limit: int = 10000) -> bytes:
        """Read file content as bytes using backend if available."""
        return self._read_file(file, limit=limit).encode("utf-8", errors="ignore")

    # ------------------------------------------------------------------ #
    # public interface — every subclass must implement these six methods
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_file_outline(self, file: str) -> list[dict]:
        """Return every class/function/method defined in *file*."""
        ...

    @abstractmethod
    def goto_definition(
        self, file: str, line: int, name: Optional[str] = None
    ) -> Optional[dict]:
        """Resolve the symbol *name* on *line* of *file* to its definition."""
        ...

    @abstractmethod
    def get_source(self, file: str, line: int, context: int = 60) -> dict:
        """Return the full source of the function/class starting on *line*."""
        ...

    @abstractmethod
    def get_callers(self, file: str, line: int) -> list[dict]:
        """Find every place in the project that references the symbol on *line* of *file*."""
        ...

    @abstractmethod
    def get_callees(self, file: str, line: int) -> list[dict]:
        """Find every symbol called by the function on *line* of *file*."""
        ...

    @abstractmethod
    def find_symbol(self, name: str) -> list[dict]:
        """Search for *name* across the project."""
        ...
