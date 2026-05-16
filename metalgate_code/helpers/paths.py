"""
Databases paths helpers.
"""

from pathlib import Path


def get_home_path(cwd: str) -> Path:
    """Get home path for a project.

    Args:
        cwd: Project working directory

    Returns:
        Path to home directory
    """
    data_dir = Path.home() / ".metalgate" / "memory" / (Path(cwd).name or "unknown")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_checkpoints_data_dir(cwd: str) -> Path:
    """Get checkpoints database path for a project.

    Args:
        cwd: Project working directory

    Returns:
        Path to checkpoints database
    """
    return get_home_path(cwd) / "checkpoints.db"


def get_memory_data_dir() -> Path:
    """Get Mem0 memory data directory.

    This is where Mem0 stores Chroma vectors and SQLite history.

    Returns:
        Path to memory data directory
    """
    data_dir = Path.home() / ".metalgate" / "memory" / "mem0"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_index_data_dir(cwd: str) -> Path:
    """Get index database path for a project.

    Args:
        cwd: Project working directory

    Returns:
        Path to index database
    """
    return get_home_path(cwd) / "index.db"
