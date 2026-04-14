"""
Checkpointer paths for ACP session persistence.

Session data is stored in ~/.metalgate/memory/<project>/checkpoints.db
using AsyncSqliteSaver, which is set up in agent.py.
"""

from pathlib import Path


def get_db_path(cwd: str) -> Path:
    """Get database path for a project.

    Args:
        cwd: Project working directory

    Returns:
        Path to checkpoints database
    """
    project = Path(cwd).name or "unknown"
    db_dir = Path.home() / ".metalgate" / "memory" / project
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "checkpoints.db"
