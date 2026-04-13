"""
Session mode configuration.
"""

from acp.schema import SessionMode, SessionModeState

AVAILABLE_MODES = [
    SessionMode(
        id="ask_before_edits",
        name="Ask before edits",
        description="Ask permission before edits, writes, shell commands, and plans",
    ),
    SessionMode(
        id="accept_edits",
        name="Accept edits",
        description="Auto-accept edit operations, but ask before shell commands and plans",
    ),
    SessionMode(
        id="accept_everything",
        name="Accept everything",
        description="Auto-accept all operations without asking permission",
    ),
]


INTERRUPT_CONFIGS = {
    "ask_before_edits": {
        "edit_file": {"allowed_decisions": ["approve", "reject"]},
        "write_file": {"allowed_decisions": ["approve", "reject"]},
        "write_todos": {"allowed_decisions": ["approve", "reject"]},
        "execute": {"allowed_decisions": ["approve", "reject"]},
    },
    "accept_edits": {
        "write_todos": {"allowed_decisions": ["approve", "reject"]},
        "execute": {"allowed_decisions": ["approve", "reject"]},
    },
    "accept_everything": {},
}


def get_interrupt_config(mode_id: str) -> dict:
    """Get interrupt configuration for a given mode."""
    return INTERRUPT_CONFIGS.get(mode_id, {})


def get_available_modes() -> SessionModeState:
    """Get default session mode configuration."""
    return SessionModeState(
        current_mode_id="accept_edits",
        available_modes=AVAILABLE_MODES,
    )
