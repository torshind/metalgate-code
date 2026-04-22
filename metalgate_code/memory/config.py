"""
Mem0 memory configuration constants.
"""

# Agent IDs for scoping memories
SEMANTIC_AGENT_ID = "semantic"  # Extracted facts, preferences, project patterns
EPISODIC_AGENT_ID = "episodic"  # Session summaries
USER_AGENT_ID = "user"  # User-level preferences, identity, cross-project conventions

# Default memory limits
DEFAULT_EPISODIC_LIMIT = 5


SEMANTIC_INSTRUCTIONS = """
Only extract facts that remain true and relevant beyond this session:
architectural decisions, technical conventions, and stable user preferences about the codebase and the tooling.
Ignore anything that is specific to the current task or temporary in nature.
"""

EPISODIC_INSTRUCTIONS = """
Extract memories from this coding session for future retrieval.

EXTRACT:
- What was attempted and the outcome
- Specific problems hit and how they were resolved
- Context that would help resume interrupted work

SKIP: Architectural decisions, conventions, or stable preferences (tracked separately). Generic knowledge.

Each memory must be self-contained with concrete identifiers (paths, names, errors). No vague references.
"""

USER_INSTRUCTIONS = """
Only extract facts that apply universally across ALL projects and sessions:
user identity, personal preferences with no scope, and project independent conventions.
Ignore anything project specific, temporary, or tied to a particular scope.
"""
