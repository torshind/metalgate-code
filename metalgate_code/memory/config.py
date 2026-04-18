"""
Mem0 memory configuration constants.
"""

# Agent IDs for scoping memories
HEURISTIC_AGENT_ID = "heuristic"  # Extracted facts, preferences, project patterns
HISTORICAL_AGENT_ID = "historical"  # Session summaries

# Default memory limits
DEFAULT_HISTORICAL_LIMIT = 5


HEURISTIC_INSTRUCTIONS = """
Only extract facts that remain true and relevant beyond this session:
architectural decisions, technical conventions, and stable user preferences about the codebase and the tooling.
Ignore anything that is specific to the current task or temporary in nature.
"""

HISTORICAL_INSTRUCTIONS = """
Extract memories from this coding session for future retrieval.

EXTRACT:
- What was attempted and the outcome
- Specific problems hit and how they were resolved
- Context that would help resume interrupted work

SKIP: Architectural decisions, conventions, or stable preferences (tracked separately). Generic knowledge.

Each memory must be self-contained with concrete identifiers (paths, names, errors). No vague references.
"""
