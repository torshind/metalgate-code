#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/agent.py"
