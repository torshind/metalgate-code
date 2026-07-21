#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"
MEM0_TELEMETRY=False uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/agent.py"
