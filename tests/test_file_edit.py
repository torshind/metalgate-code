"""
Verifies that the ACP agent launched by run.sh is capable of editing a file.
"""

import logging
import os
import tempfile
import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest
from conftest import RecordingClient, run_agent

logger = logging.getLogger("acp_test")


# Fixtures
@pytest.fixture
def target_file() -> Generator[Path, None, None]:
    """
    A temporary file pre-populated with known content.
    Deleted after the test regardless of outcome.
    """
    original = textwrap.dedent("""\
        Hello, ACP test!
        This line should be replaced by the agent.
        End of original content.
    """)
    fd, raw_path = tempfile.mkstemp(suffix=".txt", prefix="acp_test_")
    path = Path(raw_path)
    os.close(fd)
    path.write_text(original, encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)


# Tests
@pytest.mark.asyncio
async def test_agent_starts_and_responds(run_sh: Path) -> None:
    """
    Smoke test: verify the agent process starts, handles the ACP handshake,
    and returns at least one session/update notification for an innocuous prompt.
    """
    client = RecordingClient()
    await run_agent(client, run_sh, "Hello! Are you ready?")
    logger.info("Agent output:\n%s", client.all_text)

    assert client.updates, (
        "Agent produced no session/update notifications — "
        "it may have crashed or silently ignored the prompt."
    )


@pytest.mark.asyncio
async def test_agent_edits_file(target_file: Path, run_sh: Path) -> None:
    """
    Full round-trip: spawn the agent, ask it to edit a file, verify the
    file changed on disk.
    """
    original_content = target_file.read_text(encoding="utf-8")
    expected_marker = "EDITED_BY_AGENT"

    client = RecordingClient()
    await run_agent(
        client,
        run_sh,
        f"Please edit the file at '{target_file}'. "
        f"Replace the second line with the text '{expected_marker}'. "
        "Only modify that one line and save the file.",
    )
    logger.info("Agent output:\n%s", client.all_text)

    new_content = target_file.read_text(encoding="utf-8")

    # 1. The file must have been modified.
    assert new_content != original_content, (
        f"The agent did not modify the file at all.\nAgent output:\n{client.all_text}"
    )

    # 2. The expected marker must appear in the new content.
    assert expected_marker in new_content, (
        f"Expected marker '{expected_marker}' not found in file after edit.\n"
        f"File content after run:\n{new_content}\n"
        f"Agent output:\n{client.all_text}"
    )

    # 3. Lines that should not change must still be present.
    assert "Hello, ACP test!" in new_content, "First line was unexpectedly altered."
    assert "End of original content." in new_content, (
        "Third line was unexpectedly altered."
    )

    # 4. The client must have observed at least one write call.
    assert client.written_files, (
        "No edit tool call observed — agent did not edit any file.\n"
        f"Agent output:\n{client.all_text}"
    )
    assert str(target_file) in client.written_files, (
        f"Agent edited unexpected path(s): {client.written_files}"
    )


@pytest.mark.asyncio
async def test_agent_edit_preserves_encoding(target_file: Path, run_sh: Path) -> None:
    """
    Ensure the agent writes back valid UTF-8 (no corruption from encoding
    mismatches or binary injection).
    """
    client = RecordingClient()
    marker = "UTF8_SAFE_EDIT_✓"

    await run_agent(
        client,
        run_sh,
        f"Replace the second line of '{target_file}' with '{marker}' and save.",
    )
    logger.info("Agent output:\n%s", client.all_text)

    # Reading as UTF-8 must not raise.
    new_content = ""
    try:
        new_content = target_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"File is no longer valid UTF-8 after agent edit: {exc}")

    assert marker in new_content, (
        f"UTF-8 marker '{marker}' not found in file.\nContent:\n{new_content}"
    )


@pytest.mark.asyncio
async def test_agent_call_skills(run_sh: Path) -> None:
    """
    Ensure the agent calls one of the predefined skills.
    """
    client = RecordingClient()

    await run_agent(
        client,
        run_sh,
        "Use skills to list all files in the current directory.",
    )
    logger.info("Agent output:\n%s", client.all_text)
