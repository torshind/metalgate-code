"""Integration tests for MicrosandboxBackend against real microVMs.

These tests boot actual microsandbox microVMs and exercise every public
method of ``MicrosandboxBackend`` end-to-end.  No mocks are used for the
sandbox itself — the goal is to verify that the backend correctly drives
a real microVM.

Requires the microsandbox runtime (``msb``) to be installed and functional.
"""

from __future__ import annotations

import base64
import textwrap
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from metalgate_code.factory.microsandbox_backend import MicrosandboxBackend

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A temp directory that gets bind-mounted into the VM as /workspace."""
    return tmp_path


@pytest_asyncio.fixture
async def backend(workspace: Path) -> AsyncGenerator[MicrosandboxBackend, None]:
    """Create a real MicrosandboxBackend, boot the VM, and tear it down after."""
    b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
    yield b
    await b.stop()


# --------------------------------------------------------------------------- #
# __init__ and properties
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestInit:
    async def test_defaults(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._image == "python"
        assert b._cpus == 4
        assert b._memory == 4096
        assert b._env == {}
        assert b._secrets == []
        assert b._sandbox is None
        assert b.id.startswith("msb-")
        await b.stop()

    async def test_custom_params(self, workspace: Path):
        b = MicrosandboxBackend(
            root_dir=str(workspace),
            image="ubuntu:24.04",
            env={"FOO": "bar"},
            cpus=2,
            memory=2048,
        )
        assert b._image == "ubuntu:24.04"
        assert b._cpus == 2
        assert b._memory == 2048
        assert b._env == {"FOO": "bar"}
        await b.stop()

    async def test_inherit_env(self, workspace: Path):
        b = MicrosandboxBackend(
            root_dir=str(workspace),
            env={"CUSTOM": "val"},
            inherit_env=True,
        )
        assert "PATH" in b._env
        assert b._env["CUSTOM"] == "val"
        await b.stop()

    async def test_root_dir_resolved(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace / "subdir"))
        assert b._root_dir == str((workspace / "subdir").resolve())
        await b.stop()

    async def test_eager_boots_sandbox(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), eager=True, memory=1024)
        assert b._sandbox is not None
        await b.stop()

    async def test_not_eager_does_not_boot(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._sandbox is None
        await b.stop()


@pytest.mark.asyncio
class TestProperties:
    async def test_id(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b.id == b._sandbox_id
        assert b.id.startswith("msb-")
        await b.stop()

    async def test_cwd(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b.cwd == "/workspace"
        await b.stop()


# --------------------------------------------------------------------------- #
# Sandbox lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestSandboxLifecycle:
    async def test_ensure_sandbox_creates_once(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        sb1 = await b._ensure_sandbox()
        sb2 = await b._ensure_sandbox()
        assert sb1 is sb2
        await b.stop()

    async def test_ensure_sandbox_idempotent_under_concurrency(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        import asyncio

        try:
            sb1, sb2 = await asyncio.gather(
                b._ensure_sandbox(),
                b._ensure_sandbox(),
            )
            assert sb1 is sb2
        finally:
            await b.stop()

    async def test_stop_clears_sandbox(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        await b._ensure_sandbox()
        assert b._sandbox is not None
        await b.stop()
        assert b._sandbox is None

    async def test_stop_when_not_started(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        await b.stop()
        assert b._sandbox is None


# --------------------------------------------------------------------------- #
# Command execution: aexecute / execute
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestExecute:
    async def test_aexecute_success(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("echo hello world")
        assert result.exit_code == 0
        assert "hello world" in result.output
        assert not result.truncated

    async def test_aexecute_with_stderr(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("echo out; echo err >&2")
        assert result.exit_code == 0
        assert "out" in result.output
        assert "[stderr] err" in result.output

    async def test_aexecute_nonzero_exit(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("exit 3")
        assert result.exit_code == 3
        assert "Exit code: 3" in result.output

    async def test_aexecute_no_output(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("true")
        assert result.exit_code == 0
        assert result.output == "<no output>"

    async def test_aexecute_empty_command(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("")
        assert result.exit_code == 1
        assert "non-empty string" in result.output

    async def test_aexecute_non_string_command(self, backend: MicrosandboxBackend):
        result = await backend.aexecute(123)  # ty: ignore[invalid-argument-type]
        assert result.exit_code == 1
        assert "non-empty string" in result.output

    async def test_aexecute_shell_exception(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("this_command_does_not_exist_xyz")
        assert result.exit_code != 0

    async def test_aexecute_timeout_default(self, backend: MicrosandboxBackend):
        # A fast command completes well within the default timeout.
        result = await backend.aexecute("echo fast")
        assert result.exit_code == 0

    async def test_aexecute_timeout_custom(self, backend: MicrosandboxBackend):
        # A custom timeout larger than the command duration succeeds.
        result = await backend.aexecute("echo fast", timeout=30)
        assert result.exit_code == 0

    async def test_aexecute_timeout_actually_enforced(
        self, backend: MicrosandboxBackend
    ):
        # A command that sleeps longer than the timeout must be killed.
        # microsandbox raises ExecTimeoutError, caught by the generic
        # exception handler → exit_code 1 (not 124, which is reserved for
        # the outer asyncio.wait_for guard).
        result = await backend.aexecute("sleep 10", timeout=2)
        assert result.exit_code != 0
        assert "timed out" in result.output.lower()

    async def test_aexecute_timeout_zero_disables(self, backend: MicrosandboxBackend):
        # timeout=0 is reset to the default (not infinite), so a fast
        # command still completes normally.
        result = await backend.aexecute("echo fast", timeout=0)
        assert result.exit_code == 0

    async def test_aexecute_truncation(self, backend: MicrosandboxBackend):
        from metalgate_code.factory.microsandbox_backend import _MAX_OUTPUT_BYTES

        big = "x" * (_MAX_OUTPUT_BYTES + 500)
        result = await backend.aexecute(f"printf '%s' '{big}'")
        assert result.truncated
        assert "truncated" in result.output.lower()

    async def test_aexecute_cwd_is_workspace(self, backend: MicrosandboxBackend):
        result = await backend.aexecute("pwd")
        assert result.exit_code == 0
        assert "/workspace" in result.output

    async def test_aexecute_env_vars(self, workspace: Path):
        b = MicrosandboxBackend(
            root_dir=str(workspace),
            env={"TEST_VAR": "env_value_123"},
            memory=1024,
        )
        try:
            result = await b.aexecute("echo $TEST_VAR")
            assert result.exit_code == 0
            assert "env_value_123" in result.output
        finally:
            await b.stop()

    async def test_execute_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            result = b.execute("echo sync works")
            assert result.exit_code == 0
            assert "sync works" in result.output
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# File read: aread / read
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestRead:
    async def test_aread_text_file(self, backend: MicrosandboxBackend, workspace: Path):
        (workspace / "test.txt").write_text("line1\nline2\nline3\n")
        result = await backend.aread("test.txt")
        assert result.error is None
        assert result.file_data is not None
        assert "line1" in result.file_data["content"]
        assert "line3" in result.file_data["content"]
        assert result.file_data["encoding"] == "utf-8"

    async def test_aread_binary_file(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        binary_data = b"\x00\x01\x02\xff\xfe"
        (workspace / "blob.bin").write_bytes(binary_data)
        result = await backend.aread("blob.bin")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        decoded = base64.standard_b64decode(result.file_data["content"])
        assert decoded == binary_data

    async def test_aread_empty_file(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "empty.txt").write_text("")
        result = await backend.aread("empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert (
            "empty" in result.file_data["content"].lower()
            or result.file_data["content"] == ""
        )

    async def test_aread_whitespace_only_file(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "ws.txt").write_text("   \n  \t\n")
        result = await backend.aread("ws.txt")
        assert result.error is None
        assert result.file_data is not None

    async def test_aread_file_not_found(self, backend: MicrosandboxBackend):
        result = await backend.aread("does_not_exist.txt")
        assert result.error is not None
        assert "does_not_exist.txt" in result.error

    async def test_aread_pagination(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        content = "".join(f"line{i}\n" for i in range(10))
        (workspace / "big.txt").write_text(content)
        result = await backend.aread("big.txt", offset=2, limit=3)
        assert result.error is None
        assert result.file_data is not None
        page = result.file_data["content"]
        assert "line2" in page
        assert "line4" in page
        assert "line5" not in page

    async def test_aread_offset_exceeds_length(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "short.txt").write_text("only one line\n")
        result = await backend.aread("short.txt", offset=100)
        assert result.error is not None
        assert "100" in result.error

    async def test_aread_absolute_host_path(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "abs.txt").write_text("abs path content")
        abs_path = str(workspace / "subdir" / "abs.txt")
        result = await backend.aread(abs_path)
        assert result.error is None
        assert result.file_data is not None
        assert "abs path content" in result.file_data["content"]

    async def test_aread_guest_path_passthrough(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "guest.txt").write_text("guest path")
        result = await backend.aread("/workspace/guest.txt")
        assert result.error is None
        assert result.file_data is not None
        assert "guest path" in result.file_data["content"]

    async def test_read_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            (workspace / "sync.txt").write_text("sync read")
            result = b.read("sync.txt")
            assert result.error is None
            assert result.file_data is not None
            assert "sync read" in result.file_data["content"]
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# File write: awrite / write
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestWrite:
    async def test_awrite_new_file(self, backend: MicrosandboxBackend, workspace: Path):
        result = await backend.awrite("new.txt", "hello world")
        assert result.error is None
        assert result.path == "new.txt"
        assert (workspace / "new.txt").read_text() == "hello world"

    async def test_awrite_existing_file_fails(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "exist.txt").write_text("old content")
        result = await backend.awrite("exist.txt", "new content")
        assert result.error is not None
        assert "already exists" in result.error
        assert (workspace / "exist.txt").read_text() == "old content"

    async def test_awrite_creates_parent_dir(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        result = await backend.awrite("sub/dir/file.txt", "nested content")
        assert result.error is None
        assert (workspace / "sub" / "dir" / "file.txt").read_text() == "nested content"

    async def test_awrite_multiline_content(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        content = "line1\nline2\nline3\n"
        result = await backend.awrite("multi.txt", content)
        assert result.error is None
        assert (workspace / "multi.txt").read_text() == content

    async def test_awrite_unicode_content(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        result = await backend.awrite("unicode.txt", "héllo wörld 日本語")
        assert result.error is None
        assert (workspace / "unicode.txt").read_text() == "héllo wörld 日本語"

    async def test_write_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            result = b.write("sync_write.txt", "sync content")
            assert result.error is None
            assert (workspace / "sync_write.txt").read_text() == "sync content"
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# File edit: aedit / edit
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestEdit:
    async def test_aedit_single_occurrence(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "edit.txt").write_text("foo bar baz")
        result = await backend.aedit("edit.txt", "bar", "qux")
        assert result.error is None
        assert result.occurrences == 1
        assert (workspace / "edit.txt").read_text() == "foo qux baz"

    async def test_aedit_replace_all(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "edit.txt").write_text("a a a")
        result = await backend.aedit("edit.txt", "a", "b", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3
        assert (workspace / "edit.txt").read_text() == "b b b"

    async def test_aedit_multiple_without_replace_all_fails(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "edit.txt").write_text("dup dup")
        result = await backend.aedit("edit.txt", "dup", "x")
        assert result.error is not None
        assert "multiple times" in result.error

    async def test_aedit_string_not_found(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "edit.txt").write_text("hello world")
        result = await backend.aedit("edit.txt", "missing", "x")
        assert result.error is not None
        assert "not found" in result.error

    async def test_aedit_file_not_found(self, backend: MicrosandboxBackend):
        result = await backend.aedit("nope.txt", "a", "b")
        assert result.error is not None
        assert "nope.txt" in result.error

    async def test_aedit_binary_file_error(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "edit.bin").write_bytes(b"\x00\x01\xff")
        result = await backend.aedit("edit.bin", "a", "b")
        assert result.error is not None
        assert "not a text file" in result.error

    async def test_aedit_normalizes_crlf(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "edit.txt").write_bytes(b"line1\nline2\n")
        result = await backend.aedit("edit.txt", "line1\r\nline2", "foo\r\nbar")
        assert result.error is None
        assert (workspace / "edit.txt").read_bytes() == b"foo\nbar\n"

    async def test_aedit_multiline_replacement(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        original = textwrap.dedent("""\
            def hello():
                pass
        """)
        (workspace / "code.py").write_text(original)
        result = await backend.aedit("code.py", "pass", "return 42")
        assert result.error is None
        assert "return 42" in (workspace / "code.py").read_text()

    async def test_edit_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            (workspace / "sync_edit.txt").write_text("old value")
            result = b.edit("sync_edit.txt", "old", "new")
            assert result.error is None
            assert "new value" in (workspace / "sync_edit.txt").read_text()
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# Directory listing: als / ls
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestLs:
    async def test_als_directory(self, backend: MicrosandboxBackend, workspace: Path):
        (workspace / "file1.txt").write_text("a")
        (workspace / "file2.txt").write_text("b")
        (workspace / "subdir").mkdir()
        result = await backend.als(".")
        assert result.error is None
        assert result.entries is not None
        paths = [e["path"] for e in result.entries]
        assert any("file1.txt" in p for p in paths)
        assert any("file2.txt" in p for p in paths)
        assert any("subdir" in p for p in paths)

    async def test_als_empty_directory(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        result = await backend.als(".")
        assert result.error is None
        assert result.entries is not None
        # The backend may create internal directories (e.g. .venv-msb) in
        # the workspace; filter those out to check there are no user files.
        user_entries = [e for e in result.entries if ".venv-msb" not in e["path"]]
        assert len(user_entries) == 0

    async def test_als_is_dir_flag(self, backend: MicrosandboxBackend, workspace: Path):
        (workspace / "file.txt").write_text("x")
        (workspace / "adir").mkdir()
        result = await backend.als(".")
        assert result.error is None
        assert result.entries is not None
        for entry in result.entries:
            if "adir" in entry["path"]:
                assert entry["is_dir"] is True
            if "file.txt" in entry["path"]:
                assert entry["is_dir"] is False

    async def test_als_nonexistent_path(self, backend: MicrosandboxBackend):
        result = await backend.als("/nonexistent_xyz")
        assert result.error is not None

    async def test_ls_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            (workspace / "sync_ls.txt").write_text("x")
            result = b.ls(".")
            assert result.error is None
            assert result.entries is not None
            assert any("sync_ls.txt" in e["path"] for e in result.entries)
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# Grep: agrep / grep
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestGrep:
    async def test_agrep_finds_matches(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "file1.txt").write_text("hello world\nfoo bar\nhello again\n")
        (workspace / "file2.py").write_text("hello from python\n")
        result = await backend.agrep("hello")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) >= 3
        texts = [m["text"] for m in result.matches]
        assert any("hello world" in t for t in texts)
        assert any("hello again" in t for t in texts)
        assert any("hello from python" in t for t in texts)

    async def test_agrep_no_matches(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "file.txt").write_text("nothing to see here\n")
        result = await backend.agrep("nonexistent_pattern_xyz")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 0

    async def test_agrep_with_path(self, backend: MicrosandboxBackend, workspace: Path):
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "a.txt").write_text("target line\n")
        (workspace / "root.txt").write_text("target line\n")
        result = await backend.agrep("target", path="subdir")
        assert result.error is None
        assert result.matches is not None
        assert all("subdir" in m["path"] for m in result.matches)

    async def test_agrep_with_glob(self, backend: MicrosandboxBackend, workspace: Path):
        (workspace / "a.py").write_text("searchme\n")
        (workspace / "b.txt").write_text("searchme\n")
        result = await backend.agrep("searchme", glob="*.py")
        assert result.error is None
        assert result.matches is not None
        assert all(".py" in m["path"] for m in result.matches)

    async def test_agrep_literal_search(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "file.txt").write_text("price: $50.00\n")
        result = await backend.agrep("$50.00")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "price: $50.00" in result.matches[0]["text"]

    async def test_agrep_returns_line_numbers(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "file.txt").write_text("line1\nmatch here\nline3\n")
        result = await backend.agrep("match")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["line"] == 2

    async def test_grep_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            (workspace / "file.txt").write_text("findme\n")
            result = b.grep("findme")
            assert result.error is None
            assert result.matches is not None
            assert len(result.matches) == 1
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# Glob: aglob / glob
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestGlob:
    async def test_aglob_simple_pattern(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "file1.py").write_text("x")
        (workspace / "file2.py").write_text("x")
        (workspace / "file3.txt").write_text("x")
        result = await backend.aglob("*.py")
        assert result.error is None
        assert result.matches is not None
        paths = [m["path"] for m in result.matches]
        assert any("file1.py" in p for p in paths)
        assert any("file2.py" in p for p in paths)
        assert not any("file3.txt" in p for p in paths)

    async def test_aglob_recursive_pattern(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "deep.py").write_text("x")
        (workspace / "top.py").write_text("x")
        result = await backend.aglob("**/*.py")
        assert result.error is None
        assert result.matches is not None
        paths = [m["path"] for m in result.matches]
        assert any("deep.py" in p for p in paths)
        assert any("top.py" in p for p in paths)

    async def test_aglob_path_pattern(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "src").mkdir()
        (workspace / "src" / "main.py").write_text("x")
        # find -path matches against the full path starting from the search root (./src/main.py),
        # so the pattern must include a wildcard prefix to match.
        result = await backend.aglob("*src/main.py")
        assert result.error is None
        assert result.matches is not None
        paths = [m["path"] for m in result.matches]
        assert any("main.py" in p for p in paths)

    async def test_aglob_no_matches(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        result = await backend.aglob("*.xyz")
        assert result.error is None
        assert result.matches is not None
        # No .xyz files exist in the workspace, so there should be no
        # real matches.  (The backend may return a spurious "<no output>"
        # entry from `find` producing empty stdout — filter that out.)
        real = [m for m in result.matches if m["path"] != "<no output>"]
        assert len(real) == 0

    async def test_aglob_with_path(self, backend: MicrosandboxBackend, workspace: Path):
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "a.py").write_text("x")
        (workspace / "b.py").write_text("x")
        result = await backend.aglob("*.py", path="subdir")
        assert result.error is None
        assert result.matches is not None
        assert all("subdir" in m["path"] for m in result.matches)

    async def test_glob_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            (workspace / "sync_glob.py").write_text("x")
            result = b.glob("*.py")
            assert result.error is None
            assert result.matches is not None
            assert any("sync_glob.py" in m["path"] for m in result.matches)
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# File upload: aupload_files / upload_files
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestUploadFiles:
    async def test_aupload_single_file(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        result = await backend.aupload_files([("upload.txt", b"file content")])
        assert len(result) == 1
        assert result[0].error is None
        assert result[0].path == "upload.txt"
        assert (workspace / "upload.txt").read_bytes() == b"file content"

    async def test_aupload_multiple_files(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        files = [("a.txt", b"aaa"), ("b.txt", b"bbb")]
        result = await backend.aupload_files(files)
        assert len(result) == 2
        assert all(r.error is None for r in result)
        assert (workspace / "a.txt").read_bytes() == b"aaa"
        assert (workspace / "b.txt").read_bytes() == b"bbb"

    async def test_aupload_creates_parent_dir(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        result = await backend.aupload_files([("sub/dir/file.txt", b"data")])
        assert len(result) == 1
        assert result[0].error is None
        assert (workspace / "sub" / "dir" / "file.txt").read_bytes() == b"data"

    async def test_aupload_binary_data(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        data = bytes(range(256))
        result = await backend.aupload_files([("binary.dat", data)])
        assert len(result) == 1
        assert result[0].error is None
        assert (workspace / "binary.dat").read_bytes() == data

    async def test_upload_files_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            result = b.upload_files([("sync_up.txt", b"sync data")])
            assert len(result) == 1
            assert result[0].error is None
            assert (workspace / "sync_up.txt").read_bytes() == b"sync data"
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# File download: adownload_files / download_files
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestDownloadFiles:
    async def test_adownload_single_file(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "dl.txt").write_bytes(b"download me")
        result = await backend.adownload_files(["dl.txt"])
        assert len(result) == 1
        assert result[0].error is None
        assert result[0].content == b"download me"

    async def test_adownload_multiple_files(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "a.txt").write_bytes(b"aaa")
        (workspace / "b.txt").write_bytes(b"bbb")
        result = await backend.adownload_files(["a.txt", "b.txt"])
        assert len(result) == 2
        assert all(r.error is None for r in result)
        assert result[0].content == b"aaa"
        assert result[1].content == b"bbb"

    async def test_adownload_file_not_found(self, backend: MicrosandboxBackend):
        result = await backend.adownload_files(["missing.txt"])
        assert len(result) == 1
        assert result[0].error is not None
        assert result[0].content is None

    async def test_adownload_partial_failure(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        (workspace / "ok.txt").write_bytes(b"ok")
        result = await backend.adownload_files(["ok.txt", "missing.txt"])
        assert len(result) == 2
        assert result[0].error is None
        assert result[0].content == b"ok"
        assert result[1].error is not None
        assert result[1].content is None

    async def test_adownload_binary_file(
        self, backend: MicrosandboxBackend, workspace: Path
    ):
        data = bytes(range(256))
        (workspace / "bin.dat").write_bytes(data)
        result = await backend.adownload_files(["bin.dat"])
        assert len(result) == 1
        assert result[0].error is None
        assert result[0].content == data

    async def test_download_files_sync(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace), memory=1024)
        try:
            (workspace / "sync_dl.txt").write_bytes(b"sync dl")
            result = b.download_files(["sync_dl.txt"])
            assert len(result) == 1
            assert result[0].error is None
            assert result[0].content == b"sync dl"
        finally:
            await b.stop()


# --------------------------------------------------------------------------- #
# Path helpers (pure logic, no VM needed)
# --------------------------------------------------------------------------- #


class TestPathHelpers:
    def test_to_guest_path_empty(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._resolve_guest_path("") == "/workspace"

    def test_to_guest_path_already_guest(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._to_guest_path("/workspace/file.txt") == "/workspace/file.txt"

    def test_to_guest_path_absolute_host(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        host_file = str(workspace / "subdir" / "file.txt")
        guest = b._to_guest_path(host_file)
        assert guest == "/workspace/subdir/file.txt"

    def test_to_guest_path_absolute_host_root(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        guest = b._to_guest_path(str(workspace))
        assert guest == "/workspace"

    def test_to_guest_path_absolute_non_host(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._to_guest_path("/etc/passwd") == "/etc/passwd"

    def test_to_guest_path_relative(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._resolve_guest_path("file.txt") == "/workspace/file.txt"

    def test_to_host_path_workspace(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        host = b._to_host_path("/workspace/subdir/file.txt")
        assert host == str(workspace / "subdir" / "file.txt")

    def test_to_host_path_workspace_root(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        host = b._to_host_path("/workspace")
        assert host == str(workspace)

    def test_to_host_path_non_workspace(self, workspace: Path):
        b = MicrosandboxBackend(root_dir=str(workspace))
        assert b._to_host_path("/etc/passwd") == "/etc/passwd"
