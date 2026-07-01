"""
meta_skills.py
"""

import ast
import logging
import subprocess

import isort
from langchain_core.tools import tool

from metalgate_code.context.backend_context import get_backend
from metalgate_code.skills.registry import registry

logger = logging.getLogger("metalgate_code")


@tool
def list_tool_skills() -> list[str]:
    """Return the names of all currently available tool skills."""
    return registry.names()


@tool
def read_tool_skill(name: str) -> str:
    """Get detailed info about a specific tool skill including its description and parameters."""
    skill = registry.get(name)
    if skill is None:
        return (
            f"Tool skill '{name}' not found. Available: {', '.join(registry.names())}"
        )
    return f"{skill.name}: {skill.description}"


def _install_dependencies(packages: list[str]) -> str:
    result = subprocess.run(
        ["uv", "pip", "install"] + packages, capture_output=True, text=True
    )
    if result.returncode != 0:
        return f"Failed:\n{result.stderr}"
    return f"Installed: {', '.join(packages)}"


@tool
def install_dependencies(packages: list[str]) -> str:
    """Install Python packages into the agent runtime using uv."""
    return _install_dependencies(packages)


@tool
def create_tool_skill(
    name: str, code: str, dependencies: list[str] | None = None
) -> str:
    """
    Create and register a new tool skill.
    code must define exactly one @tool decorated function named `name`, with a docstring.
    dependencies: optional list of pip package names to install before registering.

    `@tool` (from langchain_core.tools) and `get_backend` are injected into the
    skills.py exec context, so they are available without importing. Do NOT add
    imports for them.

    Shell commands and file operations MUST run inside the agent's sandbox, not on
    the host. Use `get_backend()` to access the sandbox backend.

    Skill code runs on the HOST, but files live in the SANDBOX. This means:
    - Do NOT use `open()`, `pd.read_csv()`, `json.load()`, or any library that
      reads files directly — they run on the host where the files don't exist.
    - Do NOT use `subprocess` or `os.system` — they run on the host.
    - Instead, use `backend.execute()` for shell commands (including running Python
      scripts that read files), or `backend.read()`/`backend.download_files()` to
      get file content back to the host for processing.

    The sandbox backend provides these methods (sync and async variants, use sync):

    - `execute(command, timeout=None) -> ExecuteResponse`
        Run a shell command. Returns `ExecuteResponse(output: str, exit_code: int|None, truncated: bool)`.
    - `read(file_path, offset=0, limit=2000) -> ReadResult`
        Read file content. Returns `ReadResult(error: str|None, file_data: dict|None)`.
        On success, `file_data["content"]` is the text and `file_data["encoding"]` is "utf-8" or "base64".
        Check `result.error` first; it is None on success.
    - `write(file_path, content) -> WriteResult`
        Create a new file. Fails if the file already exists.
        Returns `WriteResult(error: str|None, path: str|None)`.
    - `edit(file_path, old_string, new_string, replace_all=False) -> EditResult`
        Replace exact string in a file. Returns `EditResult(error, path, occurrences)`.
    - `ls(path) -> LsResult`
        List directory. Returns `LsResult(error: str|None, entries: list[FileInfo]|None)`.
        Each `FileInfo` is a dict with keys: `path`, `is_dir`, `size`, `modified_at`.
    - `grep(pattern, path=None, glob=None) -> GrepResult`
        Search for a literal string in files. Returns `GrepResult(error, matches)`.
        Each match is a dict with keys: `path`, `line`, `text`.
    - `glob(pattern, path=None) -> GlobResult`
        Find files matching a glob pattern. Returns `GlobResult(error, matches)`.
        `matches` is a list of `FileInfo` dicts.
    - `upload_files(files: list[tuple[str, bytes]]) -> list[FileUploadResponse]`
        Upload files to the sandbox. Each tuple is (path, content_bytes).
    - `download_files(paths: list[str]]) -> list[FileDownloadResponse]`
        Download files from the sandbox. Returns content as bytes.

    Host paths are automatically translated to sandbox paths (`/workspace` prefix).
    The sandbox working directory is `/workspace`, which is bind-mounted to the project root.

    Example (read a JSON file in the sandbox via backend.read, then parse on the host):
        @tool
        def count_keys(path: str) -> str:
            \"\"\"Count top-level keys in a JSON file.\"\"\"
            import json
            backend = get_backend()
            if backend is None:
                return "No sandbox available"
            result = backend.read(path)
            if result.error:
                return result.error
            data = json.loads(result.file_data["content"])
            return f"{len(data)} keys"
    """
    # 1. Install dependencies if needed
    if dependencies:
        result = _install_dependencies(dependencies)
        if result.startswith("Failed"):
            return result

    # 2. Parse and validate
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    tool_funcs = [
        f
        for f in funcs
        if any(
            (isinstance(d, ast.Name) and d.id == "tool")
            or (isinstance(d, ast.Attribute) and d.attr == "tool")
            for d in f.decorator_list
        )
    ]
    if not tool_funcs:
        return "Error: code must define at least one function decorated with @tool"
    for f in tool_funcs:
        if not ast.get_docstring(f):
            return f"Error: @tool function '{f.name}' must have a docstring"

    # 3. Check for duplicate function name
    skills_path = registry.skills_path
    if not skills_path.exists():
        skills_path.mkdir(parents=True, exist_ok=True)
        skills_path.write_text('"""Project tool skills."""\n')

    existing_source = skills_path.read_text()
    existing_funcs = {
        n.name
        for n in ast.walk(ast.parse(existing_source))
        if isinstance(n, ast.FunctionDef)
    }
    if name in existing_funcs:
        return f"Error: '{name}' already exists. Use delete_tool_skill first."

    # 4. Dry-run (inject tool and get_backend, same as registry.reload)
    try:
        exec(
            compile(code, "<skill>", "exec"), {"tool": tool, "get_backend": get_backend}
        )
    except Exception as e:
        return f"Execution error: {e}"

    # 5. Combine and sort imports before writing
    combined = existing_source + f"\n\n{code}\n"
    sorted_source = isort.code(combined, float_to_top=True)
    skills_path.write_text(sorted_source)

    # 6. Hot reload
    registry.reload()
    return f"Tool skill '{name}' created and registered."


@tool
def reload_tool_skills() -> str:
    """Reload all tool skills from skills.py.
    Use this after the user has manually edited a tool skill to pick up changes without restarting."""
    logger.info("META: reload_tool_skills called")
    registry.reload()
    result = (
        f"Reloaded {len(registry.names())} tool skills: {', '.join(registry.names())}"
    )
    logger.info(f"META: reload_tool_skills returning: {result}")
    return result


@tool
def delete_tool_skill(name: str) -> str:
    """Remove a tool skill by name from skills.py and the live registry."""
    skills_path = registry.skills_path
    if not skills_path.exists():
        return "No tool skill to delete."

    source = skills_path.read_text()

    tree = ast.parse(source)

    target = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        ),
        None,
    )
    if target is None:
        return f"No tool skill named '{name}'"

    lines = source.splitlines(keepends=True)
    start = (
        target.decorator_list[0].lineno - 1
        if target.decorator_list
        else target.lineno - 1
    )
    skills_path.write_text("".join(lines[:start] + lines[target.end_lineno :]))
    registry.reload()
    return f"Tool skill '{name}' deleted."
