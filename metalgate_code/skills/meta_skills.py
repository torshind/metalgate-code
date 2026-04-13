"""
meta_skills.py
"""

import ast
import subprocess

import isort
from langchain_core.tools import tool

from metalgate_code.skills.registry import registry


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
    if len(funcs) != 1 or funcs[0].name != name:
        return f"Error: code must define exactly one function named '{name}'"
    if "tool" not in [ast.unparse(d) for d in funcs[0].decorator_list]:
        return "Error: function must be decorated with @tool"
    if not (
        funcs[0].body
        and isinstance(funcs[0].body[0], ast.Expr)
        and isinstance(funcs[0].body[0].value, ast.Constant)
    ):
        return "Error: function must have a docstring"

    # 3. Check for duplicate function name
    skills_path = registry.skills_path
    if not skills_path.exists():
        skills_path.write_text("from langchain_core.tools import tool\n")

    existing_source = skills_path.read_text()
    existing_funcs = {
        n.name
        for n in ast.walk(ast.parse(existing_source))
        if isinstance(n, ast.FunctionDef)
    }
    if name in existing_funcs:
        return f"Error: '{name}' already exists. Use delete_tool_skill first."

    # 4. Dry-run
    try:
        exec(compile(code, "<skill>", "exec"), {})
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
def delete_tool_skill(name: str) -> str:
    """Remove a tool skill by name from skills.py and the live registry."""
    with open("skills.py") as f:
        source = f.read()
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
    with open("skills.py", "w") as f:
        f.write("".join(lines[:start] + lines[target.end_lineno :]))
    registry.reload()
    return f"Tool skill '{name}' deleted."
