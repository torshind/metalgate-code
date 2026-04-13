"""
Project skills.
"""

import subprocess
from typing import Tuple

import httpx
import yaml
from langchain_core.tools import tool
from markdownify import markdownify


#
# Validation tools
#
@tool
def valid_yaml(target: str) -> str | None:
    """Validate that target is parseable YAML. Returns None on success, error message on failure."""
    try:
        yaml.safe_load(target)
        return None
    except yaml.YAMLError as e:
        return f"YAML parse error: {e}"


#
# Remote data tools
#
@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a URL. Returns the page content as plain text."""
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()
    return markdownify(response.text)


#
# Subprocess tools
#
def _run(cmd: str) -> Tuple[int, str]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    output = (result.stdout + result.stderr).strip()
    return (result.returncode, (output or f"exited {result.returncode}"))


@tool
def compile_c(target: str) -> Tuple[int, str]:
    """Compile a C file with strict warnings. target = source path.
    Returns a tuple of (returncode, output)."""
    return _run(f"gcc -Wall -Wextra -Werror -fsyntax-only {target}")


@tool
def run_python(project_path: str, target: str) -> Tuple[int, str]:
    """Run a python command using the correct project environment.
    project_path = path to the project root.
    target = python command to run.
    Returns a tuple of (returncode, output)."""
    return _run(f"uv run --project {project_path} python {target}")


@tool
def run_pytest(project_path: str, target: str) -> Tuple[int, str]:
    """Run unit tests with pytest using the correct project environment.
    project_path = path to the project root.
    target = optional path (file or directory).
    Returns a tuple of (returncode, output)."""
    return _run(f"uv run --project {project_path} pytest {target} -v --tb=short")


@tool
def run_pytest_verbose(project_path: str, target: str) -> Tuple[int, str]:
    """Run unit tests with pytest with verbose output, using the correct project environment.
    project_path = path to the project root.
    target = optional path (file or directory).
    Returns a tuple of (returncode, output)."""
    return _run(
        f"uv run --project {project_path} pytest -v -s --log-cli-level=INFO {target}"
    )


@tool
def lint_python(target: str) -> Tuple[int, str]:
    """Lint with ruff. target = path (file or directory).
    Returns a tuple of (returncode, output)."""
    return _run(f"ruff check {target}")


@tool
def list_all_files(path: str) -> Tuple[int, str]:
    """List all files and directories at the given path.
    Returns a tuple of (returncode, output)."""
    return _run(f"ls -al {path}")


@tool
def run_ty(project_path: str, ty_args: str) -> Tuple[int, str]:
    """Run ty (type checking) with arguments.
    project_path = path to the project root.
    ty_args = arguments to pass to ty.
    Returns a tuple of (returncode, output)."""
    return _run(f"uv run --project {project_path} ty {ty_args}")
