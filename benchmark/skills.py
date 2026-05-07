"""
Benchmark skills.
"""

import os
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


@tool
def web_search(query: str, num: int = 10) -> dict:
    """Search the web using Serper API.
    query = the search query string.
    num = number of results to return (default 10, max 100).
    Returns a dict with search results."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return {"error": "SERPER_API_KEY environment variable not set"}

    response = httpx.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": min(num, 100)},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


#
# Subprocess tools
#
def _run(cmd: str) -> Tuple[int, str]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    output = (result.stdout + result.stderr).strip()
    return (result.returncode, (output or f"exited {result.returncode}"))


@tool
def install_missing_python_package(package: str) -> Tuple[int, str]:
    """Install a missing Python package using pip.
    package = the name of the package to install.
    Returns a tuple of (returncode, output)."""
    return _run(f"pip install {package}")


@tool
def lint_python(target: str) -> Tuple[int, str]:
    """Lint with ruff. target = path (file or directory).
    Returns a tuple of (returncode, output)."""
    return _run(f"ruff check {target}")


@tool
def run_ty(cwd: str, ty_args: str) -> Tuple[int, str]:
    """Run ty (type checking) with arguments.
    cwd = path to the project root.
    ty_args = ty command arguments.
    Returns a tuple of (returncode, output)."""
    return _run(f"ty check {ty_args}")
