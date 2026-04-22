"""
Project skills.
"""

import os
import subprocess
from typing import Tuple

import httpx
import yaml
from langchain_core.tools import tool
from markdownify import markdownify
from tavily import TavilyClient


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
def tavily_search(query: str, max_results: int = 5) -> dict:
    """Search the web using Tavily API.
    query = the search query string.
    max_results = number of results to return (default 5, max 20).
    Returns a dict with search results."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable not set"}

    client = TavilyClient(api_key=api_key)
    return client.search(query=query, max_results=min(max_results, 20))


@tool
def tavily_crawl(
    url: str, max_depth: int = 3, limit: int = 50, instructions: str = ""
) -> dict:
    """Crawl a website starting from a base URL using Tavily API.
    url = the starting URL to crawl from.
    max_depth = maximum depth to crawl (default 3).
    limit = maximum number of pages to crawl (default 50).
    instructions = optional instructions to focus the crawl.
    Returns a dict with crawl results."""
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable not set"}

    client = TavilyClient(api_key=api_key)
    return client.crawl(
        url=url, max_depth=max_depth, limit=limit, instructions=instructions
    )


@tool
def serper_search(query: str, num: int = 10) -> dict:
    """Search Google using Serper API.
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
def compile_c(target: str) -> Tuple[int, str]:
    """Compile a C file with strict warnings. target = source path.
    Returns a tuple of (returncode, output)."""
    return _run(f"gcc -Wall -Wextra -Werror -fsyntax-only {target}")


@tool
def run_python(project_path: str, target: str) -> Tuple[int, str]:
    """Run a python command using the correct project environment.
    project_path = path to the project root.
    target = python command arguments.
    Returns a tuple of (returncode, output)."""
    return _run(f"uv run --project {project_path} python {target}")


@tool
def run_pytest(project_path: str, target: str) -> Tuple[int, str]:
    """Run unit tests with pytest using the correct project environment.
    project_path = path to the project root.
    target = optional path (file or directory).
    Returns a tuple of (returncode, output)."""
    return _run(f"uv run --project {project_path} pytest -v --tb=short {target}")


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
    ty_args = ty command arguments.
    Returns a tuple of (returncode, output)."""
    return _run(f"uv run --project {project_path} ty {ty_args}")
