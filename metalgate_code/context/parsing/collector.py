"""File collection utilities for finding Python source files."""

import json
from pathlib import Path

from deepagents.backends.protocol import SandboxBackendProtocol


async def afind_site_packages(
    backend: SandboxBackendProtocol | None, python: str | None = None
) -> list[str]:
    """Find site-packages directories using the backend.

    Args:
        backend: The sandbox backend to use for execution. If None, falls back to local.
        python: Optional Python interpreter path. If None, uses 'python'.

    Returns:
        List of site-packages directory paths as strings.
    """
    import subprocess

    interp = python or "python"

    if backend is None:
        # Fallback to local execution
        result = subprocess.run(
            [
                interp,
                "-c",
                "import sysconfig, json; "
                "print(json.dumps([sysconfig.get_paths()['purelib'], "
                "                  sysconfig.get_paths()['platlib']]))",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        seen: list[str] = []
        for p in json.loads(result.stdout):
            pp = Path(p)
            if pp.is_dir() and p not in seen:
                seen.append(p)
        return seen

    # Use backend for remote execution
    result = await backend.aexecute(
        f'{interp} -c "import sysconfig, json; '
        "print(json.dumps([sysconfig.get_paths()['purelib'], "
        "                  sysconfig.get_paths()['platlib']]))\""
    )
    if result.exit_code != 0:
        return []
    seen = []
    for p in json.loads(result.output.strip()):
        # Check if directory exists via backend
        check_result = await backend.aexecute(f'test -d "{p}" && echo "exists"')
        if check_result.output.strip() == "exists" and p not in seen:
            seen.append(p)
    return seen


async def acollect_files(
    backend: "SandboxBackendProtocol | None", roots: list[str]
) -> list[str]:
    """Collect Python files from the given roots using backend, preferring .pyi over .py.

    Args:
        backend: The sandbox backend to use for glob operations. If None, falls back to local.
        roots: List of root directory paths to search.

    Returns:
        List of file paths sorted.
    """
    if backend is None:
        # Fallback to local filesystem
        by_stem: dict[str, str] = {}
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                continue
            for f in sorted(root.rglob("*.py")):
                key = str(f.with_suffix(""))
                if key not in by_stem:
                    by_stem[key] = str(f)
            for f in sorted(root.rglob("*.pyi")):
                key = str(f.with_suffix(""))
                by_stem[key] = str(f)
        return sorted(by_stem.values())

    # Use backend for remote glob
    by_stem = {}
    for root in roots:
        # Get .py files
        py_result = await backend.aglob("*.py", root)
        if py_result.matches:
            for match in py_result.matches:
                # match["path"] may be relative or absolute depending on backend
                path = match["path"]
                if not path.startswith("/"):
                    # Relative path - prepend root
                    path = f"{root}/{path}"
                key = path.removesuffix(".py")
                if key not in by_stem:
                    by_stem[key] = path

        # Get .pyi files (override .py)
        pyi_result = await backend.aglob("*.pyi", root)
        if pyi_result.matches:
            for match in pyi_result.matches:
                # match["path"] may be relative or absolute depending on backend
                path = match["path"]
                if not path.startswith("/"):
                    # Relative path - prepend root
                    path = f"{root}/{path}"
                key = path.removesuffix(".pyi")
                by_stem[key] = path

    # Return sorted list of paths
    return sorted(by_stem.values())


__all__ = ["afind_site_packages", "acollect_files"]
