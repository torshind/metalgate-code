"""File collection utilities for finding Python source files."""

import json
import subprocess
import sys
from pathlib import Path


def find_site_packages(python: str | None = None) -> list[Path]:
    """Find site-packages directories for the given Python interpreter."""
    interp = python or sys.executable
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
    seen: list[Path] = []
    for p in json.loads(result.stdout):
        pp = Path(p)
        if pp.is_dir() and pp not in seen:
            seen.append(pp)
    return seen


def collect_files(roots: list[Path]) -> list[Path]:
    """Collect Python files from the given roots, preferring .pyi over .py."""
    by_stem: dict[str, Path] = {}
    for root in roots:
        for f in sorted(root.rglob("*.py")):
            key = str(f.with_suffix(""))
            if key not in by_stem:
                by_stem[key] = f
        for f in sorted(root.rglob("*.pyi")):
            by_stem[str(f.with_suffix(""))] = f
    return sorted(by_stem.values())


__all__ = ["find_site_packages", "collect_files"]
