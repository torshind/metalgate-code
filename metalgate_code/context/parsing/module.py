"""Module path resolution utilities."""

from pathlib import Path


def _parts_to_module(parts: list[str]) -> str | None:
    """Convert path parts to module name. Returns None if empty."""
    if not parts:
        return None
    result = list(parts)  # Copy to avoid mutating input
    if result[0].endswith("-stubs"):
        result[0] = result[0].removesuffix("-stubs")
    result[-1] = Path(result[-1]).stem
    if result[-1] == "__init__":
        result.pop()
    return ".".join(result) if result else None


def _file_to_module(path: Path, site_roots: list[Path]) -> str:
    """Convert a file path to its module name."""
    resolved = path.resolve()

    # Try site_roots first
    for sr in site_roots:
        try:
            rel = resolved.relative_to(sr.resolve())
        except ValueError:
            continue
        if mod := _parts_to_module(list(rel.parts)):
            return mod

    # Fallback: find site-packages in path
    try:
        sp_idx = resolved.parts.index("site-packages")
        if mod := _parts_to_module(list(resolved.parts[sp_idx + 1 :])):
            return mod
    except ValueError:
        pass

    return path.stem


__all__ = ["_parts_to_module", "_file_to_module"]
