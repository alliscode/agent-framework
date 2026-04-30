# Copyright (c) Microsoft. All rights reserved.

"""Path-security helpers shared by skill loading and the filesystem tool.

These helpers guard against two classes of path attacks:

1. **Path traversal** — paths that resolve outside an expected root directory
   via ``..`` segments, absolute paths, or platform-specific quirks.
2. **Symlink escape** — symlinks inside an otherwise-valid path that point
   outside the root.

All functions accept already-resolved absolute path strings; callers are
responsible for resolving (e.g. via :meth:`pathlib.Path.resolve`) before
invoking these checks.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["has_symlink_in_path", "is_path_within_directory"]


def is_path_within_directory(path: str, directory: str) -> bool:
    """Return whether *path* resides under *directory*.

    Comparison uses :meth:`pathlib.Path.is_relative_to`, which respects
    per-platform case-sensitivity rules.

    Args:
        path: Absolute path to check.
        directory: Directory that must be an ancestor of *path*.

    Returns:
        ``True`` if *path* is a descendant of *directory*.
    """
    try:
        return Path(path).is_relative_to(directory)
    except (ValueError, OSError):
        return False


def has_symlink_in_path(path: str, directory: str) -> bool:
    """Detect symlinks in the portion of *path* below *directory*.

    Only segments below *directory* are inspected; the directory itself
    and anything above it are not checked.

    **Precondition:** *path* must be a descendant of *directory*.
    Call :func:`is_path_within_directory` first to verify containment.

    Args:
        path: Absolute path to inspect.
        directory: Root directory; segments above it are not checked.

    Returns:
        ``True`` if any intermediate segment below *directory* is a symlink.

    Raises:
        ValueError: If *path* is not relative to *directory*.
    """
    dir_path = Path(directory)
    try:
        relative = Path(path).relative_to(dir_path)
    except ValueError as exc:
        raise ValueError(f"path {path!r} does not start with directory {directory!r}") from exc

    current = dir_path
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False
