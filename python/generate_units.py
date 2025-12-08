#!/usr/bin/env python
import json
from pathlib import Path

# Directory you run this from is treated as %REPO%
BASE = Path.cwd()

# Directories to ignore when collecting sources
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
}

def is_ignored(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)

def find_project_roots(base: Path):
    """
    Heuristic: any directory that has a pyproject.toml
    is considered a project root (unit root parent).
    e.g. %REPO%/a2a, %REPO%/ag-ui, etc.
    """
    roots = set()
    for pyproj in base.rglob("pyproject.toml"):
        roots.add(pyproj.parent)
    return sorted(roots)

def collect_sources_for_root(root: Path):
    """
    Collect all .py files under this root, excluding
    ignored dirs (and tests/* by default).
    """
    sources = []
    for f in root.rglob("*.py"):
        if is_ignored(f):
            continue
        # optional: skip tests; comment this block out if you want them included
        if "tests" in f.parts:
            continue
        sources.append(f)
    return sorted(sources)

def to_repo_placeholder(path: Path) -> str:
    """
    Convert an absolute Path under BASE to a %REPO%/... path
    with forward slashes.
    """
    rel = path.relative_to(BASE).as_posix()
    return f"%REPO%/{rel}"

def main():
    project_units = []

    project_roots = find_project_roots(BASE)
    if not project_roots:
        raise SystemExit("No pyproject.toml files found under this directory.")

    for root in project_roots:
        py_files = collect_sources_for_root(root)
        if not py_files:
            continue

        # root field: %REPO%/<relative-path-to-this-root>
        rel_root = root.relative_to(BASE).as_posix()
        unit_root = f"%REPO%/{rel_root}" if rel_root != "." else "%REPO%"

        unit = {
            "language": "Python",
            "root": unit_root,
            "sources": [to_repo_placeholder(f) for f in py_files],
        }
        project_units.append(unit)

    # If you prefer a single unit, you could instead
    # merge all sources into one. For now we emit a list.
    print(json.dumps(project_units, indent=2))

if __name__ == "__main__":
    main()
