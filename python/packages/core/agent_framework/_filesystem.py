# Copyright (c) Microsoft. All rights reserved.

"""Built-in filesystem tool for agents.

Provides :class:`FileSystemTool`, a security-first, cross-platform set of
filesystem operations exposed to agents as :class:`FunctionTool` instances:

* ``fs_view`` — read a file with optional line range
* ``fs_create`` — create a new file (fails if it exists)
* ``fs_edit`` — unique-match string replacement
* ``fs_multi_edit`` — apply multiple unique-match edits to one file atomically
* ``fs_glob`` — match files by pattern
* ``fs_grep`` — search file contents (uses ``rg`` if available)
* ``fs_list_dir`` — list directory entries
* ``fs_delete`` / ``fs_move`` / ``fs_rename`` — destructive ops; **always
  require user approval** (non-overridable)

**Security model.**  Every operation is sandboxed under a mandatory ``root``
directory.  Paths that escape the root via ``..``, absolute paths outside
the root, symlinks pointing outside the root, or symlinks anywhere in the
intermediate path are rejected.  A configurable denylist blocks access to
common sensitive files (``.env``, ``.git``, SSH keys, etc.) even inside
``root``.  Read and write operations can be restricted to disjoint sub-paths
via :class:`FileSystemPolicy`.

**Approvals.**  ``fs_delete``, ``fs_move``, and ``fs_rename`` are exposed
with ``approval_mode="always_require"`` and **cannot be opted out of**.
Hosting code must wire up the agent's approval flow.

This module is experimental; the API may change without notice.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess  # noqa: S404 - argv list, no shell; used only with --no-config
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypedDict

from ._feature_stage import ExperimentalFeature, experimental
from ._path_security import has_symlink_in_path, is_path_within_directory
from ._tools import FunctionTool

__all__ = ["FileSystemPolicy", "FileSystemTool"]

logger = logging.getLogger(__name__)

# region Defaults

# Conservative default denylist applied even inside the workspace root.
# Pattern semantics: forward-slash POSIX globs matched via fnmatch against
# the path *relative to root*. Patterns are always matched on POSIX form
# of the relative path (backslashes converted to forward slashes).
_DEFAULT_DENYLIST: Final[tuple[str, ...]] = (
    "**/.git/**",
    ".git/**",
    "**/.env",
    "**/.env.*",
    ".env",
    ".env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_rsa.*",
    "**/id_ed25519",
    "**/id_ed25519.*",
    "**/.ssh/**",
    ".ssh/**",
    "**/.aws/credentials",
    ".aws/credentials",
    "**/.npmrc",
    ".npmrc",
    "**/.pypirc",
    ".pypirc",
    "**/.netrc",
    ".netrc",
)

_DEFAULT_MAX_FILE_BYTES: Final[int] = 5 * 1024 * 1024  # 5 MiB
_DEFAULT_MAX_RESULTS: Final[int] = 1000
_DEFAULT_MAX_VIEW_LINES: Final[int] = 2000
_DEFAULT_LIST_DEPTH: Final[int] = 1
_DEFAULT_MAX_LIST_DEPTH: Final[int] = 5

# endregion

# region TypedDict result shapes


class ViewResult(TypedDict):
    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int
    truncated: bool


class CreateResult(TypedDict):
    path: str
    bytes_written: int


class EditResult(TypedDict):
    path: str
    replacements: int
    bytes_written: int


class MultiEditOpInput(TypedDict, total=False):
    old_str: str
    new_str: str
    count: int  # optional; default 1


class MultiEditResult(TypedDict):
    path: str
    edits_applied: int
    total_replacements: int
    bytes_written: int


class GlobResult(TypedDict):
    matches: list[str]
    truncated: bool


class GrepHit(TypedDict):
    path: str
    line_number: int
    line: str


class GrepResult(TypedDict):
    hits: list[GrepHit]
    truncated: bool
    backend: str  # "ripgrep" or "python"


class DirEntry(TypedDict):
    path: str
    type: str  # "file" or "dir"
    size: int | None


class ListDirResult(TypedDict):
    path: str
    entries: list[DirEntry]
    truncated: bool


class DeleteResult(TypedDict):
    path: str


class MoveResult(TypedDict):
    source: str
    destination: str


# endregion

# region Policy


@dataclass(frozen=True)
class FileSystemPolicy:
    """Security policy controlling :class:`FileSystemTool` operations.

    All path lists are interpreted as POSIX-style globs relative to ``root``.

    Attributes:
        read_paths: Optional allowlist of paths readable by the tool. ``None``
            allows reads anywhere under ``root``. When set, only files whose
            relative path matches at least one pattern are readable.
        write_paths: Optional allowlist of paths writable by the tool. ``None``
            allows writes anywhere under ``root``. When set, only files whose
            relative path matches at least one pattern are writable.
        denylist: Patterns that block access regardless of allowlist.
            Defaults to a conservative list including ``.git``, ``.env*``,
            SSH keys, and credential files. Pass an empty tuple to disable.
        max_file_bytes: Maximum file size in bytes for ``view``/``edit``.
            ``create`` content is also bounded by this size.
        max_results: Maximum number of results for ``glob``/``grep``/``list_dir``.
        max_view_lines: Default upper bound on lines returned by ``view``
            when no explicit ``view_range`` is supplied.
        allow_grep_ripgrep: When ``True`` (default), use ``rg`` if found on
            PATH. Set to ``False`` to force pure-Python grep.
        respect_gitignore: When ``True`` (default), ``glob``, ``grep``, and
            ``list_dir`` skip files matched by ``.gitignore`` rules found in
            the workspace (root and nested). Direct reads/writes via ``view``,
            ``create``, ``edit``, etc. are unaffected — gitignore controls
            *discovery*, not *access*. Ripgrep already honors ``.gitignore``
            natively when present.
    """

    read_paths: tuple[str, ...] | None = None
    write_paths: tuple[str, ...] | None = None
    denylist: tuple[str, ...] = _DEFAULT_DENYLIST
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES
    max_results: int = _DEFAULT_MAX_RESULTS
    max_view_lines: int = _DEFAULT_MAX_VIEW_LINES
    allow_grep_ripgrep: bool = True
    respect_gitignore: bool = True

    DEFAULT_DENYLIST: Final[tuple[str, ...]] = _DEFAULT_DENYLIST


# endregion

# region Errors


class FileSystemSecurityError(PermissionError):
    """Raised when an operation is blocked by sandbox or policy."""


# endregion

# region Tool


@experimental(feature_id=ExperimentalFeature.FILESYSTEM)
class FileSystemTool:
    """Sandboxed filesystem operations exposed as agent tools.

    Construct with a workspace ``root`` directory. All operations are
    confined to that directory; attempts to escape via ``..``, absolute
    paths, or symlinks are rejected.

    Examples:
        Basic usage::

            from agent_framework import FileSystemTool, FileSystemPolicy

            fs = FileSystemTool(
                root="/workspace/repo",
                policy=FileSystemPolicy(write_paths=("src/**", "tests/**")),
            )
            agent = ChatAgent(chat_client=client, tools=fs.as_tools())
    """

    # Tool names exposed to the model.
    _NAME_VIEW: Final[str] = "fs_view"
    _NAME_CREATE: Final[str] = "fs_create"
    _NAME_EDIT: Final[str] = "fs_edit"
    _NAME_MULTI_EDIT: Final[str] = "fs_multi_edit"
    _NAME_GLOB: Final[str] = "fs_glob"
    _NAME_GREP: Final[str] = "fs_grep"
    _NAME_LIST_DIR: Final[str] = "fs_list_dir"
    _NAME_DELETE: Final[str] = "fs_delete"
    _NAME_MOVE: Final[str] = "fs_move"
    _NAME_RENAME: Final[str] = "fs_rename"

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        policy: FileSystemPolicy | None = None,
    ) -> None:
        """Initialize the filesystem tool.

        Args:
            root: The workspace directory. Must exist and be a directory.
                The path is resolved (following symlinks on the root itself
                is permitted) before sandbox checks are applied.
            policy: Optional :class:`FileSystemPolicy` controlling allowlists
                and limits. Defaults to a security-conservative policy.

        Raises:
            ValueError: If ``root`` does not exist or is not a directory.
        """
        resolved = Path(root).expanduser().resolve(strict=False)
        if not resolved.exists():
            raise ValueError(f"FileSystemTool root does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"FileSystemTool root is not a directory: {resolved}")
        self._root: Path = resolved
        self.policy: FileSystemPolicy = policy or FileSystemPolicy()

        # Cache ripgrep detection. Re-detected by tests via shutil.which.
        self._ripgrep_path: str | None = shutil.which("rg") if self.policy.allow_grep_ripgrep else None
        # Lazy-loaded gitignore matcher.
        self._gitignore: _GitignoreMatcher | None = None
        self._gitignore_loaded: bool = False

    def _get_gitignore(self) -> _GitignoreMatcher | None:
        """Lazily load the workspace's gitignore matcher, or ``None`` if disabled."""
        if not self.policy.respect_gitignore:
            return None
        if not self._gitignore_loaded:
            self._gitignore = _GitignoreMatcher.load(self._root)
            self._gitignore_loaded = True
        return self._gitignore

    @property
    def root(self) -> Path:
        """The resolved workspace root."""
        return self._root

    # region Sandbox helpers

    def _resolve_in_root(self, relative_or_absolute: str) -> Path:
        """Resolve a user-supplied path and verify it is inside the root.

        Returns the resolved absolute :class:`Path`. Raises
        :class:`FileSystemSecurityError` if the path escapes the root,
        traverses any symlink under the root, or matches the denylist.
        """
        if not relative_or_absolute or relative_or_absolute.strip() == "":
            raise FileSystemSecurityError("Empty path is not allowed.")

        # Reject explicit parent-directory traversal up front. Even though
        # resolve() collapses ``..``, an explicit ``..`` in the user-supplied
        # input is a strong signal of intent we want to reject.
        normalized = PurePosixPath(relative_or_absolute.replace("\\", "/")).as_posix()
        if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
            raise FileSystemSecurityError(f"Path contains parent-directory traversal: {relative_or_absolute!r}")

        candidate = Path(relative_or_absolute)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        # resolve(strict=False) collapses .. and follows existing symlinks.
        resolved = candidate.resolve(strict=False)

        if not is_path_within_directory(str(resolved), str(self._root)):
            raise FileSystemSecurityError(f"Path escapes workspace root: {relative_or_absolute!r}")

        # Reject any symlink under the root in the *existing* portion of the
        # path. has_symlink_in_path inspects only existing segments; missing
        # tails (e.g., when creating a new file) are skipped.
        try:
            if has_symlink_in_path(str(resolved), str(self._root)):
                raise FileSystemSecurityError(
                    f"Path traverses a symlink under the workspace root: {relative_or_absolute!r}"
                )
        except ValueError:
            # Should not happen after is_path_within_directory passed, but
            # treat defensively as a sandbox failure.
            raise FileSystemSecurityError(f"Path failed sandbox check: {relative_or_absolute!r}") from None

        self._check_denylist(resolved)
        return resolved

    def _relative_posix(self, path: Path) -> str:
        """Return *path* relative to root in POSIX form."""
        try:
            rel = path.relative_to(self._root)
        except ValueError:
            return path.as_posix()
        return rel.as_posix() if rel.parts else "."

    def _check_denylist(self, path: Path) -> None:
        rel = self._relative_posix(path)
        for pattern in self.policy.denylist:
            if _match_glob(rel, pattern):
                raise FileSystemSecurityError(f"Path is blocked by denylist pattern {pattern!r}: {rel}")

    def _check_read_allowed(self, path: Path) -> None:
        if self.policy.read_paths is None:
            return
        rel = self._relative_posix(path)
        if not any(_match_glob(rel, p) for p in self.policy.read_paths):
            raise FileSystemSecurityError(f"Read not permitted by policy.read_paths: {rel}")

    def _check_write_allowed(self, path: Path) -> None:
        if self.policy.write_paths is None:
            return
        rel = self._relative_posix(path)
        if not any(_match_glob(rel, p) for p in self.policy.write_paths):
            raise FileSystemSecurityError(f"Write not permitted by policy.write_paths: {rel}")

    # endregion

    # region Operations

    def view(self, path: str, view_range: list[int] | None = None) -> ViewResult:
        """Read a text file with line numbers.

        Args:
            path: Path relative to workspace root (or an absolute path
                inside the root).
            view_range: Optional ``[start, end]`` 1-based inclusive line
                range. Use ``-1`` for ``end`` to read until EOF.

        Returns:
            A :class:`ViewResult` containing the requested slice.
        """
        resolved = self._resolve_in_root(path)
        self._check_read_allowed(resolved)
        if not resolved.exists():
            raise FileNotFoundError(f"File does not exist: {self._relative_posix(resolved)}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Path is not a regular file: {self._relative_posix(resolved)}")
        size = resolved.stat().st_size
        if size > self.policy.max_file_bytes:
            raise FileSystemSecurityError(
                f"File size {size} exceeds max_file_bytes "
                f"{self.policy.max_file_bytes}: {self._relative_posix(resolved)}"
            )

        data = resolved.read_bytes()
        if b"\x00" in data:
            raise ValueError(f"Refusing to view binary file: {self._relative_posix(resolved)}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8: {self._relative_posix(resolved)}") from exc

        lines = text.splitlines()
        total = len(lines)
        if view_range is None:
            start, end = 1, min(total, self.policy.max_view_lines)
        else:
            if len(view_range) != 2:
                raise ValueError("view_range must be [start, end].")
            start, end = view_range
            if end == -1:
                end = total
            if start < 1 or end < start:
                raise ValueError(f"Invalid view_range: {view_range}")
            end = min(end, total)
            if end - start + 1 > self.policy.max_view_lines:
                end = start + self.policy.max_view_lines - 1

        truncated = total == 0 or end < total or (view_range is None and total > end)
        slice_lines = lines[start - 1 : end] if total else []
        return ViewResult(
            path=self._relative_posix(resolved),
            content="\n".join(slice_lines),
            start_line=start if total else 0,
            end_line=end if total else 0,
            total_lines=total,
            truncated=truncated,
        )

    def create(self, path: str, content: str) -> CreateResult:
        """Create a new file with ``content``. Fails if the path exists."""
        resolved = self._resolve_in_root(path)
        self._check_write_allowed(resolved)
        if resolved.exists():
            raise FileExistsError(f"File already exists: {self._relative_posix(resolved)}")
        encoded = content.encode("utf-8")
        if len(encoded) > self.policy.max_file_bytes:
            raise FileSystemSecurityError(
                f"Content size {len(encoded)} exceeds max_file_bytes {self.policy.max_file_bytes}"
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(resolved, encoded)
        logger.info("fs_create %s (%d bytes)", self._relative_posix(resolved), len(encoded))
        return CreateResult(
            path=self._relative_posix(resolved),
            bytes_written=len(encoded),
        )

    def edit(
        self,
        path: str,
        old_str: str,
        new_str: str,
        count: int | None = None,
    ) -> EditResult:
        """Replace ``old_str`` with ``new_str`` in a text file.

        By default ``old_str`` must match exactly once. Pass an explicit
        ``count`` (>= 1) to require exactly that many occurrences. Line
        endings detected in the original file are preserved.
        """
        resolved = self._resolve_in_root(path)
        self._check_read_allowed(resolved)
        self._check_write_allowed(resolved)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File does not exist: {self._relative_posix(resolved)}")
        size = resolved.stat().st_size
        if size > self.policy.max_file_bytes:
            raise FileSystemSecurityError(f"File size {size} exceeds max_file_bytes {self.policy.max_file_bytes}")

        data = resolved.read_bytes()
        if b"\x00" in data:
            raise ValueError(f"Refusing to edit binary file: {self._relative_posix(resolved)}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8: {self._relative_posix(resolved)}") from exc

        if old_str == "":
            raise ValueError("old_str must be non-empty.")
        occurrences = text.count(old_str)
        expected = 1 if count is None else count
        if expected < 1:
            raise ValueError("count must be >= 1.")
        if occurrences != expected:
            raise ValueError(
                f"Expected {expected} occurrence(s) of old_str in "
                f"{self._relative_posix(resolved)}, found {occurrences}. "
                "Add surrounding context to make old_str unique."
            )

        new_text = text.replace(old_str, new_str)
        new_bytes = new_text.encode("utf-8")
        if len(new_bytes) > self.policy.max_file_bytes:
            raise FileSystemSecurityError(f"Resulting file size {len(new_bytes)} exceeds max_file_bytes.")
        _atomic_write_bytes(resolved, new_bytes)
        logger.info(
            "fs_edit %s (%d replacement(s), %d bytes)",
            self._relative_posix(resolved),
            occurrences,
            len(new_bytes),
        )
        return EditResult(
            path=self._relative_posix(resolved),
            replacements=occurrences,
            bytes_written=len(new_bytes),
        )

    def multi_edit(self, path: str, edits: list[dict[str, Any]]) -> MultiEditResult:
        """Apply a sequence of unique-match edits to a single file atomically.

        Each edit is a dict with keys ``old_str`` (required), ``new_str``
        (required), and optional ``count`` (default 1). Edits are applied
        sequentially in memory; each ``old_str`` must occur exactly the
        expected number of times *after* preceding edits have been applied.
        On any failure no changes are written.

        Returns a :class:`MultiEditResult` summarizing the operation.
        """
        if not edits:
            raise ValueError("edits must be a non-empty list.")

        resolved = self._resolve_in_root(path)
        self._check_read_allowed(resolved)
        self._check_write_allowed(resolved)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File does not exist: {self._relative_posix(resolved)}")
        size = resolved.stat().st_size
        if size > self.policy.max_file_bytes:
            raise FileSystemSecurityError(f"File size {size} exceeds max_file_bytes {self.policy.max_file_bytes}")

        data = resolved.read_bytes()
        if b"\x00" in data:
            raise ValueError(f"Refusing to edit binary file: {self._relative_posix(resolved)}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8: {self._relative_posix(resolved)}") from exc

        total_replacements = 0
        for index, raw_edit in enumerate(edits):
            if not isinstance(raw_edit, dict):
                raise ValueError(f"edits[{index}] must be an object with old_str/new_str.")
            old_str = raw_edit.get("old_str")
            new_str = raw_edit.get("new_str")
            if not isinstance(old_str, str) or not isinstance(new_str, str):
                raise ValueError(f"edits[{index}] requires string old_str and new_str.")
            if old_str == "":
                raise ValueError(f"edits[{index}].old_str must be non-empty.")
            count_value = raw_edit.get("count", 1)
            if not isinstance(count_value, int) or count_value < 1:
                raise ValueError(f"edits[{index}].count must be an integer >= 1.")
            occurrences = text.count(old_str)
            if occurrences != count_value:
                raise ValueError(
                    f"edits[{index}]: expected {count_value} occurrence(s) of old_str in "
                    f"{self._relative_posix(resolved)}, found {occurrences}. "
                    "Add surrounding context to make old_str unique."
                )
            text = text.replace(old_str, new_str)
            total_replacements += occurrences

        new_bytes = text.encode("utf-8")
        if len(new_bytes) > self.policy.max_file_bytes:
            raise FileSystemSecurityError(f"Resulting file size {len(new_bytes)} exceeds max_file_bytes.")
        _atomic_write_bytes(resolved, new_bytes)
        logger.info(
            "fs_multi_edit %s (%d edit(s), %d replacement(s), %d bytes)",
            self._relative_posix(resolved),
            len(edits),
            total_replacements,
            len(new_bytes),
        )
        return MultiEditResult(
            path=self._relative_posix(resolved),
            edits_applied=len(edits),
            total_replacements=total_replacements,
            bytes_written=len(new_bytes),
        )

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Match files under the workspace by a recursive glob pattern.

        Patterns use POSIX-style globs (``**`` matches across segments).
        Symlinks are not followed and denied paths are skipped. When
        ``policy.respect_gitignore`` is true, gitignored files are skipped.
        """
        base = self._resolve_in_root(path) if path else self._root
        if not base.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self._relative_posix(base)}")
        matches: list[str] = []
        truncated = False
        gi = self._get_gitignore()
        for entry in _safe_walk(base, self._root, self.policy.denylist, gi):
            rel = entry.relative_to(self._root).as_posix()
            # Match against path relative to ``base`` so the pattern is
            # naturally scoped, mirroring ripgrep/git-style behavior.
            relative_to_base = entry.relative_to(base).as_posix() if base != self._root else rel
            if _match_glob(relative_to_base, pattern):
                matches.append(rel)
                if len(matches) >= self.policy.max_results:
                    truncated = True
                    break
        return GlobResult(matches=matches, truncated=truncated)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        case_insensitive: bool = False,
    ) -> GrepResult:
        """Search file contents for a regex pattern.

        Uses ripgrep when available, falling back to a pure-Python walk.
        """
        base = self._resolve_in_root(path) if path else self._root
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {self._relative_posix(base)}")
        if base.is_file():
            return self._grep_python(pattern, [base], glob, case_insensitive)

        if self._ripgrep_path:
            try:
                return self._grep_ripgrep(pattern, base, glob, case_insensitive)
            except _RipgrepUnavailable:
                # Fall through to Python implementation.
                pass

        return self._grep_python(
            pattern,
            list(_safe_walk(base, self._root, self.policy.denylist, self._get_gitignore())),
            glob,
            case_insensitive,
        )

    def list_dir(self, path: str = ".", depth: int = _DEFAULT_LIST_DEPTH) -> ListDirResult:
        """List directory entries up to ``depth`` levels deep."""
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        if depth > _DEFAULT_MAX_LIST_DEPTH:
            raise ValueError(f"depth must be <= {_DEFAULT_MAX_LIST_DEPTH}.")
        resolved = self._resolve_in_root(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self._relative_posix(resolved)}")

        entries: list[DirEntry] = []
        truncated = False
        for entry in _walk_with_depth(resolved, self._root, self.policy.denylist, depth, self._get_gitignore()):
            rel = entry.relative_to(self._root).as_posix()
            try:
                stat = entry.stat()
                size = stat.st_size if entry.is_file() else None
            except OSError:
                size = None
            entries.append(
                DirEntry(
                    path=rel,
                    type="dir" if entry.is_dir() else "file",
                    size=size,
                )
            )
            if len(entries) >= self.policy.max_results:
                truncated = True
                break
        return ListDirResult(
            path=self._relative_posix(resolved),
            entries=entries,
            truncated=truncated,
        )

    def delete(self, path: str) -> DeleteResult:
        """Delete a file. Always requires user approval at the tool layer."""
        resolved = self._resolve_in_root(path)
        self._check_write_allowed(resolved)
        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {self._relative_posix(resolved)}")
        if resolved.is_dir():
            raise IsADirectoryError(
                f"Refusing to delete a directory (only files are supported): {self._relative_posix(resolved)}"
            )
        if resolved.is_symlink():
            raise FileSystemSecurityError(f"Refusing to delete a symlink: {self._relative_posix(resolved)}")
        rel = self._relative_posix(resolved)
        os.remove(resolved)
        logger.warning("fs_delete %s", rel)
        return DeleteResult(path=rel)

    def move(self, source: str, destination: str) -> MoveResult:
        """Move a file. Always requires user approval at the tool layer."""
        return self._move_impl(source, destination, op_name="fs_move")

    def rename(self, source: str, destination: str) -> MoveResult:
        """Rename a file. Always requires user approval at the tool layer.

        Functionally identical to :meth:`move` but exposed under a separate
        name so models can request renames distinctly from cross-directory
        moves.
        """
        return self._move_impl(source, destination, op_name="fs_rename")

    def _move_impl(self, source: str, destination: str, op_name: str) -> MoveResult:
        src = self._resolve_in_root(source)
        dst = self._resolve_in_root(destination)
        self._check_read_allowed(src)
        self._check_write_allowed(src)
        self._check_write_allowed(dst)
        if not src.exists():
            raise FileNotFoundError(f"Source does not exist: {self._relative_posix(src)}")
        if src.is_symlink() or dst.is_symlink():
            raise FileSystemSecurityError("Refusing to move/rename across symlinks.")
        if dst.exists():
            raise FileExistsError(f"Destination already exists: {self._relative_posix(dst)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        rel_src = self._relative_posix(src)
        rel_dst = self._relative_posix(dst)
        logger.warning("%s %s -> %s", op_name, rel_src, rel_dst)
        return MoveResult(source=rel_src, destination=rel_dst)

    # endregion

    # region Grep backends

    def _grep_ripgrep(
        self,
        pattern: str,
        base: Path,
        glob: str | None,
        case_insensitive: bool,
    ) -> GrepResult:
        if self._ripgrep_path is None:
            raise _RipgrepUnavailable
        cmd = [
            self._ripgrep_path,
            "--no-config",
            "--no-messages",
            "--line-number",
            "--no-heading",
            "--with-filename",
            "--color=never",
            "--max-count",
            str(self.policy.max_results),
        ]
        if case_insensitive:
            cmd.append("-i")
        if glob:
            cmd.extend(["--glob", glob])
        # Apply our denylist via --glob '!pattern' so rg also honors it.
        for deny in self.policy.denylist:
            cmd.extend(["--glob", f"!{deny}"])
        cmd.extend(["--", pattern, str(base)])

        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("ripgrep failed (%s); falling back to Python grep", exc)
            raise _RipgrepUnavailable from exc

        # rg exits 0 with results, 1 with no matches, 2 on error.
        if proc.returncode not in (0, 1):
            logger.warning(
                "ripgrep returned %d: %s; falling back to Python grep",
                proc.returncode,
                proc.stderr.strip(),
            )
            raise _RipgrepUnavailable

        hits: list[GrepHit] = []
        truncated = False
        for raw_line in proc.stdout.splitlines():
            parsed = _parse_rg_line(raw_line)
            if parsed is None:
                continue
            file_path, line_number, content_line = parsed
            try:
                rel = Path(file_path).resolve().relative_to(self._root).as_posix()
            except (ValueError, OSError):
                continue
            hits.append(GrepHit(path=rel, line_number=line_number, line=content_line))
            if len(hits) >= self.policy.max_results:
                truncated = True
                break
        return GrepResult(hits=hits, truncated=truncated, backend="ripgrep")

    def _grep_python(
        self,
        pattern: str,
        files: Iterable[Path],
        glob: str | None,
        case_insensitive: bool,
    ) -> GrepResult:
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid grep pattern: {exc}") from exc

        hits: list[GrepHit] = []
        truncated = False
        for file_path in files:
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self._root).as_posix()
            if glob and not _match_glob(rel, glob):
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            if stat.st_size > self.policy.max_file_bytes:
                continue
            try:
                with file_path.open("rb") as fh:
                    chunk = fh.read(8192)
                    if b"\x00" in chunk:
                        continue
                    data = chunk + fh.read()
            except OSError:
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(GrepHit(path=rel, line_number=lineno, line=line))
                    if len(hits) >= self.policy.max_results:
                        truncated = True
                        break
            if truncated:
                break
        return GrepResult(hits=hits, truncated=truncated, backend="python")

    # endregion

    # region Tool exposure

    def as_tools(self) -> list[FunctionTool]:
        """Return the operations as :class:`FunctionTool` instances."""
        return [
            FunctionTool(
                name=self._NAME_VIEW,
                description=(
                    "Read a text file inside the workspace. Returns content as a "
                    "single string with a 1-based start_line/end_line slice. Refuses "
                    "binary files and files larger than the configured limit. "
                    "Use view_range=[start, end] (end may be -1) to read a slice."
                ),
                func=self.view,
                input_model={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to workspace root."},
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Optional 1-based [start, end] line range; end=-1 means EOF.",
                        },
                    },
                    "required": ["path"],
                },
            ),
            FunctionTool(
                name=self._NAME_CREATE,
                description=(
                    "Create a new text file under the workspace root. Fails if the "
                    "path already exists. Parent directories are created automatically."
                ),
                func=self.create,
                input_model={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to workspace root."},
                        "content": {"type": "string", "description": "UTF-8 file contents."},
                    },
                    "required": ["path", "content"],
                },
            ),
            FunctionTool(
                name=self._NAME_EDIT,
                description=(
                    "Replace exactly one (or `count`) occurrence(s) of `old_str` "
                    "with `new_str` in a UTF-8 text file. Fails if the match count "
                    "differs; add surrounding context to disambiguate."
                ),
                func=self.edit,
                input_model={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Required match count; default 1.",
                        },
                    },
                    "required": ["path", "old_str", "new_str"],
                },
            ),
            FunctionTool(
                name=self._NAME_MULTI_EDIT,
                description=(
                    "Apply multiple unique-match edits to one file atomically. "
                    "Each edit must have `old_str` and `new_str`; optional `count` "
                    "defaults to 1. Edits are applied sequentially in memory and "
                    "all-or-nothing — if any edit's match count is wrong, no "
                    "changes are written. Cheaper than calling fs_edit repeatedly "
                    "for refactors that touch many sites in the same file."
                ),
                func=self.multi_edit,
                input_model={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "edits": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_str": {"type": "string"},
                                    "new_str": {"type": "string"},
                                    "count": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "description": "Required match count; default 1.",
                                    },
                                },
                                "required": ["old_str", "new_str"],
                            },
                        },
                    },
                    "required": ["path", "edits"],
                },
            ),
            FunctionTool(
                name=self._NAME_GLOB,
                description=(
                    "Find files matching a POSIX glob pattern under the workspace "
                    "(or under an optional sub-path). Symlinks are not followed."
                ),
                func=self.glob,
                input_model={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern, e.g. '**/*.py'.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional sub-path to scope the search.",
                        },
                    },
                    "required": ["pattern"],
                },
            ),
            FunctionTool(
                name=self._NAME_GREP,
                description=(
                    "Search file contents for a regular expression. Uses ripgrep when "
                    "available and a pure-Python fallback otherwise. Optional `glob` "
                    "filters which files are searched."
                ),
                func=self.grep,
                input_model={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern."},
                        "path": {"type": "string", "description": "Optional sub-path."},
                        "glob": {"type": "string", "description": "Optional file glob filter."},
                        "case_insensitive": {"type": "boolean", "default": False},
                    },
                    "required": ["pattern"],
                },
            ),
            FunctionTool(
                name=self._NAME_LIST_DIR,
                description=(
                    "List directory entries up to `depth` levels deep. Symlinks and denylisted paths are skipped."
                ),
                func=self.list_dir,
                input_model={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "depth": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _DEFAULT_MAX_LIST_DEPTH,
                            "default": _DEFAULT_LIST_DEPTH,
                        },
                    },
                },
            ),
            FunctionTool(
                name=self._NAME_DELETE,
                description=(
                    "Delete a file inside the workspace. Always requires user "
                    "approval. Refuses to delete directories or symlinks."
                ),
                func=self.delete,
                approval_mode="always_require",
                input_model={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
            FunctionTool(
                name=self._NAME_MOVE,
                description=(
                    "Move a file inside the workspace. Always requires user "
                    "approval. Refuses to move symlinks or overwrite an existing "
                    "destination."
                ),
                func=self.move,
                approval_mode="always_require",
                input_model={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["source", "destination"],
                },
            ),
            FunctionTool(
                name=self._NAME_RENAME,
                description=(
                    "Rename a file inside the workspace. Always requires user "
                    "approval. Refuses to overwrite an existing destination."
                ),
                func=self.rename,
                approval_mode="always_require",
                input_model={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["source", "destination"],
                },
            ),
        ]

    # endregion


# endregion

# region Helpers


class _RipgrepUnavailable(RuntimeError):
    """Internal sentinel: ripgrep failed; fall back to Python grep."""


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically by staging in a sibling temp file."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".fstool-", dir=str(parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of the temp file.
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _match_glob(rel_posix: str, pattern: str) -> bool:
    """Match a relative POSIX path against a glob pattern.

    Supports ``**`` (any number of segments, including zero) in addition
    to standard :mod:`fnmatch` semantics. ``**/foo`` matches ``foo``,
    ``a/foo``, and ``a/b/foo``; ``foo/**`` matches ``foo`` and ``foo/x``;
    ``foo/**/bar`` matches ``foo/bar`` and ``foo/a/bar``.
    """
    pattern = pattern.replace("\\", "/")
    rel_posix = rel_posix.replace("\\", "/")
    regex = _glob_to_regex(pattern)
    return re.match(regex, rel_posix) is not None


def _glob_to_regex(pattern: str) -> str:
    """Translate a forward-slash glob pattern to a full-match regex.

    Handles ``**`` (any number of segments, including zero) plus standard
    fnmatch tokens within each segment.
    """
    segments = pattern.split("/")
    parts: list[str] = []
    for i, seg in enumerate(segments):
        is_last = i == len(segments) - 1
        if seg == "**":
            if i == 0 and not is_last:
                # Leading ``**/`` — zero or more leading segments.
                parts.append("(?:.*/)?")
                continue
            if is_last:
                # Trailing ``/**`` — zero or more trailing segments,
                # consuming the slash optionally.
                if parts and parts[-1] == "/":
                    parts[-1] = "(?:/.*)?"
                else:
                    parts.append("(?:.*)?")
                continue
            # Middle ``/**/`` — zero or more middle segments.
            if parts and parts[-1] == "/":
                parts[-1] = "(?:/.*/|/)"
            else:
                parts.append(".*/")
            continue
        parts.append(_fnmatch_to_regex(seg))
        if not is_last:
            parts.append("/")
    return "^" + "".join(parts) + "$"


def _fnmatch_to_regex(pattern: str) -> str:
    """Translate a non-``**`` fnmatch pattern to a regex source string."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(ch))
            else:
                out.append(pattern[i : j + 1])
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def _safe_walk(
    base: Path,
    root: Path,
    denylist: Sequence[str],
    gitignore: _GitignoreMatcher | None = None,
) -> Iterable[Path]:
    """Yield files under *base* without following symlinks or denylisted paths.

    Skips symlinks entirely (both files and directories). Directories that
    match the denylist or *gitignore* are pruned from descent.
    """
    stack: list[Path] = [base]
    while stack:
        current = stack.pop()
        try:
            scan = os.scandir(current)
        except OSError:
            continue
        with scan as it:
            for entry in it:
                entry_path = Path(entry.path)
                try:
                    rel = entry_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if entry.is_symlink():
                    continue
                if any(_match_glob(rel, p) for p in denylist):
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                if gitignore is not None and gitignore.is_ignored(rel, is_dir=is_dir):
                    continue
                if is_dir:
                    stack.append(entry_path)
                elif entry.is_file(follow_symlinks=False):
                    yield entry_path


def _walk_with_depth(
    base: Path,
    root: Path,
    denylist: Sequence[str],
    max_depth: int,
    gitignore: _GitignoreMatcher | None = None,
) -> Iterable[Path]:
    """Walk *base* like :func:`_safe_walk` but with depth + dirs.

    Bounded to *max_depth* levels and yields both files and directories.
    """
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            scan = os.scandir(current)
        except OSError:
            continue
        with scan as it:
            for entry in it:
                entry_path = Path(entry.path)
                try:
                    rel = entry_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if entry.is_symlink():
                    continue
                if any(_match_glob(rel, p) for p in denylist):
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                if gitignore is not None and gitignore.is_ignored(rel, is_dir=is_dir):
                    continue
                yield entry_path
                if is_dir and depth + 1 < max_depth:
                    stack.append((entry_path, depth + 1))


def _parse_rg_line(line: str) -> tuple[str, int, str] | None:
    """Parse a ``--no-heading --line-number --with-filename`` rg line.

    Format: ``<path>:<lineno>:<text>`` (path may contain ``:`` on Windows).
    """
    if not line:
        return None
    # On Windows paths look like ``C:\foo\bar.py:42:content``. Search from
    # the right for ``:<digits>:`` to disambiguate.
    match = re.search(r":(\d+):(.*)$", line)
    if match is None:
        return None
    lineno_idx = match.start(1)
    file_path = line[: lineno_idx - 1]
    try:
        line_number = int(match.group(1))
    except ValueError:
        return None
    return file_path, line_number, match.group(2)


@dataclass(frozen=True)
class _GitignoreRule:
    """A single compiled gitignore rule.

    *base* is the POSIX-style path of the directory containing the
    ``.gitignore`` (relative to root, ``""`` for the root itself).
    *regex* is a compiled pattern that matches paths *relative to root*.
    *negate* indicates a leading ``!``. *dir_only* indicates a trailing ``/``.
    """

    base: str
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


class _GitignoreMatcher:
    """Minimal but correct ``.gitignore`` matcher.

    Loads the workspace ``.gitignore`` plus any nested ``.gitignore`` files
    discovered during loading. Rules from deeper files take precedence over
    rules from shallower ones, and within a file later rules override
    earlier ones. Negation (``!``) is supported.

    This implementation covers the common gitignore subset used in coding
    agents: ``*``, ``?``, ``**``, character classes, leading ``/`` for
    anchored patterns, trailing ``/`` for directory-only matches, leading
    ``!`` for negation, and ``#`` comments. It does *not* implement all
    edge cases of the real ``git`` matcher; for full fidelity, users may
    rely on ripgrep's native gitignore support (the default for ``fs_grep``).
    """

    def __init__(self, rules: list[_GitignoreRule]) -> None:
        self._rules = rules

    @classmethod
    def load(cls, root: Path) -> _GitignoreMatcher | None:
        """Walk *root* discovering ``.gitignore`` files and compile rules.

        Returns ``None`` if no rules were found.
        """
        rules: list[_GitignoreRule] = []
        # BFS over directories starting at root. We don't follow symlinks.
        stack: list[Path] = [root]
        while stack:
            current = stack.pop()
            gi_path = current / ".gitignore"
            if gi_path.is_file():
                try:
                    text = gi_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = ""
                try:
                    base_rel = current.relative_to(root).as_posix()
                except ValueError:
                    base_rel = ""
                if base_rel == ".":
                    base_rel = ""
                rules.extend(_compile_gitignore_text(text, base_rel))
            try:
                scan = os.scandir(current)
            except OSError:
                continue
            with scan as it:
                for entry in it:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        # Always descend into .git? No — skip dot-git to avoid noise,
                        # but keep other dot directories so their .gitignore is read.
                        if entry.name == ".git":
                            continue
                        stack.append(Path(entry.path))
        if not rules:
            return None
        return cls(rules)

    def is_ignored(self, rel_posix: str, *, is_dir: bool) -> bool:
        """Return ``True`` if *rel_posix* (path relative to root) is ignored."""
        ignored = False
        # Rules are evaluated in order; later rules override earlier ones.
        # Rules from deeper bases are appended later by load() because of
        # the LIFO stack, but BFS order between sibling subdirs is not
        # significant: gitignore semantics treat each directory's rules
        # independently for paths under that directory.
        for rule in self._rules:
            if rule.dir_only and not is_dir:
                # A dir-only rule still applies if the path is *under* an
                # ignored directory; that case is handled by walk pruning,
                # so here we only test the directory itself.
                continue
            # Only apply rules whose base is a prefix of rel_posix.
            if rule.base and not (rel_posix == rule.base or rel_posix.startswith(rule.base + "/")):
                continue
            target = rel_posix
            if rule.regex.match(target):
                ignored = not rule.negate
        return ignored


def _compile_gitignore_text(text: str, base: str) -> list[_GitignoreRule]:
    """Compile a single ``.gitignore`` file's contents into rules."""
    rules: list[_GitignoreRule] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        # Strip trailing unescaped whitespace
        line = re.sub(r"(?<!\\)\s+$", "", line)
        if not line or line.startswith("#"):
            continue
        negate = False
        if line.startswith("!"):
            negate = True
            line = line[1:]
        if line.startswith("\\!") or line.startswith("\\#"):
            line = line[1:]
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        # Anchored if pattern starts with '/' or contains a non-trailing '/'.
        anchored = line.startswith("/") or ("/" in line)
        if line.startswith("/"):
            line = line[1:]
        # Build regex relative to *base*. The pattern matches a path
        # relative to root that lives under base/.
        pattern_regex = _gitignore_pattern_to_regex(line, anchored=anchored)
        prefix = ""
        if base:
            prefix = re.escape(base) + "/"
        if anchored:
            full = "^" + prefix + pattern_regex + "(?:/.*)?$"
        else:
            # Non-anchored: pattern can match at any depth at or below base.
            full = "^" + prefix + "(?:.*/)?" + pattern_regex + "(?:/.*)?$"
        try:
            compiled = re.compile(full)
        except re.error:
            continue
        rules.append(_GitignoreRule(base=base, regex=compiled, negate=negate, dir_only=dir_only))
    return rules


def _gitignore_pattern_to_regex(pattern: str, *, anchored: bool) -> str:
    """Translate a gitignore pattern (single segment or path) to a regex source.

    Handles ``**``, ``*``, ``?``, and character classes. The result matches
    the pattern itself; callers wrap it with prefix/suffix for anchoring.
    """
    # Reuse our segment-aware glob translator. Trailing ``/**`` semantics
    # differ slightly: gitignore's ``a/**`` means "everything under a", which
    # our translator already produces. Strip any leading slash; caller
    # handles anchoring.
    pattern = pattern.replace("\\", "/")
    if not anchored and "/" not in pattern:
        # Single segment, non-anchored: use fnmatch translation.
        return _fnmatch_to_regex(pattern)
    # Multi-segment or anchored: use full glob-to-regex but strip the
    # surrounding ``^...$`` so the caller can wrap with prefix/suffix.
    full = _glob_to_regex(pattern)
    return full[1:-1]  # drop ^ and $


# endregion
