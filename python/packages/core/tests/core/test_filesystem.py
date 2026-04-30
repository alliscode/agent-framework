# Copyright (c) Microsoft. All rights reserved.

"""Tests for FileSystemTool."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_framework import FileSystemPolicy, FileSystemTool
from agent_framework._filesystem import (
    FileSystemSecurityError,
    _atomic_write_bytes,
    _match_glob,
    _parse_rg_line,
)

pytestmark = pytest.mark.filterwarnings(r"ignore:\[FILESYSTEM\].*:FutureWarning")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A populated workspace fixture."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# project\n\nhello world\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=abc\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    return tmp_path


def _tool(workspace: Path, **policy_kwargs: Any) -> FileSystemTool:
    return FileSystemTool(root=workspace, policy=FileSystemPolicy(**policy_kwargs))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_root_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            FileSystemTool(root=tmp_path / "nope")

    def test_root_must_be_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("x")
        with pytest.raises(ValueError):
            FileSystemTool(root=f)

    def test_as_tools_returns_all_ops(self, workspace: Path) -> None:
        names = {t.name for t in _tool(workspace).as_tools()}
        assert names == {
            "fs_view",
            "fs_create",
            "fs_edit",
            "fs_multi_edit",
            "fs_glob",
            "fs_grep",
            "fs_list_dir",
            "fs_delete",
            "fs_move",
            "fs_rename",
        }

    def test_destructive_ops_require_approval(self, workspace: Path) -> None:
        by_name = {t.name: t for t in _tool(workspace).as_tools()}
        for name in ("fs_delete", "fs_move", "fs_rename"):
            assert by_name[name].approval_mode == "always_require", name

    def test_non_destructive_ops_do_not_require_approval(self, workspace: Path) -> None:
        by_name = {t.name: t for t in _tool(workspace).as_tools()}
        for name in ("fs_view", "fs_create", "fs_edit", "fs_multi_edit", "fs_glob", "fs_grep", "fs_list_dir"):
            assert by_name[name].approval_mode == "never_require", name


# ---------------------------------------------------------------------------
# Sandbox + denylist
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_rejects_parent_traversal(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view("../etc/passwd")

    def test_rejects_absolute_outside_root(self, workspace: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
        outside = tmp_path_factory.mktemp("outside")
        (outside / "secret.txt").write_text("nope")
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view(str(outside / "secret.txt"))

    def test_absolute_inside_root_allowed(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.view(str(workspace / "README.md"))
        assert "hello world" in result["content"]

    def test_empty_path_rejected(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view("")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires admin on Windows")
    def test_symlink_escape_rejected(self, workspace: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
        outside = tmp_path_factory.mktemp("outside")
        target = outside / "leak.txt"
        target.write_text("leak")
        link = workspace / "link.txt"
        os.symlink(target, link)
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view("link.txt")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires admin on Windows")
    def test_symlink_inside_root_rejected_for_safety(self, workspace: Path) -> None:
        # Even a symlink pointing inside the root is rejected because we
        # cannot reason about what it might point to in the future.
        target = workspace / "src" / "main.py"
        link = workspace / "link.py"
        os.symlink(target, link)
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view("link.py")

    def test_denylist_blocks_dotenv(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view(".env")

    def test_denylist_blocks_dot_git(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.view(".git/config")

    def test_empty_denylist_allows_dotenv(self, workspace: Path) -> None:
        fs = _tool(workspace, denylist=())
        result = fs.view(".env")
        assert "SECRET" in result["content"]

    def test_read_paths_allowlist(self, workspace: Path) -> None:
        fs = _tool(workspace, read_paths=("src/**",))
        fs.view("src/main.py")
        with pytest.raises(FileSystemSecurityError):
            fs.view("README.md")

    def test_write_paths_allowlist(self, workspace: Path) -> None:
        fs = _tool(workspace, write_paths=("src/**",))
        fs.create("src/new.py", "x = 1\n")
        with pytest.raises(FileSystemSecurityError):
            fs.create("docs/new.md", "x")


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


class TestView:
    def test_view_full_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.view("src/util.py")
        assert result["content"] == "def add(a, b):\n    return a + b"
        assert result["start_line"] == 1
        assert result["end_line"] == 2
        assert result["total_lines"] == 2

    def test_view_range(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.view("src/util.py", view_range=[2, 2])
        assert result["content"] == "    return a + b"
        assert result["start_line"] == 2
        assert result["end_line"] == 2

    def test_view_range_until_eof(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.view("src/util.py", view_range=[1, -1])
        assert result["total_lines"] == 2

    def test_view_invalid_range(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(ValueError):
            fs.view("src/util.py", view_range=[0, 1])
        with pytest.raises(ValueError):
            fs.view("src/util.py", view_range=[3, 1])

    def test_view_missing_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileNotFoundError):
            fs.view("nope.txt")

    def test_view_directory(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(IsADirectoryError):
            fs.view("src")

    def test_view_max_size_enforced(self, workspace: Path) -> None:
        fs = _tool(workspace, max_file_bytes=8)
        (workspace / "big.txt").write_text("x" * 100, encoding="utf-8")
        with pytest.raises(FileSystemSecurityError):
            fs.view("big.txt")

    def test_view_refuses_binary(self, workspace: Path) -> None:
        (workspace / "binary.bin").write_bytes(b"\x00\x01\x02")
        fs = _tool(workspace)
        with pytest.raises(ValueError, match="binary"):
            fs.view("binary.bin")

    def test_view_refuses_invalid_utf8(self, workspace: Path) -> None:
        (workspace / "bad.txt").write_bytes(b"\xff\xfe\xfd")
        fs = _tool(workspace)
        with pytest.raises(ValueError, match="UTF-8"):
            fs.view("bad.txt")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.create("src/new.py", "x = 1\n")
        assert (workspace / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"
        assert result["bytes_written"] == 6

    def test_create_makes_parent_dirs(self, workspace: Path) -> None:
        fs = _tool(workspace)
        fs.create("docs/api/reference.md", "# ref\n")
        assert (workspace / "docs" / "api" / "reference.md").exists()

    def test_create_fails_if_exists(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileExistsError):
            fs.create("README.md", "overwrite")

    def test_create_size_limited(self, workspace: Path) -> None:
        fs = _tool(workspace, max_file_bytes=4)
        with pytest.raises(FileSystemSecurityError):
            fs.create("big.txt", "x" * 100)


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    def test_edit_unique_match(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.edit("src/util.py", "a + b", "a - b")
        assert (workspace / "src" / "util.py").read_text(encoding="utf-8") == ("def add(a, b):\n    return a - b\n")
        assert result["replacements"] == 1

    def test_edit_zero_match_fails(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(ValueError, match="found 0"):
            fs.edit("src/util.py", "nonexistent", "x")

    def test_edit_two_match_fails(self, workspace: Path) -> None:
        (workspace / "dup.txt").write_text("hello\nhello\n", encoding="utf-8")
        fs = _tool(workspace)
        with pytest.raises(ValueError, match="found 2"):
            fs.edit("dup.txt", "hello", "world")

    def test_edit_explicit_count(self, workspace: Path) -> None:
        (workspace / "dup.txt").write_text("hello\nhello\n", encoding="utf-8")
        fs = _tool(workspace)
        result = fs.edit("dup.txt", "hello", "world", count=2)
        assert result["replacements"] == 2
        assert (workspace / "dup.txt").read_text(encoding="utf-8") == "world\nworld\n"

    def test_edit_empty_old_str_fails(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(ValueError):
            fs.edit("src/util.py", "", "x")

    def test_edit_missing_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileNotFoundError):
            fs.edit("nope.txt", "a", "b")

    def test_edit_atomic(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force os.replace to fail mid-flight; original file must be intact.
        target = workspace / "src" / "util.py"
        original = target.read_bytes()

        def boom(*a: Any, **kw: Any) -> None:
            raise OSError("simulated crash")

        monkeypatch.setattr("agent_framework._filesystem.os.replace", boom)
        fs = _tool(workspace)
        with pytest.raises(OSError):
            fs.edit("src/util.py", "a + b", "a - b")
        assert target.read_bytes() == original
        # Temp file should have been cleaned up.
        siblings = list(target.parent.glob(".fstool-*"))
        assert siblings == []


# ---------------------------------------------------------------------------
# glob / list_dir
# ---------------------------------------------------------------------------


class TestGlobAndListDir:
    def test_glob_recursive_py(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.glob("**/*.py")
        assert sorted(result["matches"]) == [
            "src/main.py",
            "src/util.py",
            "tests/test_main.py",
        ]

    def test_glob_scoped_to_subpath(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.glob("**/*.py", path="src")
        assert sorted(result["matches"]) == ["src/main.py", "src/util.py"]

    def test_glob_skips_denylist(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.glob("**/*")
        assert all(".git/" not in m for m in result["matches"])
        assert ".env" not in result["matches"]

    def test_glob_truncated_at_max_results(self, workspace: Path) -> None:
        for i in range(20):
            (workspace / f"f{i}.txt").write_text("x", encoding="utf-8")
        fs = _tool(workspace, max_results=5)
        result = fs.glob("*.txt")
        assert len(result["matches"]) == 5
        assert result["truncated"] is True

    def test_list_dir_default_depth_1(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.list_dir(".")
        names = {e["path"] for e in result["entries"]}
        assert "src" in names
        assert "README.md" in names
        # depth=1: should not descend into src/
        assert "src/main.py" not in names

    def test_list_dir_depth_2(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.list_dir(".", depth=2)
        names = {e["path"] for e in result["entries"]}
        assert "src/main.py" in names

    def test_list_dir_skips_denylist(self, workspace: Path) -> None:
        fs = _tool(workspace)
        result = fs.list_dir(".", depth=3)
        names = {e["path"] for e in result["entries"]}
        assert ".env" not in names
        assert all(not n.startswith(".git") for n in names)

    def test_list_dir_depth_bounds(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(ValueError):
            fs.list_dir(".", depth=0)
        with pytest.raises(ValueError):
            fs.list_dir(".", depth=99)


# ---------------------------------------------------------------------------
# grep (forces python backend)
# ---------------------------------------------------------------------------


class TestGrepPython:
    @pytest.fixture()
    def fs(self, workspace: Path) -> FileSystemTool:
        return _tool(workspace, allow_grep_ripgrep=False)

    def test_basic_match(self, fs: FileSystemTool) -> None:
        result = fs.grep(r"def \w+")
        assert result["backend"] == "python"
        paths = {h["path"] for h in result["hits"]}
        assert "src/util.py" in paths

    def test_case_insensitive(self, fs: FileSystemTool, workspace: Path) -> None:
        (workspace / "case.txt").write_text("Hello World\n", encoding="utf-8")
        ci = fs.grep(r"hello", case_insensitive=True)
        assert any(h["path"] == "case.txt" for h in ci["hits"])
        cs = fs.grep(r"hello")
        assert not any(h["path"] == "case.txt" for h in cs["hits"])

    def test_glob_filter(self, fs: FileSystemTool) -> None:
        result = fs.grep(r"def \w+", glob="**/*.py")
        for h in result["hits"]:
            assert h["path"].endswith(".py")

    def test_skips_denylist(self, fs: FileSystemTool, workspace: Path) -> None:
        (workspace / ".env").write_text("SECRET=secret_pattern\n", encoding="utf-8")
        result = fs.grep("secret_pattern")
        assert all(h["path"] != ".env" for h in result["hits"])

    def test_invalid_regex(self, fs: FileSystemTool) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            fs.grep("(unclosed")

    def test_skips_binary(self, fs: FileSystemTool, workspace: Path) -> None:
        (workspace / "bin.dat").write_bytes(b"\x00\x01match\n")
        result = fs.grep("match")
        assert all(h["path"] != "bin.dat" for h in result["hits"])


# ---------------------------------------------------------------------------
# grep ripgrep parity (mocked)
# ---------------------------------------------------------------------------


class TestGrepRipgrepParity:
    def test_python_and_ripgrep_agree(self, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        py_fs = _tool(workspace, allow_grep_ripgrep=False)
        py_result = py_fs.grep(r"def \w+", glob="**/*.py")

        # Force the ripgrep path: since rg may not be installed in CI, we
        # synthesize a fake `rg` invocation by monkeypatching subprocess.run.
        rg_fs = _tool(workspace)
        rg_fs._ripgrep_path = "rg-fake"  # type: ignore[attr-defined]

        # Build a synthetic stdout in the format `<path>:<line>:<text>`.
        # Use absolute paths since that is what rg prints when given an
        # absolute base.
        synthesized: list[str] = []
        for h in py_result["hits"]:
            abs_path = (workspace / h["path"]).as_posix()
            synthesized.append(f"{abs_path}:{h['line_number']}:{h['line']}")
        stdout = "\n".join(synthesized) + ("\n" if synthesized else "")

        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        proc = _Proc()
        proc.stdout = stdout
        monkeypatch.setattr(
            "agent_framework._filesystem.subprocess.run",
            lambda *a, **k: proc,
        )
        rg_result = rg_fs.grep(r"def \w+", glob="**/*.py")
        assert rg_result["backend"] == "ripgrep"
        assert sorted((h["path"], h["line_number"]) for h in rg_result["hits"]) == sorted(
            (h["path"], h["line_number"]) for h in py_result["hits"]
        )


# ---------------------------------------------------------------------------
# delete / move / rename
# ---------------------------------------------------------------------------


class TestDestructive:
    def test_delete_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        fs.delete("README.md")
        assert not (workspace / "README.md").exists()

    def test_delete_directory_refused(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(IsADirectoryError):
            fs.delete("src")

    def test_delete_missing_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileNotFoundError):
            fs.delete("nope")

    def test_delete_denylisted_blocked(self, workspace: Path) -> None:
        fs = _tool(workspace)
        with pytest.raises(FileSystemSecurityError):
            fs.delete(".env")

    def test_move_file(self, workspace: Path) -> None:
        fs = _tool(workspace)
        fs.move("README.md", "docs/README.md")
        assert (workspace / "docs" / "README.md").exists()
        assert not (workspace / "README.md").exists()

    def test_move_existing_destination_refused(self, workspace: Path) -> None:
        fs = _tool(workspace)
        (workspace / "other.md").write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError):
            fs.move("README.md", "other.md")

    def test_rename_alias_of_move(self, workspace: Path) -> None:
        fs = _tool(workspace)
        fs.rename("README.md", "PROJECT.md")
        assert (workspace / "PROJECT.md").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize(
        ("rel", "pattern", "expected"),
        [
            ("src/main.py", "**/*.py", True),
            ("main.py", "**/*.py", True),
            ("src/main.py", "*.py", False),
            ("main.py", "*.py", True),
            ("src/main.py", "src/*.py", True),
            ("a/b/c/d.py", "**/*.py", True),
            (".git/config", "**/.git/**", True),
            (".git/config", ".git/**", True),
            (".env", "**/.env", True),
            (".env", ".env", True),
            (".env.local", ".env.*", True),
            ("src/util.py", "**/*.md", False),
        ],
    )
    def test_match_glob(self, rel: str, pattern: str, expected: bool) -> None:
        assert _match_glob(rel, pattern) is expected

    def test_match_glob_normalizes_backslashes(self) -> None:
        assert _match_glob("src\\main.py", "**/*.py") is True

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("/a/b.py:42:matched text", ("/a/b.py", 42, "matched text")),
            ("C:\\repo\\a.py:7:hello", ("C:\\repo\\a.py", 7, "hello")),
            ("nocolon", None),
            ("", None),
            ("file:notdigits:line", None),
        ],
    )
    def test_parse_rg_line(self, line: str, expected: tuple[str, int, str] | None) -> None:
        assert _parse_rg_line(line) == expected

    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "f.txt"
        _atomic_write_bytes(target, b"hello")
        assert target.read_bytes() == b"hello"

    def test_atomic_write_cleans_temp_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "f.txt"

        def boom(*a: Any, **kw: Any) -> None:
            raise OSError("simulated")

        monkeypatch.setattr("agent_framework._filesystem.os.replace", boom)
        with pytest.raises(OSError):
            _atomic_write_bytes(target, b"hello")
        assert list(tmp_path.glob(".fstool-*")) == []


# ---------------------------------------------------------------------------
# MultiEdit
# ---------------------------------------------------------------------------


class TestMultiEdit:
    def _seed(self, workspace: Path) -> Path:
        f = workspace / "src" / "module.py"
        f.write_text(
            "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
            encoding="utf-8",
        )
        return f

    def test_applies_sequential_edits_atomically(self, workspace: Path) -> None:
        f = self._seed(workspace)
        result = _tool(workspace).multi_edit(
            "src/module.py",
            [
                {"old_str": "def foo():", "new_str": "def alpha():"},
                {"old_str": "def bar():", "new_str": "def beta():"},
                {"old_str": "return 1", "new_str": "return 10"},
            ],
        )
        assert result["edits_applied"] == 3
        assert result["total_replacements"] == 3
        body = f.read_text(encoding="utf-8")
        assert "def alpha():" in body
        assert "def beta():" in body
        assert "return 10" in body

    def test_supports_explicit_count(self, workspace: Path) -> None:
        f = workspace / "src" / "dup.py"
        f.write_text("X\nX\nY\n", encoding="utf-8")
        _tool(workspace).multi_edit(
            "src/dup.py",
            [{"old_str": "X", "new_str": "Z", "count": 2}],
        )
        assert f.read_text(encoding="utf-8") == "Z\nZ\nY\n"

    def test_rejects_when_any_edit_does_not_match(self, workspace: Path) -> None:
        f = self._seed(workspace)
        original = f.read_bytes()
        with pytest.raises(ValueError, match=r"edits\[1\]"):
            _tool(workspace).multi_edit(
                "src/module.py",
                [
                    {"old_str": "def foo():", "new_str": "def alpha():"},
                    {"old_str": "DOES_NOT_EXIST", "new_str": "x"},
                ],
            )
        # Atomicity: file unchanged.
        assert f.read_bytes() == original

    def test_sequential_match_count_after_prior_edit(self, workspace: Path) -> None:
        # First edit produces a duplicate; second edit's count must reflect that.
        f = workspace / "src" / "seq.py"
        f.write_text("A\nB\n", encoding="utf-8")
        _tool(workspace).multi_edit(
            "src/seq.py",
            [
                {"old_str": "A", "new_str": "B"},
                {"old_str": "B", "new_str": "C", "count": 2},
            ],
        )
        assert f.read_text(encoding="utf-8") == "C\nC\n"

    def test_rejects_empty_edits_list(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _tool(workspace).multi_edit("src/main.py", [])

    def test_rejects_empty_old_str(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="old_str must be non-empty"):
            _tool(workspace).multi_edit(
                "src/main.py",
                [{"old_str": "", "new_str": "x"}],
            )

    def test_rejects_invalid_count(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="count"):
            _tool(workspace).multi_edit(
                "src/main.py",
                [{"old_str": "print", "new_str": "log", "count": 0}],
            )

    def test_rejects_non_dict_edit(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            _tool(workspace).multi_edit("src/main.py", ["bad"])  # type: ignore[list-item]

    def test_rejects_missing_strings(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="string old_str and new_str"):
            _tool(workspace).multi_edit(
                "src/main.py",
                [{"old_str": "print"}],  # missing new_str
            )

    def test_atomic_failure_leaves_file_unchanged(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = self._seed(workspace)
        original = f.read_bytes()

        def boom(*a: Any, **kw: Any) -> None:
            raise OSError("simulated")

        monkeypatch.setattr("agent_framework._filesystem.os.replace", boom)
        with pytest.raises(OSError):
            _tool(workspace).multi_edit(
                "src/module.py",
                [{"old_str": "def foo():", "new_str": "def alpha():"}],
            )
        assert f.read_bytes() == original


# ---------------------------------------------------------------------------
# Gitignore filtering
# ---------------------------------------------------------------------------


class TestGitignore:
    def test_glob_skips_gitignored_files_by_default(self, workspace: Path) -> None:
        (workspace / "build").mkdir()
        (workspace / "build" / "out.py").write_text("# generated\n", encoding="utf-8")
        (workspace / "src" / "ignored.py").write_text("# ignored\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("build/\nignored.py\n", encoding="utf-8")

        matches = _tool(workspace).glob("**/*.py")["matches"]
        assert "build/out.py" not in matches
        assert "src/ignored.py" not in matches
        # Non-ignored files still appear.
        assert "src/main.py" in matches

    def test_can_disable_gitignore(self, workspace: Path) -> None:
        (workspace / "build").mkdir()
        (workspace / "build" / "out.py").write_text("x\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("build/\n", encoding="utf-8")

        matches = _tool(workspace, respect_gitignore=False).glob("**/*.py")["matches"]
        assert "build/out.py" in matches

    def test_negation_unignores(self, workspace: Path) -> None:
        (workspace / "build").mkdir()
        (workspace / "build" / "keep.py").write_text("x\n", encoding="utf-8")
        (workspace / "build" / "drop.py").write_text("x\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("build/*.py\n!build/keep.py\n", encoding="utf-8")

        matches = _tool(workspace).glob("**/*.py")["matches"]
        assert "build/keep.py" in matches
        assert "build/drop.py" not in matches

    def test_nested_gitignore_scopes_to_subdir(self, workspace: Path) -> None:
        (workspace / "pkg").mkdir()
        (workspace / "pkg" / "a.py").write_text("x\n", encoding="utf-8")
        (workspace / "pkg" / "b.py").write_text("x\n", encoding="utf-8")
        (workspace / "pkg" / ".gitignore").write_text("a.py\n", encoding="utf-8")
        # A sibling top-level a.py must not be ignored by the nested rule.
        (workspace / "a.py").write_text("x\n", encoding="utf-8")

        matches = _tool(workspace).glob("**/*.py")["matches"]
        assert "pkg/a.py" not in matches
        assert "pkg/b.py" in matches
        assert "a.py" in matches

    def test_directory_only_pattern(self, workspace: Path) -> None:
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "x.js").write_text("x\n", encoding="utf-8")
        # A file with the same name should NOT be ignored by 'node_modules/'.
        (workspace / "node_modules.txt").write_text("note\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("node_modules/\n", encoding="utf-8")

        glob_matches = _tool(workspace).glob("**/*")["matches"]
        assert "node_modules/x.js" not in glob_matches
        assert "node_modules.txt" in glob_matches

    def test_anchored_pattern(self, workspace: Path) -> None:
        # Leading slash anchors to gitignore base. '/foo' matches root foo only.
        (workspace / "foo.py").write_text("x\n", encoding="utf-8")
        (workspace / "src" / "foo.py").write_text("x\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("/foo.py\n", encoding="utf-8")

        matches = _tool(workspace).glob("**/*.py")["matches"]
        assert "foo.py" not in matches
        assert "src/foo.py" in matches

    def test_grep_python_respects_gitignore(self, workspace: Path) -> None:
        (workspace / "build").mkdir()
        (workspace / "build" / "out.py").write_text("SECRET=match\n", encoding="utf-8")
        (workspace / "src" / "main.py").write_text("SECRET=match\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("build/\n", encoding="utf-8")

        # Force python backend to validate filter logic.
        result = _tool(workspace, allow_grep_ripgrep=False).grep("SECRET")
        paths = {hit["path"] for hit in result["hits"]}
        assert "build/out.py" not in paths
        assert "src/main.py" in paths

    def test_list_dir_respects_gitignore(self, workspace: Path) -> None:
        (workspace / "build").mkdir()
        (workspace / "build" / "out.py").write_text("x\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("build/\n", encoding="utf-8")

        result = _tool(workspace).list_dir(".", depth=2)
        paths = {e["path"] for e in result["entries"]}
        assert "build" not in paths
        assert "build/out.py" not in paths

    def test_view_still_works_on_gitignored_file(self, workspace: Path) -> None:
        # Gitignore controls discovery, NOT direct access. Explicit reads
        # of a gitignored file must still succeed.
        (workspace / "build").mkdir()
        target = workspace / "build" / "out.py"
        target.write_text("hello\n", encoding="utf-8")
        (workspace / ".gitignore").write_text("build/\n", encoding="utf-8")

        result = _tool(workspace).view("build/out.py")
        assert "hello" in result["content"]

    def test_denylist_still_wins_over_gitignore_negation(self, workspace: Path) -> None:
        # Even if a user un-ignores a sensitive file, the denylist still blocks it.
        (workspace / ".gitignore").write_text("!.env\n", encoding="utf-8")
        matches = _tool(workspace).glob("**/*")["matches"]
        assert ".env" not in matches

    def test_no_gitignore_file_means_no_filtering(self, workspace: Path) -> None:
        # Workspace fixture has no .gitignore — everything (except denylist) shows.
        matches = _tool(workspace).glob("**/*.py")["matches"]
        assert "src/main.py" in matches
