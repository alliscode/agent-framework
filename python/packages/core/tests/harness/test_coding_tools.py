# Copyright (c) Microsoft. All rights reserved.

"""Tests for smart file tools in CodingTools (Phase 11)."""

import sys
from pathlib import Path

import pytest

# CodingTools lives in the samples directory, add it to sys.path
_samples_harness = Path(__file__).resolve().parents[4] / "samples" / "getting_started" / "workflows" / "harness"
if str(_samples_harness) not in sys.path:
    sys.path.insert(0, str(_samples_harness))

from coding_tools import CodingTools  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    """Create a sandbox directory with test files."""
    # Create a simple Python file
    (tmp_path / "hello.py").write_text(
        "# A sample file\n"
        "import os\n"
        "\n"
        "def greet(name):\n"
        "    return f'Hello, {name}!'\n"
        "\n"
        "def farewell(name):\n"
        "    return f'Goodbye, {name}!'\n"
        "\n"
        "class Greeter:\n"
        "    pass\n",
        encoding="utf-8",
    )

    # Create a subdirectory with files
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "main.py").write_text("print('main')\n", encoding="utf-8")
    (sub / "utils.py").write_text("def helper():\n    pass\n", encoding="utf-8")

    # Create a nested subdirectory
    deep = sub / "core"
    deep.mkdir()
    (deep / "engine.py").write_text("class Engine:\n    pass\n", encoding="utf-8")
    (deep / "config.json").write_text('{"key": "value"}\n', encoding="utf-8")

    # Create a markdown file at root
    (tmp_path / "README.md").write_text("# Project\nThis is a test.\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def tools(sandbox: Path) -> CodingTools:
    """Create CodingTools instance scoped to sandbox."""
    return CodingTools(working_directory=sandbox)


class TestReadFileLineRange:
    """Tests for read_file with line range support (Phase 11a)."""

    def test_full_file_backward_compat(self, tools: CodingTools) -> None:
        """Omitting start_line/end_line reads the entire file unchanged."""
        result = tools.read_file("hello.py")
        assert "def greet(name):" in result
        assert "class Greeter:" in result
        # Should NOT have line-number prefix when no range specified
        assert "[Lines" not in result

    def test_line_range_basic(self, tools: CodingTools) -> None:
        """Reading a specific line range returns numbered lines with header."""
        result = tools.read_file("hello.py", start_line=4, end_line=5)
        assert "[Lines 4-5 of 11 in hello.py]" in result
        assert "4. def greet(name):" in result
        assert "5." in result

    def test_start_line_only(self, tools: CodingTools) -> None:
        """start_line without end_line reads to end of file."""
        result = tools.read_file("hello.py", start_line=10)
        assert "[Lines 10-11 of 11 in hello.py]" in result
        assert "10. class Greeter:" in result

    def test_end_line_only(self, tools: CodingTools) -> None:
        """end_line without start_line reads from beginning."""
        result = tools.read_file("hello.py", end_line=2)
        assert "[Lines 1-2 of 11 in hello.py]" in result
        assert "1. # A sample file" in result
        assert "2. import os" in result

    def test_end_line_minus_one(self, tools: CodingTools) -> None:
        """end_line=-1 reads to end of file."""
        result = tools.read_file("hello.py", start_line=10, end_line=-1)
        assert "[Lines 10-11 of 11 in hello.py]" in result
        assert "11. " in result

    def test_end_line_exceeds_total(self, tools: CodingTools) -> None:
        """end_line beyond file length is clamped."""
        result = tools.read_file("hello.py", start_line=10, end_line=999)
        assert "[Lines 10-11 of 11 in hello.py]" in result

    def test_nonexistent_file(self, tools: CodingTools) -> None:
        """Reading a missing file returns an error."""
        result = tools.read_file("nope.py")
        assert "Error" in result


class TestGrepFiles:
    """Tests for grep_files search tool (Phase 11b)."""

    def test_basic_pattern(self, tools: CodingTools) -> None:
        """grep_files finds files containing a pattern."""
        result = tools.grep_files("def greet")
        assert "hello.py" in result
        assert "Found 1 file(s)" in result

    def test_file_glob_filter(self, tools: CodingTools) -> None:
        """file_glob limits search to matching files."""
        result = tools.grep_files("pass", file_glob="*.py")
        assert "utils.py" in result or "engine.py" in result
        assert ".json" not in result

    def test_include_lines(self, tools: CodingTools) -> None:
        """include_lines=True shows line numbers and content."""
        result = tools.grep_files("class", include_lines=True)
        assert "hello.py" in result
        # Should show line number prefix
        assert ": class" in result or ":  class" not in result

    def test_regex_pattern(self, tools: CodingTools) -> None:
        """Regex patterns work for searching."""
        result = tools.grep_files(r"def \w+\(")
        assert "hello.py" in result

    def test_invalid_regex_falls_back(self, tools: CodingTools) -> None:
        """Invalid regex falls back to literal match."""
        result = tools.grep_files("[invalid")
        # Should not raise — falls back to literal search
        assert "No files matching" in result or "Found" in result

    def test_no_matches(self, tools: CodingTools) -> None:
        """Returns friendly message when nothing matches."""
        result = tools.grep_files("zzz_nonexistent_pattern_zzz")
        assert "No files matching" in result

    def test_max_results(self, tools: CodingTools) -> None:
        """max_results limits output."""
        result = tools.grep_files("pass", max_results=1)
        assert "Found 1 file(s)" in result

    def test_invalid_directory(self, tools: CodingTools) -> None:
        """Searching in a nonexistent directory returns error."""
        result = tools.grep_files("pattern", path="nonexistent_dir")
        assert "Error" in result


class TestFindFiles:
    """Tests for find_files glob tool (Phase 11c)."""

    def test_find_py_files(self, tools: CodingTools) -> None:
        """find_files locates Python files recursively."""
        result = tools.find_files("**/*.py")
        assert "hello.py" in result
        assert "main.py" in result
        assert "engine.py" in result

    def test_find_md_files(self, tools: CodingTools) -> None:
        """find_files finds markdown files."""
        result = tools.find_files("*.md")
        assert "README.md" in result

    def test_find_in_subdir(self, tools: CodingTools) -> None:
        """find_files works within a subdirectory."""
        result = tools.find_files("*.py", path="src")
        assert "main.py" in result
        assert "hello.py" not in result

    def test_find_no_matches(self, tools: CodingTools) -> None:
        """Returns message when no files match."""
        result = tools.find_files("*.xyz")
        assert "No files matching" in result

    def test_max_results(self, tools: CodingTools) -> None:
        """max_results limits output."""
        result = tools.find_files("**/*.py", max_results=2)
        assert "Found 2 file(s)" in result

    def test_invalid_directory(self, tools: CodingTools) -> None:
        """Searching in a nonexistent directory returns error."""
        result = tools.find_files("*.py", path="nonexistent_dir")
        assert "Error" in result


class TestListDirectoryDepth:
    """Tests for list_directory with depth support (Phase 11d)."""

    def test_depth_1(self, tools: CodingTools) -> None:
        """depth=1 shows only immediate children."""
        result = tools.list_directory(".", depth=1)
        assert "[DIR]" in result
        assert "hello.py" in result
        # Should NOT show nested files with depth=1
        assert "engine.py" not in result
        assert "main.py" not in result

    def test_depth_2_default(self, tools: CodingTools) -> None:
        """Default depth=2 shows children and grandchildren."""
        result = tools.list_directory(".")
        assert "hello.py" in result
        # src/ children should appear at depth 2
        assert "main.py" in result
        # But src/core/engine.py is depth 3, should not appear
        assert "engine.py" not in result

    def test_depth_3(self, tools: CodingTools) -> None:
        """depth=3 shows three levels deep."""
        result = tools.list_directory(".", depth=3)
        assert "engine.py" in result

    def test_tree_indentation(self, tools: CodingTools) -> None:
        """Nested items are indented."""
        result = tools.list_directory(".", depth=2)
        lines = result.splitlines()
        # Find an indented line (child of a directory)
        indented_lines = [ln for ln in lines if ln.startswith("  ")]
        assert len(indented_lines) > 0

    def test_nonexistent_directory(self, tools: CodingTools) -> None:
        """Listing a missing directory returns error."""
        result = tools.list_directory("nonexistent")
        assert "Error" in result

    def test_skips_hidden(self, tools: CodingTools, sandbox: Path) -> None:
        """Hidden files and directories are skipped."""
        (sandbox / ".hidden_dir").mkdir()
        (sandbox / ".hidden_file").write_text("secret", encoding="utf-8")
        result = tools.list_directory(".", depth=2)
        assert ".hidden_dir" not in result
        assert ".hidden_file" not in result


class TestGetToolsList:
    """Tests for get_tools including new tools (Phase 11e)."""

    def test_includes_grep_files(self, tools: CodingTools) -> None:
        """grep_files is in the tools list."""
        tool_list = tools.get_tools()
        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tool_list]
        assert "grep_files" in tool_names

    def test_includes_find_files(self, tools: CodingTools) -> None:
        """find_files is in the tools list."""
        tool_list = tools.get_tools()
        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tool_list]
        assert "find_files" in tool_names

    def test_all_original_tools_present(self, tools: CodingTools) -> None:
        """All original tools are still present."""
        tool_list = tools.get_tools()
        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tool_list]
        for name in ["read_file", "write_file", "list_directory", "run_command", "create_directory"]:
            assert name in tool_names
