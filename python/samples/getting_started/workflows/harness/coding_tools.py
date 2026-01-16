# Copyright (c) Microsoft. All rights reserved.

"""Simple coding tools for testing the Agent Harness.

These tools provide basic file system and command execution capabilities
for coding tasks. They are sandboxed to a specific working directory.

Usage:
    tools = CodingTools(working_directory="/path/to/sandbox")
    agent = chat_client.create_agent(
        instructions="You are a coding assistant.",
        tools=tools.get_tools(),
    )
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from agent_framework import ai_function


class CodingTools:
    """A collection of coding tools sandboxed to a working directory."""

    def __init__(self, working_directory: str | Path):
        """Initialize coding tools.

        Args:
            working_directory: The directory to sandbox all operations to.
                All file paths will be relative to this directory.
        """
        self.working_directory = Path(working_directory).resolve()
        self.working_directory.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the working directory.

        Ensures the path stays within the sandbox.

        Args:
            path: The path to resolve.

        Returns:
            The resolved absolute path.

        Raises:
            ValueError: If the path escapes the sandbox.
        """
        # Handle absolute paths by making them relative
        if os.path.isabs(path):
            path = os.path.relpath(path, "/")

        resolved = (self.working_directory / path).resolve()

        # Security check: ensure we stay within sandbox
        try:
            resolved.relative_to(self.working_directory)
        except ValueError:
            raise ValueError(f"Path '{path}' escapes the sandbox directory")

        return resolved

    def get_tools(self) -> list:
        """Get all coding tools as a list for the agent.

        Returns:
            List of tool functions.
        """
        return [
            self.read_file,
            self.write_file,
            self.list_directory,
            self.run_command,
            self.create_directory,
        ]

    @ai_function(approval_mode="never_require")
    def read_file(
        self,
        path: Annotated[str, "The path to the file to read, relative to the working directory"],
    ) -> str:
        """Read the contents of a file.

        Use this to examine existing code, configuration files, or any text file.
        """
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"Error: File '{path}' does not exist"
            if not resolved.is_file():
                return f"Error: '{path}' is not a file"

            content = resolved.read_text(encoding="utf-8")
            return content
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"

    @ai_function(approval_mode="never_require")
    def write_file(
        self,
        path: Annotated[str, "The path to the file to write, relative to the working directory"],
        content: Annotated[str, "The content to write to the file"],
    ) -> str:
        """Write content to a file, creating it if it doesn't exist.

        Use this to create new files or modify existing ones.
        Parent directories will be created automatically.
        """
        try:
            resolved = self._resolve_path(path)

            # Create parent directories if needed
            resolved.parent.mkdir(parents=True, exist_ok=True)

            resolved.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to '{path}'"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"

    @ai_function(approval_mode="never_require")
    def list_directory(
        self,
        path: Annotated[str, "The path to the directory to list, relative to the working directory. Use '.' for current directory"] = ".",
    ) -> str:
        """List the contents of a directory.

        Use this to explore the file structure and find files.
        Returns a tree-like view of files and directories.
        """
        try:
            resolved = self._resolve_path(path)
            if not resolved.exists():
                return f"Error: Directory '{path}' does not exist"
            if not resolved.is_dir():
                return f"Error: '{path}' is not a directory"

            items = []
            for item in sorted(resolved.iterdir()):
                rel_path = item.relative_to(self.working_directory)
                if item.is_dir():
                    items.append(f"[DIR]  {rel_path}/")
                else:
                    size = item.stat().st_size
                    items.append(f"[FILE] {rel_path} ({size} bytes)")

            if not items:
                return f"Directory '{path}' is empty"

            return "\n".join(items)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"

    @ai_function(approval_mode="never_require")
    def create_directory(
        self,
        path: Annotated[str, "The path to the directory to create, relative to the working directory"],
    ) -> str:
        """Create a directory and any necessary parent directories.

        Use this to set up directory structures for your project.
        """
        try:
            resolved = self._resolve_path(path)
            resolved.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory '{path}'"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating directory: {e}"

    @ai_function(approval_mode="never_require")
    async def run_command(
        self,
        command: Annotated[str, "The shell command to run"],
    ) -> str:
        """Run a shell command in the working directory.

        Use this to run tests, install dependencies, or execute scripts.
        The command runs with the working directory as the current directory.
        Commands have a 10-minute timeout. Long-running commands will show
        progress dots while executing.

        Common commands:
        - python script.py: Run a Python script
        - pytest: Run tests
        - pip install package: Install a package
        - npx create-react-app myapp: Create a React app (takes a few minutes)
        """
        timeout_seconds = 600  # 10 minute timeout

        try:
            # Print progress indicator for long-running commands
            print(f"\n  [Running: {command[:50]}{'...' if len(command) > 50 else ''}]", end="", flush=True)

            # Use asyncio subprocess for non-blocking execution
            # DEVNULL for stdin prevents commands from waiting for input
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
            )

            # Create a task for communicate() and poll for progress
            async def wait_with_progress() -> tuple[bytes, bytes]:
                start_time = asyncio.get_event_loop().time()
                dots_printed = 0

                # Start the communicate task
                communicate_task = asyncio.create_task(process.communicate())

                while not communicate_task.done():
                    # Wait a bit and check if done
                    done, _ = await asyncio.wait({communicate_task}, timeout=5.0)

                    if done:
                        break

                    # Still running - check timeout
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout_seconds:
                        process.kill()
                        await communicate_task  # Clean up
                        raise TimeoutError(f"Command timed out after {timeout_seconds // 60} minutes")

                    # Print a dot every 5 seconds to show progress
                    print(".", end="", flush=True)
                    dots_printed += 1

                    # Every 60 seconds, show elapsed time
                    if dots_printed % 12 == 0:
                        print(f" ({int(elapsed)}s)", end="", flush=True)

                return await communicate_task

            stdout, stderr = await wait_with_progress()
            print()  # Newline after progress dots

            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            output_parts = []
            if stdout_text:
                output_parts.append(f"STDOUT:\n{stdout_text}")
            if stderr_text:
                output_parts.append(f"STDERR:\n{stderr_text}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if process.returncode == 0:
                return f"Command succeeded (exit code 0):\n{output}"
            else:
                return f"Command failed (exit code {process.returncode}):\n{output}"

        except TimeoutError as e:
            print()  # Newline if we were printing progress
            return f"Error: {e}"
        except Exception as e:
            print()  # Newline if we were printing progress
            return f"Error running command: {e}"
