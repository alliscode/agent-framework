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
import contextlib
import os
import signal
import sys
from pathlib import Path
from typing import Annotated

from agent_framework import ai_function

# Windows-specific: flag to create a new process group
CREATE_NEW_PROCESS_GROUP = 0x00000200 if sys.platform == "win32" else 0


class BackgroundProcess:
    """Tracks a background process and its output."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        command: str,
        stdout_buffer: list[str],
        stderr_buffer: list[str],
    ):
        self.process = process
        self.command = command
        self.stdout_buffer = stdout_buffer
        self.stderr_buffer = stderr_buffer
        self.started_at = asyncio.get_event_loop().time()

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None

    @property
    def pid(self) -> int:
        return self.process.pid


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
        # Track background processes by ID
        self._background_processes: dict[str, BackgroundProcess] = {}
        self._next_bg_id = 1

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
        except ValueError as e:
            raise ValueError(f"Path '{path}' escapes the sandbox directory") from e

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
            self.get_background_output,
            self.stop_background_process,
            self.list_background_processes,
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

            return resolved.read_text(encoding="utf-8")
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
        path: Annotated[str, "The directory path relative to working directory. Use '.' for current."] = ".",
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

    async def _kill_process_tree(self, process: asyncio.subprocess.Process) -> None:
        """Kill a process and all its children.

        On Windows, we need to use taskkill to kill the entire process tree.
        On Unix, we send SIGTERM to the process group.
        """
        if process.returncode is not None:
            return  # Already terminated

        pid = process.pid
        try:
            if sys.platform == "win32":
                # Use taskkill to kill the entire process tree
                # /F = force, /T = tree (kill child processes)
                kill_proc = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await kill_proc.wait()
            else:
                # On Unix, send SIGTERM to the process group
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(pid), signal.SIGTERM)

            # Give it a moment to terminate gracefully
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill if still running
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
        except Exception:
            # Last resort
            with contextlib.suppress(Exception):
                process.kill()

    @ai_function(approval_mode="never_require")
    async def run_command(
        self,
        command: Annotated[str, "The shell command to run"],
        timeout_seconds: Annotated[int, "Timeout in seconds (default: 60)"] = 60,
        background: Annotated[bool, "Run in background for long-running processes"] = False,
    ) -> str:
        """Run a shell command in the working directory.

        Use this to run tests, install dependencies, or execute scripts.
        The command runs with the working directory as the current directory.

        For server commands (npm start, flask run, python -m http.server), use
        background=True to run them without blocking. You can then use
        get_background_output() to check their status and stop_background_process()
        to terminate them.

        Common commands:
        - python script.py: Run a Python script
        - pytest: Run tests
        - pip install package: Install a package
        - npm install: Install node dependencies
        - npm start (with background=True): Start a dev server
        - flask run (with background=True): Start Flask server
        """
        # Cap timeout at 10 minutes max
        timeout_seconds = min(timeout_seconds, 600)

        try:
            # Platform-specific subprocess creation
            if sys.platform == "win32":
                # On Windows, create a new process group for proper tree killing
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_directory,
                    creationflags=CREATE_NEW_PROCESS_GROUP,
                )
            else:
                # On Unix, use start_new_session for process group
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_directory,
                    start_new_session=True,
                )

            # Handle background mode
            if background:
                return await self._start_background_process(process, command)

            # Print progress indicator for long-running commands
            print(
                f"\n  [Running: {command[:50]}{'...' if len(command) > 50 else ''}]",
                end="",
                flush=True,
            )

            # Create a task for communicate() and poll for progress
            async def wait_with_progress() -> tuple[bytes, bytes]:
                start_time = asyncio.get_event_loop().time()
                dots_printed = 0

                # Start the communicate task
                communicate_task = asyncio.create_task(process.communicate())

                try:
                    while not communicate_task.done():
                        # Wait a bit and check if done
                        done, _ = await asyncio.wait({communicate_task}, timeout=5.0)

                        if done:
                            break

                        # Still running - check timeout
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > timeout_seconds:
                            # Kill the entire process tree
                            await self._kill_process_tree(process)
                            communicate_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await communicate_task
                            raise TimeoutError(
                                f"Command timed out after {int(elapsed)} seconds. "
                                f"If this is a long-running server command, use "
                                f"background=True to run it without blocking."
                            )

                        # Print a dot every 5 seconds to show progress
                        print(".", end="", flush=True)
                        dots_printed += 1

                        # Every 30 seconds, show elapsed time
                        if dots_printed % 6 == 0:
                            print(f" ({int(elapsed)}s)", end="", flush=True)

                    return await communicate_task
                except asyncio.CancelledError:
                    # If our coroutine is cancelled, clean up the process
                    await self._kill_process_tree(process)
                    communicate_task.cancel()
                    raise

            stdout, stderr = await wait_with_progress()
            print()  # Newline after progress dots

            # Decode output
            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Truncate very long output
            max_output_len = 50000
            if len(stdout_text) > max_output_len:
                stdout_text = stdout_text[:max_output_len] + f"\n... (truncated, {len(stdout_text)} total chars)"
            if len(stderr_text) > max_output_len:
                stderr_text = stderr_text[:max_output_len] + f"\n... (truncated, {len(stderr_text)} total chars)"

            output_parts = []
            if stdout_text:
                output_parts.append(f"STDOUT:\n{stdout_text}")
            if stderr_text:
                output_parts.append(f"STDERR:\n{stderr_text}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if process.returncode == 0:
                return f"Command succeeded (exit code 0):\n{output}"
            return f"Command failed (exit code {process.returncode}):\n{output}"

        except TimeoutError as e:
            print()  # Newline if we were printing progress
            return f"Error: {e}"
        except asyncio.CancelledError:
            print()  # Newline if we were printing progress
            return "Error: Command was cancelled"
        except Exception as e:
            print()  # Newline if we were printing progress
            return f"Error running command: {e}"

    async def _start_background_process(
        self,
        process: asyncio.subprocess.Process,
        command: str,
    ) -> str:
        """Start tracking a background process and return immediately."""
        bg_id = f"bg_{self._next_bg_id}"
        self._next_bg_id += 1

        stdout_buffer: list[str] = []
        stderr_buffer: list[str] = []

        # Create background process tracker
        bg_process = BackgroundProcess(process, command, stdout_buffer, stderr_buffer)
        self._background_processes[bg_id] = bg_process

        # Start async tasks to read output without blocking
        async def read_stream(
            stream: asyncio.StreamReader | None,
            buffer: list[str],
        ) -> None:
            if stream is None:
                return
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    buffer.append(line.decode("utf-8", errors="replace"))
            except Exception:
                pass  # Stream closed or error

        # Start output readers in background
        asyncio.create_task(read_stream(process.stdout, stdout_buffer))
        asyncio.create_task(read_stream(process.stderr, stderr_buffer))

        return (
            f"Started background process '{bg_id}' (PID: {process.pid})\n"
            f"Command: {command}\n\n"
            f"Use get_background_output('{bg_id}') to check output.\n"
            f"Use stop_background_process('{bg_id}') to terminate it."
        )

    @ai_function(approval_mode="never_require")
    async def get_background_output(
        self,
        process_id: Annotated[str, "The background process ID (e.g., 'bg_1')"],
        clear: Annotated[bool, "Clear the output buffer after reading"] = False,
    ) -> str:
        """Get output from a background process.

        Use this to check on long-running processes started with background=True.
        By default, output accumulates. Set clear=True to clear after reading.
        """
        if process_id not in self._background_processes:
            available = list(self._background_processes.keys())
            if available:
                return f"Error: Process '{process_id}' not found. Available: {available}"
            return f"Error: Process '{process_id}' not found. No background processes running."

        bg = self._background_processes[process_id]

        # Check if process is still running
        status = "RUNNING" if bg.is_running else f"EXITED (code: {bg.process.returncode})"

        # Get buffered output
        stdout = "".join(bg.stdout_buffer)
        stderr = "".join(bg.stderr_buffer)

        if clear:
            bg.stdout_buffer.clear()
            bg.stderr_buffer.clear()

        output_parts = [f"Process {process_id} - Status: {status}"]
        output_parts.append(f"Command: {bg.command}")
        output_parts.append(f"PID: {bg.pid}")

        if stdout:
            output_parts.append(f"\nSTDOUT:\n{stdout}")
        if stderr:
            output_parts.append(f"\nSTDERR:\n{stderr}")
        if not stdout and not stderr:
            output_parts.append("\n(no output yet)")

        return "\n".join(output_parts)

    @ai_function(approval_mode="never_require")
    async def stop_background_process(
        self,
        process_id: Annotated[str, "The background process ID (e.g., 'bg_1')"],
    ) -> str:
        """Stop a background process.

        Use this to terminate a long-running process started with background=True.
        """
        if process_id not in self._background_processes:
            available = list(self._background_processes.keys())
            if available:
                return f"Error: Process '{process_id}' not found. Available: {available}"
            return f"Error: Process '{process_id}' not found. No background processes running."

        bg = self._background_processes[process_id]

        if not bg.is_running:
            # Already stopped, just clean up
            del self._background_processes[process_id]
            return f"Process {process_id} already exited (code: {bg.process.returncode}). Removed from tracking."

        # Kill the process
        await self._kill_process_tree(bg.process)

        # Get final output
        stdout = "".join(bg.stdout_buffer)
        stderr = "".join(bg.stderr_buffer)

        # Remove from tracking
        del self._background_processes[process_id]

        output_parts = [f"Stopped process {process_id} (PID: {bg.pid})"]
        if stdout:
            # Truncate if too long
            if len(stdout) > 5000:
                stdout = stdout[-5000:] + "\n... (truncated, showing last 5000 chars)"
            output_parts.append(f"\nFinal STDOUT:\n{stdout}")
        if stderr:
            if len(stderr) > 5000:
                stderr = stderr[-5000:] + "\n... (truncated, showing last 5000 chars)"
            output_parts.append(f"\nFinal STDERR:\n{stderr}")

        return "\n".join(output_parts)

    @ai_function(approval_mode="never_require")
    def list_background_processes(self) -> str:
        """List all background processes.

        Shows running and recently exited background processes.
        """
        if not self._background_processes:
            return "No background processes."

        lines = ["Background processes:"]
        for bg_id, bg in self._background_processes.items():
            status = "RUNNING" if bg.is_running else f"EXITED ({bg.process.returncode})"
            cmd_preview = bg.command[:50] + "..." if len(bg.command) > 50 else bg.command
            lines.append(f"  {bg_id}: [{status}] PID {bg.pid} - {cmd_preview}")

        return "\n".join(lines)
