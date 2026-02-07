# Copyright (c) Microsoft. All rights reserved.

"""Interactive REPL using the Agent Harness.

This script provides an interactive command-line interface for working with
an AI agent that has access to file system and command execution tools.

Prerequisites:
- Azure OpenAI configured with AZURE_OPENAI_ENDPOINT environment variable
- Authentication via azure-identity (run `az login` before executing)

Usage:
    python harness_repl.py [--sandbox PATH]

The agent can read/write files, list directories, and run commands within
a sandboxed directory.
"""

import argparse
import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path

from agent_framework import AgentRunUpdateEvent, WorkflowOutputEvent
from agent_framework._harness import (
    AgentHarness,
    HarnessResult,
    HarnessStatus,
    get_task_complete_tool,
)
from agent_framework._harness._compaction import (
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
)
from agent_framework._workflows._events import (
    ExecutorCompletedEvent,
    ExecutorInvokedEvent,
    SuperStepStartedEvent,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from coding_tools import CodingTools

# Debug logger - writes to file for analysis
debug_logger: logging.Logger | None = None


def setup_debug_logging(log_file: Path) -> logging.Logger:
    """Set up debug logging to a file."""
    logger = logging.getLogger("harness_repl_debug")
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Format with timestamps
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def debug(msg: str) -> None:
    """Write a debug message to the log file."""
    if debug_logger:
        debug_logger.debug(msg)


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_lifecycle(message: str) -> None:
    """Print a lifecycle event message."""
    print(f"{Colors.DIM}[{Colors.YELLOW}⚡{Colors.RESET}{Colors.DIM}] {message}{Colors.RESET}")


def print_system(message: str) -> None:
    """Print a system message."""
    print(f"{Colors.DIM}[{Colors.CYAN}ℹ{Colors.RESET}{Colors.DIM}] {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[✗] {message}{Colors.RESET}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[✓] {message}{Colors.RESET}")


AGENT_INSTRUCTIONS = """You are a capable AI coding assistant with access to a local workspace.
You can read and write files, list directories, and run shell commands.
When asked to investigate code, be thorough — read every relevant source file
before drawing conclusions or writing deliverables.
"""
"""


async def run_repl(sandbox_dir: Path, max_turns: int = 20, verbose: bool = False) -> None:
    """Run the interactive REPL."""
    global debug_logger

    # Set up debug logging
    log_file = sandbox_dir / "debug.log"
    debug_logger = setup_debug_logging(log_file)

    # Also enable harness-level logging to the same file
    harness_logger = logging.getLogger("agent_framework._harness")
    harness_logger.setLevel(logging.DEBUG)
    harness_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    harness_handler.setLevel(logging.DEBUG)
    harness_handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | HARNESS | %(message)s", datefmt="%H:%M:%S"))
    harness_logger.addHandler(harness_handler)

    debug(f"=== REPL Started at {datetime.now().isoformat()} ===")
    debug(f"Sandbox: {sandbox_dir}")
    debug(f"Max turns: {max_turns}, Verbose: {verbose}")

    print(f"\n{Colors.BOLD}Agent Harness REPL{Colors.RESET}")
    print("=" * 50)
    print(f"Sandbox: {Colors.CYAN}{sandbox_dir}{Colors.RESET}")
    print(f"Debug log: {Colors.DIM}{log_file}{Colors.RESET}")
    print(f"Max turns: {max_turns} | Type {Colors.BOLD}help{Colors.RESET} for commands")
    print("=" * 50)

    # Create tools sandboxed to the directory
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    # Create Azure OpenAI client and agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions=AGENT_INSTRUCTIONS,
        name="assistant",
        tools=all_tools,
    )

    # Create harness with all features enabled
    harness = AgentHarness(
        agent,
        max_turns=max_turns,
        enable_stall_detection=True,
        stall_threshold=3,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        enable_work_items=True,
        enable_compaction=True,
        compaction_store=InMemoryCompactionStore(),
        artifact_store=InMemoryArtifactStore(),
        summary_cache=InMemorySummaryCache(max_entries=100),
        max_input_tokens=100_000,
        soft_threshold_percent=0.85,
        sandbox_path=str(sandbox_dir),
    )

    # Track verbose mode
    show_verbose = verbose
    message_count = 0  # Track conversation length

    while True:
        try:
            # Get user input
            print()
            user_input = input(f"{Colors.GREEN}You:{Colors.RESET} ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ("exit", "quit"):
                print_system("Goodbye!")
                break

            if user_input.lower() == "files":
                print_system("Files in sandbox:")
                for f in sandbox_dir.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(sandbox_dir)
                        if "__pycache__" not in str(rel) and ".pytest_cache" not in str(rel):
                            print(f"  {rel}")
                continue

            if user_input.lower() == "clear":
                print("\033[2J\033[H", end="")  # Clear screen
                continue

            if user_input.lower() == "reset":
                # Create a fresh harness to clear conversation history
                harness = AgentHarness(
                    agent,
                    max_turns=max_turns,
                    enable_stall_detection=True,
                    stall_threshold=3,
                    enable_continuation_prompts=True,
                    max_continuation_prompts=2,
                    enable_work_items=True,
                    enable_compaction=True,
                    compaction_store=InMemoryCompactionStore(),
                    artifact_store=InMemoryArtifactStore(),
                    summary_cache=InMemorySummaryCache(max_entries=100),
                    max_input_tokens=100_000,
                    soft_threshold_percent=0.85,
                    sandbox_path=str(sandbox_dir),
                )
                message_count = 0
                print_system("Conversation history cleared (new harness instance)")
                continue

            if user_input.lower() == "help":
                print(f"\n{Colors.BOLD}Commands:{Colors.RESET}")
                print(f"  {Colors.CYAN}exit, quit{Colors.RESET} - Exit the REPL")
                print(f"  {Colors.CYAN}files{Colors.RESET}      - List files in sandbox")
                print(f"  {Colors.CYAN}clear{Colors.RESET}      - Clear the screen")
                print(f"  {Colors.CYAN}reset{Colors.RESET}      - Clear conversation history")
                print(f"  {Colors.CYAN}help{Colors.RESET}       - Show this help")
                print(f"\n{Colors.BOLD}Example tasks:{Colors.RESET}")
                print("  Create a hello world Python script and run it")
                print("  Write a function to calculate fibonacci numbers")
                print("  Create a simple REST API client")
                print("  Read the files in this directory and summarize them")
                continue

            # Show conversation indicator
            message_count += 1
            if message_count > 1:
                print_system(f"Message #{message_count} (conversation history preserved)")

            debug(f"--- User message #{message_count}: {user_input[:100]}...")

            # Run the harness - it natively accumulates conversation history
            print(f"\n{Colors.BLUE}Agent:{Colors.RESET} ", end="", flush=True)

            turn_count = 0
            continuation_count = 0
            last_executor = None
            event_count = 0

            debug("Starting harness.run_stream()")
            async for event in harness.run_stream(user_input):
                event_count += 1
                event_type = type(event).__name__
                debug(f"Event #{event_count}: {event_type}")
                # Track lifecycle events
                if isinstance(event, ExecutorInvokedEvent):
                    executor_name = event.executor_id
                    debug(f"  ExecutorInvoked: {executor_name}")
                    if executor_name != last_executor:
                        last_executor = executor_name
                        # Show interesting lifecycle events
                        if "context_pressure" in executor_name:
                            print()
                            print_lifecycle("Context pressure check...")
                        elif "stop_decision" in executor_name and show_verbose:
                            print()
                            print_lifecycle("Evaluating stop conditions...")
                        elif "repair" in executor_name and turn_count > 0 and show_verbose:
                            print()
                            print_lifecycle("Repair check...")

                if isinstance(event, ExecutorCompletedEvent):
                    executor_name = event.executor_id
                    debug(f"  ExecutorCompleted: {executor_name}")
                    if "agent_turn" in executor_name and show_verbose:
                        print()
                        print_lifecycle(f"Turn {turn_count + 1} completed")

                if isinstance(event, SuperStepStartedEvent):
                    debug("  SuperStepStarted")
                    # Could track supersteps if needed

                # Show agent streaming updates
                if isinstance(event, AgentRunUpdateEvent):
                    update = event.data

                    # Debug: log update details
                    update_role = getattr(update, "role", None)
                    update_text_preview = (getattr(update, "text", "") or "")[:50].replace("\n", "\\n")
                    update_contents = getattr(update, "contents", []) or []
                    content_types = [type(c).__name__ for c in update_contents]
                    finish_reason = getattr(update, "finish_reason", None)
                    debug(f"  AgentRunUpdate: role={update_role}, text='{update_text_preview}...', contents={content_types}, finish={finish_reason}")

                    # Detect continuation prompts by checking for the pattern
                    if hasattr(update, "role") and str(update.role) == "user":
                        if hasattr(update, "text") and "completed ALL" in str(update.text):
                            print()
                            print_lifecycle("Continuation prompt sent (verifying task completion)")
                            continuation_count += 1
                            print(f"\n{Colors.BLUE}Agent:{Colors.RESET} ", end="", flush=True)
                            continue

                    # Print agent's text as it streams
                    if hasattr(update, "text") and update.text:
                        print(update.text, end="", flush=True)

                    # Show tool calls if present in contents
                    if hasattr(update, "contents") and update.contents:
                        for content in update.contents:
                            content_type = type(content).__name__
                            if "FunctionCall" in content_type:
                                tool_name = getattr(content, "name", None)
                                tool_args = getattr(content, "arguments", None)
                                debug(f"    FunctionCall: {tool_name}({str(tool_args)[:100]}...)")
                                # Only print if we have a valid tool name (skip task_complete)
                                if tool_name and tool_name != "task_complete":
                                    print(f"\n{Colors.DIM}  → {tool_name}(){Colors.RESET}")
                            elif "FunctionResult" in content_type:
                                result_str = str(getattr(content, "result", ""))[:200]
                                debug(f"    FunctionResult: {result_str}...")
                                if show_verbose and result_str:
                                    print(f"\n{Colors.DIM}  ← {result_str[:80]}{Colors.RESET}", end="", flush=True)

                    # Track turns
                    if hasattr(update, "finish_reason") and update.finish_reason:
                        turn_count += 1
                        debug(f"  Turn incremented to {turn_count}, finish_reason={finish_reason}")

                # Handle final result
                if isinstance(event, WorkflowOutputEvent):
                    result = event.data
                    debug(f"  WorkflowOutput: {type(result).__name__}")
                    if isinstance(result, HarnessResult):
                        debug(f"  HarnessResult: status={result.status}, turns={result.turn_count}, reason={result.reason}")
                        print()  # Newline after streaming

                        # Show result status with appropriate styling
                        if result.status == HarnessStatus.DONE:
                            print_success(f"Completed in {result.turn_count} turns")
                        elif result.status == HarnessStatus.STALLED:
                            print_error(f"Stalled after {result.turn_count} turns - no progress detected")
                        elif result.status == HarnessStatus.FAILED:
                            print_error(f"Failed: {result.reason.message if result.reason else 'Unknown error'}")
                        else:
                            print_system(f"Status: {result.status.value} ({result.turn_count} turns)")

                        # Show if continuation prompts were used
                        if continuation_count > 0:
                            print_system(f"Used {continuation_count} continuation prompt(s)")

            debug(f"--- Message #{message_count} completed, total events: {event_count}")

        except KeyboardInterrupt:
            debug("KeyboardInterrupt received")
            print()
            print_system("Interrupted. Type 'exit' to quit.")
            continue
        except EOFError:
            debug("EOFError received")
            print()
            print_system("Goodbye!")
            break
        except Exception as e:
            import traceback
            debug(f"Exception: {e}")
            debug(f"Traceback:\n{traceback.format_exc()}")
            print()
            print_error(f"Error: {e}")
            continue


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive Agent Harness REPL")
    parser.add_argument(
        "--sandbox",
        type=Path,
        default=None,
        help="Directory for agent workspace (default: temp directory)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns per task (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show more detailed lifecycle events",
    )
    args = parser.parse_args()

    # Determine sandbox directory
    if args.sandbox:
        sandbox_dir = args.sandbox.resolve()
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        print_system(f"Using sandbox: {sandbox_dir}")
        await run_repl(sandbox_dir, max_turns=args.max_turns, verbose=args.verbose)
    else:
        # Use temporary directory
        with tempfile.TemporaryDirectory(prefix="harness_repl_") as temp_dir:
            await run_repl(Path(temp_dir), max_turns=args.max_turns, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
