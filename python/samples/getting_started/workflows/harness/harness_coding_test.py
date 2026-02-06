# Copyright (c) Microsoft. All rights reserved.

"""Test the Agent Harness on a coding task.

This script demonstrates using the Agent Harness to run a coding agent
that can read/write files and execute commands.

Prerequisites:
- Azure OpenAI configured with AZURE_OPENAI_ENDPOINT environment variable
- Authentication via azure-identity (run `az login` before executing)

Usage:
    python harness_coding_test.py

The script creates a sandbox directory and gives the agent a coding task.
"""

import asyncio
import tempfile
from pathlib import Path

from agent_framework import AgentRunUpdateEvent, WorkflowOutputEvent
from agent_framework._harness import (
    AgentHarness,
    HarnessResult,
    Predicate,
    RequiredOutput,
    TaskContract,
    get_task_complete_tool,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from coding_tools import CodingTools


def is_binary_file(path: Path) -> bool:
    """Check if a file is binary."""
    try:
        with Path(path).open("rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True


async def run_interactive_mode(sandbox_dir: Path) -> None:
    """Run the harness in interactive mode (model judgment).

    This is like how Claude Code works - the model decides when it's done.
    """
    print("=" * 60)
    print("Interactive Mode - Model Judgment")
    print("=" * 60)

    # Create tools sandboxed to the directory
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    # Create Azure OpenAI client and agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions="""You are a skilled Python developer with access to these tools:
- write_file(path, content): Create or modify files
- read_file(path): Read file contents
- list_directory(path): List directory contents
- run_command(command): Execute shell commands like 'python', 'pytest', etc.
- task_complete(summary): Signal that you have finished the task

IMPORTANT: When asked to run tests, you MUST use run_command to actually execute them.
Do not just create files - verify they work by running them.

Workflow:
1. Create the required files using write_file
2. Run the code/tests using run_command (e.g., 'pytest test_file.py' or 'python -m unittest test_file')
3. Report results
4. Call task_complete with a summary when done

STYLE: Think out loud as you work. Briefly explain what you're about to do before doing it.
For example: "I'll create the calculator module first..." or "Now let me run the tests to verify..."

Be concise but communicative.""",
        name="coding_assistant",
        tools=all_tools,
    )

    # Create harness with stall detection
    harness = AgentHarness(
        agent,
        max_turns=10,
        enable_stall_detection=True,
        stall_threshold=3,
    )

    # The coding task
    task = """Create a Python file called 'calculator.py' that implements a simple calculator
with add, subtract, multiply, and divide functions. Then create a test file
'test_calculator.py' that tests all four operations. Finally, run the tests."""

    print(f"\nTask: {task}")
    print(f"Sandbox: {sandbox_dir}")
    print("-" * 60)

    # Run the harness and show agent activity
    async for event in harness.run_stream(task):
        # Show agent streaming updates
        if isinstance(event, AgentRunUpdateEvent):
            update = event.data
            # Print agent's text as it streams
            if hasattr(update, "text") and update.text:
                print(update.text, end="", flush=True)

        if isinstance(event, WorkflowOutputEvent):
            result = event.data
            if isinstance(result, HarnessResult):
                print(f"\n{'=' * 60}")
                print("HARNESS RESULT")
                print(f"{'=' * 60}")
                print(f"Status: {result.status.value}")
                print(f"Turns: {result.turn_count}")
                print(f"Stop reason: {result.reason.kind if result.reason else 'N/A'}")
                if result.reason and result.reason.message:
                    print(f"Message: {result.reason.message}")

    # Show what was created
    print(f"\n{'=' * 60}")
    print("FILES CREATED")
    print(f"{'=' * 60}")
    for f in sandbox_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(sandbox_dir)
            # Skip cache files
            if "__pycache__" in str(rel_path) or ".pytest_cache" in str(rel_path):
                continue
            if is_binary_file(f):
                print(f"  {rel_path} (binary)")
                continue
            print(f"  {rel_path}")
            # Show file contents
            print("    --- Contents ---")
            try:
                content = f.read_text(encoding="utf-8")
                for line in content.split("\n")[:10]:  # First 10 lines
                    print(f"    {line}")
                if len(content.split("\n")) > 10:
                    print(f"    ... ({len(content.split(chr(10)))} lines total)")
            except Exception as e:
                print(f"    (error reading: {e})")


async def run_automation_mode(sandbox_dir: Path) -> None:
    """Run the harness in automation mode (contract verification).

    This uses a formal TaskContract to verify the work is actually done.
    """
    print("\n" + "=" * 60)
    print("Automation Mode - Contract Verification")
    print("=" * 60)

    # Create tools sandboxed to the directory
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    # Create Azure OpenAI client and agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions="""You are a skilled Python developer with access to these tools:
- write_file(path, content): Create or modify files
- read_file(path): Read file contents
- list_directory(path): List directory contents
- run_command(command): Execute shell commands like 'python', 'pytest', etc.
- task_complete(summary): Signal that you have finished the task

IMPORTANT: When asked to run tests, you MUST use run_command to actually execute them.
Do not just create files - verify they work by running them.

Workflow:
1. Create the required files using write_file
2. Run the code/tests using run_command (e.g., 'pytest test_file.py' or 'python -m unittest test_file')
3. Report results
4. Call task_complete with a summary when done

STYLE: Think out loud as you work. Briefly explain what you're about to do before doing it.
For example: "I'll create the greeter module first..." or "Now let me run the tests to verify..."

Be concise but communicative.""",
        name="coding_assistant",
        tools=all_tools,
    )

    # Define the contract - what "done" means
    contract = TaskContract(
        goal="Create a greeting module with tests",
        required_outputs=[
            RequiredOutput(
                id="R1",
                description="greeter.py file exists",
                predicate=Predicate.file_exists(str(sandbox_dir / "greeter.py")),
            ),
            RequiredOutput(
                id="R2",
                description="test_greeter.py file exists",
                predicate=Predicate.file_exists(str(sandbox_dir / "test_greeter.py")),
            ),
            # Soft requirement - model judgment
            RequiredOutput(
                id="R3",
                description="Tests pass",
                predicate=Predicate.always_true("Verified by running tests"),
            ),
        ],
    )

    # Create harness with contract
    harness = AgentHarness(
        agent,
        max_turns=10,
        task_contract=contract,
        enable_stall_detection=True,
    )

    # The coding task
    task = """Create a Python module called 'greeter.py' with a function greet(name)
that returns 'Hello, {name}!'. Then create 'test_greeter.py' with tests.
Run the tests to verify everything works."""

    print(f"\nTask: {task}")
    print(f"Sandbox: {sandbox_dir}")
    print(f"Contract: {len(contract.required_outputs)} requirements")
    for req in contract.required_outputs:
        print(f"  - {req.id}: {req.description}")
    print("-" * 60)

    # Run the harness and show agent activity
    async for event in harness.run_stream(task):
        # Show agent streaming updates
        if isinstance(event, AgentRunUpdateEvent):
            update = event.data
            # Print agent's text as it streams
            if hasattr(update, "text") and update.text:
                print(update.text, end="", flush=True)

        if isinstance(event, WorkflowOutputEvent):
            result = event.data
            if isinstance(result, HarnessResult):
                print(f"\n{'=' * 60}")
                print("HARNESS RESULT")
                print(f"{'=' * 60}")
                print(f"Status: {result.status.value}")
                print(f"Turns: {result.turn_count}")
                print(f"Stop reason: {result.reason.kind if result.reason else 'N/A'}")
                if result.reason and result.reason.message:
                    print(f"Message: {result.reason.message}")

    # Show what was created
    print(f"\n{'=' * 60}")
    print("FILES CREATED")
    print(f"{'=' * 60}")
    for f in sandbox_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(sandbox_dir)
            # Skip cache and binary files
            if "__pycache__" in str(rel_path) or ".pytest_cache" in str(rel_path):
                continue
            if is_binary_file(f):
                print(f"  {rel_path} (binary)")
                continue
            print(f"  {rel_path}")
            # Show file contents
            print("    --- Contents ---")
            try:
                content = f.read_text(encoding="utf-8")
                for line in content.split("\n")[:10]:  # First 10 lines
                    print(f"    {line}")
                if len(content.split("\n")) > 10:
                    print(f"    ... ({len(content.split(chr(10)))} lines total)")
            except Exception as e:
                print(f"    (error reading: {e})")


async def main() -> None:
    """Run harness coding tests."""
    print("Agent Harness Coding Test")
    print("=" * 60)
    print("Testing the harness on real coding tasks with file I/O")
    print()

    # Create temporary sandbox directories
    with tempfile.TemporaryDirectory(prefix="harness_test_interactive_") as sandbox1:
        await run_interactive_mode(Path(sandbox1))

    with tempfile.TemporaryDirectory(prefix="harness_test_automation_") as sandbox2:
        await run_automation_mode(Path(sandbox2))

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
