# Copyright (c) Microsoft. All rights reserved.

"""Run the Agent Harness in DevUI.

This script demonstrates running a harness-wrapped agent in DevUI for
interactive testing and debugging. The harness provides:
- Turn limits and stall detection
- Continuation prompts to verify task completion
- Task contract verification (optional)
- Context pressure management (optional)

Usage:
    python devui_harness.py [--sandbox PATH] [--port PORT]
"""

import argparse
import logging
import tempfile
from pathlib import Path

from agent_framework.azure import AzureOpenAIChatClient
from agent_framework._harness import (
    AgentHarness,
    get_task_complete_tool,
)
from agent_framework_devui import serve
from azure.identity import AzureCliCredential

from coding_tools import CodingTools

# Enable debug logging for harness and mapper
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("agent_framework._harness").setLevel(logging.DEBUG)
logging.getLogger("agent_framework._workflows").setLevel(logging.DEBUG)
logging.getLogger("agent_framework_devui._mapper").setLevel(logging.DEBUG)
logging.getLogger("agent_framework_devui._executor").setLevel(logging.DEBUG)
# Also log raw events from the harness
logging.getLogger("agent_framework_devui._server").setLevel(logging.DEBUG)


AGENT_INSTRUCTIONS = """You are a capable AI assistant with access to a local workspace.

AVAILABLE TOOLS:
- write_file(path, content): Create or modify files
- read_file(path): Read file contents
- list_directory(path): List directory contents (use "." for current directory)
- run_command(command): Execute shell commands (python, pip, git, etc.)
- create_directory(path): Create directories
- task_complete(summary): Signal that you have finished the task

GUIDELINES:
1. Think step-by-step and explain your reasoning briefly before taking actions
2. When you complete a task, verify your work (e.g., read files you created, run code you wrote)
3. If something fails, diagnose the issue and try to fix it
4. Be proactive - if you see a better approach, suggest it
5. When you have finished all requested work, call task_complete with a brief summary

STYLE:
- Be concise but informative
- Explain what you're about to do before doing it
- Report results clearly after each action
"""


def create_harness_agent(sandbox_dir: Path) -> AgentHarness:
    """Create a harness-wrapped agent with coding tools.

    Returns an AgentHarness that can be registered with DevUI.
    The AgentHarness has a run_stream(message) method that DevUI
    can call directly with the user's input.
    """
    # Create tools sandboxed to the directory
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    # Create the underlying agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        name="coding-assistant",
        description="A coding assistant that can read/write files and run commands",
        instructions=AGENT_INSTRUCTIONS,
        tools=all_tools,
    )

    # Wrap in harness with all features enabled
    harness = AgentHarness(
        agent,
        max_turns=20,
        enable_stall_detection=True,
        stall_threshold=3,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
    )

    # Add id, name and description for DevUI discovery
    # DevUI requires both id and name for agent-like entities
    harness.id = "coding-harness"
    harness.name = "coding-harness"
    harness.description = (
        f"Coding assistant with harness infrastructure. "
        f"Sandbox: {sandbox_dir}"
    )

    return harness


def main():
    parser = argparse.ArgumentParser(description="Run coding agent harness in DevUI")
    parser.add_argument(
        "--sandbox",
        type=Path,
        default=None,
        help="Directory for agent workspace (default: temp directory)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for DevUI server (default: 8080)",
    )
    args = parser.parse_args()

    # Determine sandbox directory
    if args.sandbox:
        sandbox_dir = args.sandbox.resolve()
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using sandbox: {sandbox_dir}")
    else:
        # Use temporary directory
        temp_dir = tempfile.mkdtemp(prefix="harness_devui_")
        sandbox_dir = Path(temp_dir)
        print(f"Using temp sandbox: {sandbox_dir}")

    # Create the harness-wrapped agent
    harness = create_harness_agent(sandbox_dir)

    print(f"\nStarting DevUI on port {args.port}...")
    print(f"Sandbox directory: {sandbox_dir}")
    print(f"Harness config: max_turns=20, stall_threshold=3")

    # Launch DevUI with the harness
    # Note: AgentHarness has run_stream(message) method that DevUI can call directly
    serve(
        entities=[harness],
        port=args.port,
        auto_open=True,
        mode="developer",
    )


if __name__ == "__main__":
    main()
