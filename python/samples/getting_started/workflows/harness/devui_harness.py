# Copyright (c) Microsoft. All rights reserved.

"""Run the Agent Harness in DevUI.

This script demonstrates running a harness-wrapped agent in DevUI for
interactive testing and debugging. The harness provides:
- Turn limits and stall detection
- Continuation prompts to verify task completion
- Task contract verification (optional)
- Context compaction (optional) - production-quality context management
- Work item tracking (optional) - self-critique loop for multi-step tasks
- MCP tool integration (optional) - connect to MCP servers for additional tools

Usage:
    python devui_harness.py [--sandbox PATH] [--port PORT] [--compaction] [--work-items] [--mcp NAME COMMAND ARGS...]

Examples:
    # Basic usage
    python devui_harness.py

    # With context compaction enabled
    python devui_harness.py --compaction

    # With work item tracking (self-critique loop)
    python devui_harness.py --work-items

    # With both compaction and work items
    python devui_harness.py --compaction --work-items

    # With an MCP stdio server
    python devui_harness.py --mcp compose compose mcp /path/to/project.json

    # With multiple MCP servers
    python devui_harness.py --mcp filesystem npx -y @modelcontextprotocol/server-filesystem /tmp --mcp compose compose mcp /path/to/project.json
"""

import argparse
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from agent_framework import MCPStdioTool
from agent_framework._harness import (
    AgentHarness,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    MarkdownRenderer,
    get_task_complete_tool,
    render_stream,
)
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework_devui import serve
from azure.identity import AzureCliCredential
from coding_tools import CodingTools

# Enable debug logging for harness and mapper
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


class RenderedHarness:
    """Wrapper that applies MarkdownRenderer to harness run_stream output.

    This wraps an AgentHarness and intercepts run_stream() calls to apply
    the MarkdownRenderer, which formats progress indicators and deliverables
    as markdown for DevUI display.
    """

    def __init__(self, harness: AgentHarness, use_renderer: bool = True):
        self._harness = harness
        self._renderer = MarkdownRenderer() if use_renderer else None
        # Copy attributes DevUI needs for discovery
        self.id = harness.id
        self.name = harness.name
        self.description = harness.description

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying harness."""
        return getattr(self._harness, name)

    async def run_stream(self, message: str, **kwargs: Any):
        """Run the harness with optional markdown rendering."""
        if self._renderer:
            async for event in render_stream(self._harness, message, self._renderer, **kwargs):
                yield event
        else:
            async for event in self._harness.run_stream(message, **kwargs):
                yield event


def create_harness_agent(
    sandbox_dir: Path,
    enable_compaction: bool = False,
    enable_work_items: bool = False,
    mcp_tools: list[MCPStdioTool] | None = None,
) -> AgentHarness:
    """Create a harness-wrapped agent with coding tools.

    Returns an AgentHarness that can be registered with DevUI.
    The AgentHarness has a run_stream(message) method that DevUI
    can call directly with the user's input.

    Args:
        sandbox_dir: Directory for agent workspace.
        enable_compaction: Whether to enable production context compaction.
        enable_work_items: Whether to enable work item tracking (self-critique loop).
        mcp_tools: Optional list of connected MCP tools to add to the agent.
    """
    # Create tools sandboxed to the directory
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    # Add MCP tool functions if provided
    if mcp_tools:
        for mcp_tool in mcp_tools:
            all_tools.extend(mcp_tool.functions)
            print(f"Added {len(mcp_tool.functions)} tools from MCP server '{mcp_tool.name}'")

    # Create the underlying agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        name="coding-assistant",
        description="A coding assistant that can read/write files and run commands",
        instructions=AGENT_INSTRUCTIONS,
        tools=all_tools,
    )

    # Configure compaction stores if enabled
    compaction_kwargs: dict[str, Any] = {}
    if enable_compaction:
        compaction_kwargs = {
            "enable_compaction": True,
            "compaction_store": InMemoryCompactionStore(),
            "artifact_store": InMemoryArtifactStore(),
            "summary_cache": InMemorySummaryCache(max_entries=100),
            "max_input_tokens": 100_000,
            "soft_threshold_percent": 0.85,
        }

    # Wrap in harness with all features enabled
    harness = AgentHarness(
        agent,
        max_turns=20,
        enable_stall_detection=True,
        stall_threshold=3,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        enable_work_items=enable_work_items,
        **compaction_kwargs,
    )

    # Add id, name and description for DevUI discovery
    # DevUI requires both id and name for agent-like entities
    harness.id = "coding-harness"
    harness.name = "coding-harness"
    compaction_status = "enabled" if enable_compaction else "disabled"
    work_items_status = "enabled" if enable_work_items else "disabled"
    harness.description = (
        f"Coding assistant with harness infrastructure. "
        f"Sandbox: {sandbox_dir}. Compaction: {compaction_status}. "
        f"Work items: {work_items_status}"
    )

    return harness


def _connect_mcp_tools_sync(mcp_tools: list[MCPStdioTool]) -> list[MCPStdioTool]:
    """Connect MCP tools synchronously before starting the event loop.

    This runs the async connect() in an event loop, keeping the connections
    alive for use with DevUI's event loop.

    Args:
        mcp_tools: List of MCP tools to connect.

    Returns:
        List of successfully connected MCP tools.
    """
    connected: list[MCPStdioTool] = []

    async def connect_all():
        for mcp_tool in mcp_tools:
            try:
                await mcp_tool.connect()
                connected.append(mcp_tool)
                print(f"Connected to MCP server '{mcp_tool.name}' - {len(mcp_tool.functions)} tools available")
            except Exception as e:
                print(f"Error connecting to MCP server '{mcp_tool.name}': {e}")

    # Get or create event loop and run connection
    # Note: We don't close the loop to keep MCP connections alive
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(connect_all())

    return connected


def main():
    """Entry point for the DevUI harness."""
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
    parser.add_argument(
        "--compaction",
        action="store_true",
        help="Enable production context compaction (Phase 9)",
    )
    parser.add_argument(
        "--work-items",
        action="store_true",
        help="Enable work item tracking (self-critique loop)",
    )
    parser.add_argument(
        "--mcp",
        action="append",
        nargs="+",
        metavar=("NAME", "COMMAND"),
        help="Add MCP stdio server: --mcp NAME COMMAND [ARGS...] (can be repeated)",
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

    # Parse and create MCP tools
    mcp_tools: list[MCPStdioTool] = []
    if args.mcp:
        for mcp_args in args.mcp:
            if len(mcp_args) < 2:
                print(f"Error: --mcp requires at least NAME and COMMAND, got: {mcp_args}")
                return
            name = mcp_args[0]
            command = mcp_args[1]
            command_args = mcp_args[2:] if len(mcp_args) > 2 else []
            mcp_tool = MCPStdioTool(
                name=name,
                command=command,
                args=command_args,
                description=f"MCP server: {name}",
                approval_mode="never_require",
            )
            mcp_tools.append(mcp_tool)
            print(f"Configured MCP server '{name}': {command} {' '.join(command_args)}")

    # Connect MCP tools before starting DevUI
    connected_tools: list[MCPStdioTool] = []
    if mcp_tools:
        connected_tools = _connect_mcp_tools_sync(mcp_tools)

    # Create the harness-wrapped agent with MCP tools
    harness = create_harness_agent(
        sandbox_dir,
        enable_compaction=args.compaction,
        enable_work_items=args.work_items,
        mcp_tools=connected_tools if connected_tools else None,
    )

    # Wrap with MarkdownRenderer when work items are enabled
    if args.work_items:
        harness = RenderedHarness(harness, use_renderer=True)

    print(f"\nStarting DevUI on port {args.port}...")
    print(f"Sandbox directory: {sandbox_dir}")
    print("Harness config: max_turns=20, stall_threshold=3")
    if args.compaction:
        print("Context compaction: ENABLED (100K tokens, 85% threshold)")
    else:
        print("Context compaction: disabled")
    if args.work_items:
        print("Work item tracking: ENABLED (self-critique loop)")
        print("Markdown renderer: ENABLED (activity verbs, progress bars)")
    else:
        print("Work item tracking: disabled")
    if connected_tools:
        total_mcp_tools = sum(len(t.functions) for t in connected_tools)
        print(f"MCP servers: {len(connected_tools)} connected ({total_mcp_tools} tools)")

    # Launch DevUI with the harness
    serve(
        entities=[harness],
        port=args.port,
        auto_open=True,
        mode="developer",
    )


if __name__ == "__main__":
    main()
