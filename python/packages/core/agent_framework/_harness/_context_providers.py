# Copyright (c) Microsoft. All rights reserved.

"""Context providers for harness-level instruction injection.

These providers use the existing ContextProvider protocol to inject
environment context and harness guidance into the system prompt,
rather than as user messages that are vulnerable to compaction.
"""

import os
import platform
from collections.abc import MutableSequence
from typing import Any

from .._memory import Context, ContextProvider
from .._types import ChatMessage


class EnvironmentContextProvider(ContextProvider):
    """Injects environment context (cwd, OS, directory listing) into system prompt.

    This provider automatically detects the working directory, operating system,
    and top-level directory contents, then injects them as system-level instructions
    so the agent has environmental awareness without guessing.

    Args:
        sandbox_path: Override for the working directory. If None, uses os.getcwd().
        max_entries: Maximum number of directory entries to include. Default is 50.
    """

    def __init__(self, sandbox_path: str | None = None, max_entries: int = 50):
        """Initialize the EnvironmentContextProvider.

        Args:
            sandbox_path: Override for the working directory. If None, uses os.getcwd().
            max_entries: Maximum number of directory entries to include.
        """
        self._sandbox = sandbox_path
        self._max_entries = max_entries

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        """Inject environment context into the system prompt.

        Args:
            messages: The messages being sent to the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Context with environment information as instructions.
        """
        cwd = self._sandbox or os.getcwd()
        os_name = platform.system()
        try:
            entries = sorted(os.listdir(cwd))[: self._max_entries]
            listing = "\n".join(f"  {e}" for e in entries)
        except OSError:
            listing = "  (unable to list directory)"

        instructions = (
            "<environment_context>\n"
            f"Working directory: {cwd}\n"
            f"Operating system: {os_name}\n"
            f"Directory contents (top-level):\n{listing}\n"
            "</environment_context>"
        )
        return Context(instructions=instructions)


class HarnessGuidanceProvider(ContextProvider):
    """Injects tool strategy, work item, planning guidance, and task completion instructions into system prompt.

    This provider consolidates guidance that was previously injected as user messages
    into the system prompt where it carries more weight and is protected from compaction.

    Args:
        enable_work_items: Whether to include work item and planning guidance.
        enable_tool_guidance: Whether to include tool strategy guidance.
        enable_sub_agents: Whether to include sub-agent delegation guidance.
    """

    TASK_COMPLETION_INSTRUCTIONS = (
        "<task_completion>\n"
        "* A task is not complete until the expected outcome is verified and persistent\n"
        "* After making changes, validate they work correctly\n"
        "* If an initial approach fails, try alternative tools or methods before concluding\n"
        "* You MUST call task_complete when finished — do not just stop responding\n"
        "* Only call task_complete after all work items are done and deliverables are written\n"
        "</task_completion>"
    )

    SUB_AGENT_GUIDANCE = (
        "<sub_agents>\n"
        "You have access to specialized sub-agents:\n"
        "- explore(prompt): Fast codebase Q&A (cheap model, <300 word answers, parallel-safe)\n"
        "- run_task(prompt): Execute commands — builds, tests, linting (brief success, verbose failure)\n"
        "- document(prompt): Produce comprehensive technical documents (reads files in fresh context)\n\n"
        "Use explore proactively for codebase questions before making changes.\n"
        "Use run_task for builds/tests where you only need success/failure status.\n"
        "Use document when you need to produce a detailed deliverable. IMPORTANT: do your\n"
        "own thorough exploration FIRST so you can give the document agent a detailed brief\n"
        "listing the specific files and directories to focus on, key classes/concepts you\n"
        "discovered, and the output file path. The more context you provide in the prompt,\n"
        "the better the document will be. Do NOT delegate to document before you understand\n"
        "the codebase yourself.\n"
        "</sub_agents>"
    )

    def __init__(
        self,
        enable_work_items: bool = False,
        enable_tool_guidance: bool = True,
        enable_sub_agents: bool = False,
    ):
        """Initialize the HarnessGuidanceProvider.

        Args:
            enable_work_items: Whether to include work item and planning guidance.
            enable_tool_guidance: Whether to include tool strategy guidance.
            enable_sub_agents: Whether to include sub-agent delegation guidance.
        """
        self._enable_work_items = enable_work_items
        self._enable_tool_guidance = enable_tool_guidance
        self._enable_sub_agents = enable_sub_agents

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        """Inject harness guidance into the system prompt.

        Args:
            messages: The messages being sent to the model.
            **kwargs: Additional keyword arguments.

        Returns:
            Context with guidance as instructions.
        """
        from ._agent_turn_executor import AgentTurnExecutor

        sections: list[str] = []
        if self._enable_tool_guidance:
            sections.append(AgentTurnExecutor.TOOL_STRATEGY_GUIDANCE)
        if self._enable_work_items:
            sections.append(AgentTurnExecutor.WORK_ITEM_GUIDANCE)
            sections.append(AgentTurnExecutor.PLANNING_PROMPT)
        if self._enable_sub_agents:
            sections.append(self.SUB_AGENT_GUIDANCE)

        # Task completion instructions are always included
        sections.append(self.TASK_COMPLETION_INSTRUCTIONS)

        return Context(instructions="\n\n".join(sections))
