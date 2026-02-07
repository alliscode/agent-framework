# Copyright (c) Microsoft. All rights reserved.

"""Sub-agent definitions and factory for harness-level agent delegation."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .._agents import ChatAgent
from .._tools import AIFunction

if TYPE_CHECKING:
    from .._clients import ChatClientProtocol
    from .._tools import ToolProtocol


def create_explore_tool(
    chat_client: "ChatClientProtocol",
    tools: "Sequence[ToolProtocol | Callable[..., Any]]",
) -> "AIFunction[Any, str]":
    """Create an explore sub-agent tool for fast codebase Q&A.

    Args:
        chat_client: Chat client (ideally a fast/cheap model like gpt-4o-mini).
        tools: Read-only tools available to the explore agent (read_file, list_directory, run_command).

    Returns:
        An AIFunction tool that the main agent can call.
    """
    explore = ChatAgent(
        chat_client=chat_client,
        name="explore",
        description="Fast codebase exploration agent for answering questions about code",
        instructions=(
            "You are an exploration agent specialized in rapid codebase analysis.\n"
            "CRITICAL: Keep your answer under 300 words.\n"
            "CRITICAL: MAXIMIZE PARALLEL TOOL CALLING — make multiple independent "
            "tool calls in a single response.\n"
            "Aim to answer questions in 1-3 tool calls when possible.\n"
            "Return a focused, factual answer. No preamble, no hedging."
        ),
        tools=list(tools),
    )
    return explore.as_tool(
        name="explore",
        description=(
            "Launch a fast exploration agent to answer codebase questions. "
            "Returns focused answers under 300 words. Safe to call in parallel. "
            "Use for: finding files, understanding code structure, answering questions. "
            "Do NOT use for: making changes, running builds, complex multi-step work."
        ),
        arg_name="prompt",
        arg_description="The question or exploration task for the agent.",
    )


def create_task_tool(
    chat_client: "ChatClientProtocol",
    tools: "Sequence[ToolProtocol | Callable[..., Any]]",
) -> "AIFunction[Any, str]":
    """Create a task sub-agent tool for command execution.

    Args:
        chat_client: Chat client (ideally a fast/cheap model).
        tools: Tools available to the task agent (typically all tools).

    Returns:
        An AIFunction tool that the main agent can call.
    """
    task_agent = ChatAgent(
        chat_client=chat_client,
        name="task",
        description="Command execution agent for builds, tests, and linting",
        instructions=(
            "You are a command execution agent that runs development commands.\n"
            "On SUCCESS: Return a brief one-line summary (e.g., 'All 247 tests passed').\n"
            "On FAILURE: Return the full error output for debugging.\n"
            "Do NOT attempt to fix errors or make suggestions — just execute and report."
        ),
        tools=list(tools),
    )
    return task_agent.as_tool(
        name="run_task",
        description=(
            "Launch a task agent to execute commands (builds, tests, linting). "
            "Returns brief success summary or verbose error output. "
            "Use for: running tests, building code, checking linting. "
            "Do NOT use for: exploring code or making changes."
        ),
        arg_name="prompt",
        arg_description="The command or task to execute.",
    )


def create_document_tool(
    chat_client: "ChatClientProtocol",
    tools: "Sequence[ToolProtocol | Callable[..., Any]]",
) -> "AIFunction[Any, str]":
    """Create a documentation sub-agent tool for producing thorough technical documents.

    Args:
        chat_client: Chat client for the documentation agent.
        tools: Tools available to the doc agent (read_file, list_directory, write_file, run_command).

    Returns:
        An AIFunction tool that the main agent can call.
    """
    doc_agent = ChatAgent(
        chat_client=chat_client,
        name="document",
        description="Documentation agent specialized in producing comprehensive technical documents",
        instructions=(
            "You are a technical documentation specialist. Your job is to produce\n"
            "thorough, high-quality technical documents about software systems.\n\n"
            "APPROACH:\n"
            "1. Read EVERY source file relevant to the topic — do not skip files or guess.\n"
            "2. Go deep on the most important components to answer the request thoroughly.\n"
            "3. Reference specific class names, method signatures, and module paths.\n"
            "4. Include source code examples where they help communicate concepts and principles.\n"
            "5. Include ASCII diagrams for architecture, data flow, or component relationships\n"
            "   where they help the reader understand the system.\n"
            "6. Organize with clear sections, tables, and hierarchical structure.\n\n"
            "QUALITY STANDARDS:\n"
            "- Every claim must be backed by something you read in a source file.\n"
            "- Do not summarize from directory listings — read the actual code.\n"
            "- Prioritize depth over breadth: deeply explain the most important parts\n"
            "  rather than shallowly listing everything.\n"
            "- Your document should give a reader genuine understanding of how the\n"
            "  system works, not just a surface inventory of its parts.\n"
            "- Include constructor signatures, key method signatures, and important\n"
            "  type definitions when documenting classes.\n\n"
            "OUTPUT:\n"
            "- Use write_file to save your document to the path specified in the prompt.\n"
            "- Return a brief summary of what you produced and its location."
        ),
        tools=list(tools),
    )
    return doc_agent.as_tool(
        name="document",
        description=(
            "Launch a documentation agent to produce a comprehensive technical document. "
            "The agent reads source files itself in a fresh context window and writes "
            "a thorough deliverable. Provide: the topic/request, which files or directories "
            "to focus on, and the output file path. "
            "Use for: creating architectural designs, API docs, code analysis reports. "
            "Do NOT use for: quick questions (use explore) or running commands (use run_task)."
        ),
        arg_name="prompt",
        arg_description="The documentation request including topic, scope, target files/directories, and output file path.",
    )
