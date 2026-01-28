# Copyright (c) Microsoft. All rights reserved.

"""Part 2: A Minimal Agent Harness using Workflows.

This module demonstrates the core concept of an agent harness:
a workflow that wraps agent execution with turn limits and
termination detection.

The harness consists of two executors in a loop:
1. AgentTurnExecutor - Runs one turn of the agent (LLM call + tool execution)
2. StopDecisionExecutor - Decides whether to continue or stop

This is the simplest useful harness: it prevents infinite loops and
detects when the agent has finished its task.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_framework import AgentProtocol, ChatMessage
from agent_framework._workflows import (
    Executor,
    Workflow,
    WorkflowContext,
    WorkflowEvent,
)

# ============================================================
# Shared State Keys
# ============================================================
# Executors communicate through shared state in the WorkflowContext.
# These keys define the "contract" between executors.

TURN_COUNT_KEY = "harness:turn_count"
MAX_TURNS_KEY = "harness:max_turns"
AGENT_MESSAGES_KEY = "harness:messages"
AGENT_DONE_KEY = "harness:agent_done"


# ============================================================
# Result Types
# ============================================================


class HarnessStatus(Enum):
    """Final status of a harness run."""

    COMPLETE = "complete"  # Agent finished its task
    MAX_TURNS = "max_turns"  # Hit turn limit
    ERROR = "error"  # Something went wrong


@dataclass
class HarnessResult:
    """The final output of a harness run."""

    status: HarnessStatus
    turn_count: int
    final_response: str = ""
    error: str | None = None


# ============================================================
# Trigger Types
# ============================================================
# Triggers are messages passed between executors to control flow.


@dataclass
class StartTurn:
    """Trigger to start an agent turn."""

    pass


@dataclass
class TurnComplete:
    """Result of an agent turn, passed to StopDecisionExecutor."""

    response_text: str
    has_tool_calls: bool


# ============================================================
# Agent Turn Executor
# ============================================================


class AgentTurnExecutor(Executor):
    """Executes one turn of agent interaction.

    A "turn" is: send messages to LLM → get response → execute any tools.

    This executor:
    1. Reads the conversation history from shared state
    2. Calls the agent's run method
    3. Stores the response back in shared state
    4. Yields TurnComplete for the stop decision executor
    """

    def __init__(self, agent: AgentProtocol):
        super().__init__()
        self._agent = agent
        self._thread = agent.get_new_thread()

    async def execute(
        self, trigger: StartTurn, ctx: WorkflowContext[Any]
    ) -> TurnComplete:
        """Run one agent turn."""
        # Get current conversation from shared state
        messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)

        # Increment turn count
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        turn_count += 1
        await ctx.set_shared_state(TURN_COUNT_KEY, turn_count)

        # Call the agent
        response = await self._agent.run(messages, thread=self._thread)

        # Extract response details
        response_text = response.text or ""
        has_tool_calls = bool(response.tool_calls)

        # Add assistant response to conversation history
        messages.append(ChatMessage(role="assistant", text=response_text))
        await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)

        # Emit an event so external observers can track progress
        await ctx.add_event(
            WorkflowEvent(data={"turn": turn_count, "response": response_text[:200]})
        )

        return TurnComplete(response_text=response_text, has_tool_calls=has_tool_calls)


# ============================================================
# Stop Decision Executor
# ============================================================


class StopDecisionExecutor(Executor):
    """Decides whether to continue or stop the harness.

    This executor checks two conditions:
    1. Has the agent finished? (no more tool calls = done)
    2. Have we hit the turn limit?

    If continuing, it yields StartTurn to loop back.
    If stopping, it yields HarnessResult as the final output.
    """

    async def execute(
        self, trigger: TurnComplete, ctx: WorkflowContext[Any]
    ) -> StartTurn | HarnessResult:
        """Decide whether to continue or stop."""
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        max_turns = await ctx.get_shared_state(MAX_TURNS_KEY)

        # Check if agent is done (no tool calls means it's finished responding)
        agent_done = not trigger.has_tool_calls

        if agent_done:
            # Agent completed its task
            await ctx.yield_output(
                HarnessResult(
                    status=HarnessStatus.COMPLETE,
                    turn_count=turn_count,
                    final_response=trigger.response_text,
                )
            )
            # Return None to end the workflow (no next executor)
            return None  # type: ignore[return-value]

        if turn_count >= max_turns:
            # Hit turn limit - stop to prevent runaway execution
            await ctx.yield_output(
                HarnessResult(
                    status=HarnessStatus.MAX_TURNS,
                    turn_count=turn_count,
                    final_response=trigger.response_text,
                )
            )
            return None  # type: ignore[return-value]

        # Continue to next turn
        return StartTurn()


# ============================================================
# Harness Builder
# ============================================================


def build_harness_workflow(agent: AgentProtocol, max_turns: int = 10) -> Workflow:
    """Build a minimal harness workflow.

    The workflow structure:
        StartTurn → AgentTurnExecutor → TurnComplete → StopDecisionExecutor → StartTurn (loop)
                                                                            ↘ HarnessResult (end)

    Args:
        agent: The agent to wrap in the harness.
        max_turns: Maximum turns before stopping (default: 10).

    Returns:
        A configured Workflow ready to run.
    """
    # Create executors
    agent_executor = AgentTurnExecutor(agent)
    stop_executor = StopDecisionExecutor()

    # Build workflow with routing:
    # StartTurn → AgentTurnExecutor → TurnComplete → StopDecisionExecutor → ...
    return (
        Workflow(name="minimal-harness")
        .add_executor(agent_executor, trigger_type=StartTurn)
        .add_executor(stop_executor, trigger_type=TurnComplete)
    )


def get_initial_state(task: str, max_turns: int = 10) -> dict[str, Any]:
    """Create the initial shared state for a harness run.

    Args:
        task: The user's task/request.
        max_turns: Maximum turns allowed.

    Returns:
        Dictionary of initial shared state values.
    """
    return {
        TURN_COUNT_KEY: 0,
        MAX_TURNS_KEY: max_turns,
        AGENT_MESSAGES_KEY: [ChatMessage(role="user", text=task)],
        AGENT_DONE_KEY: False,
    }


# ============================================================
# Simple Harness Class
# ============================================================


class SimpleHarness:
    """A minimal agent harness with turn limits.

    This class provides a clean interface over the workflow,
    handling setup and providing a simple run() method.

    Example:
        harness = SimpleHarness(agent, max_turns=10)
        result = await harness.run("Write a hello world program")
        print(f"Status: {result.status}, Turns: {result.turn_count}")
    """

    def __init__(self, agent: AgentProtocol, max_turns: int = 10):
        """Initialize the harness.

        Args:
            agent: The agent to wrap.
            max_turns: Maximum turns before stopping (default: 10).
        """
        self._agent = agent
        self._max_turns = max_turns
        self._workflow = build_harness_workflow(agent, max_turns)

    async def run(self, task: str) -> HarnessResult:
        """Run the agent with harness protections.

        Args:
            task: The task for the agent to perform.

        Returns:
            HarnessResult with status and final response.
        """
        initial_state = get_initial_state(task, self._max_turns)

        # Start the workflow with a StartTurn trigger
        result: HarnessResult | None = None

        async for event in self._workflow.run_stream(StartTurn(), **initial_state):
            # Capture the final result when yielded
            if isinstance(event, WorkflowEvent) and isinstance(event.data, HarnessResult):
                result = event.data

        if result is None:
            # Should not happen, but handle gracefully
            result = HarnessResult(
                status=HarnessStatus.ERROR,
                turn_count=0,
                error="No result produced by workflow",
            )

        return result

    async def run_stream(self, task: str):
        """Run the agent and stream events.

        Yields workflow events as they occur, allowing real-time
        observation of the harness execution.

        Args:
            task: The task for the agent to perform.

        Yields:
            WorkflowEvent objects as execution progresses.
        """
        initial_state = get_initial_state(task, self._max_turns)

        async for event in self._workflow.run_stream(StartTurn(), **initial_state):
            yield event
