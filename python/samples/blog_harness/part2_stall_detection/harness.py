# Copyright (c) Microsoft. All rights reserved.

"""Part 3: Agent Harness with Stall Detection.

Building on Part 2, this module adds stall detection and recovery:
- Response fingerprinting to detect repetitive behavior
- Stall counting when fingerprints repeat
- Continuation prompt injection to break stalls

The Problem:
Agents sometimes get "stuck" - they say "I'll do X" repeatedly without
actually doing X, or they loop through the same reasoning. Without
detection, this wastes tokens and time.

The Solution:
1. Fingerprint each response (hash of key characteristics)
2. Track recent fingerprints in a sliding window
3. When fingerprints repeat, increment a stall counter
4. After threshold stalls, inject a continuation prompt
5. If stalls persist, stop the harness
"""

import hashlib
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

TURN_COUNT_KEY = "harness:turn_count"
MAX_TURNS_KEY = "harness:max_turns"
AGENT_MESSAGES_KEY = "harness:messages"

# New keys for stall detection
STALL_THRESHOLD_KEY = "harness:stall_threshold"
STALL_COUNT_KEY = "harness:stall_count"
RECENT_FINGERPRINTS_KEY = "harness:recent_fingerprints"
CONTINUATION_INJECTED_KEY = "harness:continuation_injected"


# ============================================================
# Result Types
# ============================================================


class HarnessStatus(Enum):
    """Final status of a harness run."""

    COMPLETE = "complete"  # Agent finished its task
    MAX_TURNS = "max_turns"  # Hit turn limit
    STALLED = "stalled"  # Agent stalled and couldn't recover
    ERROR = "error"  # Something went wrong


@dataclass
class HarnessResult:
    """The final output of a harness run."""

    status: HarnessStatus
    turn_count: int
    final_response: str = ""
    stall_count: int = 0
    error: str | None = None


# ============================================================
# Trigger Types
# ============================================================


@dataclass
class StartTurn:
    """Trigger to start an agent turn."""

    pass


@dataclass
class TurnComplete:
    """Result of an agent turn."""

    response_text: str
    has_tool_calls: bool
    fingerprint: str  # New: hash of response characteristics


# ============================================================
# Fingerprinting
# ============================================================


def compute_fingerprint(response_text: str, has_tool_calls: bool) -> str:
    """Compute a fingerprint for stall detection.

    The fingerprint captures the "shape" of a response, not exact content.
    This helps detect when the agent is stuck in a loop even if the
    exact wording varies slightly.

    We hash:
    - First 200 chars (captures intent/structure)
    - Last 100 chars (captures conclusion)
    - Whether tools were called
    - Response length bucket (short/medium/long)

    Args:
        response_text: The agent's response text.
        has_tool_calls: Whether the response included tool calls.

    Returns:
        A hex string fingerprint.
    """
    # Normalize whitespace
    normalized = " ".join(response_text.split())

    # Extract features
    prefix = normalized[:200] if len(normalized) > 200 else normalized
    suffix = normalized[-100:] if len(normalized) > 100 else ""
    length_bucket = "short" if len(normalized) < 100 else "medium" if len(normalized) < 500 else "long"

    # Combine into fingerprint input
    fingerprint_input = f"{prefix}|{suffix}|{has_tool_calls}|{length_bucket}"

    # Hash it
    return hashlib.md5(fingerprint_input.encode()).hexdigest()[:12]


# ============================================================
# Agent Turn Executor
# ============================================================


class AgentTurnExecutor(Executor):
    """Executes one turn of agent interaction.

    Enhanced from Part 2 to compute response fingerprints for stall detection.
    """

    def __init__(self, agent: AgentProtocol):
        super().__init__()
        self._agent = agent
        self._thread = agent.get_new_thread()

    async def execute(self, trigger: StartTurn, ctx: WorkflowContext[Any]) -> TurnComplete:
        """Run one agent turn and compute fingerprint."""
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

        # Compute fingerprint for stall detection
        fingerprint = compute_fingerprint(response_text, has_tool_calls)

        # Add assistant response to conversation history
        messages.append(ChatMessage(role="assistant", text=response_text))
        await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)

        # Emit progress event
        await ctx.add_event(
            WorkflowEvent(
                data={
                    "turn": turn_count,
                    "response": response_text[:200],
                    "fingerprint": fingerprint,
                    "has_tools": has_tool_calls,
                }
            )
        )

        return TurnComplete(
            response_text=response_text,
            has_tool_calls=has_tool_calls,
            fingerprint=fingerprint,
        )


# ============================================================
# Stop Decision Executor (with Stall Detection)
# ============================================================


class StopDecisionExecutor(Executor):
    """Decides whether to continue, inject a prompt, or stop.

    Enhanced from Part 2 with stall detection:
    1. Track fingerprints in a sliding window
    2. Detect repeated fingerprints as stalls
    3. Inject continuation prompts to break stalls
    4. Stop if stalls persist beyond threshold
    """

    # The continuation prompt injected when a stall is detected
    CONTINUATION_PROMPT = (
        "You seem to be repeating yourself or stuck in a loop. "
        "Please take a different approach or make concrete progress. "
        "If you've completed the task, say so clearly. "
        "If you're blocked, explain what's preventing progress."
    )

    async def execute(self, trigger: TurnComplete, ctx: WorkflowContext[Any]) -> StartTurn | HarnessResult:
        """Decide next action with stall detection."""
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        max_turns = await ctx.get_shared_state(MAX_TURNS_KEY)
        stall_threshold = await ctx.get_shared_state(STALL_THRESHOLD_KEY)
        stall_count = await ctx.get_shared_state(STALL_COUNT_KEY)
        recent_fingerprints: list[str] = await ctx.get_shared_state(RECENT_FINGERPRINTS_KEY)

        # --- Check for stall ---
        is_stall = trigger.fingerprint in recent_fingerprints

        if is_stall:
            stall_count += 1
            await ctx.set_shared_state(STALL_COUNT_KEY, stall_count)

            # Emit stall event for observability
            await ctx.add_event(
                WorkflowEvent(
                    data={
                        "event": "stall_detected",
                        "stall_count": stall_count,
                        "fingerprint": trigger.fingerprint,
                    }
                )
            )

            # Check if we've exceeded stall threshold
            if stall_count >= stall_threshold:
                await ctx.yield_output(
                    HarnessResult(
                        status=HarnessStatus.STALLED,
                        turn_count=turn_count,
                        final_response=trigger.response_text,
                        stall_count=stall_count,
                    )
                )
                return None  # type: ignore[return-value]

            # Inject continuation prompt to break the stall
            messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)
            messages.append(ChatMessage(role="user", text=self.CONTINUATION_PROMPT))
            await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)
            await ctx.set_shared_state(CONTINUATION_INJECTED_KEY, True)

            # Continue to give agent a chance to recover
            return StartTurn()

        # --- Not a stall: update fingerprint window ---
        recent_fingerprints.append(trigger.fingerprint)
        # Keep only last 5 fingerprints (sliding window)
        if len(recent_fingerprints) > 5:
            recent_fingerprints = recent_fingerprints[-5:]
        await ctx.set_shared_state(RECENT_FINGERPRINTS_KEY, recent_fingerprints)

        # Reset stall count on progress
        if stall_count > 0:
            await ctx.set_shared_state(STALL_COUNT_KEY, 0)

        # --- Check completion conditions ---
        agent_done = not trigger.has_tool_calls

        if agent_done:
            await ctx.yield_output(
                HarnessResult(
                    status=HarnessStatus.COMPLETE,
                    turn_count=turn_count,
                    final_response=trigger.response_text,
                    stall_count=stall_count,
                )
            )
            return None  # type: ignore[return-value]

        if turn_count >= max_turns:
            await ctx.yield_output(
                HarnessResult(
                    status=HarnessStatus.MAX_TURNS,
                    turn_count=turn_count,
                    final_response=trigger.response_text,
                    stall_count=stall_count,
                )
            )
            return None  # type: ignore[return-value]

        # Continue to next turn
        return StartTurn()


# ============================================================
# Harness Builder
# ============================================================


def build_harness_workflow(agent: AgentProtocol) -> Workflow:
    """Build a harness workflow with stall detection.

    Args:
        agent: The agent to wrap in the harness.

    Returns:
        A configured Workflow ready to run.
    """
    agent_executor = AgentTurnExecutor(agent)
    stop_executor = StopDecisionExecutor()

    return (
        Workflow(name="stall-detection-harness")
        .add_executor(agent_executor, trigger_type=StartTurn)
        .add_executor(stop_executor, trigger_type=TurnComplete)
    )


def get_initial_state(
    task: str,
    max_turns: int = 10,
    stall_threshold: int = 3,
) -> dict[str, Any]:
    """Create initial shared state for a harness run.

    Args:
        task: The user's task/request.
        max_turns: Maximum turns allowed.
        stall_threshold: Number of stalls before giving up.

    Returns:
        Dictionary of initial shared state values.
    """
    return {
        TURN_COUNT_KEY: 0,
        MAX_TURNS_KEY: max_turns,
        AGENT_MESSAGES_KEY: [ChatMessage(role="user", text=task)],
        STALL_THRESHOLD_KEY: stall_threshold,
        STALL_COUNT_KEY: 0,
        RECENT_FINGERPRINTS_KEY: [],
        CONTINUATION_INJECTED_KEY: False,
    }


# ============================================================
# Simple Harness Class
# ============================================================


class SimpleHarness:
    """Agent harness with turn limits and stall detection.

    This harness detects when an agent is stuck in a loop and
    attempts to break the stall with continuation prompts.

    Example:
        harness = SimpleHarness(agent, max_turns=10, stall_threshold=3)
        result = await harness.run("Complex task here")

        if result.status == HarnessStatus.STALLED:
            print(f"Agent stalled after {result.stall_count} repeated responses")
    """

    def __init__(
        self,
        agent: AgentProtocol,
        max_turns: int = 10,
        stall_threshold: int = 3,
    ):
        """Initialize the harness.

        Args:
            agent: The agent to wrap.
            max_turns: Maximum turns before stopping.
            stall_threshold: Number of stalls before stopping.
        """
        self._agent = agent
        self._max_turns = max_turns
        self._stall_threshold = stall_threshold
        self._workflow = build_harness_workflow(agent)

    async def run(self, task: str) -> HarnessResult:
        """Run the agent with harness protections.

        Args:
            task: The task for the agent to perform.

        Returns:
            HarnessResult with status, response, and stall info.
        """
        initial_state = get_initial_state(
            task,
            self._max_turns,
            self._stall_threshold,
        )

        result: HarnessResult | None = None

        async for event in self._workflow.run_stream(StartTurn(), **initial_state):
            if isinstance(event, WorkflowEvent) and isinstance(event.data, HarnessResult):
                result = event.data

        if result is None:
            result = HarnessResult(
                status=HarnessStatus.ERROR,
                turn_count=0,
                error="No result produced by workflow",
            )

        return result

    async def run_stream(self, task: str):
        """Run the agent and stream events.

        Yields workflow events including stall detection events.

        Args:
            task: The task for the agent to perform.

        Yields:
            WorkflowEvent objects as execution progresses.
        """
        initial_state = get_initial_state(
            task,
            self._max_turns,
            self._stall_threshold,
        )

        async for event in self._workflow.run_stream(StartTurn(), **initial_state):
            yield event
