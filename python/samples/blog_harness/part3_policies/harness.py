# Copyright (c) Microsoft. All rights reserved.

"""Part 4: Agent Harness with Policy Enforcement.

Building on Part 3, this module adds policy enforcement:
- Pluggable policy system for custom rules
- Policy executor that runs after each agent turn
- Built-in policies for common needs (tool limits, content checks)

The Key Insight:
The workflow architecture makes it trivial to add new behaviors.
We insert a PolicyExecutor between AgentTurnExecutor and StopDecisionExecutor
without modifying either. This is the power of composable executors.

Workflow Flow:
    StartTurn → AgentTurnExecutor → TurnComplete
                                        ↓
                                  PolicyExecutor → PolicyChecked
                                                       ↓
                                              StopDecisionExecutor → StartTurn (loop)
                                                                   ↘ HarnessResult (end)
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

# Stall detection keys (from Part 3)
STALL_THRESHOLD_KEY = "harness:stall_threshold"
STALL_COUNT_KEY = "harness:stall_count"
RECENT_FINGERPRINTS_KEY = "harness:recent_fingerprints"
CONTINUATION_INJECTED_KEY = "harness:continuation_injected"

# New: Policy tracking keys
TOOL_CALL_COUNT_KEY = "harness:tool_call_count"
POLICY_VIOLATIONS_KEY = "harness:policy_violations"


# ============================================================
# Result Types
# ============================================================


class HarnessStatus(Enum):
    """Final status of a harness run."""

    COMPLETE = "complete"  # Agent finished its task
    MAX_TURNS = "max_turns"  # Hit turn limit
    STALLED = "stalled"  # Agent stalled and couldn't recover
    POLICY_VIOLATION = "policy_violation"  # Policy blocked continuation
    ERROR = "error"  # Something went wrong


@dataclass
class HarnessResult:
    """The final output of a harness run."""

    status: HarnessStatus
    turn_count: int
    final_response: str = ""
    stall_count: int = 0
    policy_violations: list[str] = field(default_factory=list)
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
    tool_call_count: int  # New: track how many tools were called
    fingerprint: str


@dataclass
class PolicyChecked:
    """Result after policy checks, passed to StopDecisionExecutor."""

    response_text: str
    has_tool_calls: bool
    fingerprint: str
    policy_passed: bool
    violation_message: str | None = None


# ============================================================
# Policy System
# ============================================================


@dataclass
class PolicyResult:
    """Result of a policy check."""

    passed: bool
    message: str | None = None  # Explanation if failed
    should_stop: bool = False  # If True, stop immediately (don't just warn)


class Policy(ABC):
    """Base class for harness policies.

    Policies are rules that can inspect agent responses and decide
    whether to allow continuation, inject guidance, or stop execution.

    To create a custom policy:
    1. Subclass Policy
    2. Implement the check() method
    3. Return PolicyResult indicating pass/fail
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this policy."""
        ...

    @abstractmethod
    async def check(
        self,
        response_text: str,
        tool_call_count: int,
        ctx: WorkflowContext[Any],
    ) -> PolicyResult:
        """Check if the response violates this policy.

        Args:
            response_text: The agent's response text.
            tool_call_count: Number of tool calls in this turn.
            ctx: Workflow context for accessing shared state.

        Returns:
            PolicyResult indicating whether the policy passed.
        """
        ...


# ============================================================
# Built-in Policies
# ============================================================


class MaxToolCallsPolicy(Policy):
    """Limits the total number of tool calls across all turns.

    This prevents runaway tool usage that could be expensive or dangerous.
    """

    def __init__(self, max_calls: int = 50):
        self._max_calls = max_calls

    @property
    def name(self) -> str:
        return f"MaxToolCalls({self._max_calls})"

    async def check(
        self,
        response_text: str,
        tool_call_count: int,
        ctx: WorkflowContext[Any],
    ) -> PolicyResult:
        total_calls = await ctx.get_shared_state(TOOL_CALL_COUNT_KEY)
        total_calls += tool_call_count
        await ctx.set_shared_state(TOOL_CALL_COUNT_KEY, total_calls)

        if total_calls > self._max_calls:
            return PolicyResult(
                passed=False,
                message=f"Exceeded maximum tool calls ({total_calls}/{self._max_calls})",
                should_stop=True,
            )

        return PolicyResult(passed=True)


class ContentFilterPolicy(Policy):
    """Filters responses containing prohibited patterns.

    Use this to catch unwanted content like:
    - Profanity or inappropriate language
    - Sensitive data patterns (SSNs, credit cards)
    - Forbidden topics
    """

    def __init__(
        self,
        patterns: list[str],
        case_sensitive: bool = False,
        stop_on_match: bool = False,
    ):
        """Initialize the content filter.

        Args:
            patterns: Regex patterns to match against response.
            case_sensitive: Whether pattern matching is case-sensitive.
            stop_on_match: If True, stop execution on match. If False, just warn.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        self._patterns = [(p, re.compile(p, flags)) for p in patterns]
        self._stop_on_match = stop_on_match

    @property
    def name(self) -> str:
        return f"ContentFilter({len(self._patterns)} patterns)"

    async def check(
        self,
        response_text: str,
        tool_call_count: int,
        ctx: WorkflowContext[Any],
    ) -> PolicyResult:
        for pattern_str, pattern in self._patterns:
            if pattern.search(response_text):
                return PolicyResult(
                    passed=False,
                    message=f"Response matched prohibited pattern: {pattern_str}",
                    should_stop=self._stop_on_match,
                )

        return PolicyResult(passed=True)


class ResponseLengthPolicy(Policy):
    """Enforces minimum and maximum response lengths.

    Useful for:
    - Ensuring responses aren't too terse
    - Preventing extremely long responses that waste tokens
    """

    def __init__(self, min_length: int = 0, max_length: int = 10000):
        self._min_length = min_length
        self._max_length = max_length

    @property
    def name(self) -> str:
        return f"ResponseLength({self._min_length}-{self._max_length})"

    async def check(
        self,
        response_text: str,
        tool_call_count: int,
        ctx: WorkflowContext[Any],
    ) -> PolicyResult:
        length = len(response_text)

        if length < self._min_length:
            return PolicyResult(
                passed=False,
                message=f"Response too short ({length} < {self._min_length} chars)",
                should_stop=False,  # Don't stop, let agent try again
            )

        if length > self._max_length:
            return PolicyResult(
                passed=False,
                message=f"Response too long ({length} > {self._max_length} chars)",
                should_stop=False,
            )

        return PolicyResult(passed=True)


# ============================================================
# Fingerprinting (from Part 3)
# ============================================================


def compute_fingerprint(response_text: str, has_tool_calls: bool) -> str:
    """Compute a fingerprint for stall detection."""
    normalized = " ".join(response_text.split())
    prefix = normalized[:200] if len(normalized) > 200 else normalized
    suffix = normalized[-100:] if len(normalized) > 100 else ""
    length_bucket = "short" if len(normalized) < 100 else "medium" if len(normalized) < 500 else "long"
    fingerprint_input = f"{prefix}|{suffix}|{has_tool_calls}|{length_bucket}"
    return hashlib.md5(fingerprint_input.encode()).hexdigest()[:12]


# ============================================================
# Agent Turn Executor
# ============================================================


class AgentTurnExecutor(Executor):
    """Executes one turn of agent interaction."""

    def __init__(self, agent: AgentProtocol):
        super().__init__()
        self._agent = agent
        self._thread = agent.get_new_thread()

    async def execute(self, trigger: StartTurn, ctx: WorkflowContext[Any]) -> TurnComplete:
        """Run one agent turn."""
        messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)

        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        turn_count += 1
        await ctx.set_shared_state(TURN_COUNT_KEY, turn_count)

        response = await self._agent.run(messages, thread=self._thread)

        response_text = response.text or ""
        has_tool_calls = bool(response.tool_calls)
        tool_call_count = len(response.tool_calls) if response.tool_calls else 0
        fingerprint = compute_fingerprint(response_text, has_tool_calls)

        messages.append(ChatMessage(role="assistant", text=response_text))
        await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)

        await ctx.add_event(
            WorkflowEvent(
                data={
                    "turn": turn_count,
                    "response": response_text[:200],
                    "fingerprint": fingerprint,
                    "has_tools": has_tool_calls,
                    "tool_count": tool_call_count,
                }
            )
        )

        return TurnComplete(
            response_text=response_text,
            has_tool_calls=has_tool_calls,
            tool_call_count=tool_call_count,
            fingerprint=fingerprint,
        )


# ============================================================
# Policy Executor (NEW in Part 4)
# ============================================================


class PolicyExecutor(Executor):
    """Runs policy checks after each agent turn.

    This executor sits between AgentTurnExecutor and StopDecisionExecutor,
    checking each response against configured policies before deciding
    whether to continue.

    If a policy fails with should_stop=True, the harness stops immediately.
    If a policy fails with should_stop=False, a warning is recorded but
    execution continues.
    """

    def __init__(self, policies: list[Policy]):
        super().__init__()
        self._policies = policies

    async def execute(self, trigger: TurnComplete, ctx: WorkflowContext[Any]) -> PolicyChecked | HarnessResult:
        """Run all policies and decide whether to continue."""
        violations: list[str] = await ctx.get_shared_state(POLICY_VIOLATIONS_KEY)

        for policy in self._policies:
            result = await policy.check(
                trigger.response_text,
                trigger.tool_call_count,
                ctx,
            )

            if not result.passed:
                violation_msg = f"[{policy.name}] {result.message}"
                violations.append(violation_msg)
                await ctx.set_shared_state(POLICY_VIOLATIONS_KEY, violations)

                # Emit policy violation event
                await ctx.add_event(
                    WorkflowEvent(
                        data={
                            "event": "policy_violation",
                            "policy": policy.name,
                            "message": result.message,
                            "should_stop": result.should_stop,
                        }
                    )
                )

                if result.should_stop:
                    # Hard stop - policy requires termination
                    turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
                    await ctx.yield_output(
                        HarnessResult(
                            status=HarnessStatus.POLICY_VIOLATION,
                            turn_count=turn_count,
                            final_response=trigger.response_text,
                            policy_violations=violations,
                        )
                    )
                    return None  # type: ignore[return-value]

        # All policies passed (or only soft failures)
        return PolicyChecked(
            response_text=trigger.response_text,
            has_tool_calls=trigger.has_tool_calls,
            fingerprint=trigger.fingerprint,
            policy_passed=len(violations) == 0,
            violation_message=violations[-1] if violations else None,
        )


# ============================================================
# Stop Decision Executor (with Stall Detection from Part 3)
# ============================================================


class StopDecisionExecutor(Executor):
    """Decides whether to continue, inject a prompt, or stop."""

    CONTINUATION_PROMPT = (
        "You seem to be repeating yourself or stuck in a loop. "
        "Please take a different approach or make concrete progress. "
        "If you've completed the task, say so clearly. "
        "If you're blocked, explain what's preventing progress."
    )

    async def execute(self, trigger: PolicyChecked, ctx: WorkflowContext[Any]) -> StartTurn | HarnessResult:
        """Decide next action with stall detection."""
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        max_turns = await ctx.get_shared_state(MAX_TURNS_KEY)
        stall_threshold = await ctx.get_shared_state(STALL_THRESHOLD_KEY)
        stall_count = await ctx.get_shared_state(STALL_COUNT_KEY)
        recent_fingerprints: list[str] = await ctx.get_shared_state(RECENT_FINGERPRINTS_KEY)
        policy_violations = await ctx.get_shared_state(POLICY_VIOLATIONS_KEY)

        # --- Check for stall ---
        is_stall = trigger.fingerprint in recent_fingerprints

        if is_stall:
            stall_count += 1
            await ctx.set_shared_state(STALL_COUNT_KEY, stall_count)

            await ctx.add_event(
                WorkflowEvent(
                    data={
                        "event": "stall_detected",
                        "stall_count": stall_count,
                        "fingerprint": trigger.fingerprint,
                    }
                )
            )

            if stall_count >= stall_threshold:
                await ctx.yield_output(
                    HarnessResult(
                        status=HarnessStatus.STALLED,
                        turn_count=turn_count,
                        final_response=trigger.response_text,
                        stall_count=stall_count,
                        policy_violations=policy_violations,
                    )
                )
                return None  # type: ignore[return-value]

            # Inject continuation prompt
            messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)
            messages.append(ChatMessage(role="user", text=self.CONTINUATION_PROMPT))
            await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)
            await ctx.set_shared_state(CONTINUATION_INJECTED_KEY, True)

            return StartTurn()

        # --- Not a stall: update fingerprint window ---
        recent_fingerprints.append(trigger.fingerprint)
        if len(recent_fingerprints) > 5:
            recent_fingerprints = recent_fingerprints[-5:]
        await ctx.set_shared_state(RECENT_FINGERPRINTS_KEY, recent_fingerprints)

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
                    policy_violations=policy_violations,
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
                    policy_violations=policy_violations,
                )
            )
            return None  # type: ignore[return-value]

        return StartTurn()


# ============================================================
# Harness Builder
# ============================================================


def build_harness_workflow(
    agent: AgentProtocol,
    policies: list[Policy] | None = None,
) -> Workflow:
    """Build a harness workflow with policies and stall detection.

    Args:
        agent: The agent to wrap in the harness.
        policies: List of policies to enforce. Defaults to empty.

    Returns:
        A configured Workflow ready to run.
    """
    agent_executor = AgentTurnExecutor(agent)
    policy_executor = PolicyExecutor(policies or [])
    stop_executor = StopDecisionExecutor()

    return (
        Workflow(name="policy-harness")
        .add_executor(agent_executor, trigger_type=StartTurn)
        .add_executor(policy_executor, trigger_type=TurnComplete)
        .add_executor(stop_executor, trigger_type=PolicyChecked)
    )


def get_initial_state(
    task: str,
    max_turns: int = 10,
    stall_threshold: int = 3,
) -> dict[str, Any]:
    """Create initial shared state for a harness run."""
    return {
        TURN_COUNT_KEY: 0,
        MAX_TURNS_KEY: max_turns,
        AGENT_MESSAGES_KEY: [ChatMessage(role="user", text=task)],
        STALL_THRESHOLD_KEY: stall_threshold,
        STALL_COUNT_KEY: 0,
        RECENT_FINGERPRINTS_KEY: [],
        CONTINUATION_INJECTED_KEY: False,
        TOOL_CALL_COUNT_KEY: 0,
        POLICY_VIOLATIONS_KEY: [],
    }


# ============================================================
# Simple Harness Class
# ============================================================


class SimpleHarness:
    """Agent harness with turn limits, stall detection, and policy enforcement.

    Example:
        # Create harness with policies
        harness = SimpleHarness(
            agent,
            max_turns=10,
            policies=[
                MaxToolCallsPolicy(max_calls=20),
                ContentFilterPolicy(patterns=[r"password", r"secret"]),
            ],
        )

        result = await harness.run("Analyze this data")

        if result.status == HarnessStatus.POLICY_VIOLATION:
            print(f"Stopped due to: {result.policy_violations}")
    """

    def __init__(
        self,
        agent: AgentProtocol,
        max_turns: int = 10,
        stall_threshold: int = 3,
        policies: list[Policy] | None = None,
    ):
        """Initialize the harness.

        Args:
            agent: The agent to wrap.
            max_turns: Maximum turns before stopping.
            stall_threshold: Number of stalls before stopping.
            policies: List of policies to enforce.
        """
        self._agent = agent
        self._max_turns = max_turns
        self._stall_threshold = stall_threshold
        self._policies = policies or []
        self._workflow = build_harness_workflow(agent, self._policies)

    async def run(self, task: str) -> HarnessResult:
        """Run the agent with harness protections."""
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
        """Run the agent and stream events."""
        initial_state = get_initial_state(
            task,
            self._max_turns,
            self._stall_threshold,
        )

        async for event in self._workflow.run_stream(StartTurn(), **initial_state):
            yield event
