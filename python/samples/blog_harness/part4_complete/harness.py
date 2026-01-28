# Copyright (c) Microsoft. All rights reserved.

"""Part 5: A Complete Custom Agent Harness.

This module brings together all concepts from the series:
- Turn limits and completion detection (Part 2)
- Stall detection with fingerprinting (Part 3)
- Pluggable policy enforcement (Part 4)
- NEW: Custom executor pattern for domain-specific logic
- NEW: HarnessBuilder for fluent configuration
- NEW: Comprehensive event system for observability

The Complete Picture:
This harness demonstrates that building sophisticated agent control
isn't magic - it's composable workflow executors with clear contracts.
Each executor handles one concern, communicating through typed triggers
and shared state.

Workflow Flow:
    StartTurn → AgentTurnExecutor → TurnComplete
                                        ↓
                                  PolicyExecutor → PolicyChecked
                                                       ↓
                                              [CustomExecutors] → ...
                                                                   ↓
                                                        StopDecisionExecutor
"""

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

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
STALL_THRESHOLD_KEY = "harness:stall_threshold"
STALL_COUNT_KEY = "harness:stall_count"
RECENT_FINGERPRINTS_KEY = "harness:recent_fingerprints"
CONTINUATION_INJECTED_KEY = "harness:continuation_injected"
TOOL_CALL_COUNT_KEY = "harness:tool_call_count"
POLICY_VIOLATIONS_KEY = "harness:policy_violations"
CUSTOM_STATE_KEY = "harness:custom"  # For user-defined state


# ============================================================
# Result Types
# ============================================================


class HarnessStatus(Enum):
    """Final status of a harness run."""

    COMPLETE = "complete"
    MAX_TURNS = "max_turns"
    STALLED = "stalled"
    POLICY_VIOLATION = "policy_violation"
    VALIDATION_FAILED = "validation_failed"  # New: output validation failed
    ERROR = "error"


@dataclass
class HarnessResult:
    """The final output of a harness run."""

    status: HarnessStatus
    turn_count: int
    final_response: str = ""
    stall_count: int = 0
    policy_violations: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    custom_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ============================================================
# Event Types (for observability)
# ============================================================


@dataclass
class HarnessEvent:
    """Structured event emitted during harness execution.

    These events enable rich observability - log them, stream them
    to a UI, or use them for debugging.
    """

    event_type: str  # "turn_started", "turn_complete", "policy_violation", etc.
    turn: int = 0
    data: dict[str, Any] = field(default_factory=dict)


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
    tool_call_count: int
    fingerprint: str


@dataclass
class PolicyChecked:
    """Result after policy checks."""

    response_text: str
    has_tool_calls: bool
    fingerprint: str
    policy_passed: bool
    violation_message: str | None = None


@dataclass
class ValidationChecked:
    """Result after output validation."""

    response_text: str
    has_tool_calls: bool
    fingerprint: str
    validation_passed: bool
    validation_errors: list[str] = field(default_factory=list)


# ============================================================
# Policy System (from Part 4)
# ============================================================


@dataclass
class PolicyResult:
    """Result of a policy check."""

    passed: bool
    message: str | None = None
    should_stop: bool = False


class Policy(ABC):
    """Base class for harness policies."""

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
        """Check if the response violates this policy."""
        ...


class MaxToolCallsPolicy(Policy):
    """Limits total tool calls across all turns."""

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
    """Filters responses containing prohibited patterns."""

    def __init__(
        self,
        patterns: list[str],
        case_sensitive: bool = False,
        stop_on_match: bool = False,
    ):
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


# ============================================================
# Output Validator (NEW in Part 5)
# ============================================================


class OutputValidator(ABC):
    """Base class for output validators.

    Validators check that agent output meets domain-specific requirements.
    Unlike policies (which are about safety/limits), validators are about
    correctness and quality.

    Example use cases:
    - JSON schema validation for structured output
    - Required fields checking
    - Format validation (dates, emails, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this validator."""
        ...

    @abstractmethod
    async def validate(
        self,
        response_text: str,
        ctx: WorkflowContext[Any],
    ) -> tuple[bool, list[str]]:
        """Validate the response.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        ...


class JsonSchemaValidator(OutputValidator):
    """Validates that response contains valid JSON matching a schema.

    This is useful when you need the agent to produce structured output.
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        field_types: dict[str, type] | None = None,
    ):
        """Initialize the validator.

        Args:
            required_fields: List of field names that must be present.
            field_types: Dict mapping field names to expected types.
        """
        self._required_fields = required_fields or []
        self._field_types = field_types or {}

    @property
    def name(self) -> str:
        return f"JsonSchema(fields={self._required_fields})"

    async def validate(
        self,
        response_text: str,
        ctx: WorkflowContext[Any],
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []

        # Try to extract JSON from the response
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_text

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return False, errors

        if not isinstance(data, dict):
            errors.append("JSON must be an object")
            return False, errors

        # Check required fields
        for field_name in self._required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        # Check field types
        for field_name, expected_type in self._field_types.items():
            if field_name in data and not isinstance(data[field_name], expected_type):
                actual_type = type(data[field_name]).__name__
                errors.append(
                    f"Field '{field_name}' has wrong type: expected {expected_type.__name__}, got {actual_type}"
                )

        return len(errors) == 0, errors


class CustomValidator(OutputValidator):
    """Validator using a custom function.

    For quick one-off validations without creating a new class.
    """

    def __init__(
        self,
        name: str,
        validate_fn: Callable[[str], tuple[bool, list[str]]],
    ):
        self._name = name
        self._validate_fn = validate_fn

    @property
    def name(self) -> str:
        return self._name

    async def validate(
        self,
        response_text: str,
        ctx: WorkflowContext[Any],
    ) -> tuple[bool, list[str]]:
        return self._validate_fn(response_text)


# ============================================================
# Fingerprinting
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
# Executors
# ============================================================


class AgentTurnExecutor(Executor):
    """Executes one turn of agent interaction."""

    def __init__(self, agent: AgentProtocol):
        super().__init__()
        self._agent = agent
        self._thread = agent.get_new_thread()

    async def execute(self, trigger: StartTurn, ctx: WorkflowContext[Any]) -> TurnComplete:
        messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)

        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        turn_count += 1
        await ctx.set_shared_state(TURN_COUNT_KEY, turn_count)

        # Emit turn started event
        await ctx.add_event(
            WorkflowEvent(
                data=HarnessEvent(
                    event_type="turn_started",
                    turn=turn_count,
                )
            )
        )

        response = await self._agent.run(messages, thread=self._thread)

        response_text = response.text or ""
        has_tool_calls = bool(response.tool_calls)
        tool_call_count = len(response.tool_calls) if response.tool_calls else 0
        fingerprint = compute_fingerprint(response_text, has_tool_calls)

        messages.append(ChatMessage(role="assistant", text=response_text))
        await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)

        # Emit turn complete event
        await ctx.add_event(
            WorkflowEvent(
                data=HarnessEvent(
                    event_type="turn_complete",
                    turn=turn_count,
                    data={
                        "response_preview": response_text[:200],
                        "fingerprint": fingerprint,
                        "has_tool_calls": has_tool_calls,
                        "tool_call_count": tool_call_count,
                    },
                )
            )
        )

        return TurnComplete(
            response_text=response_text,
            has_tool_calls=has_tool_calls,
            tool_call_count=tool_call_count,
            fingerprint=fingerprint,
        )


class PolicyExecutor(Executor):
    """Runs policy checks after each agent turn."""

    def __init__(self, policies: list[Policy]):
        super().__init__()
        self._policies = policies

    async def execute(self, trigger: TurnComplete, ctx: WorkflowContext[Any]) -> PolicyChecked | HarnessResult:
        violations: list[str] = await ctx.get_shared_state(POLICY_VIOLATIONS_KEY)
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)

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

                await ctx.add_event(
                    WorkflowEvent(
                        data=HarnessEvent(
                            event_type="policy_violation",
                            turn=turn_count,
                            data={
                                "policy": policy.name,
                                "message": result.message,
                                "should_stop": result.should_stop,
                            },
                        )
                    )
                )

                if result.should_stop:
                    await ctx.yield_output(
                        HarnessResult(
                            status=HarnessStatus.POLICY_VIOLATION,
                            turn_count=turn_count,
                            final_response=trigger.response_text,
                            policy_violations=violations,
                        )
                    )
                    return None  # type: ignore[return-value]

        return PolicyChecked(
            response_text=trigger.response_text,
            has_tool_calls=trigger.has_tool_calls,
            fingerprint=trigger.fingerprint,
            policy_passed=len(violations) == 0,
            violation_message=violations[-1] if violations else None,
        )


class ValidationExecutor(Executor):
    """Runs output validators after policies (NEW in Part 5).

    This executor demonstrates adding domain-specific logic to the harness.
    Validators run only when the agent appears to be done (no tool calls).
    """

    def __init__(
        self,
        validators: list[OutputValidator],
        max_retries: int = 2,
    ):
        super().__init__()
        self._validators = validators
        self._max_retries = max_retries
        self._retry_counts: dict[str, int] = {}

    async def execute(
        self, trigger: PolicyChecked, ctx: WorkflowContext[Any]
    ) -> ValidationChecked | StartTurn | HarnessResult:
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)

        # Only validate when agent appears done (no tool calls)
        if trigger.has_tool_calls:
            return ValidationChecked(
                response_text=trigger.response_text,
                has_tool_calls=trigger.has_tool_calls,
                fingerprint=trigger.fingerprint,
                validation_passed=True,
            )

        # Run all validators
        all_errors: list[str] = []
        for validator in self._validators:
            is_valid, errors = await validator.validate(trigger.response_text, ctx)
            if not is_valid:
                all_errors.extend([f"[{validator.name}] {e}" for e in errors])

        if all_errors:
            # Track retries
            retry_key = f"validation_retry_{turn_count}"
            retries = self._retry_counts.get(retry_key, 0)

            await ctx.add_event(
                WorkflowEvent(
                    data=HarnessEvent(
                        event_type="validation_failed",
                        turn=turn_count,
                        data={
                            "errors": all_errors,
                            "retry_count": retries,
                            "max_retries": self._max_retries,
                        },
                    )
                )
            )

            if retries >= self._max_retries:
                # Give up after max retries
                await ctx.yield_output(
                    HarnessResult(
                        status=HarnessStatus.VALIDATION_FAILED,
                        turn_count=turn_count,
                        final_response=trigger.response_text,
                        validation_errors=all_errors,
                    )
                )
                return None  # type: ignore[return-value]

            # Inject correction prompt and retry
            self._retry_counts[retry_key] = retries + 1
            correction_prompt = (
                "Your response has validation errors:\n"
                + "\n".join(f"- {e}" for e in all_errors)
                + "\n\nPlease fix these issues and try again."
            )

            messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)
            messages.append(ChatMessage(role="user", text=correction_prompt))
            await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)

            return StartTurn()  # Retry

        return ValidationChecked(
            response_text=trigger.response_text,
            has_tool_calls=trigger.has_tool_calls,
            fingerprint=trigger.fingerprint,
            validation_passed=True,
        )


class StopDecisionExecutor(Executor):
    """Decides whether to continue, inject a prompt, or stop."""

    CONTINUATION_PROMPT = (
        "You seem to be repeating yourself or stuck in a loop. "
        "Please take a different approach or make concrete progress. "
        "If you've completed the task, say so clearly. "
        "If you're blocked, explain what's preventing progress."
    )

    async def execute(self, trigger: ValidationChecked, ctx: WorkflowContext[Any]) -> StartTurn | HarnessResult:
        turn_count = await ctx.get_shared_state(TURN_COUNT_KEY)
        max_turns = await ctx.get_shared_state(MAX_TURNS_KEY)
        stall_threshold = await ctx.get_shared_state(STALL_THRESHOLD_KEY)
        stall_count = await ctx.get_shared_state(STALL_COUNT_KEY)
        recent_fingerprints: list[str] = await ctx.get_shared_state(RECENT_FINGERPRINTS_KEY)
        policy_violations = await ctx.get_shared_state(POLICY_VIOLATIONS_KEY)

        # Check for stall
        is_stall = trigger.fingerprint in recent_fingerprints

        if is_stall:
            stall_count += 1
            await ctx.set_shared_state(STALL_COUNT_KEY, stall_count)

            await ctx.add_event(
                WorkflowEvent(
                    data=HarnessEvent(
                        event_type="stall_detected",
                        turn=turn_count,
                        data={
                            "stall_count": stall_count,
                            "fingerprint": trigger.fingerprint,
                        },
                    )
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

            messages: list[ChatMessage] = await ctx.get_shared_state(AGENT_MESSAGES_KEY)
            messages.append(ChatMessage(role="user", text=self.CONTINUATION_PROMPT))
            await ctx.set_shared_state(AGENT_MESSAGES_KEY, messages)
            await ctx.set_shared_state(CONTINUATION_INJECTED_KEY, True)

            return StartTurn()

        # Update fingerprint window
        recent_fingerprints.append(trigger.fingerprint)
        if len(recent_fingerprints) > 5:
            recent_fingerprints = recent_fingerprints[-5:]
        await ctx.set_shared_state(RECENT_FINGERPRINTS_KEY, recent_fingerprints)

        if stall_count > 0:
            await ctx.set_shared_state(STALL_COUNT_KEY, 0)

        # Check completion
        agent_done = not trigger.has_tool_calls

        if agent_done:
            await ctx.add_event(
                WorkflowEvent(
                    data=HarnessEvent(
                        event_type="complete",
                        turn=turn_count,
                        data={"final_response_preview": trigger.response_text[:200]},
                    )
                )
            )

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
# Harness Builder (NEW in Part 5)
# ============================================================


class HarnessBuilder:
    """Fluent builder for constructing custom harnesses.

    This builder makes it easy to configure a harness with exactly
    the features you need.

    Example:
        harness = (
            HarnessBuilder(agent)
            .with_max_turns(15)
            .with_stall_detection(threshold=3)
            .with_policy(MaxToolCallsPolicy(max_calls=30))
            .with_policy(ContentFilterPolicy(patterns=["password"]))
            .with_validator(JsonSchemaValidator(required_fields=["result"]))
            .build()
        )
    """

    def __init__(self, agent: AgentProtocol):
        self._agent = agent
        self._max_turns = 10
        self._stall_threshold = 3
        self._policies: list[Policy] = []
        self._validators: list[OutputValidator] = []
        self._validation_retries = 2

    def with_max_turns(self, max_turns: int) -> "HarnessBuilder":
        """Set the maximum number of turns."""
        self._max_turns = max_turns
        return self

    def with_stall_detection(self, threshold: int = 3) -> "HarnessBuilder":
        """Configure stall detection threshold."""
        self._stall_threshold = threshold
        return self

    def with_policy(self, policy: Policy) -> "HarnessBuilder":
        """Add a policy to enforce."""
        self._policies.append(policy)
        return self

    def with_validator(self, validator: OutputValidator) -> "HarnessBuilder":
        """Add an output validator."""
        self._validators.append(validator)
        return self

    def with_validation_retries(self, max_retries: int) -> "HarnessBuilder":
        """Set how many times to retry on validation failure."""
        self._validation_retries = max_retries
        return self

    def build(self) -> "CompleteHarness":
        """Build the configured harness."""
        return CompleteHarness(
            agent=self._agent,
            max_turns=self._max_turns,
            stall_threshold=self._stall_threshold,
            policies=self._policies,
            validators=self._validators,
            validation_retries=self._validation_retries,
        )


# ============================================================
# Complete Harness
# ============================================================


def build_harness_workflow(
    agent: AgentProtocol,
    policies: list[Policy],
    validators: list[OutputValidator],
    validation_retries: int,
) -> Workflow:
    """Build the complete harness workflow."""
    agent_executor = AgentTurnExecutor(agent)
    policy_executor = PolicyExecutor(policies)
    validation_executor = ValidationExecutor(validators, validation_retries)
    stop_executor = StopDecisionExecutor()

    return (
        Workflow(name="complete-harness")
        .add_executor(agent_executor, trigger_type=StartTurn)
        .add_executor(policy_executor, trigger_type=TurnComplete)
        .add_executor(validation_executor, trigger_type=PolicyChecked)
        .add_executor(stop_executor, trigger_type=ValidationChecked)
    )


def get_initial_state(
    task: str,
    max_turns: int,
    stall_threshold: int,
) -> dict[str, Any]:
    """Create initial shared state."""
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
        CUSTOM_STATE_KEY: {},
    }


class CompleteHarness:
    """A complete agent harness with all features.

    This harness includes:
    - Turn limits
    - Stall detection with fingerprinting
    - Pluggable policy enforcement
    - Output validation with retries
    - Comprehensive event streaming

    Use the HarnessBuilder for easy configuration:

        harness = (
            HarnessBuilder(agent)
            .with_max_turns(10)
            .with_policy(MaxToolCallsPolicy(20))
            .with_validator(JsonSchemaValidator(["result"]))
            .build()
        )

    Or instantiate directly:

        harness = CompleteHarness(
            agent=agent,
            max_turns=10,
            policies=[MaxToolCallsPolicy(20)],
        )
    """

    def __init__(
        self,
        agent: AgentProtocol,
        max_turns: int = 10,
        stall_threshold: int = 3,
        policies: list[Policy] | None = None,
        validators: list[OutputValidator] | None = None,
        validation_retries: int = 2,
    ):
        self._agent = agent
        self._max_turns = max_turns
        self._stall_threshold = stall_threshold
        self._policies = policies or []
        self._validators = validators or []
        self._validation_retries = validation_retries
        self._workflow = build_harness_workflow(
            agent,
            self._policies,
            self._validators,
            self._validation_retries,
        )

    async def run(self, task: str) -> HarnessResult:
        """Run the agent with full harness protections."""
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

        Yields HarnessEvent objects for observability.
        """
        initial_state = get_initial_state(
            task,
            self._max_turns,
            self._stall_threshold,
        )

        async for event in self._workflow.run_stream(StartTurn(), **initial_state):
            yield event


# Convenience alias
SimpleHarness = CompleteHarness
