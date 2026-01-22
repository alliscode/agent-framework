# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness Context Pressure module (Phase 2)."""

from dataclasses import dataclass
from typing import Any

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework._harness import (
    ClearEdit,
    ClearToolResultsStrategy,
    CompactConversationStrategy,
    CompactEdit,
    ContextEditPlan,
    ContextPressureComplete,
    ContextPressureExecutor,
    DropEdit,
    DropOldestStrategy,
    HarnessStatus,
    RepairExecutor,
    RepairTrigger,
    TokenBudget,
    TranscriptRange,
    estimate_tokens,
    estimate_transcript_tokens,
    get_default_strategies,
)
from agent_framework._harness._constants import (
    HARNESS_CONTEXT_EDIT_HISTORY_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)

# Test TokenBudget


def test_token_budget_defaults() -> None:
    """Test that TokenBudget has sensible defaults."""
    budget = TokenBudget()

    assert budget.max_input_tokens == 100000
    assert budget.soft_threshold_percent == 0.85
    assert budget.current_estimate == 0
    assert budget.soft_threshold == 85000
    assert not budget.is_under_pressure
    assert budget.tokens_over_threshold == 0


def test_token_budget_pressure_detection() -> None:
    """Test that TokenBudget correctly detects pressure."""
    budget = TokenBudget(max_input_tokens=10000, soft_threshold_percent=0.80)

    # Under threshold - no pressure
    budget.current_estimate = 7000
    assert not budget.is_under_pressure
    assert budget.tokens_over_threshold == 0

    # At threshold - under pressure
    budget.current_estimate = 8000
    assert budget.is_under_pressure
    assert budget.tokens_over_threshold == 0

    # Over threshold - under pressure
    budget.current_estimate = 9000
    assert budget.is_under_pressure
    assert budget.tokens_over_threshold == 1000


def test_token_budget_serialization() -> None:
    """Test TokenBudget serialization round-trip."""
    budget = TokenBudget(max_input_tokens=50000, soft_threshold_percent=0.90, current_estimate=40000)

    serialized = budget.to_dict()
    restored = TokenBudget.from_dict(serialized)

    assert restored.max_input_tokens == 50000
    assert restored.soft_threshold_percent == 0.90
    assert restored.current_estimate == 40000


# Test TranscriptRange


def test_transcript_range_defaults() -> None:
    """Test TranscriptRange defaults."""
    range = TranscriptRange()

    assert range.start_index == 0
    assert range.end_index is None
    assert range.event_types is None


def test_transcript_range_serialization() -> None:
    """Test TranscriptRange serialization round-trip."""
    range = TranscriptRange(start_index=5, end_index=10, event_types=["tool_result", "agent_response"])

    serialized = range.to_dict()
    restored = TranscriptRange.from_dict(serialized)

    assert restored.start_index == 5
    assert restored.end_index == 10
    assert restored.event_types == ["tool_result", "agent_response"]


# Test ContextEditPlan


def test_context_edit_plan_creation() -> None:
    """Test creating a ContextEditPlan."""
    plan = ContextEditPlan(
        reason="token_budget",
        estimated_token_reduction=5000,
        edits=[
            ClearEdit(scope=TranscriptRange(start_index=0, end_index=5)),
        ],
    )

    assert plan.reason == "token_budget"
    assert plan.estimated_token_reduction == 5000
    assert len(plan.edits) == 1


def test_context_edit_plan_serialization() -> None:
    """Test ContextEditPlan serialization."""
    plan = ContextEditPlan(
        reason="token_budget",
        estimated_token_reduction=3000,
        edits=[
            ClearEdit(scope=TranscriptRange(start_index=0, end_index=3)),
            CompactEdit(scope=TranscriptRange(start_index=3, end_index=6)),
        ],
    )

    serialized = plan.to_dict()

    assert serialized["reason"] == "token_budget"
    assert serialized["estimated_token_reduction"] == 3000
    assert len(serialized["edits"]) == 2


# Test Token Estimation


def test_estimate_tokens_basic() -> None:
    """Test basic token estimation."""
    # Rough approximation: 1 token ≈ 4 characters
    text = "Hello, world!"  # 13 characters
    estimate = estimate_tokens(text)

    # Should be around 3-4 tokens
    assert estimate >= 1
    assert estimate <= 5


def test_estimate_tokens_empty() -> None:
    """Test token estimation for empty string."""
    estimate = estimate_tokens("")

    assert estimate == 1  # Minimum of 1


def test_estimate_transcript_tokens() -> None:
    """Test estimating tokens for a transcript."""
    transcript = [
        {"event_type": "turn_start", "data": {"turn_number": 1}},
        {"event_type": "agent_response", "data": {"turn_number": 1, "message": "Hello"}},
        {"event_type": "tool_call", "data": {"tool_name": "test", "args": {"x": 1}}},
    ]

    estimate = estimate_transcript_tokens(transcript)

    # Should be positive
    assert estimate > 0


# Test Strategies


def test_clear_tool_results_strategy_applicability() -> None:
    """Test ClearToolResultsStrategy.is_applicable()."""
    strategy = ClearToolResultsStrategy()
    budget = TokenBudget()

    # Not applicable if transcript too short
    assert not strategy.is_applicable(budget, [])
    assert not strategy.is_applicable(budget, [{"event_type": "tool_result"}])
    assert not strategy.is_applicable(budget, [{"event_type": "tool_result"}, {"event_type": "turn_start"}])

    # Not applicable if no tool results in older entries
    transcript = [
        {"event_type": "turn_start", "data": {}},
        {"event_type": "agent_response", "data": {}},
        {"event_type": "turn_start", "data": {}},
        {"event_type": "agent_response", "data": {}},
    ]
    assert not strategy.is_applicable(budget, transcript)

    # Applicable if tool_result in older entries
    transcript = [
        {"event_type": "tool_result", "data": {}},  # Old - can be cleared
        {"event_type": "turn_start", "data": {}},
        {"event_type": "agent_response", "data": {}},
        {"event_type": "turn_start", "data": {}},
    ]
    assert strategy.is_applicable(budget, transcript)


def test_compact_conversation_strategy_applicability() -> None:
    """Test CompactConversationStrategy.is_applicable()."""
    strategy = CompactConversationStrategy()
    budget = TokenBudget()

    # Not applicable if transcript too short
    assert not strategy.is_applicable(budget, [])
    assert not strategy.is_applicable(budget, [{"event_type": "turn_start"}] * 4)

    # Applicable if 5+ events
    transcript = [{"event_type": "turn_start", "data": {}}] * 5
    assert strategy.is_applicable(budget, transcript)


def test_drop_oldest_strategy_applicability() -> None:
    """Test DropOldestStrategy.is_applicable()."""
    strategy = DropOldestStrategy()
    budget = TokenBudget()

    # Not applicable if transcript too short
    assert not strategy.is_applicable(budget, [])
    assert not strategy.is_applicable(budget, [{"event_type": "turn_start"}] * 5)

    # Applicable if 6+ events
    transcript = [{"event_type": "turn_start", "data": {}}] * 6
    assert strategy.is_applicable(budget, transcript)


def test_get_default_strategies() -> None:
    """Test get_default_strategies returns correct order."""
    strategies = get_default_strategies()

    assert len(strategies) == 3
    assert strategies[0].name == "clear_tool_results"
    assert strategies[1].name == "compact_conversation"
    assert strategies[2].name == "drop_oldest"


# Test ContextPressureExecutor


async def test_context_pressure_executor_no_pressure() -> None:
    """Test ContextPressureExecutor when not under pressure."""

    @dataclass
    class TestOutput:
        edits_applied: int
        tokens_freed: int

    class TestExecutor(Executor):
        @handler
        async def handle(
            self, msg: ContextPressureComplete, ctx: WorkflowContext[None, TestOutput]
        ) -> None:
            await ctx.yield_output(TestOutput(edits_applied=msg.edits_applied, tokens_freed=msg.tokens_freed))

    # Create small transcript that won't trigger pressure
    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairTrigger]) -> None:
            # Small transcript - won't trigger pressure
            await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, [
                {"event_type": "turn_start", "data": {"turn_number": 1}},
            ])
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 1)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.send_message(RepairTrigger())

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(
            lambda: ContextPressureExecutor(
                max_input_tokens=100000,
                soft_threshold_percent=0.85,
                id="pressure",
            ),
            name="pressure",
        )
        .register_executor(lambda: TestExecutor(id="test"), name="test")
        .add_edge("setup", "repair")
        .add_edge("repair", "pressure")
        .add_edge("pressure", "test")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.edits_applied == 0
    assert output.tokens_freed == 0


async def test_context_pressure_executor_under_pressure() -> None:
    """Test ContextPressureExecutor when under pressure with tool results to clear."""

    @dataclass
    class TestOutput:
        edits_applied: int
        tokens_freed: int
        edit_history: list[Any]

    class TestExecutor(Executor):
        @handler
        async def handle(
            self, msg: ContextPressureComplete, ctx: WorkflowContext[None, TestOutput]
        ) -> None:
            edit_history = await ctx.get_shared_state(HARNESS_CONTEXT_EDIT_HISTORY_KEY) or []
            await ctx.yield_output(
                TestOutput(
                    edits_applied=msg.edits_applied,
                    tokens_freed=msg.tokens_freed,
                    edit_history=edit_history,
                )
            )

    # Create large transcript with tool results that will trigger pressure
    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairTrigger]) -> None:
            # Large transcript with tool results
            transcript = []
            for i in range(20):
                transcript.extend([
                    {"event_type": "turn_start", "data": {"turn_number": i + 1}},
                    {"event_type": "tool_call", "data": {"tool_name": "test", "args": {"x": "y" * 100}}},
                    {"event_type": "tool_result", "data": {"result": "z" * 1000}},  # Large result
                    {"event_type": "agent_response", "data": {"message": "Response " + "x" * 500}},
                ])
            await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, transcript)
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 20)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.send_message(RepairTrigger())

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(
            lambda: ContextPressureExecutor(
                max_input_tokens=1000,  # Very low threshold to trigger pressure
                soft_threshold_percent=0.10,  # Very low to ensure pressure
                id="pressure",
            ),
            name="pressure",
        )
        .register_executor(lambda: TestExecutor(id="test"), name="test")
        .add_edge("setup", "repair")
        .add_edge("repair", "pressure")
        .add_edge("pressure", "test")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    # Should have applied at least one edit
    assert output.edits_applied > 0
    # Should have recorded edit history
    assert len(output.edit_history) > 0


async def test_context_pressure_executor_saves_budget() -> None:
    """Test that ContextPressureExecutor saves token budget to state."""

    @dataclass
    class TestOutput:
        budget: dict[str, Any] | None

    class TestExecutor(Executor):
        @handler
        async def handle(
            self, msg: ContextPressureComplete, ctx: WorkflowContext[None, TestOutput]
        ) -> None:
            budget = await ctx.get_shared_state(HARNESS_TOKEN_BUDGET_KEY)
            await ctx.yield_output(TestOutput(budget=budget))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairTrigger]) -> None:
            await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, [
                {"event_type": "turn_start", "data": {"turn_number": 1}},
            ])
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 1)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.send_message(RepairTrigger())

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(
            lambda: ContextPressureExecutor(
                max_input_tokens=50000,
                soft_threshold_percent=0.90,
                id="pressure",
            ),
            name="pressure",
        )
        .register_executor(lambda: TestExecutor(id="test"), name="test")
        .add_edge("setup", "repair")
        .add_edge("repair", "pressure")
        .add_edge("pressure", "test")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.budget is not None
    assert output.budget["max_input_tokens"] == 50000
    assert output.budget["soft_threshold_percent"] == 0.90
    assert "current_estimate" in output.budget


# Test Edit types


def test_clear_edit_defaults() -> None:
    """Test ClearEdit defaults."""
    edit = ClearEdit()

    assert edit.kind == "clear"
    assert edit.mode == "results_only"
    assert edit.placeholder_template == "[Tool result cleared to save context]"


def test_compact_edit_defaults() -> None:
    """Test CompactEdit defaults."""
    edit = CompactEdit()

    assert edit.kind == "compact"
    assert edit.summary_type == "bullet"
    assert edit.keep_recent_count == 3


def test_drop_edit_defaults() -> None:
    """Test DropEdit defaults."""
    edit = DropEdit()

    assert edit.kind == "drop"
