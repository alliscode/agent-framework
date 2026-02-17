# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent_framework import ChatMessage, FunctionCallContent, FunctionResultContent
from agent_framework._harness._compaction import ClearRecord, CompactionPlan, DropRecord, SpanReference, TokenBudget
from agent_framework._harness._compaction._tokenizer import get_tokenizer
from agent_framework._harness._compaction_owner import (
    CompactionOwnerService,
    OwnerCompactionResult,
    OwnerFallbackReason,
)
from agent_framework._harness._compaction_shadow import CompactionShadowService
from agent_framework._harness._compaction_state import CompactionStateStore
from agent_framework._harness._compaction_policy import CompactionPolicy
from agent_framework._harness._compaction_lifecycle import CompactionLifecycleEmitter
from agent_framework._harness._compaction_thread import CompactionThreadService
from agent_framework._harness._compaction_helpers import summarize_applied_strategies, tokens_to_free_for_pressure
from agent_framework._harness._compaction_telemetry import OwnerCompactionOutcome
from agent_framework._harness._state import HarnessLifecycleEvent
from agent_framework._harness._constants import (
    HARNESS_COMPACTION_METRICS_KEY,
    HARNESS_COMPACTION_OWNER_MODE_KEY,
    HARNESS_COMPACTION_PLAN_KEY,
    HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY,
    HARNESS_SHARED_TURN_BUFFER_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from agent_framework._workflows._conversation_state import encode_chat_messages
from agent_framework._harness._compaction import CacheThreadAdapter


class _FakeContext:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self._state = dict(state or {})
        self.events: list[Any] = []

    async def get_shared_state(self, key: str) -> Any:
        if key not in self._state:
            raise KeyError(key)
        return self._state[key]

    async def set_shared_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    async def add_event(self, event: Any) -> None:
        self.events.append(event)


@dataclass
class _CompactResult:
    plan: CompactionPlan | None
    tokens_freed: int
    proposals_applied: int
    proposals_generated: int = 0


class _StubCoordinator:
    def __init__(self, result: _CompactResult | None = None, *, should_raise: bool = False) -> None:
        self._result = result
        self._should_raise = should_raise

    async def compact(self, *args: Any, **kwargs: Any) -> _CompactResult:
        if self._should_raise:
            raise RuntimeError("coordinator failed")
        assert self._result is not None
        return self._result


@pytest.mark.asyncio
async def test_compaction_state_store_budget_plan_metrics_roundtrip() -> None:
    ctx = _FakeContext()
    store = CompactionStateStore(max_input_tokens=10000, soft_threshold_percent=0.75)

    budget = await store.get_or_create_budget(ctx)
    assert budget.max_input_tokens == 10000
    assert budget.soft_threshold_percent == 0.75

    budget.current_estimate = 444
    await store.save_budget(ctx, budget)
    loaded_budget = await store.get_or_create_budget(ctx)
    assert loaded_budget.current_estimate == 444

    assert await store.get_turn_count(ctx) == 0
    await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 7)
    assert await store.get_turn_count(ctx) == 7

    assert await store.get_owner_mode(ctx) == "compaction_executor"
    await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "shadow")
    assert await store.get_owner_mode(ctx) == "shadow"

    plan = CompactionPlan.create_empty(thread_id="t")
    await store.save_plan(ctx, plan, version=3)
    loaded_plan, loaded_version = await store.load_plan(ctx)
    assert loaded_plan is not None
    assert loaded_version == 4

    for i in range(103):
        await store.append_metrics(ctx, {"i": i})
    metrics = await ctx.get_shared_state(HARNESS_COMPACTION_METRICS_KEY)
    assert isinstance(metrics, list)
    assert len(metrics) == 100
    assert metrics[0]["i"] == 3
    assert metrics[-1]["i"] == 102


@pytest.mark.asyncio
async def test_compaction_shadow_service_clears_candidate_when_not_shadow() -> None:
    ctx = _FakeContext()
    service = CompactionShadowService(
        coordinator=_StubCoordinator(_CompactResult(plan=None, tokens_freed=0, proposals_applied=0)),  # type: ignore[arg-type]
        tokenizer=get_tokenizer(),
    )
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=800)

    await service.publish_candidate(
        ctx=ctx,
        owner_mode="agent_turn",
        turn_number=1,
        current_tokens=budget.current_estimate,
        budget=budget,
        strategies_available=["clear"],
    )
    assert await ctx.get_shared_state(HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY) is None


@pytest.mark.asyncio
async def test_compaction_shadow_service_publishes_simulation_metadata() -> None:
    messages = [
        ChatMessage(role="user", text="u1", message_id="u1"),
        ChatMessage(role="tool", contents=[FunctionResultContent(call_id="c1", result="R" * 3000)], message_id="t1"),
    ]
    ctx = _FakeContext(
        {
            HARNESS_SHARED_TURN_BUFFER_KEY: {
                "version": 2,
                "message_count": len(messages),
                "messages": encode_chat_messages(messages),
            }
        }
    )
    plan = CompactionPlan.create_empty(thread_id="t")
    plan.clearings.append(ClearRecord(span=SpanReference(message_ids=["t1"], first_turn=1, last_turn=1), preserved_fields={}))
    plan = CompactionPlan(
        thread_id=plan.thread_id,
        thread_version=plan.thread_version,
        created_at=plan.created_at,
        clearings=plan.clearings,
    )
    service = CompactionShadowService(
        coordinator=_StubCoordinator(_CompactResult(plan=plan, tokens_freed=1200, proposals_applied=1)),  # type: ignore[arg-type]
        tokenizer=get_tokenizer(),
    )
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=800)

    await service.publish_candidate(
        ctx=ctx,
        owner_mode="shadow",
        turn_number=3,
        current_tokens=budget.current_estimate,
        budget=budget,
        strategies_available=["clear", "drop"],
    )
    candidate = await ctx.get_shared_state(HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY)
    assert candidate["candidate_owner"] == "compaction_executor"
    assert candidate["simulation_available"] is True
    assert candidate["candidate_tokens_freed"] == 1200
    assert candidate["candidate_effective_compaction"] is True


@pytest.mark.asyncio
async def test_compaction_owner_service_fallback_without_snapshot() -> None:
    saved_budgets: list[TokenBudget] = []

    async def _save_budget(_: Any, budget: TokenBudget) -> None:
        saved_budgets.append(budget)

    service = CompactionOwnerService(
        coordinator=_StubCoordinator(_CompactResult(plan=None, tokens_freed=0, proposals_applied=0)),  # type: ignore[arg-type]
        tokenizer=get_tokenizer(),
        save_budget=_save_budget,
    )
    ctx = _FakeContext()
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=900)

    result = await service.try_apply(ctx=ctx, budget=budget, current_tokens=900, turn_number=2)
    assert result.applied is False
    assert result.fallback_reason == OwnerFallbackReason.MISSING_SHARED_TURN_BUFFER
    assert saved_budgets == []


@pytest.mark.asyncio
async def test_compaction_owner_service_applies_direct_clear_fallback() -> None:
    saved_budgets: list[TokenBudget] = []

    async def _save_budget(_: Any, budget: TokenBudget) -> None:
        saved_budgets.append(TokenBudget.from_dict(budget.to_dict()))

    assistant = ChatMessage(
        role="assistant",
        contents=[FunctionCallContent(call_id="c1", name="read_file", arguments={"path": "x"})],
        message_id="a1",
    )
    tool = ChatMessage(
        role="tool",
        contents=[FunctionResultContent(call_id="c1", result="R" * 4000)],
        message_id="t1",
    )
    recent_user = ChatMessage(role="user", text="recent", message_id="u1")
    snapshot = {
        "version": 9,
        "message_count": 3,
        "messages": encode_chat_messages([tool, assistant, recent_user]),
    }
    ctx = _FakeContext({HARNESS_SHARED_TURN_BUFFER_KEY: snapshot})

    empty_plan = CompactionPlan.create_empty(thread_id="t")
    service = CompactionOwnerService(
        coordinator=_StubCoordinator(_CompactResult(plan=empty_plan, tokens_freed=0, proposals_applied=0)),  # type: ignore[arg-type]
        tokenizer=get_tokenizer(),
        save_budget=_save_budget,
    )
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=900)

    result = await service.try_apply(ctx=ctx, budget=budget, current_tokens=900, turn_number=3)
    assert result.applied is True
    assert result.fallback_reason is None
    assert result.strategies_applied == ["clear"]
    assert result.shared_snapshot_version == 10
    assert (result.shared_message_count or 0) >= 1
    assert result.tokens_freed > 0
    assert result.proposals_applied >= 1
    updated_snapshot = await ctx.get_shared_state(HARNESS_SHARED_TURN_BUFFER_KEY)
    assert updated_snapshot["version"] == 10
    assert updated_snapshot["message_count"] >= 1
    assert await ctx.get_shared_state(HARNESS_COMPACTION_PLAN_KEY) is None
    assert len(saved_budgets) == 1
    assert saved_budgets[0].current_estimate < 900


def test_compaction_policy_decisions() -> None:
    policy = CompactionPolicy()
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=960)
    assert policy.is_under_pressure(budget) is True
    assert policy.is_blocking(budget) is True
    assert policy.should_attempt_owner_path("compaction_executor") is True
    assert policy.should_attempt_owner_path("agent_turn") is False
    assert policy.is_owner_fallback_allowed(
        owner_mode="compaction_executor",
        owner_fallback_reason="missing_shared_turn_buffer",
        turn_number=1,
    )
    assert not policy.is_owner_fallback_allowed(
        owner_mode="compaction_executor",
        owner_fallback_reason="missing_shared_turn_buffer",
        turn_number=5,
    )
    assert policy.is_owner_fallback_allowed(
        owner_mode="compaction_executor",
        owner_fallback_reason="owner_compaction_failed",
        turn_number=99,
    )
    assert policy.is_owner_fallback_gate_violation(
        owner_mode="compaction_executor",
        owner_fallback_reason="missing_shared_turn_buffer",
        turn_number=5,
    )
    ensured = policy.ensure_plan(None)
    assert isinstance(ensured, CompactionPlan)


@pytest.mark.asyncio
async def test_compaction_lifecycle_emitter_emits_expected_events() -> None:
    ctx = _FakeContext()
    emitter = CompactionLifecycleEmitter()
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=900)
    owner_outcome = OwnerCompactionOutcome(
        turn_number=2,
        tokens_before=900,
        tokens_freed=300,
        proposals_applied=1,
        strategies_applied=["clear"],
        owner_mode="compaction_executor",
        under_pressure=True,
    )

    await emitter.emit_owner_completed(ctx, turn_number=2, owner_outcome=owner_outcome)
    await emitter.emit_started(
        ctx,
        turn_number=2,
        current_tokens=900,
        budget=budget,
        strategies_available=["clear"],
        owner_mode="compaction_executor",
        owner_fallback_reason="empty_plan",
    )
    await emitter.emit_pressure(
        ctx,
        turn_number=2,
        plan_updated=False,
        tokens_freed=0,
        proposals_applied=0,
        owner_mode="compaction_executor",
        owner_fallback_reason="empty_plan",
    )

    event_types = [e.event_type for e in ctx.events if isinstance(e, HarnessLifecycleEvent)]
    assert event_types == ["compaction_completed", "compaction_started", "context_pressure"]
    assert ctx.events[0].data["owner_path_applied"] is True
    assert ctx.events[1].data["owner_fallback_reason"] == "empty_plan"
    assert ctx.events[2].data["owner_fallback_reason"] == "empty_plan"
    assert ctx.events[1].data["owner_fallback_allowed"] is None
    assert ctx.events[2].data["owner_fallback_gate_violation"] is None


def test_compaction_helpers_tokens_to_free_and_strategy_summary() -> None:
    assert tokens_to_free_for_pressure(tokens_over=0, max_input_tokens=1000) == 100
    assert tokens_to_free_for_pressure(tokens_over=250, max_input_tokens=1000) == 350

    empty = CompactionPlan.create_empty(thread_id="h")
    assert summarize_applied_strategies(empty) == []

    plan = CompactionPlan.create_empty(thread_id="h")
    plan.clearings.append(ClearRecord(span=SpanReference(message_ids=["m1"], first_turn=1, last_turn=1), preserved_fields={}))
    plan.drops.append(DropRecord(span=SpanReference(message_ids=["m2"], first_turn=1, last_turn=1), reason="x"))
    plan = CompactionPlan(
        thread_id=plan.thread_id,
        thread_version=plan.thread_version,
        created_at=plan.created_at,
        clearings=plan.clearings,
        drops=plan.drops,
    )
    assert summarize_applied_strategies(plan) == ["clear", "drop"]


@pytest.mark.asyncio
async def test_compaction_thread_service_skips_under_threshold() -> None:
    service = CompactionThreadService(
        coordinator=_StubCoordinator(_CompactResult(plan=CompactionPlan.create_empty(thread_id="h"), tokens_freed=0, proposals_applied=0)),  # type: ignore[arg-type]
        tokenizer=get_tokenizer(),
    )
    thread = CacheThreadAdapter([ChatMessage(role="user", text="short")])
    budget = TokenBudget(max_input_tokens=10000, soft_threshold_percent=0.9, current_estimate=0)
    result_plan = await service.compact_thread(thread, None, budget, 1)
    assert isinstance(result_plan, CompactionPlan)


@pytest.mark.asyncio
async def test_compaction_thread_service_runs_coordinator_when_over_threshold() -> None:
    expected_plan = CompactionPlan.create_empty(thread_id="h")
    service = CompactionThreadService(
        coordinator=_StubCoordinator(_CompactResult(plan=expected_plan, tokens_freed=500, proposals_applied=1)),  # type: ignore[arg-type]
        tokenizer=get_tokenizer(),
    )
    thread = CacheThreadAdapter([ChatMessage(role="user", text="X" * 12000)])
    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=0)
    result_plan = await service.compact_thread(thread, None, budget, 2)
    assert result_plan is expected_plan
