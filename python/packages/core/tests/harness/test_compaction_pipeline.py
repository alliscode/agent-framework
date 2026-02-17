# Copyright (c) Microsoft. All rights reserved.

"""Tests for Phase 4 — Full Compaction Pipeline Wiring.

These tests cover:
- CacheThreadAdapter and CacheMessageStore
- _run_full_compaction pathway in AgentTurnExecutor
- Dual thresholds (soft + blocking) in TokenBudget
- CompactionComplete.blocking signal
"""

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

import pytest

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Executor,
    FunctionResultContent,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework._workflows._conversation_state import encode_chat_messages
from agent_framework._harness import (
    CompactionComplete,
    CompactionExecutor,
    HarnessStatus,
    OwnerCompactionResult,
    RepairComplete,
    TokenBudget,
)
from agent_framework._harness._compaction import (
    CacheMessageStore,
    CacheThreadAdapter,
    CompactionPlan,
)
from agent_framework._harness._constants import (
    HARNESS_COMPACTION_METRICS_KEY,
    HARNESS_COMPACTION_OWNER_MODE_KEY,
    HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_SHARED_TURN_BUFFER_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TURN_COUNT_KEY,
)

# ============================================================================
# CacheThreadAdapter Tests
# ============================================================================


class TestCacheMessageStore:
    """Tests for CacheMessageStore."""

    @pytest.mark.asyncio
    async def test_list_messages_returns_copy(self) -> None:
        """Test that list_messages returns a copy of the cache."""
        msg1 = ChatMessage(role="user", text="Hello")
        msg2 = ChatMessage(role="assistant", text="Hi there")
        cache: list[Any] = [msg1, msg2]

        store = CacheMessageStore(cache)
        messages = await store.list_messages()

        assert len(messages) == 2
        assert messages[0] is msg1
        assert messages[1] is msg2
        # Verify it's a copy (modifying returned list doesn't affect cache)
        messages.append(ChatMessage(role="user", text="Extra"))
        assert len(cache) == 2

    @pytest.mark.asyncio
    async def test_list_messages_empty_cache(self) -> None:
        """Test list_messages with an empty cache."""
        store = CacheMessageStore([])
        messages = await store.list_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_add_messages_is_noop(self) -> None:
        """Test that add_messages does not modify the cache."""
        cache: list[Any] = [ChatMessage(role="user", text="Hello")]
        store = CacheMessageStore(cache)

        await store.add_messages([ChatMessage(role="assistant", text="New")])
        assert len(cache) == 1  # Cache unchanged


class TestCacheThreadAdapter:
    """Tests for CacheThreadAdapter."""

    def test_has_message_store(self) -> None:
        """Test that adapter exposes a message_store attribute."""
        cache: list[Any] = [ChatMessage(role="user", text="Hello")]
        adapter = CacheThreadAdapter(cache)

        assert adapter.message_store is not None
        assert isinstance(adapter.message_store, CacheMessageStore)

    @pytest.mark.asyncio
    async def test_message_store_reads_cache(self) -> None:
        """Test that message_store reads from the cache."""
        msg = ChatMessage(role="user", text="Test message")
        cache: list[Any] = [msg]
        adapter = CacheThreadAdapter(cache)

        messages = await adapter.message_store.list_messages()
        assert len(messages) == 1
        assert messages[0] is msg

    @pytest.mark.asyncio
    async def test_reflects_cache_mutations(self) -> None:
        """Test that adapter reflects changes to the underlying cache."""
        cache: list[Any] = []
        adapter = CacheThreadAdapter(cache)

        # Initially empty
        messages = await adapter.message_store.list_messages()
        assert len(messages) == 0

        # Add to cache externally (as AgentTurnExecutor does)
        cache.append(ChatMessage(role="user", text="Added"))
        messages = await adapter.message_store.list_messages()
        assert len(messages) == 1


# ============================================================================
# Dual Threshold Tests
# ============================================================================


class TestDualThresholds:
    """Tests for dual threshold support in TokenBudget."""

    def test_soft_threshold_default(self) -> None:
        """Test that soft threshold defaults to 80%."""
        budget = TokenBudget()
        assert budget.soft_threshold_percent == 0.80
        assert budget.soft_threshold == 102400

    def test_blocking_threshold_default(self) -> None:
        """Test that blocking threshold defaults to 95%."""
        budget = TokenBudget()
        assert budget.blocking_threshold_percent == 0.95
        assert budget.blocking_threshold == 121600

    def test_under_soft_threshold(self) -> None:
        """Test state when under soft threshold."""
        budget = TokenBudget(current_estimate=50000)
        assert not budget.is_under_pressure
        assert not budget.is_blocking

    def test_between_thresholds(self) -> None:
        """Test state between soft and blocking thresholds."""
        budget = TokenBudget(current_estimate=110000)
        assert budget.is_under_pressure
        assert not budget.is_blocking

    def test_above_blocking_threshold(self) -> None:
        """Test state above blocking threshold."""
        budget = TokenBudget(current_estimate=125000)
        assert budget.is_under_pressure
        assert budget.is_blocking

    def test_serialization_roundtrip(self) -> None:
        """Test that blocking_threshold_percent survives serialization."""
        budget = TokenBudget(
            max_input_tokens=50000,
            soft_threshold_percent=0.75,
            blocking_threshold_percent=0.90,
            current_estimate=40000,
        )
        data = budget.to_dict()
        restored = TokenBudget.from_dict(data)

        assert restored.soft_threshold_percent == 0.75
        assert restored.blocking_threshold_percent == 0.90
        assert restored.current_estimate == 40000
        assert restored.max_input_tokens == 50000

    def test_from_dict_defaults(self) -> None:
        """Test that from_dict provides defaults for new fields."""
        # Simulate loading old-format data without blocking_threshold_percent
        old_data = {
            "max_input_tokens": 100000,
            "soft_threshold_percent": 0.85,
            "current_estimate": 50000,
        }
        budget = TokenBudget.from_dict(old_data)
        assert budget.blocking_threshold_percent == 0.95  # default
        assert budget.soft_threshold_percent == 0.85  # preserved from data


# ============================================================================
# CompactionComplete.blocking Tests
# ============================================================================


class TestCompactionCompleteBlocking:
    """Tests for the blocking field on CompactionComplete."""

    def test_defaults_to_false(self) -> None:
        """Test that blocking defaults to False."""
        complete = CompactionComplete(repairs_made=0)
        assert not complete.blocking
        assert not complete.compaction_needed

    def test_blocking_true_when_set(self) -> None:
        """Test that blocking can be set to True."""
        complete = CompactionComplete(
            repairs_made=0,
            compaction_needed=True,
            blocking=True,
        )
        assert complete.blocking
        assert complete.compaction_needed


# ============================================================================
# Full Compaction Pipeline Integration Tests
# ============================================================================


class MockAgentForCompaction(BaseAgent):
    """A mock agent that produces large tool results to trigger compaction."""

    def __init__(self, *, name: str = "mock_agent"):
        super().__init__(name=name)

    def get_new_thread(self) -> AgentThread:
        return AgentThread()

    async def run(
        self,
        messages: list[Any],
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        response_message = ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text="Turn response")],
        )
        return AgentRunResponse(
            messages=[response_message],
            user_input_requests=[],
        )

    async def run_stream(
        self,
        messages: list[Any],
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        response = await self.run(messages, thread, **kwargs)
        for message in response.messages:
            yield AgentRunResponseUpdate(message=message)


@pytest.mark.asyncio
async def test_compaction_executor_signals_blocking() -> None:
    """Test that CompactionExecutor signals blocking when above blocking threshold."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        blocking: bool

    class TestCaptureExecutor(Executor):
        """Captures the CompactionComplete signal."""

        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    blocking=msg.blocking,
                )
            )

    class SetupExecutor(Executor):
        """Sets up state with high token count."""

        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            # Set up state simulating 96% utilization (above blocking threshold)
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=96000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 5)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                id="compaction",
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is True
    assert output.blocking is True


@pytest.mark.asyncio
async def test_compaction_executor_signals_nonblocking_at_soft_threshold() -> None:
    """Test that CompactionExecutor signals non-blocking at soft threshold."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        blocking: bool

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    blocking=msg.blocking,
                )
            )

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            # 85% utilization — above soft (80%) but below blocking (95%)
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=85000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 5)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                id="compaction",
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is True
    assert output.blocking is False


@pytest.mark.asyncio
async def test_compaction_executor_publishes_shadow_candidate_simulation() -> None:
    """Shadow mode publishes candidate simulation metadata from shared turn buffer snapshot."""

    @dataclass
    class TestOutput:
        candidate: dict[str, Any] | None

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            candidate = await ctx.get_shared_state(HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY)
            await ctx.yield_output(TestOutput(candidate=candidate if isinstance(candidate, dict) else None))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 3)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "shadow")

            snapshot_messages = [
                ChatMessage(role="user", text="Analyze files"),
                ChatMessage(
                    role="tool",
                    contents=[FunctionResultContent(call_id="c1", result="R" * 6000)],
                ),
            ]
            await ctx.set_shared_state(
                HARNESS_SHARED_TURN_BUFFER_KEY,
                {
                    "version": 4,
                    "message_count": len(snapshot_messages),
                    "messages": encode_chat_messages(snapshot_messages),
                },
            )
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                id="compaction",
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    candidate = outputs[0].candidate
    assert candidate is not None
    assert candidate.get("candidate_owner") == "compaction_executor"
    assert candidate.get("would_compact") is True
    assert "simulation_available" in candidate
    assert "candidate_effective_compaction" in candidate


@pytest.mark.asyncio
async def test_compaction_executor_owner_mode_falls_back_without_shared_buffer() -> None:
    """Owner mode should fall back to agent_turn path when shared buffer is missing."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        plan_updated: bool

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    plan_updated=msg.plan_updated,
                )
            )

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 2)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "compaction_executor")
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: CompactionExecutor(id="compaction"), name="compaction")
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is True
    assert output.plan_updated is False


@pytest.mark.asyncio
async def test_compaction_executor_owner_fallback_gate_violation_after_bootstrap() -> None:
    """Owner fallback outside bootstrap window should be marked as gate violation."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        fallback_allowed: bool | None
        gate_violation: bool | None

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            metrics = await ctx.get_shared_state(HARNESS_COMPACTION_METRICS_KEY)
            latest = metrics[-1] if isinstance(metrics, list) and metrics else {}
            fallback_allowed = latest.get("owner_fallback_allowed") if isinstance(latest, dict) else None
            gate_violation = latest.get("owner_fallback_gate_violation") if isinstance(latest, dict) else None
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    fallback_allowed=fallback_allowed,
                    gate_violation=gate_violation,
                )
            )

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 6)  # beyond bootstrap fallback turns
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "compaction_executor")
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                id="compaction",
                owner_bootstrap_fallback_turn_limit=2,
                enforce_owner_fallback_gate=True,
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is True
    assert output.fallback_allowed is False
    assert output.gate_violation is True


@pytest.mark.asyncio
async def test_compaction_executor_owner_mode_canary_path_applies_when_available() -> None:
    """Owner mode canary path sets compaction_needed=False when owner path applies."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        plan_updated: bool
        tokens_freed: int

    class CanaryCompactionExecutor(CompactionExecutor):
        async def _try_compaction_executor_ownership(self, **kwargs: Any) -> OwnerCompactionResult:
            return OwnerCompactionResult.applied_result(
                tokens_freed=123,
                proposals_applied=2,
                strategies_applied=["clear"],
                shared_snapshot_version=1,
                shared_message_count=1,
            )

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    plan_updated=msg.plan_updated,
                    tokens_freed=msg.tokens_freed,
                )
            )

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 2)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "compaction_executor")
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: CanaryCompactionExecutor(id="compaction"), name="compaction")
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is False
    assert output.plan_updated is True
    assert output.tokens_freed == 123


@pytest.mark.asyncio
async def test_compaction_executor_respects_injected_policy_for_owner_attempt() -> None:
    """Injected policy can disable owner-path attempt even in compaction_executor mode."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        plan_updated: bool

    class DisableOwnerAttemptPolicy:
        def is_under_pressure(self, budget: TokenBudget) -> bool:
            return budget.is_under_pressure

        def should_attempt_owner_path(self, owner_mode: str) -> bool:
            return False

        def is_blocking(self, budget: TokenBudget) -> bool:
            return budget.is_blocking

        def ensure_plan(self, plan: Any) -> Any:
            from agent_framework._harness._compaction import CompactionPlan

            return plan if plan is not None else CompactionPlan.create_empty(thread_id="harness")

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    plan_updated=msg.plan_updated,
                )
            )

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 2)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "compaction_executor")
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                id="compaction",
                policy=DisableOwnerAttemptPolicy(),  # type: ignore[arg-type]
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is True
    assert output.plan_updated is False


@pytest.mark.asyncio
async def test_compaction_executor_uses_injected_owner_service() -> None:
    """Injected owner service can drive owner-path completion without subclassing."""

    @dataclass
    class TestOutput:
        compaction_needed: bool
        plan_updated: bool
        tokens_freed: int

    class StubOwnerService:
        async def try_apply(self, **kwargs: Any) -> OwnerCompactionResult:
            return OwnerCompactionResult.applied_result(
                tokens_freed=222,
                proposals_applied=3,
                strategies_applied=["clear"],
                shared_snapshot_version=1,
                shared_message_count=1,
            )

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(
                TestOutput(
                    compaction_needed=msg.compaction_needed,
                    plan_updated=msg.plan_updated,
                    tokens_freed=msg.tokens_freed,
                )
            )

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 2)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "compaction_executor")
            await ctx.send_message(RepairComplete(repairs_made=0))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                id="compaction",
                owner_service=StubOwnerService(),  # type: ignore[arg-type]
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    output = outputs[0]
    assert output.compaction_needed is False
    assert output.plan_updated is True
    assert output.tokens_freed == 222


@pytest.mark.asyncio
async def test_compaction_executor_compact_thread_uses_injected_thread_service() -> None:
    """compact_thread delegates to injected thread service."""

    class StubThreadService:
        def __init__(self, plan: Any) -> None:
            self.plan = plan
            self.called = False

        async def compact_thread(self, thread: Any, current_plan: Any, budget: Any, turn_number: int) -> Any:
            self.called = True
            assert turn_number == 7
            return self.plan

    expected_plan = CompactionPlan.create_empty(thread_id="harness")
    stub = StubThreadService(expected_plan)
    executor = CompactionExecutor(thread_service=stub)  # type: ignore[arg-type]

    class _NoStoreThread:
        message_store = None

    budget = TokenBudget(max_input_tokens=1000, soft_threshold_percent=0.5, current_estimate=900)
    result_plan = await executor.compact_thread(_NoStoreThread(), None, budget, 7)  # type: ignore[arg-type]
    assert stub.called is True
    assert result_plan is expected_plan


@pytest.mark.asyncio
async def test_compaction_executor_uses_injected_lifecycle_on_fallback_path() -> None:
    """Injected lifecycle emitter is used for started/pressure on fallback flow."""

    @dataclass
    class TestOutput:
        compaction_needed: bool

    class StubLifecycle:
        def __init__(self) -> None:
            self.started = 0
            self.pressure = 0
            self.owner_completed = 0

        async def emit_started(self, *args: Any, **kwargs: Any) -> None:
            self.started += 1

        async def emit_pressure(self, *args: Any, **kwargs: Any) -> None:
            self.pressure += 1

        async def emit_owner_completed(self, *args: Any, **kwargs: Any) -> None:
            self.owner_completed += 1

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(TestOutput(compaction_needed=msg.compaction_needed))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 2)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "agent_turn")
            await ctx.send_message(RepairComplete(repairs_made=0))

    lifecycle = StubLifecycle()
    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: CompactionExecutor(id="compaction", lifecycle=lifecycle), name="compaction")  # type: ignore[arg-type]
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].compaction_needed is True
    assert lifecycle.started == 1
    assert lifecycle.pressure == 1
    assert lifecycle.owner_completed == 0


@pytest.mark.asyncio
async def test_compaction_executor_uses_injected_lifecycle_on_owner_path() -> None:
    """Injected lifecycle emitter is used for owner completion flow."""

    @dataclass
    class TestOutput:
        compaction_needed: bool

    class StubLifecycle:
        def __init__(self) -> None:
            self.started = 0
            self.pressure = 0
            self.owner_completed = 0

        async def emit_started(self, *args: Any, **kwargs: Any) -> None:
            self.started += 1

        async def emit_pressure(self, *args: Any, **kwargs: Any) -> None:
            self.pressure += 1

        async def emit_owner_completed(self, *args: Any, **kwargs: Any) -> None:
            self.owner_completed += 1

    class StubOwnerService:
        async def try_apply(self, **kwargs: Any) -> OwnerCompactionResult:
            return OwnerCompactionResult.applied_result(
                tokens_freed=10,
                proposals_applied=1,
                strategies_applied=["clear"],
                shared_snapshot_version=1,
                shared_message_count=1,
            )

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(TestOutput(compaction_needed=msg.compaction_needed))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=90000,
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 2)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "compaction_executor")
            await ctx.send_message(RepairComplete(repairs_made=0))

    lifecycle = StubLifecycle()
    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                id="compaction",
                owner_service=StubOwnerService(),  # type: ignore[arg-type]
                lifecycle=lifecycle,  # type: ignore[arg-type]
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].compaction_needed is False
    assert lifecycle.owner_completed == 1
    assert lifecycle.started == 0
    assert lifecycle.pressure == 0


@pytest.mark.asyncio
async def test_compaction_executor_uses_injected_shadow_service() -> None:
    """Injected shadow service is invoked during compaction check."""

    @dataclass
    class TestOutput:
        compaction_needed: bool

    class StubShadowService:
        def __init__(self) -> None:
            self.calls = 0

        async def publish_candidate(self, **kwargs: Any) -> None:
            self.calls += 1

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(TestOutput(compaction_needed=msg.compaction_needed))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            budget = TokenBudget(
                max_input_tokens=100000,
                soft_threshold_percent=0.80,
                blocking_threshold_percent=0.95,
                current_estimate=1000,  # under threshold
            )
            await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 1)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, "shadow")
            await ctx.send_message(RepairComplete(repairs_made=0))

    stub_shadow = StubShadowService()
    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                id="compaction",
                shadow_service=stub_shadow,  # type: ignore[arg-type]
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].compaction_needed is False
    assert stub_shadow.calls == 1


@pytest.mark.asyncio
async def test_compaction_executor_uses_injected_state_store() -> None:
    """Injected state store is used for control-flow reads/writes."""

    @dataclass
    class TestOutput:
        compaction_needed: bool

    class StubStateStore:
        def __init__(self) -> None:
            self.get_budget_calls = 0
            self.get_turn_calls = 0
            self.get_mode_calls = 0
            self.load_plan_calls = 0
            self.save_budget_calls = 0

        async def get_or_create_budget(self, ctx: Any) -> TokenBudget:
            self.get_budget_calls += 1
            return TokenBudget(max_input_tokens=100000, soft_threshold_percent=0.80, current_estimate=1000)

        async def get_turn_count(self, ctx: Any) -> int:
            self.get_turn_calls += 1
            return 3

        async def get_owner_mode(self, ctx: Any) -> str:
            self.get_mode_calls += 1
            return "agent_turn"

        async def load_plan(self, ctx: Any) -> tuple[Any, int]:
            self.load_plan_calls += 1
            return None, 0

        async def save_budget(self, ctx: Any, budget: Any) -> None:
            self.save_budget_calls += 1

        async def save_plan(self, ctx: Any, plan: Any, version: int) -> None:
            return None

        async def append_metrics(self, ctx: Any, metrics: Any) -> None:
            return None

    class TestCaptureExecutor(Executor):
        @handler
        async def handle(self, msg: CompactionComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            await ctx.yield_output(TestOutput(compaction_needed=msg.compaction_needed))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairComplete]) -> None:
            await ctx.send_message(RepairComplete(repairs_made=0))

    store = StubStateStore()
    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(
            lambda: CompactionExecutor(
                id="compaction",
                state_store=store,  # type: ignore[arg-type]
            ),
            name="compaction",
        )
        .register_executor(lambda: TestCaptureExecutor(id="capture"), name="capture")
        .add_edge("setup", "compaction")
        .add_edge("compaction", "capture")
        .set_start_executor("setup")
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].compaction_needed is False
    assert store.get_budget_calls == 1
    assert store.get_turn_calls == 1
    assert store.get_mode_calls == 1
    assert store.load_plan_calls == 1
    assert store.save_budget_calls == 1


# ============================================================================
# Phase 4e — ChatClientSummarizer Tests
# ============================================================================


class FakeChatResponse:
    """Minimal fake ChatResponse for testing."""

    def __init__(self, text: str) -> None:
        self.messages = [ChatMessage(role="assistant", text=text)]


class FakeChatClient:
    """Fake chat client that returns a configurable response."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.call_count = 0
        self.last_messages: list[Any] = []

    @property
    def additional_properties(self) -> dict[str, Any]:
        return {}

    async def get_response(self, messages: Any, **kwargs: Any) -> FakeChatResponse:
        self.call_count += 1
        self.last_messages = messages
        return FakeChatResponse(self._response_text)

    def get_streaming_response(self, messages: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class TestChatClientSummarizer:
    """Tests for the ChatClientSummarizer."""

    @pytest.mark.asyncio
    async def test_summarize_returns_structured_summary(self) -> None:
        """Test that summarize calls the LLM and returns a StructuredSummary."""
        from agent_framework._harness._compaction._summarizer import ChatClientSummarizer

        llm_response = json.dumps({
            "facts": ["The user wants to build a web app"],
            "decisions": [{"decision": "Use React", "rationale": "User preference", "turn_number": 1}],
            "open_items": [{"description": "Set up CI/CD", "context": "Needed for deployment", "priority": "high"}],
            "tool_outcomes": [{"tool_name": "grep", "outcome": "success", "key_fields": {"matches": 5}}],
            "current_task": "Building the frontend",
            "current_plan": ["Create components", "Add routing"],
        })
        client = FakeChatClient(llm_response)
        summarizer = ChatClientSummarizer(client)  # type: ignore[arg-type]

        messages = [
            ChatMessage(role="user", text="Build me a web app", message_id="m1"),
            ChatMessage(role="assistant", text="Sure, I'll use React.", message_id="m2"),
            ChatMessage(role="user", text="Add routing too", message_id="m3"),
        ]

        summary = await summarizer.summarize(messages, target_token_ratio=0.25)

        assert client.call_count == 1
        assert "The user wants to build a web app" in summary.facts
        assert len(summary.decisions) == 1
        assert summary.decisions[0].decision == "Use React"
        assert len(summary.open_items) == 1
        assert summary.open_items[0].priority == "high"
        assert len(summary.tool_outcomes) == 1
        assert summary.tool_outcomes[0].tool_name == "grep"
        assert summary.current_task == "Building the frontend"
        assert summary.current_plan == ["Create components", "Add routing"]
        assert summary.span.message_ids == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_summarize_preserves_required_facts(self) -> None:
        """Test that preserve_facts are included even if LLM omits them."""
        from agent_framework._harness._compaction._summarizer import ChatClientSummarizer

        llm_response = json.dumps({"facts": ["LLM fact"]})
        client = FakeChatClient(llm_response)
        summarizer = ChatClientSummarizer(client)  # type: ignore[arg-type]

        messages = [ChatMessage(role="user", text="Hello", message_id="m1")]
        summary = await summarizer.summarize(messages, preserve_facts=["Must keep this"])

        assert "Must keep this" in summary.facts
        assert "LLM fact" in summary.facts

    @pytest.mark.asyncio
    async def test_summarize_handles_llm_failure(self) -> None:
        """Test that summarize returns empty summary on LLM failure."""
        from agent_framework._harness._compaction._summarizer import ChatClientSummarizer

        class FailingClient(FakeChatClient):
            async def get_response(self, messages: Any, **kwargs: Any) -> Any:
                raise RuntimeError("LLM unavailable")

        client = FailingClient("unused")
        summarizer = ChatClientSummarizer(client)  # type: ignore[arg-type]

        messages = [ChatMessage(role="user", text="Hello", message_id="m1")]
        summary = await summarizer.summarize(messages)

        assert summary.facts == []
        assert summary.decisions == []
        assert summary.span.message_ids == ["m1"]

    @pytest.mark.asyncio
    async def test_summarize_handles_invalid_json(self) -> None:
        """Test that summarize returns empty summary on invalid JSON."""
        from agent_framework._harness._compaction._summarizer import ChatClientSummarizer

        client = FakeChatClient("not valid json {{{")
        summarizer = ChatClientSummarizer(client)  # type: ignore[arg-type]

        messages = [ChatMessage(role="user", text="Hello", message_id="m1")]
        summary = await summarizer.summarize(messages)

        assert summary.facts == []
        assert summary.span.message_ids == ["m1"]

    @pytest.mark.asyncio
    async def test_summarize_strips_markdown_fencing(self) -> None:
        """Test that markdown code fences are stripped before parsing."""
        from agent_framework._harness._compaction._summarizer import ChatClientSummarizer

        fenced = '```json\n{"facts": ["fenced fact"]}\n```'
        client = FakeChatClient(fenced)
        summarizer = ChatClientSummarizer(client)  # type: ignore[arg-type]

        messages = [ChatMessage(role="user", text="Hello", message_id="m1")]
        summary = await summarizer.summarize(messages)

        assert "fenced fact" in summary.facts

    @pytest.mark.asyncio
    async def test_summarize_passes_model_id(self) -> None:
        """Test that model_id override is passed to chat client."""
        from agent_framework._harness._compaction._summarizer import ChatClientSummarizer

        llm_response = json.dumps({"facts": []})

        class TrackingClient(FakeChatClient):
            def __init__(self) -> None:
                super().__init__(llm_response)
                self.last_kwargs: dict[str, Any] = {}

            async def get_response(self, messages: Any, **kwargs: Any) -> FakeChatResponse:
                self.last_kwargs = kwargs
                return await super().get_response(messages, **kwargs)

        client = TrackingClient()
        summarizer = ChatClientSummarizer(client, model_id="gpt-4o-mini")  # type: ignore[arg-type]

        messages = [ChatMessage(role="user", text="Hello", message_id="m1")]
        await summarizer.summarize(messages)

        assert client.last_kwargs.get("model_id") == "gpt-4o-mini"


# ============================================================================
# Phase 4e — Turn 0 Protection Tests
# ============================================================================


class TestTurn0Protection:
    """Tests that turn 0 (original user request) is protected from summarization."""

    @pytest.mark.asyncio
    async def test_turn0_excluded_from_summarization(self) -> None:
        """Turn 0 messages should never appear in summarization proposals."""
        from agent_framework._harness._compaction import (
            SimpleTokenizer,
            SummarizeStrategy,
            TokenBudget,
        )

        class NoopSummarizer:
            async def summarize(self, messages: Any, **kwargs: Any) -> Any:
                raise AssertionError("Should not be called")

        strategy = SummarizeStrategy(
            NoopSummarizer(),  # type: ignore[arg-type]
            preserve_recent_messages=4,
            min_span_messages=1,
            min_span_tokens=1,
        )

        # Build 8 turns worth of messages — first user message should be protected
        messages = []
        for turn in range(8):
            messages.append(ChatMessage(role="user", text=f"User turn {turn}", message_id=f"u{turn}"))
            messages.append(ChatMessage(role="assistant", text=f"Assistant turn {turn}", message_id=f"a{turn}"))

        adapter = CacheThreadAdapter(messages)
        tokenizer = SimpleTokenizer()
        budget = TokenBudget()

        proposals = await strategy.analyze(adapter, None, budget, tokenizer)

        proposed_ids: set[str] = set()
        for p in proposals:
            proposed_ids.update(p.span.message_ids)

        assert "u0" not in proposed_ids, "First user message should be protected"

    @pytest.mark.asyncio
    async def test_turn0_protected_even_with_many_turns(self) -> None:
        """With many turns, first user message stays protected while other messages are summarizable."""
        from agent_framework._harness._compaction import (
            SimpleTokenizer,
            SummarizeStrategy,
            TokenBudget,
        )

        class NoopSummarizer:
            async def summarize(self, messages: Any, **kwargs: Any) -> Any:
                raise AssertionError("Should not be called")

        strategy = SummarizeStrategy(
            NoopSummarizer(),  # type: ignore[arg-type]
            preserve_recent_messages=4,
            min_span_messages=1,
            min_span_tokens=1,
        )

        # 15 turns (30 messages), preserve last 4 → 26 candidates minus first user msg
        messages = []
        for turn in range(15):
            messages.append(ChatMessage(role="user", text=f"User turn {turn} " * 20, message_id=f"u{turn}"))
            messages.append(ChatMessage(role="assistant", text=f"Asst turn {turn} " * 20, message_id=f"a{turn}"))

        adapter = CacheThreadAdapter(messages)
        tokenizer = SimpleTokenizer()
        budget = TokenBudget()

        proposals = await strategy.analyze(adapter, None, budget, tokenizer)

        proposed_ids: set[str] = set()
        for p in proposals:
            proposed_ids.update(p.span.message_ids)

        assert "u0" not in proposed_ids
        assert len(proposals) > 0, "Should have proposals for middle messages"

    @pytest.mark.asyncio
    async def test_no_proposals_when_only_recent_and_turn0(self) -> None:
        """With only a few messages, nothing should be summarizable."""
        from agent_framework._harness._compaction import (
            SimpleTokenizer,
            SummarizeStrategy,
            TokenBudget,
        )

        class NoopSummarizer:
            async def summarize(self, messages: Any, **kwargs: Any) -> Any:
                raise AssertionError("Should not be called")

        strategy = SummarizeStrategy(
            NoopSummarizer(),  # type: ignore[arg-type]
            preserve_recent_messages=12,
            min_span_messages=1,
            min_span_tokens=1,
        )

        # 6 turns = 12 messages. With preserve_recent_messages=12, everything is recent.
        messages = []
        for turn in range(6):
            messages.append(ChatMessage(role="user", text=f"Turn {turn}", message_id=f"u{turn}"))
            messages.append(ChatMessage(role="assistant", text=f"Reply {turn}", message_id=f"a{turn}"))

        adapter = CacheThreadAdapter(messages)
        tokenizer = SimpleTokenizer()
        budget = TokenBudget()

        proposals = await strategy.analyze(adapter, None, budget, tokenizer)
        assert len(proposals) == 0


# ============================================================================
# Phase 4e — Pipeline Wiring Tests
# ============================================================================


class TestSummarizeStrategyWiring:
    """Tests that SummarizeStrategy is wired into _run_full_compaction."""

    def test_executor_accepts_summarizer_param(self) -> None:
        """AgentTurnExecutor.__init__ accepts a summarizer parameter."""
        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        class DummyAgent(BaseAgent):
            async def run(self, *args: Any, **kwargs: Any) -> AgentRunResponse:
                return AgentRunResponse(messages=[], response_id="r")

            def run_stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[AgentRunResponseUpdate]:
                raise NotImplementedError

        class DummySummarizer:
            async def summarize(self, messages: Any, **kwargs: Any) -> Any:
                pass

        executor = AgentTurnExecutor(
            DummyAgent(),
            enable_compaction=True,
            summarizer=DummySummarizer(),  # type: ignore[arg-type]
        )
        assert executor._summarizer is not None

    def test_executor_summarizer_default_none(self) -> None:
        """AgentTurnExecutor defaults to None summarizer."""
        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        class DummyAgent(BaseAgent):
            async def run(self, *args: Any, **kwargs: Any) -> AgentRunResponse:
                return AgentRunResponse(messages=[], response_id="r")

            def run_stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[AgentRunResponseUpdate]:
                raise NotImplementedError

        executor = AgentTurnExecutor(DummyAgent(), enable_compaction=True)
        assert executor._summarizer is None
