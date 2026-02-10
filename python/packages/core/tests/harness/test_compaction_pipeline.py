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
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework._harness import (
    CompactionComplete,
    CompactionExecutor,
    HarnessStatus,
    RepairComplete,
    TokenBudget,
)
from agent_framework._harness._compaction import (
    CacheMessageStore,
    CacheThreadAdapter,
)
from agent_framework._harness._constants import (
    HARNESS_MAX_TURNS_KEY,
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
            preserve_recent_turns=2,
            min_span_messages=1,
            min_span_tokens=1,
        )

        # Build 8 turns worth of messages — turn 0 should be protected
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

        assert "u0" not in proposed_ids, "Turn 0 user message should be protected"
        assert "a0" not in proposed_ids, "Turn 0 assistant message should be protected"

    @pytest.mark.asyncio
    async def test_turn0_protected_even_with_many_turns(self) -> None:
        """With many turns, turn 0 stays protected while middle turns are summarizable."""
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
            preserve_recent_turns=2,
            min_span_messages=1,
            min_span_tokens=1,
        )

        # 15 turns
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
        assert "a0" not in proposed_ids
        # Middle turns should be proposed for summarization
        assert len(proposals) > 0, "Should have proposals for middle turns"

    @pytest.mark.asyncio
    async def test_no_proposals_when_only_recent_and_turn0(self) -> None:
        """With only turn 0 + recent turns, no summarization should happen."""
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
            preserve_recent_turns=6,
            min_span_messages=1,
            min_span_tokens=1,
        )

        # 6 user messages produce 7 turn groups (due to user-boundary grouping).
        # With preserve_recent_turns=6, only turns[:1] is a candidate.
        # Turn 0 is protected → nothing to summarize.
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
