# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness Context Compaction module (v2).

These tests cover:
- Core types: SpanReference, CompactionPlan, records
- Tokenizer: TiktokenTokenizer, SimpleTokenizer, TokenBudget
- Storage: InMemoryCompactionStore, InMemorySummaryCache, InMemoryArtifactStore
- Events: CompactionMetrics
"""

from datetime import datetime, timezone

import pytest

from agent_framework._harness import (
    # Core types
    ClearRecord,
    CompactionAction,
    # Event types
    CompactionMetrics,
    CompactionPlan,
    # Summary types
    Decision,
    DropRecord,
    # Tokenizer types
    OpenItem,
    SimpleTokenizer,
    SpanReference,
    StructuredSummary,
    SummaryCacheKey,
    TokenBudgetV2,
    ToolOutcome,
)
from agent_framework._harness._compaction import (
    # Storage types
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    compute_content_hash,
    create_summary_cache_key,
)

# ============================================================================
# SpanReference Tests
# ============================================================================


class TestSpanReference:
    """Tests for SpanReference."""

    def test_create_span_reference(self) -> None:
        """Test creating a span reference with message IDs."""
        span = SpanReference(
            message_ids=["msg-1", "msg-2", "msg-3"],
            first_turn=1,
            last_turn=3,
        )

        assert span.message_ids == ["msg-1", "msg-2", "msg-3"]
        assert span.message_count == 3
        assert span.first_turn == 1
        assert span.last_turn == 3

    def test_span_reference_properties(self) -> None:
        """Test span reference property methods."""
        span = SpanReference(
            message_ids=["msg-a", "msg-b", "msg-c"],
            first_turn=1,
            last_turn=2,
        )

        assert span.start_message_id == "msg-a"
        assert span.end_message_id == "msg-c"

    def test_span_reference_contains(self) -> None:
        """Test span reference contains method."""
        span = SpanReference(
            message_ids=["msg-1", "msg-2"],
            first_turn=1,
            last_turn=1,
        )

        assert span.contains("msg-1")
        assert span.contains("msg-2")
        assert not span.contains("msg-3")

    def test_span_reference_empty_raises(self) -> None:
        """Test that empty message_ids raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SpanReference(
                message_ids=[],
                first_turn=1,
                last_turn=1,
            )


# ============================================================================
# CompactionPlan Tests
# ============================================================================


class TestCompactionPlan:
    """Tests for CompactionPlan."""

    def test_create_empty_plan(self) -> None:
        """Test creating an empty compaction plan."""
        plan = CompactionPlan.create_empty("thread-123", thread_version=0)

        assert plan.thread_id == "thread-123"
        assert plan.thread_version == 0
        assert len(plan.clearings) == 0
        assert len(plan.summarizations) == 0
        assert len(plan.externalizations) == 0
        assert len(plan.drops) == 0
        assert plan.is_empty

    def test_plan_with_clear_record(self) -> None:
        """Test plan with a clear record."""
        span = SpanReference(
            message_ids=["msg-1", "msg-2"],
            first_turn=1,
            last_turn=2,
        )
        clear = ClearRecord(
            span=span,
            preserved_fields={"tool_name": "read_file"},
        )

        plan = CompactionPlan(
            thread_id="thread-123",
            thread_version=1,
            created_at=datetime.now(timezone.utc),
            clearings=[clear],
        )

        assert len(plan.clearings) == 1
        assert plan.clearings[0].preserved_fields == {"tool_name": "read_file"}
        assert not plan.is_empty

    def test_plan_get_action_for_message(self) -> None:
        """Test getting the action for a specific message."""
        span = SpanReference(
            message_ids=["msg-1", "msg-2"],
            first_turn=1,
            last_turn=1,
        )
        clear = ClearRecord(
            span=span,
            preserved_fields={"tool_name": "read_file"},
        )

        plan = CompactionPlan(
            thread_id="thread-123",
            thread_version=1,
            created_at=datetime.now(timezone.utc),
            clearings=[clear],
        )

        action, record = plan.get_action("msg-1")
        assert action == CompactionAction.CLEAR
        assert record == clear

        action2, record2 = plan.get_action("msg-unknown")
        assert action2 == CompactionAction.INCLUDE
        assert record2 is None

    def test_compaction_precedence(self) -> None:
        """Test that higher precedence actions take priority."""
        span = SpanReference(
            message_ids=["msg-1"],
            first_turn=1,
            last_turn=1,
        )

        # Create both clear and drop for same message
        clear = ClearRecord(
            span=span,
            preserved_fields={},
        )
        drop = DropRecord(
            span=span,
            reason="test",
        )

        plan = CompactionPlan(
            thread_id="thread-123",
            thread_version=1,
            created_at=datetime.now(timezone.utc),
            clearings=[clear],
            drops=[drop],
        )

        # DROP should take precedence over CLEAR
        action, _ = plan.get_action("msg-1")
        assert action == CompactionAction.DROP


# ============================================================================
# StructuredSummary Tests
# ============================================================================


class TestStructuredSummary:
    """Tests for StructuredSummary."""

    def test_create_structured_summary(self) -> None:
        """Test creating a structured summary."""
        span = SpanReference(
            message_ids=["msg-1", "msg-2"],
            first_turn=1,
            last_turn=2,
        )
        summary = StructuredSummary(
            span=span,
            facts=["User prefers Python"],
            decisions=[
                Decision(
                    decision="Use Python for implementation",
                    rationale="User's existing codebase is Python",
                    turn_number=1,
                )
            ],
            open_items=[
                OpenItem(
                    description="Add unit tests",
                    context="Tests for new feature",
                    priority="high",
                )
            ],
            tool_outcomes=[
                ToolOutcome(
                    tool_name="read_file",
                    outcome="success",
                )
            ],
        )

        assert len(summary.decisions) == 1
        assert len(summary.open_items) == 1
        assert len(summary.tool_outcomes) == 1
        assert summary.facts == ["User prefers Python"]


# ============================================================================
# Tokenizer Tests
# ============================================================================


class TestSimpleTokenizer:
    """Tests for SimpleTokenizer."""

    def test_simple_tokenizer_count(self) -> None:
        """Test simple tokenizer counting."""
        tokenizer = SimpleTokenizer(chars_per_token=4.0)

        # 20 characters / 4 = 5 tokens (rounded)
        count = tokenizer.count_tokens("Hello, world! Test.")
        assert count == 5  # 19 chars / 4 = 4.75 -> 5

    def test_simple_tokenizer_empty_string(self) -> None:
        """Test simple tokenizer with empty string."""
        tokenizer = SimpleTokenizer()
        assert tokenizer.count_tokens("") == 0


class TestTokenBudgetV2:
    """Tests for TokenBudgetV2."""

    def test_budget_defaults(self) -> None:
        """Test default budget values."""
        budget = TokenBudgetV2()

        assert budget.max_input_tokens == 128_000
        assert budget.soft_threshold_percent == 0.85
        assert budget.safety_buffer_tokens == 500

    def test_budget_soft_threshold(self) -> None:
        """Test budget soft threshold calculation."""
        budget = TokenBudgetV2(max_input_tokens=10000, soft_threshold_percent=0.80)

        # 10000 * 0.80 = 8000
        assert budget.soft_threshold == 8000

    def test_budget_pressure_detection(self) -> None:
        """Test budget pressure detection."""
        budget = TokenBudgetV2(
            max_input_tokens=10000,
            soft_threshold_percent=0.80,
            system_prompt_tokens=1000,
            tool_schema_tokens=500,
            safety_buffer_tokens=500,
            rehydration_reserve_tokens=500,
        )

        # Overhead = 1000 + 500 + 500 + 500 = 2500
        # Soft threshold = 10000 * 0.8 = 8000
        # Available = 8000 - 2500 = 5500

        assert not budget.is_under_pressure(5000)
        assert budget.is_under_pressure(6000)

    def test_budget_tokens_over_threshold(self) -> None:
        """Test tokens over threshold calculation."""
        budget = TokenBudgetV2(
            max_input_tokens=10000,
            soft_threshold_percent=0.80,
            safety_buffer_tokens=0,
            rehydration_reserve_tokens=0,
        )

        # With no overhead, available = 8000
        assert budget.tokens_over_threshold(7000) == 0
        assert budget.tokens_over_threshold(9000) > 0


# ============================================================================
# Storage Tests
# ============================================================================


class TestInMemoryCompactionStore:
    """Tests for InMemoryCompactionStore."""

    @pytest.mark.asyncio
    async def test_store_get_nonexistent(self) -> None:
        """Test getting a non-existent plan returns None."""
        store = InMemoryCompactionStore()

        plan = await store.get_plan("thread-123")

        assert plan is None

    @pytest.mark.asyncio
    async def test_store_save_and_get_plan(self) -> None:
        """Test saving and retrieving a plan."""
        store = InMemoryCompactionStore()
        plan = CompactionPlan.create_empty("thread-123", thread_version=0)

        await store.save_plan("thread-123", plan)
        stored_plan = await store.get_plan("thread-123")

        assert stored_plan is not None
        assert stored_plan.thread_id == "thread-123"

    @pytest.mark.asyncio
    async def test_store_overwrite_plan(self) -> None:
        """Test that saving a plan overwrites the previous one."""
        store = InMemoryCompactionStore()
        plan1 = CompactionPlan.create_empty("thread-123", thread_version=0)
        plan2 = CompactionPlan.create_empty("thread-123", thread_version=1)

        await store.save_plan("thread-123", plan1)
        await store.save_plan("thread-123", plan2)
        stored_plan = await store.get_plan("thread-123")

        assert stored_plan is not None
        assert stored_plan.thread_version == 1

    @pytest.mark.asyncio
    async def test_store_delete_plan(self) -> None:
        """Test deleting a plan."""
        store = InMemoryCompactionStore()
        plan = CompactionPlan.create_empty("thread-123", thread_version=0)

        await store.save_plan("thread-123", plan)
        deleted = await store.delete_plan("thread-123")

        assert deleted
        stored_plan = await store.get_plan("thread-123")
        assert stored_plan is None


class TestInMemorySummaryCache:
    """Tests for InMemorySummaryCache."""

    @pytest.mark.asyncio
    async def test_cache_get_miss(self) -> None:
        """Test cache miss returns None."""
        cache = InMemorySummaryCache()
        key = SummaryCacheKey(
            content_hash="abc123",
            schema_version="v1.0",
            policy_version="v1.0",
            model_id="gpt-4",
            prompt_version="v1.0",
        )

        result = await cache.get(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_put_and_get(self) -> None:
        """Test putting and getting a summary."""
        cache = InMemorySummaryCache()
        key = SummaryCacheKey(
            content_hash="abc123",
            schema_version="v1.0",
            policy_version="v1.0",
            model_id="gpt-4",
            prompt_version="v1.0",
        )
        span = SpanReference(message_ids=["msg-1"], first_turn=1, last_turn=1)
        summary = StructuredSummary(span=span, facts=["Test fact"])

        await cache.put(key, summary)
        result = await cache.get(key)

        assert result is not None
        assert result.facts == ["Test fact"]

    @pytest.mark.asyncio
    async def test_cache_invalidate(self) -> None:
        """Test invalidating a cached summary."""
        cache = InMemorySummaryCache()
        key = SummaryCacheKey(
            content_hash="abc123",
            schema_version="v1.0",
            policy_version="v1.0",
            model_id="gpt-4",
            prompt_version="v1.0",
        )
        span = SpanReference(message_ids=["msg-1"], first_turn=1, last_turn=1)
        summary = StructuredSummary(span=span, facts=["Test fact"])

        await cache.put(key, summary)
        invalidated = await cache.invalidate(key)

        assert invalidated
        result = await cache.get(key)
        assert result is None


class TestInMemoryArtifactStore:
    """Tests for InMemoryArtifactStore."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_artifact(self) -> None:
        """Test storing and retrieving an artifact."""
        store = InMemoryArtifactStore()

        artifact_id = await store.store("Hello, world!", {"type": "text"})
        content = await store.retrieve(artifact_id)

        assert content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_artifact(self) -> None:
        """Test retrieving non-existent artifact returns None."""
        store = InMemoryArtifactStore()

        content = await store.retrieve("nonexistent-id")
        assert content is None

    @pytest.mark.asyncio
    async def test_delete_artifact(self) -> None:
        """Test deleting an artifact."""
        store = InMemoryArtifactStore()

        artifact_id = await store.store("Test content", {})
        deleted = await store.delete(artifact_id)

        assert deleted
        content = await store.retrieve(artifact_id)
        assert content is None


# ============================================================================
# Event Tests
# ============================================================================


class TestCompactionMetrics:
    """Tests for CompactionMetrics."""

    def test_record_compaction(self) -> None:
        """Test recording a compaction cycle."""
        metrics = CompactionMetrics()

        metrics.record_compaction(
            tokens_freed=10000,
            proposals_applied=3,
            proposals_rejected=1,
            duration_ms=150.5,
        )

        assert metrics.total_compactions == 1
        assert metrics.total_tokens_freed == 10000
        assert metrics.total_proposals_applied == 3
        assert metrics.total_proposals_rejected == 1

    def test_record_summary_compression(self) -> None:
        """Test recording summary compression ratio."""
        metrics = CompactionMetrics()

        metrics.record_summary(0.25)  # 75% compression
        metrics.record_summary(0.30)  # 70% compression

        assert metrics.average_compression_ratio == pytest.approx(0.275)

    def test_record_rehydration(self) -> None:
        """Test recording rehydration events."""
        metrics = CompactionMetrics()

        metrics.record_rehydration(blocked=False)
        metrics.record_rehydration(blocked=True)
        metrics.record_rehydration(blocked=False)

        assert metrics.total_rehydrations == 2
        assert metrics.total_rehydrations_blocked == 1


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_content_hash(self) -> None:
        """Test content hash computation."""
        hash1 = compute_content_hash("Hello, world!")
        hash2 = compute_content_hash("Hello, world!")
        hash3 = compute_content_hash("Different content")

        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        # Hash should be 16 characters
        assert len(hash1) == 16

    def test_create_summary_cache_key(self) -> None:
        """Test creating a summary cache key."""
        span = SpanReference(
            message_ids=["msg-1"],
            first_turn=1,
            last_turn=1,
        )

        key = create_summary_cache_key(
            span=span,
            messages_content="Hello, this is a test message.",
            policy_version="v1.0",
            model_id="gpt-4",
            prompt_version="v1.0",
        )

        assert key.schema_version == "v1.0"
        assert key.policy_version == "v1.0"
        assert key.model_id == "gpt-4"
        assert len(key.content_hash) == 16
