# Copyright (c) Microsoft. All rights reserved.

"""Lifecycle events for context compaction observability.

This module provides:
- CompactionEvent base class for all compaction events
- Specific event types for each compaction operation
- CompactionMetrics for aggregated statistics
- EventEmitter protocol for observability integration

Event Flow:
1. compaction_check_started - Budget check begins
2. compaction_proposal_generated - Strategy proposes action
3. compaction_cleared / summarized / externalized / dropped - Action executed
4. compaction_rehydrated - Content restored to prompt
5. compaction_completed - Full compaction cycle done

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CompactionEventType(Enum):
    """Types of compaction events."""

    # Lifecycle events
    COMPACTION_CHECK_STARTED = "compaction_check_started"
    COMPACTION_COMPLETED = "compaction_completed"

    # Proposal events
    PROPOSAL_GENERATED = "compaction_proposal_generated"
    PROPOSAL_REJECTED = "compaction_proposal_rejected"

    # Action events
    CONTENT_CLEARED = "compaction_cleared"
    CONTENT_SUMMARIZED = "compaction_summarized"
    CONTENT_EXTERNALIZED = "compaction_externalized"
    CONTENT_DROPPED = "compaction_dropped"

    # Rehydration events
    CONTENT_REHYDRATED = "compaction_rehydrated"
    REHYDRATION_BLOCKED = "compaction_rehydration_blocked"

    # Error events
    COMPACTION_ERROR = "compaction_error"
    VERSION_CONFLICT = "compaction_version_conflict"


@dataclass
class CompactionEvent:
    """Base class for all compaction events.

    Attributes:
        event_type: Type of the event.
        thread_id: ID of the thread being compacted.
        turn_number: Current turn number.
        timestamp: When the event occurred.
        event_id: Unique identifier for this event.
        metadata: Additional event-specific data.
    """

    event_type: CompactionEventType
    thread_id: str
    turn_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = ""
    metadata: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Generate event_id if not provided."""
        if not self.event_id:
            import uuid

            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "thread_id": self.thread_id,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CompactionCheckStartedEvent(CompactionEvent):
    """Event emitted when compaction check begins.

    Attributes:
        current_tokens: Current token count before compaction.
        budget_limit: Maximum allowed tokens.
        tokens_over_budget: How many tokens over budget.
    """

    current_tokens: int = 0
    budget_limit: int = 0
    tokens_over_budget: int = 0

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.COMPACTION_CHECK_STARTED
        super().__post_init__()
        self.metadata.update(
            {
                "current_tokens": self.current_tokens,
                "budget_limit": self.budget_limit,
                "tokens_over_budget": self.tokens_over_budget,
            }
        )


@dataclass
class CompactionCompletedEvent(CompactionEvent):
    """Event emitted when compaction cycle completes.

    Attributes:
        tokens_freed: Total tokens freed by compaction.
        proposals_applied: Number of proposals that were applied.
        proposals_rejected: Number of proposals that were rejected.
        final_tokens: Token count after compaction.
        duration_ms: Time taken for compaction in milliseconds.
    """

    tokens_freed: int = 0
    proposals_applied: int = 0
    proposals_rejected: int = 0
    final_tokens: int = 0
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.COMPACTION_COMPLETED
        super().__post_init__()
        self.metadata.update(
            {
                "tokens_freed": self.tokens_freed,
                "proposals_applied": self.proposals_applied,
                "proposals_rejected": self.proposals_rejected,
                "final_tokens": self.final_tokens,
                "duration_ms": self.duration_ms,
            }
        )


@dataclass
class ProposalGeneratedEvent(CompactionEvent):
    """Event emitted when a compaction proposal is generated.

    Attributes:
        strategy_name: Name of the strategy that generated the proposal.
        action: Proposed compaction action.
        span_id: ID of the span being targeted.
        estimated_savings: Estimated token savings.
    """

    strategy_name: str = ""
    action: str = ""  # CompactionAction.value
    span_id: str = ""
    estimated_savings: int = 0

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.PROPOSAL_GENERATED
        super().__post_init__()
        self.metadata.update(
            {
                "strategy_name": self.strategy_name,
                "action": self.action,
                "span_id": self.span_id,
                "estimated_savings": self.estimated_savings,
            }
        )


@dataclass
class ProposalRejectedEvent(CompactionEvent):
    """Event emitted when a proposal is rejected.

    Attributes:
        strategy_name: Name of the strategy whose proposal was rejected.
        action: Rejected compaction action.
        span_id: ID of the span that was targeted.
        rejection_reason: Why the proposal was rejected.
    """

    strategy_name: str = ""
    action: str = ""
    span_id: str = ""
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.PROPOSAL_REJECTED
        super().__post_init__()
        self.metadata.update(
            {
                "strategy_name": self.strategy_name,
                "action": self.action,
                "span_id": self.span_id,
                "rejection_reason": self.rejection_reason,
            }
        )


@dataclass
class ContentClearedEvent(CompactionEvent):
    """Event emitted when content is cleared.

    Attributes:
        span_id: ID of the cleared span.
        message_count: Number of messages in the span.
        tokens_cleared: Tokens removed by clearing.
        preserved_fields: Fields that were preserved.
    """

    span_id: str = ""
    message_count: int = 0
    tokens_cleared: int = 0
    preserved_fields: list[str] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.CONTENT_CLEARED
        super().__post_init__()
        self.metadata.update(
            {
                "span_id": self.span_id,
                "message_count": self.message_count,
                "tokens_cleared": self.tokens_cleared,
                "preserved_fields": self.preserved_fields,
            }
        )


@dataclass
class ContentSummarizedEvent(CompactionEvent):
    """Event emitted when content is summarized.

    Attributes:
        span_id: ID of the summarized span.
        message_count: Number of messages summarized.
        original_tokens: Tokens before summarization.
        summary_tokens: Tokens in the summary.
        compression_ratio: Ratio of summary to original.
    """

    span_id: str = ""
    message_count: int = 0
    original_tokens: int = 0
    summary_tokens: int = 0
    compression_ratio: float = 0.0

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.CONTENT_SUMMARIZED
        super().__post_init__()
        if self.original_tokens > 0:
            self.compression_ratio = self.summary_tokens / self.original_tokens
        self.metadata.update(
            {
                "span_id": self.span_id,
                "message_count": self.message_count,
                "original_tokens": self.original_tokens,
                "summary_tokens": self.summary_tokens,
                "compression_ratio": self.compression_ratio,
            }
        )


@dataclass
class ContentExternalizedEvent(CompactionEvent):
    """Event emitted when content is externalized.

    Attributes:
        span_id: ID of the externalized span.
        artifact_id: ID of the stored artifact.
        message_count: Number of messages externalized.
        content_bytes: Size of externalized content.
        tokens_freed: Tokens freed by externalization.
    """

    span_id: str = ""
    artifact_id: str = ""
    message_count: int = 0
    content_bytes: int = 0
    tokens_freed: int = 0

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.CONTENT_EXTERNALIZED
        super().__post_init__()
        self.metadata.update(
            {
                "span_id": self.span_id,
                "artifact_id": self.artifact_id,
                "message_count": self.message_count,
                "content_bytes": self.content_bytes,
                "tokens_freed": self.tokens_freed,
            }
        )


@dataclass
class ContentDroppedEvent(CompactionEvent):
    """Event emitted when content is dropped.

    Attributes:
        span_id: ID of the dropped span.
        message_count: Number of messages dropped.
        tokens_dropped: Tokens removed by dropping.
        drop_reason: Why the content was dropped.
    """

    span_id: str = ""
    message_count: int = 0
    tokens_dropped: int = 0
    drop_reason: str = ""

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.CONTENT_DROPPED
        super().__post_init__()
        self.metadata.update(
            {
                "span_id": self.span_id,
                "message_count": self.message_count,
                "tokens_dropped": self.tokens_dropped,
                "drop_reason": self.drop_reason,
            }
        )


@dataclass
class ContentRehydratedEvent(CompactionEvent):
    """Event emitted when content is rehydrated.

    Attributes:
        artifact_id: ID of the rehydrated artifact.
        content_tokens: Tokens in the rehydrated content.
        was_truncated: Whether content was truncated.
        trigger: What triggered the rehydration (message/tool_call).
    """

    artifact_id: str = ""
    content_tokens: int = 0
    was_truncated: bool = False
    trigger: str = ""

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.CONTENT_REHYDRATED
        super().__post_init__()
        self.metadata.update(
            {
                "artifact_id": self.artifact_id,
                "content_tokens": self.content_tokens,
                "was_truncated": self.was_truncated,
                "trigger": self.trigger,
            }
        )


@dataclass
class RehydrationBlockedEvent(CompactionEvent):
    """Event emitted when rehydration is blocked.

    Attributes:
        artifact_id: ID of the artifact that wasn't rehydrated.
        block_reason: Why rehydration was blocked.
        cooldown_remaining_seconds: Seconds until cooldown expires.
    """

    artifact_id: str = ""
    block_reason: str = ""
    cooldown_remaining_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.REHYDRATION_BLOCKED
        super().__post_init__()
        self.metadata.update(
            {
                "artifact_id": self.artifact_id,
                "block_reason": self.block_reason,
                "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            }
        )


@dataclass
class CompactionErrorEvent(CompactionEvent):
    """Event emitted when a compaction error occurs.

    Attributes:
        error_type: Type of error that occurred.
        error_message: Error description.
        strategy_name: Strategy that caused the error (if applicable).
        recoverable: Whether the error is recoverable.
    """

    error_type: str = ""
    error_message: str = ""
    strategy_name: str = ""
    recoverable: bool = True

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.COMPACTION_ERROR
        super().__post_init__()
        self.metadata.update(
            {
                "error_type": self.error_type,
                "error_message": self.error_message,
                "strategy_name": self.strategy_name,
                "recoverable": self.recoverable,
            }
        )


@dataclass
class VersionConflictEvent(CompactionEvent):
    """Event emitted when a version conflict occurs.

    Attributes:
        expected_version: Version that was expected.
        actual_version: Actual version found.
        will_retry: Whether the operation will be retried.
    """

    expected_version: int = 0
    actual_version: int = 0
    will_retry: bool = False

    def __post_init__(self) -> None:
        """Set event type and generate ID."""
        self.event_type = CompactionEventType.VERSION_CONFLICT
        super().__post_init__()
        self.metadata.update(
            {
                "expected_version": self.expected_version,
                "actual_version": self.actual_version,
                "will_retry": self.will_retry,
            }
        )


@dataclass
class CompactionMetrics:
    """Aggregated metrics for compaction operations.

    Attributes:
        total_compactions: Total number of compaction cycles.
        total_tokens_freed: Total tokens freed across all compactions.
        total_proposals_generated: Total proposals generated.
        total_proposals_applied: Total proposals applied.
        total_proposals_rejected: Total proposals rejected.
        total_errors: Total errors encountered.
        total_version_conflicts: Total version conflicts.
        total_rehydrations: Total rehydration operations.
        total_rehydrations_blocked: Total blocked rehydrations.
        average_compression_ratio: Average compression ratio for summaries.
        average_duration_ms: Average compaction duration.
    """

    total_compactions: int = 0
    total_tokens_freed: int = 0
    total_proposals_generated: int = 0
    total_proposals_applied: int = 0
    total_proposals_rejected: int = 0
    total_errors: int = 0
    total_version_conflicts: int = 0
    total_rehydrations: int = 0
    total_rehydrations_blocked: int = 0
    average_compression_ratio: float = 0.0
    average_duration_ms: float = 0.0

    # Internal tracking
    _compression_ratios: list[float] = field(default_factory=lambda: [])
    _durations_ms: list[float] = field(default_factory=lambda: [])

    def record_compaction(
        self,
        tokens_freed: int,
        proposals_applied: int,
        proposals_rejected: int,
        duration_ms: float,
    ) -> None:
        """Record a completed compaction cycle."""
        self.total_compactions += 1
        self.total_tokens_freed += tokens_freed
        self.total_proposals_applied += proposals_applied
        self.total_proposals_rejected += proposals_rejected
        self._durations_ms.append(duration_ms)
        if self._durations_ms:
            self.average_duration_ms = sum(self._durations_ms) / len(self._durations_ms)

    def record_proposal(self, applied: bool) -> None:
        """Record a proposal event."""
        self.total_proposals_generated += 1
        if applied:
            self.total_proposals_applied += 1
        else:
            self.total_proposals_rejected += 1

    def record_summary(self, compression_ratio: float) -> None:
        """Record a summarization with its compression ratio."""
        self._compression_ratios.append(compression_ratio)
        if self._compression_ratios:
            self.average_compression_ratio = sum(self._compression_ratios) / len(
                self._compression_ratios
            )

    def record_error(self) -> None:
        """Record an error."""
        self.total_errors += 1

    def record_version_conflict(self) -> None:
        """Record a version conflict."""
        self.total_version_conflicts += 1

    def record_rehydration(self, blocked: bool = False) -> None:
        """Record a rehydration attempt."""
        if blocked:
            self.total_rehydrations_blocked += 1
        else:
            self.total_rehydrations += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_compactions": self.total_compactions,
            "total_tokens_freed": self.total_tokens_freed,
            "total_proposals_generated": self.total_proposals_generated,
            "total_proposals_applied": self.total_proposals_applied,
            "total_proposals_rejected": self.total_proposals_rejected,
            "total_errors": self.total_errors,
            "total_version_conflicts": self.total_version_conflicts,
            "total_rehydrations": self.total_rehydrations,
            "total_rehydrations_blocked": self.total_rehydrations_blocked,
            "average_compression_ratio": self.average_compression_ratio,
            "average_duration_ms": self.average_duration_ms,
        }


class CompactionEventEmitter(Protocol):
    """Protocol for emitting compaction events.

    Implementations can send events to logging, metrics systems,
    or other observability infrastructure.
    """

    def emit(self, event: CompactionEvent) -> None:
        """Emit a compaction event.

        Args:
            event: The event to emit.
        """
        ...


class LoggingEventEmitter:
    """Event emitter that logs events.

    Suitable for development and debugging.
    """

    def __init__(self, logger_name: str = "compaction.events") -> None:
        """Initialize the logging emitter.

        Args:
            logger_name: Name of the logger to use.
        """
        self._logger = logging.getLogger(logger_name)

    def emit(self, event: CompactionEvent) -> None:
        """Emit event to logger.

        Args:
            event: The event to emit.
        """
        level = logging.INFO
        if event.event_type == CompactionEventType.COMPACTION_ERROR:
            level = logging.ERROR
        elif event.event_type == CompactionEventType.VERSION_CONFLICT:
            level = logging.WARNING

        self._logger.log(
            level,
            "Compaction event: %s [thread=%s, turn=%d] %s",
            event.event_type.value,
            event.thread_id,
            event.turn_number,
            event.metadata,
        )


class MetricsCollector:
    """Collects metrics from compaction events.

    Maintains running statistics and can export to
    various metrics backends.
    """

    def __init__(self) -> None:
        """Initialize the collector."""
        self._metrics: dict[str, CompactionMetrics] = {}
        self._global_metrics = CompactionMetrics()

    def process_event(self, event: CompactionEvent) -> None:
        """Process an event and update metrics.

        Args:
            event: The event to process.
        """
        # Get or create per-thread metrics
        if event.thread_id not in self._metrics:
            self._metrics[event.thread_id] = CompactionMetrics()
        thread_metrics = self._metrics[event.thread_id]

        # Update based on event type
        if isinstance(event, CompactionCompletedEvent):
            thread_metrics.record_compaction(
                event.tokens_freed,
                event.proposals_applied,
                event.proposals_rejected,
                event.duration_ms,
            )
            self._global_metrics.record_compaction(
                event.tokens_freed,
                event.proposals_applied,
                event.proposals_rejected,
                event.duration_ms,
            )
        elif isinstance(event, ProposalGeneratedEvent):
            thread_metrics.record_proposal(applied=True)
            self._global_metrics.record_proposal(applied=True)
        elif isinstance(event, ProposalRejectedEvent):
            thread_metrics.record_proposal(applied=False)
            self._global_metrics.record_proposal(applied=False)
        elif isinstance(event, ContentSummarizedEvent):
            thread_metrics.record_summary(event.compression_ratio)
            self._global_metrics.record_summary(event.compression_ratio)
        elif isinstance(event, CompactionErrorEvent):
            thread_metrics.record_error()
            self._global_metrics.record_error()
        elif isinstance(event, VersionConflictEvent):
            thread_metrics.record_version_conflict()
            self._global_metrics.record_version_conflict()
        elif isinstance(event, ContentRehydratedEvent):
            thread_metrics.record_rehydration(blocked=False)
            self._global_metrics.record_rehydration(blocked=False)
        elif isinstance(event, RehydrationBlockedEvent):
            thread_metrics.record_rehydration(blocked=True)
            self._global_metrics.record_rehydration(blocked=True)

    def get_thread_metrics(self, thread_id: str) -> CompactionMetrics | None:
        """Get metrics for a specific thread.

        Args:
            thread_id: The thread ID.

        Returns:
            Metrics for the thread, or None if not found.
        """
        return self._metrics.get(thread_id)

    def get_global_metrics(self) -> CompactionMetrics:
        """Get aggregated global metrics.

        Returns:
            Global metrics across all threads.
        """
        return self._global_metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._global_metrics = CompactionMetrics()


class CompositeEventEmitter:
    """Event emitter that delegates to multiple emitters.

    Useful for sending events to both logging and metrics.
    """

    def __init__(self, emitters: list[CompactionEventEmitter] | None = None) -> None:
        """Initialize with emitters.

        Args:
            emitters: List of emitters to delegate to.
        """
        self._emitters = emitters or []

    def add_emitter(self, emitter: CompactionEventEmitter) -> None:
        """Add an emitter.

        Args:
            emitter: Emitter to add.
        """
        self._emitters.append(emitter)

    def emit(self, event: CompactionEvent) -> None:
        """Emit event to all registered emitters.

        Args:
            event: The event to emit.
        """
        for emitter in self._emitters:
            try:
                emitter.emit(event)
            except Exception:
                logger.exception("Error emitting event to %s", type(emitter).__name__)
