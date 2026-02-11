# Copyright (c) Microsoft. All rights reserved.

"""Lifecycle events for context compaction observability.

This module provides:
- CompactionEvent base class for all compaction events
- CompactionEventType enum for event classification
- CompactionMetrics for aggregated statistics
- CompactionEventEmitter protocol for observability integration

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


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

    # Error events
    COMPACTION_ERROR = "compaction_error"


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
    metadata: dict[str, Any] = field(default_factory=dict)

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
    _compression_ratios: list[float] = field(default_factory=list)
    _durations_ms: list[float] = field(default_factory=list)

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
            self.average_compression_ratio = sum(self._compression_ratios) / len(self._compression_ratios)

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
