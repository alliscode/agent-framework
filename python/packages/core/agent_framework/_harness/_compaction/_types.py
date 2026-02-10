# Copyright (c) Microsoft. All rights reserved.

"""Core data structures for context compaction.

This module defines the foundational types for the immutable-log + compaction-plan
architecture. Key principles:

1. AgentThread is never mutated - it's the append-only source of truth
2. CompactionPlan is pure data describing what spans are compacted
3. PromptRenderer takes Thread + Plan to produce the actual model request

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._summary import StructuredSummary


class CompactionAction(Enum):
    """Action to take for a message during rendering.

    Actions are ordered by precedence (higher value = takes precedence).
    When spans overlap, the higher-precedence action wins.
    """

    INCLUDE = "include"  # Include as-is (default)
    CLEAR = "clear"  # Replace with placeholder
    SUMMARIZE = "summarize"  # Replace with summary
    EXTERNALIZE = "externalize"  # Replace with pointer + summary
    DROP = "drop"  # Omit entirely


# Precedence order (higher = takes precedence)
COMPACTION_PRECEDENCE: dict[CompactionAction, int] = {
    CompactionAction.INCLUDE: 0,  # Lowest - default
    CompactionAction.CLEAR: 1,
    CompactionAction.SUMMARIZE: 2,
    CompactionAction.EXTERNALIZE: 3,
    CompactionAction.DROP: 4,  # Highest - overrides everything
}


@dataclass
class SpanReference:
    """Reference to a contiguous span of messages.

    SpanReference is self-sufficient - it stores the explicit list of message IDs,
    not just start/end. This ensures:
    - Plans are truly "pure data" resolvable without thread access
    - No ambiguity if messages are appended later
    - No off-by-one bugs from start/end resolution

    Invariants:
    - message_ids is non-empty
    - message_ids preserves the original thread ordering
    - All IDs must exist in the thread at plan creation time

    Attributes:
        message_ids: Explicit list of message IDs in this span.
        first_turn: Turn number of first message (for display).
        last_turn: Turn number of last message (for display).
    """

    message_ids: list[str]
    first_turn: int
    last_turn: int

    def __post_init__(self) -> None:
        """Validate span invariants."""
        if not self.message_ids:
            raise ValueError("SpanReference.message_ids cannot be empty")

    @property
    def message_count(self) -> int:
        """Number of messages in this span."""
        return len(self.message_ids)

    @property
    def start_message_id(self) -> str:
        """ID of the first message in this span."""
        return self.message_ids[0]

    @property
    def end_message_id(self) -> str:
        """ID of the last message in this span."""
        return self.message_ids[-1]

    def contains(self, message_id: str) -> bool:
        """Check if span contains a message ID.

        O(n) but spans are typically small.
        """
        return message_id in self.message_ids

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_ids": self.message_ids,
            "first_turn": self.first_turn,
            "last_turn": self.last_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpanReference:
        """Deserialize from dictionary."""
        return cls(
            message_ids=data["message_ids"],
            first_turn=data["first_turn"],
            last_turn=data["last_turn"],
        )


@dataclass
class ExternalizationRecord:
    """Record of externalized content.

    When content is externalized, it's written to an ArtifactStore and
    replaced with a pointer + summary in the rendered prompt.

    Attributes:
        span: The span of messages that were externalized.
        artifact_id: ID of the stored artifact.
        summary: Structured summary of the externalized content.
        rehydrate_hint: Guidance for when agent should read this back.
    """

    span: SpanReference
    artifact_id: str
    summary: StructuredSummary
    rehydrate_hint: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "span": self.span.to_dict(),
            "artifact_id": self.artifact_id,
            "summary": self.summary.to_dict(),
            "rehydrate_hint": self.rehydrate_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalizationRecord:
        """Deserialize from dictionary."""
        from ._summary import StructuredSummary as _StructuredSummary

        return cls(
            span=SpanReference.from_dict(data["span"]),
            artifact_id=data["artifact_id"],
            summary=_StructuredSummary.from_dict(data["summary"]),
            rehydrate_hint=data["rehydrate_hint"],
        )


@dataclass
class SummarizationRecord:
    """Record of summarized content.

    When content is summarized, it's replaced with a structured summary
    in the rendered prompt. The original content is preserved in the
    canonical log.

    Attributes:
        span: The span of messages that were summarized.
        summary: Structured summary of the content.
        summary_token_count: Token count of the rendered summary.
    """

    span: SpanReference
    summary: StructuredSummary
    summary_token_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "span": self.span.to_dict(),
            "summary": self.summary.to_dict(),
            "summary_token_count": self.summary_token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummarizationRecord:
        """Deserialize from dictionary."""
        from ._summary import StructuredSummary as _StructuredSummary

        return cls(
            span=SpanReference.from_dict(data["span"]),
            summary=_StructuredSummary.from_dict(data["summary"]),
            summary_token_count=data["summary_token_count"],
        )


@dataclass
class ClearRecord:
    """Record of cleared content.

    When content is cleared, it's replaced with a minimal placeholder
    that preserves key fields (tool name, outcome, important IDs).

    Attributes:
        span: The span of messages that were cleared.
        preserved_fields: Key fields preserved from the original content.
    """

    span: SpanReference
    preserved_fields: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "span": self.span.to_dict(),
            "preserved_fields": self.preserved_fields,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClearRecord:
        """Deserialize from dictionary."""
        return cls(
            span=SpanReference.from_dict(data["span"]),
            preserved_fields=data.get("preserved_fields", {}),
        )


@dataclass
class DropRecord:
    """Record of dropped content.

    Dropped content is omitted entirely from the rendered prompt.
    This is the most aggressive compaction and should be used as a last resort.

    Attributes:
        span: The span of messages that were dropped.
        reason: Why this content was dropped.
    """

    span: SpanReference
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "span": self.span.to_dict(),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DropRecord:
        """Deserialize from dictionary."""
        return cls(
            span=SpanReference.from_dict(data["span"]),
            reason=data.get("reason", ""),
        )


@dataclass
class CompactionPlan:
    """Complete compaction plan for a thread.

    A CompactionPlan is pure data describing how to render a thread. It references
    message IDs, not message content. The plan is normalized: each message ID
    appears in at most one record list, with precedence enforced at build time.

    Precedence (highest wins): Drop > Externalize > Summarize > Clear > Include

    Attributes:
        thread_id: ID of the thread this plan applies to.
        thread_version: Version for optimistic concurrency control.
        created_at: When this plan was created.
        externalizations: Records of externalized content.
        summarizations: Records of summarized content.
        clearings: Records of cleared content.
        drops: Records of dropped content.
        original_token_count: Token count before compaction.
        compacted_token_count: Token count after compaction.
    """

    thread_id: str
    thread_version: int
    created_at: datetime
    externalizations: list[ExternalizationRecord] = field(default_factory=list)
    summarizations: list[SummarizationRecord] = field(default_factory=list)
    clearings: list[ClearRecord] = field(default_factory=list)
    drops: list[DropRecord] = field(default_factory=list)
    original_token_count: int = 0
    compacted_token_count: int = 0

    # Internal: normalized action map (built in __post_init__)
    _action_map: dict[str, tuple[CompactionAction, Any]] = field(default_factory=dict, repr=False)
    _normalization_warnings: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Build normalized action map from records."""
        self._action_map, self._normalization_warnings = self._build_action_map()

    def _build_action_map(
        self,
    ) -> tuple[dict[str, tuple[CompactionAction, Any]], list[str]]:
        """Build message_id -> (action, record) map with precedence.

        Also validates that no message ID appears in multiple record lists
        at the same precedence level. Higher precedence overwrites lower.

        Returns:
            Tuple of (action_map, warnings).
        """
        action_map: dict[str, tuple[CompactionAction, Any]] = {}
        seen_ids: dict[str, tuple[CompactionAction, str]] = {}
        warnings: list[str] = []

        def process_records(
            records: list[Any],
            action: CompactionAction,
            record_type: str,
        ) -> None:
            """Process records, track overlaps, collect warnings."""
            for record in records:
                for msg_id in record.span.message_ids:
                    if msg_id in seen_ids:
                        prev_action, prev_type = seen_ids[msg_id]
                        if COMPACTION_PRECEDENCE[action] > COMPACTION_PRECEDENCE[prev_action]:
                            warnings.append(
                                f"Message {msg_id} in both {prev_type} and {record_type}; "
                                f"{record_type} takes precedence",
                            )
                        else:
                            # Lower or equal precedence, skip
                            continue
                    action_map[msg_id] = (action, record)
                    seen_ids[msg_id] = (action, record_type)

        # Process in precedence order (lowest first, higher overwrites)
        process_records(self.clearings, CompactionAction.CLEAR, "clearings")
        process_records(self.summarizations, CompactionAction.SUMMARIZE, "summarizations")
        process_records(self.externalizations, CompactionAction.EXTERNALIZE, "externalizations")
        process_records(self.drops, CompactionAction.DROP, "drops")

        return action_map, warnings

    def get_action(self, message_id: str) -> tuple[CompactionAction, Any | None]:
        """Get the action for a message ID.

        Args:
            message_id: The message ID to look up.

        Returns:
            (action, record) tuple. Record is None for INCLUDE.
        """
        return self._action_map.get(message_id, (CompactionAction.INCLUDE, None))

    @property
    def has_overlaps(self) -> bool:
        """True if plan had overlapping spans that were normalized."""
        return bool(self._normalization_warnings)

    @property
    def normalization_warnings(self) -> list[str]:
        """Warnings from plan normalization (overlapping spans)."""
        return self._normalization_warnings

    @property
    def tokens_freed(self) -> int:
        """Number of tokens freed by this compaction."""
        return max(0, self.original_token_count - self.compacted_token_count)

    @property
    def is_empty(self) -> bool:
        """True if this plan has no compaction records."""
        return not self.externalizations and not self.summarizations and not self.clearings and not self.drops

    def rebuild_action_map(self) -> None:
        """Rebuild the action map after modifying records.

        Call this after adding records to the plan to update the
        internal lookup structures.
        """
        self._action_map, self._normalization_warnings = self._build_action_map()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "thread_id": self.thread_id,
            "thread_version": self.thread_version,
            "created_at": self.created_at.isoformat(),
            "externalizations": [r.to_dict() for r in self.externalizations],
            "summarizations": [r.to_dict() for r in self.summarizations],
            "clearings": [r.to_dict() for r in self.clearings],
            "drops": [r.to_dict() for r in self.drops],
            "original_token_count": self.original_token_count,
            "compacted_token_count": self.compacted_token_count,
        }

    @classmethod
    def create_empty(cls, thread_id: str, thread_version: int = 0) -> CompactionPlan:
        """Create an empty compaction plan.

        Args:
            thread_id: ID of the thread.
            thread_version: Version number for concurrency control.

        Returns:
            An empty CompactionPlan.
        """
        return cls(
            thread_id=thread_id,
            thread_version=thread_version,
            created_at=datetime.now(timezone.utc),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompactionPlan:
        """Deserialize from dictionary.

        Args:
            data: Dictionary produced by to_dict().

        Returns:
            A CompactionPlan with all records restored.
        """
        created_at_str = data.get("created_at")
        if created_at_str and isinstance(created_at_str, str):
            created_at = datetime.fromisoformat(created_at_str)
        else:
            created_at = datetime.now(timezone.utc)

        externalizations_data: list[dict[str, Any]] = data.get("externalizations", [])
        summarizations_data: list[dict[str, Any]] = data.get("summarizations", [])
        clearings_data: list[dict[str, Any]] = data.get("clearings", [])
        drops_data: list[dict[str, Any]] = data.get("drops", [])

        return cls(
            thread_id=data.get("thread_id", ""),
            thread_version=data.get("thread_version", 0),
            created_at=created_at,
            externalizations=[ExternalizationRecord.from_dict(r) for r in externalizations_data],
            summarizations=[SummarizationRecord.from_dict(r) for r in summarizations_data],
            clearings=[ClearRecord.from_dict(r) for r in clearings_data],
            drops=[DropRecord.from_dict(r) for r in drops_data],
            original_token_count=data.get("original_token_count", 0),
            compacted_token_count=data.get("compacted_token_count", 0),
        )
