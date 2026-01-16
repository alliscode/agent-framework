# Copyright (c) Microsoft. All rights reserved.

"""Context pressure management for Agent Harness.

This module implements context pressure strategies that manage context window
utilization. When the conversation approaches token limits, strategies can
be applied to reduce context while preserving important information.

Strategies (in recommended order):
1. Externalize: Write large tool results to files, replace with pointer + summary
2. Clear: Clear older tool results with placeholders
3. Compact: Summarize older messages, keep recent slice
4. Drop: Remove older content entirely (last resort)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .._workflows._workflow_context import WorkflowContext


class ContextEditKind(Enum):
    """Types of context edits that can be applied."""

    EXTERNALIZE = "externalize"  # Write to file, replace with pointer
    CLEAR = "clear"  # Clear tool results with placeholder
    COMPACT = "compact"  # Summarize messages
    DROP = "drop"  # Remove content entirely


@dataclass
class TranscriptRange:
    """Range of transcript entries to target for editing.

    Attributes:
        start_index: Starting index in transcript (inclusive).
        end_index: Ending index in transcript (exclusive). None means to end.
        event_types: Optional filter for specific event types.
    """

    start_index: int = 0
    end_index: int | None = None
    event_types: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "event_types": self.event_types,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptRange":
        """Deserialize from dictionary."""
        return cls(
            start_index=data.get("start_index", 0),
            end_index=data.get("end_index"),
            event_types=data.get("event_types"),
        )


@dataclass
class ExternalizeEdit:
    """Edit to externalize content to a file.

    Attributes:
        scope: Range of transcript entries to externalize.
        artifact_kind: Type of artifact to create (file or blob).
        pointer_style: How to reference the externalized content.
        summary_template: Template for generating summary text.
    """

    kind: Literal["externalize"] = "externalize"
    scope: TranscriptRange = field(default_factory=TranscriptRange)
    artifact_kind: Literal["file", "blob"] = "file"
    pointer_style: Literal["path", "artifact_id"] = "path"
    summary_template: str = "Content externalized to {path}. Read the file for full details."

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kind": self.kind,
            "scope": self.scope.to_dict(),
            "artifact_kind": self.artifact_kind,
            "pointer_style": self.pointer_style,
            "summary_template": self.summary_template,
        }


@dataclass
class ClearEdit:
    """Edit to clear tool results with placeholders.

    Attributes:
        scope: Range of transcript entries to clear.
        mode: What to clear - just results or inputs too.
        placeholder_template: Template for placeholder text.
    """

    kind: Literal["clear"] = "clear"
    scope: TranscriptRange = field(default_factory=TranscriptRange)
    mode: Literal["results_only", "inputs_and_results"] = "results_only"
    placeholder_template: str = "[Tool result cleared to save context]"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kind": self.kind,
            "scope": self.scope.to_dict(),
            "mode": self.mode,
            "placeholder_template": self.placeholder_template,
        }


@dataclass
class CompactEdit:
    """Edit to summarize/compact messages.

    Attributes:
        scope: Range of transcript entries to compact.
        summary_type: Type of summary to generate.
        keep_recent_count: Number of recent messages to keep intact.
    """

    kind: Literal["compact"] = "compact"
    scope: TranscriptRange = field(default_factory=TranscriptRange)
    summary_type: Literal["bullet", "narrative", "state_json"] = "bullet"
    keep_recent_count: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kind": self.kind,
            "scope": self.scope.to_dict(),
            "summary_type": self.summary_type,
            "keep_recent_count": self.keep_recent_count,
        }


@dataclass
class DropEdit:
    """Edit to drop content entirely (last resort).

    Attributes:
        scope: Range of transcript entries to drop.
    """

    kind: Literal["drop"] = "drop"
    scope: TranscriptRange = field(default_factory=TranscriptRange)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kind": self.kind,
            "scope": self.scope.to_dict(),
        }


# Union type for all context edit types
ContextEdit = ExternalizeEdit | ClearEdit | CompactEdit | DropEdit


@dataclass
class ContextEditPlan:
    """Plan of context edits to apply.

    Attributes:
        reason: Why the edit is needed.
        estimated_token_reduction: Approximate tokens to be freed.
        edits: List of edits to apply.
    """

    reason: Literal["token_budget", "cost_budget", "time_budget", "policy"]
    estimated_token_reduction: int
    edits: list[ContextEdit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reason": self.reason,
            "estimated_token_reduction": self.estimated_token_reduction,
            "edits": [e.to_dict() for e in self.edits],
        }


@dataclass
class TokenBudget:
    """Token budget configuration for context pressure.

    Attributes:
        max_input_tokens: Maximum tokens allowed in input context.
        soft_threshold_percent: Percentage at which to trigger pressure strategies.
        current_estimate: Current estimated token count.
    """

    max_input_tokens: int = 100000
    soft_threshold_percent: float = 0.85
    current_estimate: int = 0

    @property
    def soft_threshold(self) -> int:
        """Calculate the soft threshold in tokens."""
        return int(self.max_input_tokens * self.soft_threshold_percent)

    @property
    def is_under_pressure(self) -> bool:
        """Check if context is under pressure (above soft threshold)."""
        return self.current_estimate >= self.soft_threshold

    @property
    def tokens_over_threshold(self) -> int:
        """Calculate how many tokens over the soft threshold."""
        return max(0, self.current_estimate - self.soft_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_input_tokens": self.max_input_tokens,
            "soft_threshold_percent": self.soft_threshold_percent,
            "current_estimate": self.current_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenBudget":
        """Deserialize from dictionary."""
        return cls(
            max_input_tokens=data.get("max_input_tokens", 100000),
            soft_threshold_percent=data.get("soft_threshold_percent", 0.85),
            current_estimate=data.get("current_estimate", 0),
        )


@runtime_checkable
class ContextPressureStrategy(Protocol):
    """Protocol for context pressure strategies.

    Strategies are responsible for proposing edits when context is under pressure.
    They should be tried in order from least to most aggressive.
    """

    @property
    def name(self) -> str:
        """Name of this strategy."""
        ...

    def is_applicable(self, budget: TokenBudget, transcript: list[dict[str, Any]]) -> bool:
        """Check if this strategy can be applied given current state.

        Args:
            budget: Current token budget information.
            transcript: Current transcript entries.

        Returns:
            True if this strategy can propose edits.
        """
        ...

    async def propose(
        self,
        budget: TokenBudget,
        transcript: list[dict[str, Any]],
        ctx: "WorkflowContext[Any]",
    ) -> ContextEditPlan | None:
        """Propose an edit plan to reduce context pressure.

        Args:
            budget: Current token budget information.
            transcript: Current transcript entries.
            ctx: Workflow context for state access.

        Returns:
            An edit plan, or None if no edits proposed.
        """
        ...


class ClearToolResultsStrategy:
    """Strategy to clear older tool results with placeholders.

    This is the least aggressive strategy that clears tool results
    while maintaining conversation flow.
    """

    @property
    def name(self) -> str:
        """Name of this strategy."""
        return "clear_tool_results"

    def is_applicable(self, budget: TokenBudget, transcript: list[dict[str, Any]]) -> bool:
        """Check if there are tool results to clear."""
        # Look for tool_result events in older parts of transcript
        if len(transcript) < 3:
            return False

        # Check for tool results in older entries (not recent)
        cutoff = max(0, len(transcript) - 3)
        for event in transcript[:cutoff]:
            if event.get("event_type") == "tool_result":
                return True
        return False

    async def propose(
        self,
        budget: TokenBudget,
        transcript: list[dict[str, Any]],
        ctx: "WorkflowContext[Any]",
    ) -> ContextEditPlan | None:
        """Propose clearing older tool results."""
        # Find tool result events to clear (keep recent 3)
        cutoff = max(0, len(transcript) - 3)

        tool_result_indices: list[int] = []
        for i, event in enumerate(transcript[:cutoff]):
            if event.get("event_type") == "tool_result":
                tool_result_indices.append(i)

        if not tool_result_indices:
            return None

        # Estimate token reduction (rough: assume average tool result is 500 tokens)
        estimated_reduction = len(tool_result_indices) * 500

        edit = ClearEdit(
            scope=TranscriptRange(
                start_index=0,
                end_index=cutoff,
                event_types=["tool_result"],
            ),
            mode="results_only",
            placeholder_template="[Tool result cleared - turn {turn_number}]",
        )

        return ContextEditPlan(
            reason="token_budget",
            estimated_token_reduction=estimated_reduction,
            edits=[edit],
        )


class CompactConversationStrategy:
    """Strategy to compact/summarize older messages.

    This strategy summarizes older conversation while keeping
    recent messages intact.
    """

    def __init__(self, keep_recent_percent: float = 0.10):
        """Initialize the strategy.

        Args:
            keep_recent_percent: Percentage of recent conversation to keep intact.
        """
        self._keep_recent_percent = keep_recent_percent

    @property
    def name(self) -> str:
        """Name of this strategy."""
        return "compact_conversation"

    def is_applicable(self, budget: TokenBudget, transcript: list[dict[str, Any]]) -> bool:
        """Check if there's enough conversation to compact."""
        # Need at least 5 events to make compaction worthwhile
        return len(transcript) >= 5

    async def propose(
        self,
        budget: TokenBudget,
        transcript: list[dict[str, Any]],
        ctx: "WorkflowContext[Any]",
    ) -> ContextEditPlan | None:
        """Propose compacting older conversation."""
        # Calculate how much to keep
        keep_count = max(3, int(len(transcript) * self._keep_recent_percent))
        compact_end = len(transcript) - keep_count

        if compact_end <= 0:
            return None

        # Estimate token reduction (rough: assume 50% reduction of compacted content)
        # Assume average of 200 tokens per event
        estimated_reduction = int(compact_end * 200 * 0.5)

        edit = CompactEdit(
            scope=TranscriptRange(start_index=0, end_index=compact_end),
            summary_type="bullet",
            keep_recent_count=keep_count,
        )

        return ContextEditPlan(
            reason="token_budget",
            estimated_token_reduction=estimated_reduction,
            edits=[edit],
        )


class DropOldestStrategy:
    """Strategy to drop oldest content entirely.

    This is the most aggressive strategy and should only be used
    as a last resort.
    """

    @property
    def name(self) -> str:
        """Name of this strategy."""
        return "drop_oldest"

    def is_applicable(self, budget: TokenBudget, transcript: list[dict[str, Any]]) -> bool:
        """Check if there's content to drop."""
        # Need at least 6 events, and keep at least 3
        return len(transcript) >= 6

    async def propose(
        self,
        budget: TokenBudget,
        transcript: list[dict[str, Any]],
        ctx: "WorkflowContext[Any]",
    ) -> ContextEditPlan | None:
        """Propose dropping oldest content."""
        # Drop oldest 25% (keep at least 3 recent)
        drop_count = max(1, len(transcript) // 4)
        keep_count = len(transcript) - drop_count

        if keep_count < 3:
            drop_count = len(transcript) - 3

        if drop_count <= 0:
            return None

        # Estimate token reduction (assume 200 tokens per event)
        estimated_reduction = drop_count * 200

        edit = DropEdit(
            scope=TranscriptRange(start_index=0, end_index=drop_count),
        )

        return ContextEditPlan(
            reason="token_budget",
            estimated_token_reduction=estimated_reduction,
            edits=[edit],
        )


def get_default_strategies() -> list[ContextPressureStrategy]:
    """Get the default ordered list of context pressure strategies.

    Returns:
        List of strategies in recommended order (least to most aggressive).
    """
    return [
        ClearToolResultsStrategy(),
        CompactConversationStrategy(keep_recent_percent=0.10),
        DropOldestStrategy(),
    ]


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    This is a rough approximation. For accurate counts, use a tokenizer.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    # Rough approximation: 1 token ≈ 4 characters for English
    # This varies by language and tokenizer
    return max(1, len(text) // 4)


def estimate_transcript_tokens(transcript: list[dict[str, Any]]) -> int:
    """Estimate total tokens in a transcript.

    Args:
        transcript: List of transcript events.

    Returns:
        Estimated total token count.
    """
    total = 0
    for event in transcript:
        # Serialize event to estimate its size
        import json

        event_str = json.dumps(event)
        total += estimate_tokens(event_str)
    return total
