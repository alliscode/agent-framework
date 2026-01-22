# Copyright (c) Microsoft. All rights reserved.

"""Structured summary types for context compaction.

Plain-text summaries drift over time. After 3-5 summarize cycles you get missing
constraints, lost decisions, merged entities. Structured summaries prevent this
by organizing information into semantic categories that are preserved across
compaction cycles.

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ._types import SpanReference

# Schema versioning is critical to avoid cache poisoning
# Bump this when StructuredSummary fields change
STRUCTURED_SUMMARY_SCHEMA_VERSION = "v1.0"

# Rendering format version for reproducibility
SUMMARY_RENDER_VERSION = "v1.0"


@dataclass
class Decision:
    """A decision made during the conversation.

    Decisions are critical to preserve - they represent choices that
    affect subsequent actions and should survive compaction.

    Attributes:
        decision: What was decided.
        rationale: Why this decision was made.
        turn_number: When this decision was made.
        timestamp: ISO 8601 timestamp.
    """

    decision: str
    rationale: str
    turn_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "decision": self.decision,
            "rationale": self.rationale,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        """Deserialize from dictionary."""
        return cls(
            decision=data["decision"],
            rationale=data["rationale"],
            turn_number=data["turn_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class OpenItem:
    """An unresolved item or TODO.

    Open items track work that needs to be completed. They should
    survive compaction so the agent doesn't forget pending tasks.

    Attributes:
        description: What needs to be done.
        context: Additional context about this item.
        priority: Importance level.
    """

    description: str
    context: str
    priority: Literal["high", "medium", "low"] = "medium"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "description": self.description,
            "context": self.context,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenItem:
        """Deserialize from dictionary."""
        return cls(
            description=data["description"],
            context=data["context"],
            priority=data.get("priority", "medium"),
        )


@dataclass
class ArtifactReference:
    """Reference to an externalized artifact.

    When content is externalized, we keep a reference to it so the
    agent can access it later if needed.

    Attributes:
        artifact_id: Unique identifier for the artifact.
        description: What this artifact contains.
        rehydrate_hint: When the agent should read this back.
    """

    artifact_id: str
    description: str
    rehydrate_hint: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "description": self.description,
            "rehydrate_hint": self.rehydrate_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactReference:
        """Deserialize from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            description=data["description"],
            rehydrate_hint=data["rehydrate_hint"],
        )


@dataclass
class ToolOutcome:
    """Summary of a tool result.

    Tool outcomes capture what happened when a tool was called.
    Key fields are preserved even when the full result is cleared.

    Attributes:
        tool_name: Name of the tool that was called.
        outcome: Whether the call succeeded, failed, or partially succeeded.
        key_fields: Important values to preserve (IDs, counts, etc.).
        error_message: Error message if the call failed.
    """

    tool_name: str
    outcome: Literal["success", "failure", "partial"]
    key_fields: dict[str, Any] = field(default_factory=lambda: {})
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tool_name": self.tool_name,
            "outcome": self.outcome,
            "key_fields": self.key_fields,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolOutcome:
        """Deserialize from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            outcome=data["outcome"],
            key_fields=data.get("key_fields", {}),
            error_message=data.get("error_message"),
        )


@dataclass
class StructuredSummary:
    """Structured summary that resists drift.

    Unlike plain-text summaries, structured summaries organize information
    into semantic categories. This prevents the common failure mode where
    repeated summarization loses constraints, decisions, and important details.

    The render_as_message() method produces a deterministic, versioned
    representation suitable for inclusion in prompts.

    Attributes:
        span: Reference to the span this summary covers.
        facts: Stable information, user preferences, constraints.
        decisions: Decisions made during the conversation.
        open_items: Unresolved items or TODOs.
        artifacts: References to externalized artifacts.
        tool_outcomes: Summaries of tool results.
        current_task: What the agent is currently working on.
        current_plan: Steps in the current plan.
    """

    span: SpanReference
    facts: list[str] = field(default_factory=lambda: [])
    decisions: list[Decision] = field(default_factory=lambda: [])
    open_items: list[OpenItem] = field(default_factory=lambda: [])
    artifacts: list[ArtifactReference] = field(default_factory=lambda: [])
    tool_outcomes: list[ToolOutcome] = field(default_factory=lambda: [])
    current_task: str | None = None
    current_plan: list[str] | None = None

    def render_as_message(self) -> str:
        """Render as a message for inclusion in prompt.

        The output is deterministic (same input -> same output) and
        versioned for reproducibility.

        Returns:
            Formatted string suitable for prompt inclusion.
        """
        lines: list[str] = []

        # Facts section
        if self.facts:
            lines.append("**Key Facts:**")
            for fact in sorted(self.facts):  # Sort for determinism
                lines.append(f"- {fact}")
            lines.append("")

        # Decisions section
        if self.decisions:
            lines.append("**Decisions Made:**")
            for d in sorted(self.decisions, key=lambda x: x.turn_number):
                lines.append(f"- [Turn {d.turn_number}] {d.decision}")
                if d.rationale:
                    lines.append(f"  Rationale: {d.rationale}")
            lines.append("")

        # Open items section
        if self.open_items:
            lines.append("**Open Items:**")
            # Sort by priority (high first) then description
            priority_order = {"high": 0, "medium": 1, "low": 2}
            for item in sorted(
                self.open_items,
                key=lambda x: (priority_order.get(x.priority, 1), x.description),
            ):
                priority_marker = {"high": "(!)", "medium": "", "low": "(low)"}
                marker = priority_marker.get(item.priority, "")
                lines.append(f"- {marker} {item.description}".strip())
            lines.append("")

        # Artifacts section
        if self.artifacts:
            lines.append("**Externalized Content:**")
            for artifact in sorted(self.artifacts, key=lambda x: x.artifact_id):
                lines.append(f"- [{artifact.artifact_id}] {artifact.description}")
                lines.append(f"  Hint: {artifact.rehydrate_hint}")
            lines.append("")

        # Tool outcomes section
        if self.tool_outcomes:
            lines.append("**Tool Results:**")
            for outcome in self.tool_outcomes:
                status = {"success": "✓", "failure": "✗", "partial": "~"}
                marker = status.get(outcome.outcome, "?")
                line = f"- {marker} {outcome.tool_name}"
                if outcome.key_fields:
                    # Sort keys for determinism
                    fields = ", ".join(
                        f"{k}={v}" for k, v in sorted(outcome.key_fields.items())
                    )
                    line += f" ({fields})"
                if outcome.error_message:
                    line += f" Error: {outcome.error_message}"
                lines.append(line)
            lines.append("")

        # Current state section
        if self.current_task:
            lines.append(f"**Current Task:** {self.current_task}")
            lines.append("")

        if self.current_plan:
            lines.append("**Current Plan:**")
            for i, step in enumerate(self.current_plan, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "schema_version": STRUCTURED_SUMMARY_SCHEMA_VERSION,
            "span": self.span.to_dict(),
            "facts": self.facts,
            "decisions": [d.to_dict() for d in self.decisions],
            "open_items": [i.to_dict() for i in self.open_items],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "tool_outcomes": [t.to_dict() for t in self.tool_outcomes],
            "current_task": self.current_task,
            "current_plan": self.current_plan,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredSummary:
        """Deserialize from dictionary."""
        return cls(
            span=SpanReference.from_dict(data["span"]),
            facts=data.get("facts", []),
            decisions=[Decision.from_dict(d) for d in data.get("decisions", [])],
            open_items=[OpenItem.from_dict(i) for i in data.get("open_items", [])],
            artifacts=[ArtifactReference.from_dict(a) for a in data.get("artifacts", [])],
            tool_outcomes=[ToolOutcome.from_dict(t) for t in data.get("tool_outcomes", [])],
            current_task=data.get("current_task"),
            current_plan=data.get("current_plan"),
        )

    @classmethod
    def create_empty(cls, span: SpanReference) -> StructuredSummary:
        """Create an empty summary for a span.

        Args:
            span: The span this summary covers.

        Returns:
            An empty StructuredSummary.
        """
        return cls(span=span)


@dataclass
class SummaryCacheKey:
    """Key for cached summaries.

    Summaries are expensive (LLM calls) and must be cached. The cache key
    includes all factors that affect the summary to avoid cache poisoning.

    Attributes:
        content_hash: SHA256 hash of span content.
        schema_version: StructuredSummary schema version.
        policy_version: Compaction policy version.
        model_id: Summarization model ID.
        prompt_version: Hash of summarization prompt.
    """

    content_hash: str
    schema_version: str
    policy_version: str
    model_id: str
    prompt_version: str

    def to_string(self) -> str:
        """Convert to cache key string."""
        return (
            f"{self.content_hash}:{self.schema_version}:"
            f"{self.policy_version}:{self.model_id}:{self.prompt_version}"
        )

    @classmethod
    def from_string(cls, key: str) -> SummaryCacheKey:
        """Parse from cache key string."""
        parts = key.split(":")
        if len(parts) != 5:
            raise ValueError(f"Invalid cache key format: {key}")
        return cls(
            content_hash=parts[0],
            schema_version=parts[1],
            policy_version=parts[2],
            model_id=parts[3],
            prompt_version=parts[4],
        )
