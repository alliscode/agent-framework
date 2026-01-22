# Copyright (c) Microsoft. All rights reserved.

"""Turn context for context compaction.

The TurnContext tracks state within a single turn execution, including
rehydration state needed to prevent compaction/rehydration oscillation.

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TurnContext:
    """Context passed through a single turn execution.

    This tracks turn-level state that's needed for:
    - Rehydration loop breaking (prevent oscillation)
    - Budget tracking within a turn
    - Observability/debugging

    Attributes:
        turn_number: Current turn number.
        rehydration_happened: Whether rehydration occurred this turn.
        rehydrated_tokens: Total tokens injected via rehydration.
        rehydrated_artifact_ids: IDs of artifacts that were rehydrated.
        started_at: When this turn started.
    """

    turn_number: int
    rehydration_happened: bool = False
    rehydrated_tokens: int = 0
    rehydrated_artifact_ids: list[str] = field(default_factory=lambda: [])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def should_skip_aggressive_compaction(self) -> bool:
        """Check if we should skip aggressive compaction this turn.

        When rehydration happens, we skip aggressive strategies (externalize, drop)
        to let the agent complete its work before compacting again.

        Returns:
            True if aggressive compaction should be skipped.
        """
        return self.rehydration_happened

    def record_rehydration(self, artifact_id: str, tokens: int) -> None:
        """Record that an artifact was rehydrated.

        Args:
            artifact_id: ID of the rehydrated artifact.
            tokens: Number of tokens injected.
        """
        self.rehydration_happened = True
        self.rehydrated_tokens += tokens
        if artifact_id not in self.rehydrated_artifact_ids:
            self.rehydrated_artifact_ids.append(artifact_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "turn_number": self.turn_number,
            "rehydration_happened": self.rehydration_happened,
            "rehydrated_tokens": self.rehydrated_tokens,
            "rehydrated_artifact_ids": self.rehydrated_artifact_ids,
            "started_at": self.started_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurnContext:
        """Deserialize from dictionary."""
        return cls(
            turn_number=data["turn_number"],
            rehydration_happened=data.get("rehydration_happened", False),
            rehydrated_tokens=data.get("rehydrated_tokens", 0),
            rehydrated_artifact_ids=data.get("rehydrated_artifact_ids", []),
            started_at=datetime.fromisoformat(data["started_at"]),
        )


@dataclass
class RehydrationResult:
    """Result of rehydrating an artifact.

    Returned by RehydrationInterceptor.maybe_rehydrate().

    Attributes:
        artifact_id: ID of the rehydrated artifact.
        content: The rehydrated content.
        token_count: Number of tokens in the content.
        truncated: Whether the content was truncated to fit budget.
    """

    artifact_id: str
    content: str
    token_count: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "content": self.content,
            "token_count": self.token_count,
            "truncated": self.truncated,
        }


@dataclass
class RehydrationConfig:
    """Configuration for automatic rehydration.

    Controls how the RehydrationInterceptor behaves.

    Attributes:
        enabled: Whether auto-rehydration is enabled.
        max_artifacts_per_turn: Maximum artifacts to inject per turn.
        max_tokens_per_artifact: Maximum tokens per artifact (truncate if larger).
        total_budget_tokens: Total token budget for rehydration.
        cooldown_turns: Don't re-inject same artifact for N turns.
        auto_rehydrate_sensitivities: Sensitivity levels that can be auto-rehydrated.
    """

    enabled: bool = True
    max_artifacts_per_turn: int = 3
    max_tokens_per_artifact: int = 4000
    total_budget_tokens: int = 8000
    cooldown_turns: int = 2
    auto_rehydrate_sensitivities: set[str] = field(
        default_factory=lambda: {"public", "internal"}
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "max_artifacts_per_turn": self.max_artifacts_per_turn,
            "max_tokens_per_artifact": self.max_tokens_per_artifact,
            "total_budget_tokens": self.total_budget_tokens,
            "cooldown_turns": self.cooldown_turns,
            "auto_rehydrate_sensitivities": list(self.auto_rehydrate_sensitivities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RehydrationConfig:
        """Deserialize from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            max_artifacts_per_turn=data.get("max_artifacts_per_turn", 3),
            max_tokens_per_artifact=data.get("max_tokens_per_artifact", 4000),
            total_budget_tokens=data.get("total_budget_tokens", 8000),
            cooldown_turns=data.get("cooldown_turns", 2),
            auto_rehydrate_sensitivities=set(
                data.get("auto_rehydrate_sensitivities", ["public", "internal"])
            ),
        )
