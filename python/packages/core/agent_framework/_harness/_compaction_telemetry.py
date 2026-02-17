# Copyright (c) Microsoft. All rights reserved.

"""Typed payload builders for compaction telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ._compaction import TokenBudget


@dataclass(frozen=True)
class OwnerCompactionOutcome:
    """Owner-mode compaction outcome payload source."""

    turn_number: int
    tokens_before: int
    tokens_freed: int
    proposals_applied: int
    strategies_applied: list[str]
    owner_mode: str
    under_pressure: bool
    owner_path_applied: bool = True

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn_number": self.turn_number,
            "tokens_before": self.tokens_before,
            "tokens_freed": self.tokens_freed,
            "proposals_applied": self.proposals_applied,
            "under_pressure": self.under_pressure,
            "compaction_owner_mode": self.owner_mode,
            "owner_path_applied": self.owner_path_applied,
            "strategies_applied": self.strategies_applied,
        }

    def lifecycle_completed_payload(self) -> dict[str, Any]:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": max(0, self.tokens_before - self.tokens_freed),
            "tokens_freed": self.tokens_freed,
            "proposals_applied": self.proposals_applied,
            "strategies_applied": self.strategies_applied,
            "compaction_owner_mode": self.owner_mode,
            "owner_path_applied": self.owner_path_applied,
        }


def pressure_metrics_payload(
    *,
    turn_number: int,
    tokens_before: int,
    tokens_freed: int,
    proposals_applied: int,
    under_pressure: bool,
    owner_mode: str,
    owner_fallback_allowed: bool | None = None,
    owner_fallback_gate_violation: bool | None = None,
) -> dict[str, Any]:
    """Metrics payload for pressure-signaled (fallback/non-owner) path."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_number": turn_number,
        "tokens_before": tokens_before,
        "tokens_freed": tokens_freed,
        "proposals_applied": proposals_applied,
        "under_pressure": under_pressure,
        "compaction_owner_mode": owner_mode,
        "owner_fallback_allowed": owner_fallback_allowed,
        "owner_fallback_gate_violation": owner_fallback_gate_violation,
    }


def compaction_started_payload(
    *,
    current_tokens: int,
    budget: TokenBudget,
    strategies_available: list[str],
    owner_mode: str,
    owner_fallback_reason: str | None,
    owner_fallback_allowed: bool | None = None,
    owner_fallback_gate_violation: bool | None = None,
) -> dict[str, Any]:
    """Lifecycle payload for compaction_started."""
    return {
        "current_tokens": current_tokens,
        "soft_threshold": budget.soft_threshold,
        "tokens_over_threshold": budget.tokens_over,
        "strategies_available": strategies_available,
        "compaction_owner_mode": owner_mode,
        "owner_fallback_reason": owner_fallback_reason,
        "owner_fallback_allowed": owner_fallback_allowed,
        "owner_fallback_gate_violation": owner_fallback_gate_violation,
    }


def context_pressure_payload(
    *,
    plan_updated: bool,
    tokens_freed: int,
    proposals_applied: int,
    owner_mode: str,
    owner_fallback_reason: str | None,
    owner_fallback_allowed: bool | None = None,
    owner_fallback_gate_violation: bool | None = None,
) -> dict[str, Any]:
    """Lifecycle payload for context_pressure."""
    return {
        "plan_updated": plan_updated,
        "tokens_freed": tokens_freed,
        "proposals_applied": proposals_applied,
        "compaction_owner_mode": owner_mode,
        "owner_fallback_reason": owner_fallback_reason,
        "owner_fallback_allowed": owner_fallback_allowed,
        "owner_fallback_gate_violation": owner_fallback_gate_violation,
    }
