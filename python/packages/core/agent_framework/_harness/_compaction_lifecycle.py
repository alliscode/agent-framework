# Copyright (c) Microsoft. All rights reserved.

"""Lifecycle event emitter for compaction workflow."""

from __future__ import annotations

from ._compaction import TokenBudget
from ._compaction_telemetry import (
    OwnerCompactionOutcome,
    compaction_started_payload,
    context_pressure_payload,
)
from ._state import HarnessLifecycleEvent


class CompactionLifecycleEmitter:
    """Emits compaction lifecycle events with normalized payloads."""

    async def emit_owner_completed(self, ctx: object, *, turn_number: int, owner_outcome: OwnerCompactionOutcome) -> None:
        await ctx.add_event(  # type: ignore[attr-defined]
            HarnessLifecycleEvent(
                event_type="compaction_completed",
                turn_number=turn_number,
                data=owner_outcome.lifecycle_completed_payload(),
            )
        )

    async def emit_started(
        self,
        ctx: object,
        *,
        turn_number: int,
        current_tokens: int,
        budget: TokenBudget,
        strategies_available: list[str],
        owner_mode: str,
        owner_fallback_reason: str | None,
        owner_fallback_allowed: bool | None = None,
        owner_fallback_gate_violation: bool | None = None,
    ) -> None:
        await ctx.add_event(  # type: ignore[attr-defined]
            HarnessLifecycleEvent(
                event_type="compaction_started",
                turn_number=turn_number,
                data=compaction_started_payload(
                    current_tokens=current_tokens,
                    budget=budget,
                    strategies_available=strategies_available,
                    owner_mode=owner_mode,
                    owner_fallback_reason=owner_fallback_reason,
                    owner_fallback_allowed=owner_fallback_allowed,
                    owner_fallback_gate_violation=owner_fallback_gate_violation,
                ),
            )
        )

    async def emit_pressure(
        self,
        ctx: object,
        *,
        turn_number: int,
        plan_updated: bool,
        tokens_freed: int,
        proposals_applied: int,
        owner_mode: str,
        owner_fallback_reason: str | None,
        owner_fallback_allowed: bool | None = None,
        owner_fallback_gate_violation: bool | None = None,
    ) -> None:
        await ctx.add_event(  # type: ignore[attr-defined]
            HarnessLifecycleEvent(
                event_type="context_pressure",
                turn_number=turn_number,
                data=context_pressure_payload(
                    plan_updated=plan_updated,
                    tokens_freed=tokens_freed,
                    proposals_applied=proposals_applied,
                    owner_mode=owner_mode,
                    owner_fallback_reason=owner_fallback_reason,
                    owner_fallback_allowed=owner_fallback_allowed,
                    owner_fallback_gate_violation=owner_fallback_gate_violation,
                ),
            )
        )
