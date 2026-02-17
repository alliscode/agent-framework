# Copyright (c) Microsoft. All rights reserved.

"""Policy decisions for compaction executor control flow."""

from __future__ import annotations

from dataclasses import dataclass

from ._compaction import CompactionPlan, TokenBudget


@dataclass(frozen=True)
class OwnerFallbackPolicy:
    """Owner-mode fallback gate policy."""

    bootstrap_turn_limit: int = 2
    enforce_gate: bool = True


class CompactionPolicy:
    """Encapsulates compaction control-flow decisions."""

    _BOOTSTRAP_FALLBACK_REASONS = {
        "missing_shared_turn_buffer",
        "invalid_shared_turn_buffer",
        "invalid_shared_turn_buffer_messages",
        "empty_shared_turn_buffer",
    }
    _SAFETY_FALLBACK_REASONS = {"owner_compaction_failed"}

    def __init__(self, *, owner_fallback_policy: OwnerFallbackPolicy | None = None) -> None:
        self._owner_fallback_policy = owner_fallback_policy or OwnerFallbackPolicy()

    @staticmethod
    def is_under_pressure(budget: TokenBudget) -> bool:
        return budget.is_under_pressure

    @staticmethod
    def should_attempt_owner_path(owner_mode: str) -> bool:
        return owner_mode == "compaction_executor"

    @staticmethod
    def is_blocking(budget: TokenBudget) -> bool:
        return budget.is_blocking

    @staticmethod
    def ensure_plan(plan: CompactionPlan | None) -> CompactionPlan:
        return plan if plan is not None else CompactionPlan.create_empty(thread_id="harness")

    def is_owner_fallback_allowed(self, *, owner_mode: str, owner_fallback_reason: str | None, turn_number: int) -> bool:
        if owner_mode != "compaction_executor":
            return True
        if not owner_fallback_reason:
            return True
        if owner_fallback_reason in self._SAFETY_FALLBACK_REASONS:
            return True
        if owner_fallback_reason in self._BOOTSTRAP_FALLBACK_REASONS:
            return turn_number <= self._owner_fallback_policy.bootstrap_turn_limit
        return False

    def is_owner_fallback_gate_violation(
        self,
        *,
        owner_mode: str,
        owner_fallback_reason: str | None,
        turn_number: int,
    ) -> bool:
        if not self._owner_fallback_policy.enforce_gate:
            return False
        return not self.is_owner_fallback_allowed(
            owner_mode=owner_mode,
            owner_fallback_reason=owner_fallback_reason,
            turn_number=turn_number,
        )
