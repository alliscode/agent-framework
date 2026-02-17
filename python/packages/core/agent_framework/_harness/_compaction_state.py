# Copyright (c) Microsoft. All rights reserved.

"""Shared-state access helpers for compaction workflow."""

from __future__ import annotations

from typing import Any, cast

from ._compaction import CompactionPlan, TokenBudget
from ._compaction_owner_mode import normalize_compaction_owner_mode
from ._state_store import HarnessStateStore


class CompactionStateStore:
    """Typed accessor for compaction-related shared workflow state."""

    def __init__(self, *, max_input_tokens: int, soft_threshold_percent: float) -> None:
        self._max_input_tokens = max_input_tokens
        self._soft_threshold_percent = soft_threshold_percent
        self._state_store = HarnessStateStore()

    async def get_or_create_budget(self, ctx: Any) -> TokenBudget:
        budget_data = await self._state_store.get_token_budget_data(ctx)
        if budget_data and isinstance(budget_data, dict):
            budget_dict = cast("dict[str, Any]", budget_data)
            return TokenBudget.from_dict(budget_dict)
        return TokenBudget(
            max_input_tokens=self._max_input_tokens,
            soft_threshold_percent=self._soft_threshold_percent,
        )

    async def save_budget(self, ctx: Any, budget: TokenBudget) -> None:
        await self._state_store.set_token_budget_data(ctx, budget.to_dict())

    async def get_turn_count(self, ctx: Any) -> int:
        return await self._state_store.get_turn_count(ctx)

    async def get_owner_mode(self, ctx: Any) -> str:
        return normalize_compaction_owner_mode(await self._state_store.get_compaction_owner_mode(ctx))

    async def load_plan(self, ctx: Any) -> tuple[CompactionPlan | None, int]:
        plan_data = await self._state_store.get_compaction_plan_data(ctx)
        if plan_data and isinstance(plan_data, dict):
            plan_dict = cast("dict[str, Any]", plan_data)
            version = int(plan_dict.get("_version", plan_dict.get("thread_version", 0)))
            plan = CompactionPlan.from_dict(plan_dict)
            return plan, version
        return None, 0

    async def save_plan(self, ctx: Any, plan: CompactionPlan, version: int) -> None:
        plan_data = plan.to_dict()
        plan_data["_version"] = version + 1
        await self._state_store.set_compaction_plan_data(ctx, plan_data)

    async def append_metrics(self, ctx: Any, metrics: dict[str, Any]) -> None:
        existing = await self._state_store.get_compaction_metrics(ctx)
        existing.append(metrics)
        if len(existing) > 100:
            existing = existing[-100:]
        await self._state_store.set_compaction_metrics(ctx, existing)
