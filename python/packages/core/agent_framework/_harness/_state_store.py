# Copyright (c) Microsoft. All rights reserved.

"""Typed shared-state accessors for harness runtime state."""

from __future__ import annotations

from typing import Any

from .._workflows._workflow_context import WorkflowContext
from .._workflows._const import WORKFLOW_RUN_KWARGS_KEY
from ._constants import (
    HARNESS_COMPACTION_METRICS_KEY,
    HARNESS_COMPACTION_OWNER_MODE_KEY,
    HARNESS_COMPACTION_PLAN_KEY,
    HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY,
    HARNESS_COVERAGE_LEDGER_KEY,
    HARNESS_INITIAL_MESSAGE_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_PENDING_TOOL_CALLS_KEY,
    HARNESS_PROGRESS_TRACKER_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_STOP_REASON_KEY,
    HARNESS_STOP_POLICY_PROFILE_KEY,
    HARNESS_TASK_CONTRACT_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
    HARNESS_SHARED_TURN_BUFFER_KEY,
    HARNESS_WORK_ITEM_LEDGER_KEY,
    HARNESS_WORK_COMPLETE_RETRY_COUNT_KEY,
)
from ._state import HarnessEvent


class HarnessStateStore:
    """Typed facade over workflow shared-state for common harness keys."""

    async def get_turn_count(self, ctx: WorkflowContext[Any, Any]) -> int:
        return await self._get_int(ctx, HARNESS_TURN_COUNT_KEY, default=0)

    async def set_turn_count(self, ctx: WorkflowContext[Any, Any], value: int) -> None:
        await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, value)

    async def get_max_turns(self, ctx: WorkflowContext[Any, Any]) -> int:
        return await self._get_int(ctx, HARNESS_MAX_TURNS_KEY, default=50)

    async def set_max_turns(self, ctx: WorkflowContext[Any, Any], value: int) -> None:
        await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, value)

    async def get_status(self, ctx: WorkflowContext[Any, Any]) -> str | None:
        return await self._get_optional(ctx, HARNESS_STATUS_KEY)

    async def set_status(self, ctx: WorkflowContext[Any, Any], value: str) -> None:
        await ctx.set_shared_state(HARNESS_STATUS_KEY, value)

    async def get_stop_reason(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_STOP_REASON_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_stop_reason(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any] | None) -> None:
        await ctx.set_shared_state(HARNESS_STOP_REASON_KEY, value)

    async def get_pending_tool_calls(self, ctx: WorkflowContext[Any, Any]) -> list[Any]:
        value = await self._get_optional(ctx, HARNESS_PENDING_TOOL_CALLS_KEY)
        return list(value) if isinstance(value, list) else []

    async def set_pending_tool_calls(self, ctx: WorkflowContext[Any, Any], value: list[Any]) -> None:
        await ctx.set_shared_state(HARNESS_PENDING_TOOL_CALLS_KEY, value)

    async def get_initial_message(self, ctx: WorkflowContext[Any, Any]) -> Any:
        return await self._get_optional(ctx, HARNESS_INITIAL_MESSAGE_KEY)

    async def set_initial_message(self, ctx: WorkflowContext[Any, Any], value: Any) -> None:
        await ctx.set_shared_state(HARNESS_INITIAL_MESSAGE_KEY, value)

    async def get_compaction_owner_mode(self, ctx: WorkflowContext[Any, Any]) -> str | None:
        value = await self._get_optional(ctx, HARNESS_COMPACTION_OWNER_MODE_KEY)
        return str(value) if value is not None else None

    async def set_compaction_owner_mode(self, ctx: WorkflowContext[Any, Any], value: str) -> None:
        await ctx.set_shared_state(HARNESS_COMPACTION_OWNER_MODE_KEY, value)

    async def get_stop_policy_profile(self, ctx: WorkflowContext[Any, Any]) -> str | None:
        value = await self._get_optional(ctx, HARNESS_STOP_POLICY_PROFILE_KEY)
        return str(value) if value is not None else None

    async def set_stop_policy_profile(self, ctx: WorkflowContext[Any, Any], value: str) -> None:
        await ctx.set_shared_state(HARNESS_STOP_POLICY_PROFILE_KEY, value)

    async def get_work_complete_retry_count(self, ctx: WorkflowContext[Any, Any]) -> int:
        return await self._get_int(ctx, HARNESS_WORK_COMPLETE_RETRY_COUNT_KEY, default=0)

    async def set_work_complete_retry_count(self, ctx: WorkflowContext[Any, Any], value: int) -> None:
        await ctx.set_shared_state(HARNESS_WORK_COMPLETE_RETRY_COUNT_KEY, value)

    async def get_task_contract_data(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_TASK_CONTRACT_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_task_contract_data(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any]) -> None:
        await ctx.set_shared_state(HARNESS_TASK_CONTRACT_KEY, value)

    async def get_coverage_ledger_data(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_COVERAGE_LEDGER_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_coverage_ledger_data(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any]) -> None:
        await ctx.set_shared_state(HARNESS_COVERAGE_LEDGER_KEY, value)

    async def get_progress_tracker_data(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_PROGRESS_TRACKER_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_progress_tracker_data(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any]) -> None:
        await ctx.set_shared_state(HARNESS_PROGRESS_TRACKER_KEY, value)

    async def get_work_item_ledger_data(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_WORK_ITEM_LEDGER_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_work_item_ledger_data(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any]) -> None:
        await ctx.set_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY, value)

    async def get_token_budget_data(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_TOKEN_BUDGET_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_token_budget_data(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any]) -> None:
        await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, value)

    async def get_compaction_plan_data(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_COMPACTION_PLAN_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_compaction_plan_data(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any] | None) -> None:
        await ctx.set_shared_state(HARNESS_COMPACTION_PLAN_KEY, value)

    async def get_compaction_metrics(self, ctx: WorkflowContext[Any, Any]) -> list[dict[str, Any]]:
        value = await self._get_optional(ctx, HARNESS_COMPACTION_METRICS_KEY)
        if isinstance(value, list):
            return [dict(v) for v in value if isinstance(v, dict)]
        return []

    async def set_compaction_metrics(self, ctx: WorkflowContext[Any, Any], value: list[dict[str, Any]]) -> None:
        await ctx.set_shared_state(HARNESS_COMPACTION_METRICS_KEY, value)

    async def get_compaction_shadow_candidate(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_compaction_shadow_candidate(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any] | None) -> None:
        await ctx.set_shared_state(HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY, value)

    async def get_shared_turn_buffer_payload(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any] | None:
        value = await self._get_optional(ctx, HARNESS_SHARED_TURN_BUFFER_KEY)
        return dict(value) if isinstance(value, dict) else None

    async def set_shared_turn_buffer_payload(self, ctx: WorkflowContext[Any, Any], value: dict[str, Any]) -> None:
        await ctx.set_shared_state(HARNESS_SHARED_TURN_BUFFER_KEY, value)

    async def get_continuation_count(self, ctx: WorkflowContext[Any, Any], *, key: str) -> int:
        return await self._get_int(ctx, key, default=0)

    async def set_continuation_count(self, ctx: WorkflowContext[Any, Any], *, key: str, value: int) -> None:
        await ctx.set_shared_state(key, value)

    async def get_run_kwargs(self, ctx: WorkflowContext[Any, Any]) -> dict[str, Any]:
        value = await self._get_optional(ctx, WORKFLOW_RUN_KWARGS_KEY)
        return dict(value) if isinstance(value, dict) else {}

    async def get_transcript_data(self, ctx: WorkflowContext[Any, Any]) -> list[dict[str, Any]]:
        try:
            transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            if transcript:
                return list(transcript)
        except KeyError:
            pass
        return []

    async def set_transcript_data(self, ctx: WorkflowContext[Any, Any], transcript: list[dict[str, Any]]) -> None:
        await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, transcript)

    async def append_transcript_event(self, ctx: WorkflowContext[Any, Any], event: HarnessEvent) -> None:
        transcript = await self.get_transcript_data(ctx)
        transcript.append(event.to_dict())
        await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, transcript)

    async def set_completion_state(
        self,
        ctx: WorkflowContext[Any, Any],
        *,
        status: str,
        stop_reason: dict[str, Any],
    ) -> None:
        await ctx.set_shared_state(HARNESS_STATUS_KEY, status)
        await ctx.set_shared_state(HARNESS_STOP_REASON_KEY, stop_reason)

    async def _get_optional(self, ctx: WorkflowContext[Any, Any], key: str) -> Any:
        try:
            return await ctx.get_shared_state(key)
        except KeyError:
            return None

    async def _get_int(self, ctx: WorkflowContext[Any, Any], key: str, *, default: int) -> int:
        try:
            return int(await ctx.get_shared_state(key) or default)
        except KeyError:
            return default
