# Copyright (c) Microsoft. All rights reserved.

"""Owner-mode compaction service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from .._workflows._conversation_state import decode_chat_messages, encode_chat_messages
from ._compaction import CacheThreadAdapter, CompactionCoordinator, ProviderAwareTokenizer, TokenBudget, TurnContext
from ._compaction_helpers import summarize_applied_strategies, tokens_to_free_for_pressure
from ._compaction_view import (
    apply_compaction_plan_to_messages,
    apply_direct_clear_to_messages,
    ensure_message_ids,
)
from ._state_store import HarnessStateStore

logger = logging.getLogger(__name__)


class OwnerFallbackReason(str, Enum):
    """Canonical fallback reasons for owner-mode compaction path."""

    MISSING_SHARED_TURN_BUFFER = "missing_shared_turn_buffer"
    INVALID_SHARED_TURN_BUFFER = "invalid_shared_turn_buffer"
    INVALID_SHARED_TURN_BUFFER_MESSAGES = "invalid_shared_turn_buffer_messages"
    EMPTY_SHARED_TURN_BUFFER = "empty_shared_turn_buffer"
    UNDER_THRESHOLD = "under_threshold"
    OWNER_COMPACTION_FAILED = "owner_compaction_failed"
    EMPTY_PLAN = "empty_plan"


@dataclass(frozen=True)
class OwnerCompactionResult:
    """Typed result of owner-mode compaction attempt."""

    applied: bool
    fallback_reason: OwnerFallbackReason | None = None
    tokens_freed: int = 0
    proposals_applied: int = 0
    strategies_applied: list[str] | None = None
    shared_snapshot_version: int | None = None
    shared_message_count: int | None = None

    @classmethod
    def fallback(cls, reason: OwnerFallbackReason) -> "OwnerCompactionResult":
        return cls(applied=False, fallback_reason=reason)

    @classmethod
    def applied_result(
        cls,
        *,
        tokens_freed: int,
        proposals_applied: int,
        strategies_applied: list[str],
        shared_snapshot_version: int,
        shared_message_count: int,
    ) -> "OwnerCompactionResult":
        return cls(
            applied=True,
            tokens_freed=tokens_freed,
            proposals_applied=proposals_applied,
            strategies_applied=strategies_applied,
            shared_snapshot_version=shared_snapshot_version,
            shared_message_count=shared_message_count,
        )


class CompactionOwnerService:
    """Executes owner-mode compaction against shared turn-buffer snapshots."""

    def __init__(
        self,
        *,
        coordinator: CompactionCoordinator,
        tokenizer: ProviderAwareTokenizer,
        save_budget: Callable[[Any, TokenBudget], Awaitable[None]],
    ) -> None:
        self._coordinator = coordinator
        self._tokenizer = tokenizer
        self._save_budget = save_budget
        self._state_store = HarnessStateStore()

    async def try_apply(
        self,
        *,
        ctx: Any,
        budget: TokenBudget,
        current_tokens: int,
        turn_number: int,
    ) -> OwnerCompactionResult:
        snapshot = await self._state_store.get_shared_turn_buffer_payload(ctx)
        if snapshot is None:
            return OwnerCompactionResult.fallback(OwnerFallbackReason.MISSING_SHARED_TURN_BUFFER)

        if not isinstance(snapshot, dict):
            return OwnerCompactionResult.fallback(OwnerFallbackReason.INVALID_SHARED_TURN_BUFFER)

        raw_messages = snapshot.get("messages", [])
        if not isinstance(raw_messages, list):
            return OwnerCompactionResult.fallback(OwnerFallbackReason.INVALID_SHARED_TURN_BUFFER_MESSAGES)

        messages = decode_chat_messages(raw_messages)
        if not messages:
            return OwnerCompactionResult.fallback(OwnerFallbackReason.EMPTY_SHARED_TURN_BUFFER)
        ensure_message_ids(messages)

        tokens_over = budget.tokens_over_threshold(current_tokens)
        tokens_to_free = tokens_to_free_for_pressure(tokens_over=tokens_over, max_input_tokens=budget.max_input_tokens)
        if tokens_to_free <= 0:
            return OwnerCompactionResult.fallback(OwnerFallbackReason.UNDER_THRESHOLD)

        turn_context = TurnContext(turn_number=turn_number)
        try:
            result = await self._coordinator.compact(
                CacheThreadAdapter(messages),  # type: ignore[arg-type]
                None,
                budget,
                self._tokenizer,
                tokens_to_free=tokens_to_free,
                turn_context=turn_context,
            )
        except Exception:
            logger.warning("CompactionExecutor: owner-mode compaction failed", exc_info=True)
            return OwnerCompactionResult.fallback(OwnerFallbackReason.OWNER_COMPACTION_FAILED)

        if result.plan is None or result.plan.is_empty:
            compacted_messages, cleared_count, direct_tokens_freed = apply_direct_clear_to_messages(
                messages,
                preserve_recent_messages=2,
                target_tokens_to_free=tokens_to_free,
            )
            if cleared_count <= 0:
                return OwnerCompactionResult.fallback(OwnerFallbackReason.EMPTY_PLAN)
            snapshot_version = int(snapshot.get("version", 0))
            await self._state_store.set_shared_turn_buffer_payload(
                ctx,
                {
                    "version": snapshot_version + 1,
                    "message_count": len(compacted_messages),
                    "messages": encode_chat_messages(compacted_messages),
                },
            )
            await self._state_store.set_compaction_plan_data(ctx, None)
            budget.current_estimate = max(0, current_tokens - max(0, direct_tokens_freed))
            await self._save_budget(ctx, budget)
            return OwnerCompactionResult.applied_result(
                tokens_freed=direct_tokens_freed,
                proposals_applied=cleared_count,
                strategies_applied=["clear"],
                shared_snapshot_version=snapshot_version + 1,
                shared_message_count=len(compacted_messages),
            )

        compacted_messages = apply_compaction_plan_to_messages(messages, result.plan)
        snapshot_version = int(snapshot.get("version", 0))
        await self._state_store.set_shared_turn_buffer_payload(
            ctx,
            {
                "version": snapshot_version + 1,
                "message_count": len(compacted_messages),
                "messages": encode_chat_messages(compacted_messages),
            },
        )

        # Owner mode applies the plan directly to shared snapshot state.
        # Keep shared compaction plan empty to avoid stale-plan carryover.
        await self._state_store.set_compaction_plan_data(ctx, None)
        budget.current_estimate = max(0, current_tokens - max(0, result.tokens_freed))
        await self._save_budget(ctx, budget)

        strategies_applied = summarize_applied_strategies(result.plan)

        return OwnerCompactionResult.applied_result(
            tokens_freed=result.tokens_freed,
            proposals_applied=result.proposals_applied,
            strategies_applied=strategies_applied,
            shared_snapshot_version=snapshot_version + 1,
            shared_message_count=len(compacted_messages),
        )
