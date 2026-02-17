# Copyright (c) Microsoft. All rights reserved.

"""Shadow-mode compaction candidate service."""

from __future__ import annotations

import logging
from typing import Any

from .._workflows._conversation_state import decode_chat_messages
from ._compaction import CacheThreadAdapter, CompactionCoordinator, ProviderAwareTokenizer, TokenBudget, TurnContext
from ._compaction_helpers import summarize_applied_strategies, tokens_to_free_for_pressure
from ._state_store import HarnessStateStore

logger = logging.getLogger(__name__)


class CompactionShadowService:
    """Publishes non-mutating compaction candidate metadata in shadow mode."""

    def __init__(
        self,
        *,
        coordinator: CompactionCoordinator,
        tokenizer: ProviderAwareTokenizer,
    ) -> None:
        self._coordinator = coordinator
        self._tokenizer = tokenizer
        self._state_store = HarnessStateStore()

    async def publish_candidate(
        self,
        *,
        ctx: Any,
        owner_mode: str,
        turn_number: int,
        current_tokens: int,
        budget: TokenBudget,
        strategies_available: list[str],
    ) -> None:
        if owner_mode != "shadow":
            await self._state_store.set_compaction_shadow_candidate(ctx, None)
            return

        candidate = {
            "candidate_owner": "compaction_executor",
            "turn_number": turn_number,
            "current_tokens": current_tokens,
            "soft_threshold": budget.soft_threshold,
            "tokens_over_threshold": budget.tokens_over,
            "would_compact": budget.is_under_pressure,
            "blocking": budget.is_blocking,
            "strategies_available": strategies_available,
        }
        if budget.is_under_pressure:
            simulation = await self._simulate_candidate_compaction(
                ctx=ctx,
                budget=budget,
                current_tokens=current_tokens,
                turn_number=turn_number,
            )
            candidate.update(simulation)
        await self._state_store.set_compaction_shadow_candidate(ctx, candidate)

    async def _simulate_candidate_compaction(
        self,
        *,
        ctx: Any,
        budget: TokenBudget,
        current_tokens: int,
        turn_number: int,
    ) -> dict[str, Any]:
        snapshot = await self._state_store.get_shared_turn_buffer_payload(ctx)
        if snapshot is None:
            return {"simulation_available": False, "simulation_reason": "missing_shared_buffer_snapshot"}

        if not isinstance(snapshot, dict):
            return {"simulation_available": False, "simulation_reason": "invalid_shared_buffer_snapshot"}

        snapshot_messages = snapshot.get("messages", [])
        if not isinstance(snapshot_messages, list):
            return {"simulation_available": False, "simulation_reason": "invalid_shared_buffer_messages"}

        messages = decode_chat_messages(snapshot_messages)
        if not messages:
            return {"simulation_available": False, "simulation_reason": "empty_shared_buffer_messages"}

        tokens_over = budget.tokens_over_threshold(current_tokens)
        tokens_to_free = tokens_to_free_for_pressure(tokens_over=tokens_over, max_input_tokens=budget.max_input_tokens)
        if tokens_to_free <= 0:
            return {"simulation_available": True, "candidate_effective_compaction": False, "candidate_tokens_freed": 0}

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
            logger.warning("CompactionExecutor: Shadow candidate simulation failed", exc_info=True)
            return {"simulation_available": False, "simulation_reason": "simulation_failed"}

        candidate_strategies_applied = summarize_applied_strategies(result.plan)

        return {
            "simulation_available": True,
            "shared_snapshot_message_count": len(messages),
            "candidate_tokens_freed": result.tokens_freed,
            "candidate_projected_tokens_after": max(0, current_tokens - result.tokens_freed),
            "candidate_proposals_generated": result.proposals_generated,
            "candidate_proposals_applied": result.proposals_applied,
            "candidate_strategies_applied": candidate_strategies_applied,
            "candidate_effective_compaction": bool(candidate_strategies_applied) or result.tokens_freed > 0,
        }
