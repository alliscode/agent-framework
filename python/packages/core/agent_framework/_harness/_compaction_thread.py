# Copyright (c) Microsoft. All rights reserved.

"""Thread-backed compaction service."""

from __future__ import annotations

import logging
from typing import Any

from ._compaction import CompactionCoordinator, CompactionPlan, ProviderAwareTokenizer, TokenBudget, TurnContext
from ._compaction_helpers import tokens_to_free_for_pressure

logger = logging.getLogger(__name__)


class CompactionThreadService:
    """Runs compaction when direct thread/message-store access is available."""

    def __init__(self, *, coordinator: CompactionCoordinator, tokenizer: ProviderAwareTokenizer) -> None:
        self._coordinator = coordinator
        self._tokenizer = tokenizer

    async def compact_thread(
        self,
        thread: Any,
        current_plan: CompactionPlan | None,
        budget: TokenBudget,
        turn_number: int,
    ) -> CompactionPlan:
        """Run compaction for a thread-like object exposing message_store."""
        if getattr(thread, "message_store", None) is not None:
            messages = await thread.message_store.list_messages()
            current_tokens = sum(self._tokenizer.count_tokens(msg.text) for msg in messages)
        else:
            current_tokens = 0

        tokens_over = budget.tokens_over_threshold(current_tokens)
        tokens_to_free = tokens_to_free_for_pressure(tokens_over=tokens_over, max_input_tokens=budget.max_input_tokens)
        if tokens_to_free == 0:
            return current_plan or CompactionPlan.create_empty(thread_id="harness")

        turn_context = TurnContext(turn_number=turn_number)
        result = await self._coordinator.compact(
            thread,
            current_plan,
            budget,
            self._tokenizer,
            tokens_to_free=tokens_to_free,
            turn_context=turn_context,
        )
        logger.info(
            "CompactionThreadService: Compaction complete - freed %d tokens, applied %d/%d proposals",
            result.tokens_freed,
            result.proposals_applied,
            result.proposals_generated,
        )
        return result.plan
