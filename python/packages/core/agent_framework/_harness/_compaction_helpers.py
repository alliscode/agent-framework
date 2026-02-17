# Copyright (c) Microsoft. All rights reserved.

"""Shared helper utilities for compaction flows."""

from __future__ import annotations

from ._compaction import CompactionPlan


def tokens_to_free_for_pressure(*, tokens_over: int, max_input_tokens: int) -> int:
    """Compute target token reduction for pressure recovery."""
    return max(0, tokens_over + int(max_input_tokens * 0.1))


def summarize_applied_strategies(plan: CompactionPlan) -> list[str]:
    """Return normalized list of strategies represented in a plan."""
    strategies: list[str] = []
    if plan.clearings:
        strategies.append("clear")
    if plan.summarizations:
        strategies.append("summarize")
    if plan.externalizations:
        strategies.append("externalize")
    if plan.drops:
        strategies.append("drop")
    return strategies
