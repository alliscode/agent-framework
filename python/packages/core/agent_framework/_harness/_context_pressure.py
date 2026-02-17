# Copyright (c) Microsoft. All rights reserved.

"""Token estimation utilities for Agent Harness.

This module provides utility functions for estimating token counts.
For the unified TokenBudget class, see ``_compaction._tokenizer.TokenBudget``.
"""

from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    This is a rough approximation. For accurate counts, use a tokenizer.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    # Rough approximation: 1 token ≈ 4 characters for English
    # This varies by language and tokenizer
    return max(1, len(text) // 4)


def estimate_transcript_tokens(transcript: list[dict[str, Any]]) -> int:
    """Estimate total tokens in a transcript.

    Args:
        transcript: List of transcript events.

    Returns:
        Estimated total token count.
    """
    total = 0
    for event in transcript:
        # Serialize event to estimate its size
        import json

        event_str = json.dumps(event)
        total += estimate_tokens(event_str)
    return total
