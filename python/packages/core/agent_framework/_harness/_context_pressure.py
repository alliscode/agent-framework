# Copyright (c) Microsoft. All rights reserved.

"""Token budget and estimation utilities for Agent Harness.

This module provides the TokenBudget class used for SharedState pressure signaling
between executors, and utility functions for estimating token counts.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenBudget:
    """Token budget configuration for context pressure.

    This is the v1 budget used for SharedState pressure signaling between executors.
    It stores a simple current_estimate and checks it against a soft threshold.
    For compaction-internal overhead tracking (system prompt, tool schemas, formatting),
    see ``_compaction._tokenizer.TokenBudget``.

    Attributes:
        max_input_tokens: Maximum tokens allowed in input context.
        soft_threshold_percent: Percentage at which to trigger pressure strategies.
        current_estimate: Current estimated token count.
    """

    max_input_tokens: int = 128000
    soft_threshold_percent: float = 0.80
    blocking_threshold_percent: float = 0.95
    current_estimate: int = 0

    @property
    def soft_threshold(self) -> int:
        """Calculate the soft threshold in tokens."""
        return int(self.max_input_tokens * self.soft_threshold_percent)

    @property
    def blocking_threshold(self) -> int:
        """Calculate the blocking threshold in tokens (must compact before LLM call)."""
        return int(self.max_input_tokens * self.blocking_threshold_percent)

    @property
    def is_under_pressure(self) -> bool:
        """Check if context is under pressure (above soft threshold)."""
        return self.current_estimate >= self.soft_threshold

    @property
    def is_blocking(self) -> bool:
        """Check if context requires blocking compaction before proceeding."""
        return self.current_estimate >= self.blocking_threshold

    @property
    def tokens_over_threshold(self) -> int:
        """Calculate how many tokens over the soft threshold."""
        return max(0, self.current_estimate - self.soft_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_input_tokens": self.max_input_tokens,
            "soft_threshold_percent": self.soft_threshold_percent,
            "blocking_threshold_percent": self.blocking_threshold_percent,
            "current_estimate": self.current_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenBudget":
        """Deserialize from dictionary."""
        return cls(
            max_input_tokens=data.get("max_input_tokens", 128000),
            soft_threshold_percent=data.get("soft_threshold_percent", 0.80),
            blocking_threshold_percent=data.get("blocking_threshold_percent", 0.95),
            current_estimate=data.get("current_estimate", 0),
        )


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
