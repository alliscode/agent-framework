# Copyright (c) Microsoft. All rights reserved.

"""Tests for TokenBudget and token estimation utilities."""

from typing import Any

from agent_framework._harness import (
    TokenBudget,
    estimate_tokens,
    estimate_transcript_tokens,
)

# Test TokenBudget


def test_token_budget_defaults() -> None:
    """Test that TokenBudget has sensible defaults."""
    budget = TokenBudget()

    assert budget.max_input_tokens == 128000
    assert budget.soft_threshold_percent == 0.80
    assert budget.blocking_threshold_percent == 0.95
    assert budget.current_estimate == 0
    assert budget.soft_threshold == 102400
    assert budget.blocking_threshold == 121600
    assert not budget.is_under_pressure
    assert not budget.is_blocking
    assert budget.tokens_over == 0


def test_token_budget_pressure_detection() -> None:
    """Test that TokenBudget correctly detects pressure."""
    budget = TokenBudget(max_input_tokens=10000, soft_threshold_percent=0.80)

    # Under threshold - no pressure
    budget.current_estimate = 7000
    assert not budget.is_under_pressure
    assert budget.tokens_over == 0

    # At threshold - under pressure
    budget.current_estimate = 8000
    assert budget.is_under_pressure
    assert budget.tokens_over == 0

    # Over threshold - under pressure
    budget.current_estimate = 9000
    assert budget.is_under_pressure
    assert budget.tokens_over == 1000


def test_token_budget_serialization() -> None:
    """Test TokenBudget serialization round-trip."""
    budget = TokenBudget(max_input_tokens=50000, soft_threshold_percent=0.90, current_estimate=40000)

    serialized = budget.to_dict()
    restored = TokenBudget.from_dict(serialized)

    assert restored.max_input_tokens == 50000
    assert restored.soft_threshold_percent == 0.90
    assert restored.current_estimate == 40000


# Test Token Estimation


def test_estimate_tokens_basic() -> None:
    """Test basic token estimation."""
    # Rough approximation: 1 token ≈ 4 characters
    text = "Hello, world!"  # 13 characters
    estimate = estimate_tokens(text)

    # Should be around 3-4 tokens
    assert estimate >= 1
    assert estimate <= 5


def test_estimate_tokens_empty() -> None:
    """Test token estimation for empty string."""
    estimate = estimate_tokens("")

    assert estimate == 1  # Minimum of 1


def test_estimate_transcript_tokens() -> None:
    """Test estimating tokens for a transcript."""
    transcript: list[dict[str, Any]] = [
        {"event_type": "turn_start", "data": {"turn_number": 1}},
        {"event_type": "agent_response", "data": {"turn_number": 1, "message": "Hello"}},
        {"event_type": "tool_call", "data": {"tool_name": "test", "args": {"x": 1}}},
    ]

    estimate = estimate_transcript_tokens(transcript)

    # Should be positive
    assert estimate > 0
