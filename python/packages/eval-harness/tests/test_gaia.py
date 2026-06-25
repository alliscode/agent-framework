# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for GAIABenchmark scorer and task loading utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework_eval_harness.benchmarks._gaia import (
    GAIABenchmark,
    _task_from_record,
    gaia_scorer,
)

# ── gaia_scorer ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_answer, ground_truth, expected",
    [
        # String matches
        ("Paris", "Paris", True),
        ("paris", "Paris", True),
        ("  Paris  ", "Paris", True),
        ("London", "Paris", False),
        # Numeric matches
        ("42", "42", True),
        ("42.0", "42", True),
        ("41", "42", False),
        ("$1,000", "1000", True),
        ("1000.0", "1000", True),
        # Punctuation normalization
        ("hello world", "helloworld", True),
        ("hello-world", "hello world", True),
        # List matches
        ("one, two", "one, two", True),
        ("one, three", "one, two", False),
        ("1, 2", "1, 2", True),
        ("1, 3", "1, 2", False),
        # None handling
        (None, "Paris", False),
        (None, "42", False),
    ],
)
def test_gaia_scorer(model_answer: str | None, ground_truth: str, expected: bool) -> None:
    assert gaia_scorer(model_answer, ground_truth) is expected


# ── _task_from_record ─────────────────────────────────────────────────────────


def test_task_from_record_basic() -> None:
    rec = {"Question": "What is 2+2?", "Final answer": "4", "task_id": "t1", "Level": 1}
    t = _task_from_record(rec)
    assert t is not None
    assert t.question == "What is 2+2?"
    assert t.answer == "4"
    assert t.task_id == "t1"
    assert t.level == 1


def test_task_from_record_missing_answer() -> None:
    assert _task_from_record({"Question": "x", "Final answer": "?"}) is None
    assert _task_from_record({"Question": "x"}) is None


def test_task_from_record_missing_question() -> None:
    assert _task_from_record({"Final answer": "Paris"}) is None


def test_task_from_record_not_dict() -> None:
    assert _task_from_record("string") is None
    assert _task_from_record(None) is None


def test_task_from_record_with_file() -> None:
    rec = {"Question": "Analyze this", "Final answer": "42", "file_name": "data.csv"}
    t = _task_from_record(rec)
    assert t is not None
    assert t.file_name == "data.csv"


def test_task_from_record_alternative_keys() -> None:
    rec = {"question": "Q?", "answer": "A"}
    t = _task_from_record(rec)
    assert t is not None
    assert t.question == "Q?"
    assert t.answer == "A"


# ── GAIABenchmark attributes ──────────────────────────────────────────────────


def test_gaia_benchmark_defaults() -> None:
    b = GAIABenchmark()
    assert b.name == "GAIA"
    assert b.level == 1
    assert b.skip_file_attachments is True
    assert b.parallel == 1
    assert b.timeout == 300.0


def test_gaia_benchmark_custom() -> None:
    b = GAIABenchmark(level=[1, 2], max_tasks=10, parallel=4, skip_file_attachments=False)
    assert b.level == [1, 2]
    assert b.max_tasks == 10
    assert b.parallel == 4
    assert b.skip_file_attachments is False


# ── GAIABenchmark.run() with mocked internals ─────────────────────────────────


async def test_gaia_run_produces_eval_results() -> None:
    """GAIABenchmark.run() produces EvalResults without real HF download."""
    from agent_framework import AgentResponse, Message

    fake_response = AgentResponse(messages=[Message("assistant", ["Paris"])])
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_response)
    mock_agent.default_options = {"tools": []}
    mock_agent.mcp_tools = []

    fake_task = MagicMock()
    fake_task.question = "Capital of France?"
    fake_task.answer = "Paris"
    fake_task.task_id = "t1"
    fake_task.level = 1
    fake_task.file_name = None

    with (
        patch("agent_framework_eval_harness.benchmarks._gaia._ensure_data"),
        patch(
            "agent_framework_eval_harness.benchmarks._gaia._load_tasks",
            return_value=[fake_task],
        ),
    ):
        b = GAIABenchmark(level=1, max_tasks=1)
        results = await b.run(mock_agent)

    assert len(results) == 1  # only the built-in LocalEvaluator
    r = results[0]
    assert r.total == 1
    assert r.passed == 1
    assert r.failed == 0


async def test_gaia_run_failed_task_scores_as_wrong() -> None:
    """Tasks that raise exceptions should count as failed."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM error"))
    mock_agent.default_options = {"tools": []}
    mock_agent.mcp_tools = []

    fake_task = MagicMock()
    fake_task.question = "Tricky question"
    fake_task.answer = "Right answer"
    fake_task.task_id = "t2"
    fake_task.level = 1
    fake_task.file_name = None

    with (
        patch("agent_framework_eval_harness.benchmarks._gaia._ensure_data"),
        patch(
            "agent_framework_eval_harness.benchmarks._gaia._load_tasks",
            return_value=[fake_task],
        ),
    ):
        b = GAIABenchmark(level=1)
        results = await b.run(mock_agent)

    assert results[0].failed == 1
    assert results[0].passed == 0


async def test_gaia_run_no_tasks_raises() -> None:
    mock_agent = MagicMock()

    with (
        patch("agent_framework_eval_harness.benchmarks._gaia._ensure_data"),
        patch("agent_framework_eval_harness.benchmarks._gaia._load_tasks", return_value=[]),
        pytest.raises(RuntimeError, match="No GAIA tasks found"),
    ):
        await GAIABenchmark(level=1).run(mock_agent)
