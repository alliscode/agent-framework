# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for EvalHarness."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import EvalResults

from agent_framework_eval_harness import EvalHarness
from agent_framework_eval_harness._types import Benchmark


class _SimpleBenchmark:
    name = "mock"

    async def run(
        self,
        agent: object,
        *,
        evaluators: Sequence[object] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        return [EvalResults(provider="mock", result_counts={"passed": 3, "failed": 1})]


class _CaptureBenchmark:
    """Records the evaluators it receives."""

    name = "capture"
    received_evaluators: list[object] = []
    received_eval_name: str | None = None

    async def run(
        self,
        agent: object,
        *,
        evaluators: Sequence[object] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        self.received_evaluators = list(evaluators or [])
        self.received_eval_name = eval_name
        return [EvalResults(provider="capture")]


@pytest.fixture()
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.run = AsyncMock()
    return agent


async def test_harness_delegates_to_benchmark(mock_agent: MagicMock) -> None:
    harness = EvalHarness(agent=mock_agent)
    results = await harness.run(_SimpleBenchmark())
    assert len(results) == 1
    assert results[0].provider == "mock"
    assert results[0].passed == 3


async def test_harness_wraps_single_evaluator(mock_agent: MagicMock) -> None:
    """A single Evaluator passed to run() should be wrapped in a list."""
    from agent_framework import LocalEvaluator

    bench = _CaptureBenchmark()
    harness = EvalHarness(agent=mock_agent)
    single_eval = LocalEvaluator()

    await harness.run(bench, evaluators=single_eval)
    assert bench.received_evaluators == [single_eval]


async def test_harness_passes_evaluator_sequence(mock_agent: MagicMock) -> None:
    from agent_framework import LocalEvaluator

    bench = _CaptureBenchmark()
    harness = EvalHarness(agent=mock_agent)
    evals = [LocalEvaluator(), LocalEvaluator()]

    await harness.run(bench, evaluators=evals)
    assert bench.received_evaluators == evals


async def test_harness_passes_none_evaluators(mock_agent: MagicMock) -> None:
    bench = _CaptureBenchmark()
    harness = EvalHarness(agent=mock_agent)

    await harness.run(bench)
    assert bench.received_evaluators == []


async def test_harness_passes_eval_name(mock_agent: MagicMock) -> None:
    bench = _CaptureBenchmark()
    harness = EvalHarness(agent=mock_agent)

    await harness.run(bench, eval_name="my-run")
    assert bench.received_eval_name == "my-run"


def test_benchmark_protocol_satisfied() -> None:
    """_SimpleBenchmark satisfies the Benchmark runtime protocol."""
    assert isinstance(_SimpleBenchmark(), Benchmark)
