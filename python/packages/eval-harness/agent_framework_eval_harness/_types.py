# Copyright (c) Microsoft. All rights reserved.

"""Protocol shared by all benchmark adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_framework import EvalResults, Evaluator, SupportsAgentRun


@runtime_checkable
class Benchmark(Protocol):
    """Protocol for benchmark suites that evaluate an agent.

    Any class with a ``name`` attribute and an async ``run`` method satisfies
    this protocol.  See :class:`~agent_framework_eval_harness.benchmarks.GAIABenchmark`
    for a reference implementation.
    """

    name: str

    async def run(
        self,
        agent: SupportsAgentRun,
        *,
        evaluators: Sequence[Evaluator] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        """Run the benchmark against *agent* and return evaluation results.

        Args:
            agent: An agent-framework agent to evaluate.  Always constructed
                by the caller; benchmarks never build agents themselves.
            evaluators: Additional evaluators (e.g. ``FoundryEvals``) run
                alongside the benchmark's built-in scorer.  The built-in
                scorer always runs first.
            eval_name: Display name for the evaluation run.

        Returns:
            A list of ``EvalResults``, one per evaluator (built-in first,
            then additional evaluators in order).
        """
        ...
