# Copyright (c) Microsoft. All rights reserved.

"""EvalHarness — evaluate an agent against benchmark suites."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_framework import EvalResults, Evaluator, SupportsAgentRun

    from ._types import Benchmark


class EvalHarness:
    """Evaluates a :class:`~agent_framework.SupportsAgentRun` agent against benchmarks.

    The agent is always constructed by the caller — the harness never tries to
    build one from config.  Any provider, tool combination, or custom middleware
    stack just works.

    Example:

    .. code-block:: python

        from azure.identity import DefaultAzureCredential
        from agent_framework import create_harness_agent
        from agent_framework.foundry import FoundryChatClient, FoundryEvals
        from agent_framework_eval_harness import EvalHarness
        from agent_framework_eval_harness.benchmarks import GAIABenchmark

        client = FoundryChatClient(credential=DefaultAzureCredential())
        agent = create_harness_agent(client=client)

        harness = EvalHarness(agent=agent)

        results = await harness.run(
            GAIABenchmark(level=1, max_tasks=20, parallel=4),
            evaluators=FoundryEvals(evaluators=[FoundryEvals.TASK_ADHERENCE]),
        )
        harness.print_summary(results)
    """

    def __init__(self, agent: SupportsAgentRun) -> None:
        """Initialize the harness with an agent instance.

        Args:
            agent: Any agent-framework agent (``Agent``, workflow, custom
                subclass, etc.) that satisfies ``SupportsAgentRun``.
        """
        self._agent = agent

    async def run(
        self,
        benchmark: Benchmark,
        *,
        evaluators: Evaluator | Sequence[Evaluator] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        """Run a benchmark and return evaluation results.

        Args:
            benchmark: The benchmark to run (e.g. ``GAIABenchmark``).
            evaluators: Additional evaluators beyond the benchmark's built-in
                scorer.  A single ``Evaluator`` is wrapped automatically.
            eval_name: Optional display name for the run.

        Returns:
            A list of ``EvalResults``, one per evaluator.
        """
        from agent_framework import Evaluator as _EvaluatorProtocol

        if evaluators is None:
            evals_seq: Sequence[Evaluator] | None = None
        elif isinstance(evaluators, _EvaluatorProtocol):
            evals_seq = [evaluators]
        else:
            evals_seq = list(evaluators)

        return await benchmark.run(self._agent, evaluators=evals_seq, eval_name=eval_name)

    def print_summary(self, results: list[EvalResults]) -> None:
        """Print a summary table of evaluation results to stdout.

        Args:
            results: Results returned by :meth:`run`.
        """
        from ._report import print_summary

        print_summary(results)
