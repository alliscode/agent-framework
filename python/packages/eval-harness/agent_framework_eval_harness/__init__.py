# Copyright (c) Microsoft. All rights reserved.

"""Benchmark evaluation harness for Microsoft Agent Framework.

Runs any ``agent-framework`` agent through external benchmark suites (GAIA,
τ²-bench) and produces :class:`~agent_framework.EvalResults` compatible with
the framework's evaluation infrastructure.

Example:

.. code-block:: python

    from agent_framework_eval_harness import EvalHarness
    from agent_framework_eval_harness.benchmarks import GAIABenchmark

    harness = EvalHarness(agent=my_agent)
    results = await harness.run(GAIABenchmark(level=1, max_tasks=20))
    harness.print_summary(results)
"""

from ._harness import EvalHarness
from ._types import Benchmark
from .benchmarks import GAIABenchmark, TauBenchmark, gaia_scorer

__all__ = [
    "Benchmark",
    "EvalHarness",
    "GAIABenchmark",
    "TauBenchmark",
    "gaia_scorer",
]
