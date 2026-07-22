# eval-harness Package (agent-framework-eval-harness)

Benchmark evaluation harness: runs any `agent-framework` agent through GAIA, τ²-bench, and other external benchmarks; returns standard `EvalResults` from `agent-framework-core`.

## Module Structure

```
agent_framework_eval_harness/
├── __init__.py         # Public API: EvalHarness, Benchmark, GAIABenchmark, TauBenchmark
├── _harness.py         # EvalHarness class
├── _report.py          # print_summary helper
├── _types.py           # Benchmark protocol
└── benchmarks/
    ├── __init__.py     # Re-exports GAIABenchmark, TauBenchmark
    ├── _gaia.py        # GAIABenchmark + gaia_scorer
    └── _tau2.py        # TauBenchmark (wraps agent-framework-lab tau2)
```

## Key Classes

- **`EvalHarness`** — accepts a `SupportsAgentRun` agent; delegates to benchmarks
- **`Benchmark`** — protocol: `name: str` + `async run(agent, *, evaluators, eval_name) -> list[EvalResults]`
- **`GAIABenchmark`** — GAIA L1/L2/L3; runs agent in parallel; scores with `gaia_scorer`; returns `EvalResults`
- **`TauBenchmark`** — τ²-bench airline domain; multi-turn simulation; returns pass/fail `EvalResults`
- **`gaia_scorer`** — public function: GAIA official normalized exact-match

## Design Constraints

- The agent is **always constructed by the caller** — benchmarks accept `SupportsAgentRun`, never build agents themselves.
- Returns `list[EvalResults]`: first entry is the benchmark's built-in scorer; additional `evaluators=` follow.
- GAIA runs tasks in parallel controlled by `parallel=` (default 1). Increase for faster runs.
- GAIA file-attachment tasks are skipped by default (`skip_file_attachments=True`). Phase 2 adds multimodal support.
- `TauBenchmark` requires `agent-framework-lab[tau2]` and `tau2` from git (see README).

## Adding a New Benchmark

Implement the `Benchmark` protocol:

```python
from collections.abc import Sequence
from agent_framework import EvalResults, Evaluator, SupportsAgentRun

class MyBenchmark:
    name = "my-benchmark"

    async def run(
        self,
        agent: SupportsAgentRun,
        *,
        evaluators: Sequence[Evaluator] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        # 1. Load tasks
        # 2. Run agent in parallel (asyncio.Semaphore)
        # 3. Build EvalItems with expected_output stamped
        # 4. Run LocalEvaluator(my_scorer) + any additional evaluators
        # 5. Return list of EvalResults
        ...
```
