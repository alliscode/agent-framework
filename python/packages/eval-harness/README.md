# Agent Framework Eval Harness

Benchmark evaluation harness for [Microsoft Agent Framework](https://aka.ms/agent-framework).

Runs any `agent-framework` agent through external benchmark suites (GAIA, τ²-bench) and produces
`EvalResults` compatible with the framework's evaluation infrastructure.

**The agent is always constructed by the caller** — the harness never tries to build one from config.
Any provider, tool combination, or middleware stack just works.

## Installation

```bash
# Core harness only
pip install agent-framework-eval-harness

# With GAIA support
pip install "agent-framework-eval-harness[gaia]"

# With tau2-bench support (also requires tau2 from git — see below)
pip install "agent-framework-eval-harness[tau2]"
pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench@5ba9e3e56db57c5e4114bf7f901291f09b2c5619"

# With Azure AI Foundry evaluators
pip install "agent-framework-eval-harness[foundry]"
```

## Quick Start

### GAIA Benchmark

```python
import asyncio
from azure.identity import DefaultAzureCredential
from agent_framework import create_harness_agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_eval_harness import EvalHarness
from agent_framework_eval_harness.benchmarks import GAIABenchmark

async def main():
    client = FoundryChatClient(credential=DefaultAzureCredential())
    agent = create_harness_agent(client=client)

    harness = EvalHarness(agent=agent)
    results = await harness.run(
        GAIABenchmark(level=1, max_tasks=20, parallel=4),
    )
    harness.print_summary(results)

asyncio.run(main())
```

Set `HF_TOKEN` in your environment for the first run (downloads the GAIA dataset):

```bash
export HF_TOKEN="hf_..."   # https://huggingface.co/settings/tokens
                           # Request dataset access: https://huggingface.co/datasets/gaia-benchmark/GAIA
```

### τ²-bench

```python
from agent_framework_eval_harness.benchmarks import TauBenchmark

results = await harness.run(
    TauBenchmark(domain="airline", max_tasks=50, user_client=user_llm_client),
)
harness.print_summary(results)
```

### Adding FoundryEvals

```python
from agent_framework.foundry import FoundryEvals

results = await harness.run(
    GAIABenchmark(level=1, max_tasks=20),
    evaluators=FoundryEvals(evaluators=[FoundryEvals.TASK_ADHERENCE]),
)
```

## Benchmark Comparison

Published scores (approximate, GPT-4o reference model):

| Benchmark        | LangGraph | Claude Opus | This harness |
|------------------|-----------|-------------|--------------|
| GAIA L1          | ~72%      | ~75%        | *(run it!)*  |
| τ²-bench airline | —         | ~62%        | *(run it!)*  |

## Design

- Agent is always `SupportsAgentRun` — constructed by the caller, never from YAML
- Returns standard `EvalResults` from `agent-framework-core`
- Runs tasks in parallel via `asyncio.Semaphore`
- GAIA uses the official normalized exact-match scorer
- Additional evaluators (e.g. `FoundryEvals`) compose naturally

## See Also

- [`agent-framework-core`](https://github.com/microsoft/agent-framework/tree/main/python/packages/core) — evaluation types
- [`agent-framework-lab`](https://github.com/microsoft/agent-framework/tree/main/python/packages/lab) — tau2 integration
- [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [τ²-bench](https://github.com/sierra-research/tau2-bench)
