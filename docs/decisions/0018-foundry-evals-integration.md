---
status: accepted
contact: bentho
date: 2026-02-27
deciders: bentho, markwallace-microsoft, westey-m
consulted: Pratyush Mishra, Shivam Shrivastava, Manni Arora (Centrica eval scenario)
informed: Agent Framework team, Foundry Evals team
---

# Agent Evaluation Architecture with Azure AI Foundry Integration

## Context and Problem Statement

Azure AI Foundry provides a rich evaluation service for AI agents — built-in evaluators for agent behavior (task adherence, intent resolution), tool usage (tool call accuracy, tool selection), quality (coherence, fluency, relevance), and safety (violence, self-harm, prohibited actions). Results are viewable in the Foundry portal with dashboards and comparison views.

However, using Foundry Evals with an agent-framework agent today requires significant manual effort. Developers must:

1. Transform agent-framework's `Message`/`Content` types into the OpenAI-style agent message schema that Foundry evaluators expect
2. Map tool definitions from agent-framework's `FunctionTool` format to evaluator-compatible schemas
3. Manually wire up the correct Foundry data source type (`azure_ai_traces`, `jsonl`, `azure_ai_target_completions`, etc.) depending on their scenario
4. Handle App Insights trace ID queries, response ID collection, and eval polling

Additionally, evaluation is a concern that extends beyond any single provider. Developers may want to use local evaluators (LLM-as-judge, regex, keyword matching), third-party evaluation libraries, or multiple providers in combination. The architecture must support this without creating a Foundry-specific lock-in at the API level.

The goal is: any agent-framework agent evaluable with any evaluation provider in 3-5 lines of code.

## Decision Drivers

- **Zero-friction evaluation**: Developers should go from "I have an agent" to "I have eval results" with minimal code.
- **Provider-agnostic API**: The core evaluation functions (`evaluate_agent`, `evaluate_response`, `evaluate_workflow`) must not be tied to any specific provider. Provider configuration should be separate from the evaluation call.
- **Lowest concept count**: Introduce the fewest possible new concepts. The evaluator IS the provider — no separate "provider" abstraction.
- **Agent discovery and tool reuse**: The framework already knows which agents exist and what tools they have. Evals should leverage this automatically.
- **Foundry-native results**: When using Foundry, results should be viewable in the Foundry portal.
- **Progressive disclosure**: Simple scenarios should be near-zero code. Advanced scenarios should build on the same primitives.
- **Cross-language parity**: Design must be implementable in both Python and .NET.

## Considered Options

1. **Provider-specific functions** — Build Foundry-specific helper functions (`evaluate_agent()`, etc.) directly in the Azure package. All eval functions take Foundry connection parameters.
2. **Evaluator protocol with core orchestration** — Define a provider-agnostic `Evaluator` protocol in core. Orchestration functions live in core. Providers implement the protocol.
3. **Full eval framework** — Build comprehensive eval infrastructure including custom evaluator definitions, scoring profiles, and reporting inside agent-framework.

## Decision Outcome

Chosen option: "Evaluator protocol with core orchestration", because it delivers the zero-friction developer experience, supports multiple providers without API changes, and keeps the concept count low (evaluator = provider).

### Architecture: Core vs Provider Split

The evaluation system is split across two layers:

**Core (`agent_framework._eval`, `agent_framework._local_eval`)**:

*Data types:*
- `EvalItem` — Provider-agnostic data format for evaluation items (includes `expected` for ground-truth comparison)
- `EvalResults` — Universal result type with pass/fail counts, portal links, per-item detail, sub_results
- `EvalItemResult` / `EvalScoreResult` — Per-item results with individual scores, error details, token usage

*Evaluator protocol:*
- `Evaluator` — Protocol that evaluation providers implement
- `LocalEvaluator` — Built-in `Evaluator` implementation that runs checks locally without API calls

*Orchestration functions:*
- `evaluate_agent()`, `evaluate_response()`, `evaluate_workflow()` — Extract data from agents/workflows and delegate to evaluators
- `AgentEvalConverter` — Converts agent-framework types (`Message`, `Content`, `FunctionTool`) to eval format

*Built-in checks (for use with `LocalEvaluator`):*
- `keyword_check(*keywords)` — Response must contain specified keywords
- `tool_called_check(*tool_names)` — Agent must have called specified tools

*Custom function evaluators:*
- `@function_evaluator` — Decorator to wrap a sync function as an eval check with signature-based parameter injection
- `@async_function_evaluator` — Same, for async functions (e.g., LLM-as-judge)

**Azure AI Provider (`agent_framework_azure_ai._foundry_evals`)**:

*Evaluator implementation:*
- `FoundryEvals` — `Evaluator` implementation backed by Azure AI Foundry (smart defaults, auto-detection, portal links)
- `Evaluators` — Constants for Foundry built-in evaluator names (agent behavior, tool usage, quality, safety)

*Foundry-specific functions:*
- `evaluate_traces()` — Evaluate from stored response IDs or OTel traces
- `evaluate_foundry_target()` — Evaluate a Foundry-registered agent or deployment
- `setup_continuous_eval()` — Continuous evaluation rules (not yet available)

### Consequences

- Good, because the same `evaluate_agent()` call works with Foundry, local, or third-party evaluators
- Good, because provider config is set once on the evaluator, not repeated on every function call
- Good, because mixing providers (e.g., Foundry quality + local keyword match) is natural
- Good, because `AgentEvalConverter` and data extraction logic are reusable across providers
- Neutral, because it requires core to define the `Evaluator` protocol (lightweight)
- Bad, because advanced Foundry features (scheduled evals, continuous eval) remain Foundry-specific functions

## Pros and Cons of the Options

### Provider-specific functions (Option 1, previous approach)

All eval functions take `openai_client=`, `model_deployment=`, etc. directly.

- Good, because simple implementation — no protocol/abstraction needed
- Bad, because every function signature is Foundry-specific (`project_client`, `model_deployment`)
- Bad, because switching providers requires rewriting all eval calls
- Bad, because mixing providers (Foundry + local) in one eval run is impossible

### Full eval framework (Option 3)

Build comprehensive eval infrastructure including custom evaluator definitions, dataset management, and reporting.

- Good, because it could provide evaluator features that Foundry doesn't support natively
- Bad, because it duplicates Foundry functionality and significantly expands maintenance burden
- Bad, because eval results would fragment across separate systems

## Usage Examples

### Basic: Evaluate an agent

```python
from agent_framework import evaluate_agent
from agent_framework_azure_ai import FoundryEvals

evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")

results = await evaluate_agent(
    agent=my_agent,
    queries=["What's the weather?"],
    evaluators=evals,  # smart defaults: relevance, coherence, task_adherence
                       # auto-adds tool_call_accuracy when agent has tools
)
for r in results:
    r.assert_passed()
```

### Evaluate a response you already have

```python
from agent_framework import evaluate_response
from agent_framework_azure_ai import FoundryEvals

evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")

# Quality evaluators — Responses API fast path (no query needed)
results = await evaluate_response(response=response, evaluators=evals)

# Tool evaluators — provide query and agent for tool definitions
results = await evaluate_response(
    response=response,
    query="What's the weather?",
    agent=agent,
    evaluators=evals,
)
```

### Evaluate a multi-agent workflow

```python
from agent_framework import evaluate_workflow
from agent_framework_azure_ai import FoundryEvals

evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")

result = await workflow.run("Plan a trip to Paris")
eval_results = await evaluate_workflow(
    workflow=workflow,
    workflow_result=result,
    evaluators=evals,
)

for r in eval_results:
    print(f"{r.provider}:")
    print(f"  overall: {r.passed}/{r.total}")
    for name, sub in r.sub_results.items():
        print(f"    {name}: {sub.passed}/{sub.total}")
```

### Select specific evaluators

```python
evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")

# Use select() to pick specific evaluators
quality_only = evals.select("relevance", "coherence")
results = await evaluate_agent(agent=agent, queries=queries, evaluators=quality_only)
```

### Mix multiple providers

```python
from agent_framework import evaluate_agent, LocalEvaluator, function_evaluator, keyword_check, tool_called_check
from agent_framework_azure_ai import FoundryEvals

# Custom function evaluator — just name your parameters
@function_evaluator
def is_helpful(response: str) -> bool:
    return len(response.split()) > 10

# Local checks — instant, no API calls
local = LocalEvaluator(
    is_helpful,
    keyword_check("weather"),
    tool_called_check("get_weather"),
)

# Foundry — deep quality assessment via LLM-as-judge
foundry = FoundryEvals(project_client=client, model_deployment="gpt-4o")

# Both evaluate the same items; one EvalResults per provider
results = await evaluate_agent(
    agent=agent,
    queries=queries,
    evaluators=[local, foundry],
)
```

### Foundry-specific: Trace-based evaluation

```python
from agent_framework_azure_ai import Evaluators, evaluate_traces

results = await evaluate_traces(
    response_ids=[response.response_id],
    evaluators=[Evaluators.RELEVANCE],
    project_client=project_client,
    model_deployment="gpt-4o",
)
```

### Direct evaluator access

```python
from agent_framework import AgentEvalConverter
from agent_framework_azure_ai import FoundryEvals

evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")

converter = AgentEvalConverter()
items = [converter.to_eval_item(query=q, response=r, agent=agent) for q, r in pairs]

# Call the evaluator directly
results = await evals.evaluate(items, eval_name="My Custom Eval")
```

## What To Build

### Core: Evaluator Protocol

A runtime-checkable protocol that any evaluation provider implements:

```python
@runtime_checkable
class Evaluator(Protocol):
    @property
    def name(self) -> str: ...

    async def evaluate(
        self, items: Sequence[EvalItem], *, eval_name: str = "Agent Framework Eval"
    ) -> EvalResults: ...
```

The protocol is minimal — just `name` and `evaluate()`. Providers are free to add methods like `select()`, but the core only needs this interface.

### Core: EvalItem

Provider-agnostic data format for items to evaluate:

```python
@dataclass
class EvalItem:
    query: str
    response: str
    conversation: list[dict[str, Any]]         # OpenAI chat format
    tool_definitions: list[dict[str, Any]] | None = None
    context: str | None = None
    expected: str | None = None                # Ground-truth for comparison
    response_id: str | None = None             # For Responses API providers
```

`EvalItem.to_dict()` serializes to flat dict with JSON-encoded strings for `conversation` and `tool_definitions`, suitable for JSONL data sources.

### Core: AgentEvalConverter

Converts agent-framework types to `EvalItem`. `to_eval_item()` returns `EvalItem` objects (not dicts), making the output strongly typed:

| Agent Framework | Eval Format |
|---|---|
| `Content.function_call` | `tool_call` in OpenAI chat format |
| `Content.function_result` | `tool_result` in OpenAI chat format |
| `FunctionTool` | `{name, description, parameters}` schema |
| `Message` history | `conversation` list + `query`/`response` extraction |

### Core: EvalResults

Rich result type with convenience properties for CI integration:

```python
results.all_passed          # bool: no failures or errors (recursive for workflow)
results.passed              # int: passing count
results.failed              # int: failure count
results.total               # int: total = passed + failed + errored
results.per_evaluator       # dict: per-evaluator breakdown
results.items               # list[EvalItemResult]: per-item scores and errors
results.error               # str | None: error details on failure
results.sub_results         # dict: per-agent breakdown (workflow evals)
results.report_url          # str | None: portal link (Foundry)
results.assert_passed()     # raises AssertionError with details

# Per-item filtering
results.passed_items        # list[EvalItemResult]: items that passed
results.failed_items        # list[EvalItemResult]: items that failed
results.errored_items       # list[EvalItemResult]: items that errored
```

### Core: Orchestration Functions

Provider-agnostic functions that extract data and delegate to evaluators:

| Function | What it does |
|---|---|
| `evaluate_agent()` | Runs agent against test queries, converts via `AgentEvalConverter`, passes `EvalItem`s to evaluator |
| `evaluate_response()` | Converts response(s) to `EvalItem`s, passes to evaluator. Includes `response_id` for providers that support server-side retrieval |
| `evaluate_workflow()` | Extracts per-agent data from `WorkflowRunResult`, evaluates each agent and overall output. Per-agent breakdown in `sub_results` |

### Azure AI: FoundryEvals

`Evaluator` implementation backed by Azure AI Foundry:

```python
class FoundryEvals:
    def __init__(self, *, project_client=None, openai_client=None,
                 model_deployment: str, evaluators=None, ...)
    def select(self, *evaluators: str) -> FoundryEvals
    async def evaluate(self, items, *, eval_name) -> EvalResults
```

**Smart auto-detection in `evaluate()`:**
- Default evaluators: relevance, coherence, task_adherence
- Auto-adds `tool_call_accuracy` when items have `tool_definitions`
- Filters out tool evaluators for items without `tool_definitions`
- Responses API fast path when all items have `response_id` and no tool evaluators

### Azure AI: Evaluators Constants

```python
from agent_framework_azure_ai import Evaluators

evaluators = [Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY]
```

Categories: Agent behavior, Tool usage, Quality, Safety.

### Azure AI: Foundry-Specific Functions

| Function | What it does |
|---|---|
| `evaluate_traces()` | Evaluate from stored response IDs or OTel traces |
| `evaluate_foundry_target()` | Evaluate a Foundry-registered agent or deployment |
| `setup_continuous_eval()` | Create evaluation rules (not yet available) |

### Core: LocalEvaluator and Function Evaluators

`LocalEvaluator` implements the `Evaluator` protocol for fast, API-free evaluation. It runs check functions locally — useful for inner-loop development, CI smoke tests, and combining with cloud-based evaluators.

Built-in checks:
- `keyword_check(*keywords)` — response must contain specified keywords
- `tool_called_check(*tool_names)` — agent must have called specified tools

Custom function evaluators use `@function_evaluator` to wrap plain Python functions. The function's **parameter names** determine what data it receives from the `EvalItem`:

```python
from agent_framework import function_evaluator, LocalEvaluator

# Tier 1: Simple check — just query + response
@function_evaluator
def is_concise(response: str) -> bool:
    return len(response.split()) < 500

# Tier 2: Ground truth — compare against expected output
@function_evaluator
def mentions_city(response: str, expected: str) -> bool:
    return expected.lower() in response.lower()

# Tier 3: Full context — inspect conversation and tools
@function_evaluator
def used_tools(conversation: list, tool_definitions: list) -> float:
    # ... scoring logic
    return score

local = LocalEvaluator(is_concise, mentions_city, used_tools)
```

Supported parameters: `query`, `response`, `expected`, `conversation`, `tool_definitions`, `context`.
Return types: `bool`, `float` (≥0.5 = pass), `dict` with `score` or `passed` key, or `CheckResult`.

Async functions use `@async_function_evaluator` for evaluators that need I/O (e.g., LLM-as-judge).

### Package Location

- Core types and orchestration: `agent_framework._eval`, `agent_framework._local_eval` (Python), `Microsoft.Agents.AI.Core` (.NET)
- Foundry provider: `agent_framework_azure_ai._foundry_evals` (Python), `Microsoft.Agents.AI.AzureAIFoundry` (.NET)
- Azure-AI re-exports core types for convenience

## Known Limitations

1. **Per-agent workflow evals use dataset path**: Workflow sub-agent evaluations always clear `response_id` to force the dataset evaluation path. The Responses API retrieval path doesn't work for agents whose input is another agent's full conversation (the evaluator can't extract a clean user query from the stored response).
2. **Tool evaluators require query + agent**: Tool evaluators need tool definition schemas, which are not available through Responses API response retrieval. When using these evaluators with `evaluate_response()`, `query=` and `agent=` must be provided.
3. **`model_deployment` always required**: Could potentially be inferred from the Foundry project configuration.

## Resolved Issues

1. **`options=` vs `default_options=` silent failure** (fixed): `Agent.__init__` now accepts `options=` as an alias for `default_options=`, preventing the common mistake of `Agent(options={"store": False})` silently dropping into `additional_properties`.
2. **`function_call` status serialization** (fixed): `_parse_response_from_openai` now preserves `status` in `additional_properties`, and serialization reads it back with a fallback to `"completed"`.
3. **Stale session state** (fixed): `AgentExecutor.reset()` clears `service_session_id`. Workflows call `executor.reset()` on all executors during `reset_for_new_run`.
4. **`response_id` gated on store** (fixed): `response_id` is only set on `ChatResponse` when `store` is not `False`. This prevents evals from attempting the Responses API path for non-stored responses (which would 404).

## Open Questions

1. **Red teaming non-registered agents**: Requires Foundry API support for callback-based flows.
2. **.NET parity timeline**: What is the priority for .NET implementation?
3. **Datasets with expected outputs**: A dataset abstraction for pre-populating `expected` values across eval runs is a natural next step but not yet designed.

## More Information

- [Foundry Evals documentation](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-approach-gen-ai) — Azure AI Foundry evaluation overview
