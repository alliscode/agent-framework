---
status: proposed
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

**Core (`agent_framework._eval`)**:
- `EvalItem` — Provider-agnostic data format for evaluation items
- `EvalResults` — Universal result type with pass/fail counts, portal links, sub_results
- `Evaluator` — Protocol that evaluation providers implement
- `AgentEvalConverter` — Converts agent-framework types to eval format
- `evaluate_agent()`, `evaluate_response()`, `evaluate_workflow()` — Orchestration functions that extract data and delegate to evaluators

**Azure AI Provider (`agent_framework_azure_ai._foundry_evals`)**:
- `FoundryEvals` — `Evaluator` implementation backed by Azure AI Foundry
- `Evaluators` — Constants for Foundry built-in evaluator names
- `evaluate_traces()` — Foundry-specific: evaluate from stored response IDs or OTel traces
- `evaluate_foundry_target()` — Foundry-specific: evaluate a registered agent or deployment
- `setup_continuous_eval()` — Foundry-specific: continuous evaluation rules (not yet available)

**Key insight**: The evaluator IS the provider. There is no separate "provider" concept. A `FoundryEvals` instance encapsulates all Foundry connection details (client, model deployment, evaluator selection). It is passed as the `evaluators` parameter to the core orchestration functions.

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
results.assert_passed()
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

for name, sub in eval_results.sub_results.items():
    print(f"  {name}: {sub.passed}/{sub.total}")
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
from agent_framework import evaluate_agent, LocalEvaluator, keyword_check, tool_called_check
from agent_framework_azure_ai import FoundryEvals

# Local checks — instant, no API calls
local = LocalEvaluator(
    keyword_check("weather"),
    tool_called_check("get_weather"),
)

# Foundry — deep quality assessment via LLM-as-judge
foundry = FoundryEvals(project_client=client, model_deployment="gpt-4o")

# Both evaluate the same items; results are merged
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
results.error               # str | None: error details on failure
results.sub_results         # dict: per-agent breakdown (workflow evals)
results.assert_passed()     # raises AssertionError with details
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

### Package Location

- Core types and orchestration: `agent_framework._eval` (Python), `Microsoft.Agents.AI.Core` (.NET)
- Foundry provider: `agent_framework_azure_ai._foundry_evals` (Python), `Microsoft.Agents.AI.AzureAIFoundry` (.NET)
- Azure-AI re-exports core types for convenience

## Known Limitations

1. **Workflow re-run with Responses API**: When using `evaluate_workflow(queries=...)` with Responses API clients, agents must be configured with `options={"store": False}`. Without this, the `AgentExecutor` session retains `previous_response_id` from prior runs, and the Responses API rejects requests that chain across independent conversations.
2. **Tool evaluators require query + agent**: Tool evaluators need tool definition schemas, which are not available through Responses API response retrieval. When using these evaluators with `evaluate_response()`, `query=` and `agent=` must be provided.
3. **`model_deployment` always required**: Could potentially be inferred from the Foundry project configuration.

## Open Questions

1. **Local evaluator implementations**: What built-in local evaluators should the core package provide (e.g., keyword match, regex, assertion-based)?
2. **Red teaming non-registered agents**: Requires Foundry API support for callback-based flows.
3. **.NET parity timeline**: What is the priority for .NET implementation?

## More Information

- [Foundry Evals documentation](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-approach-gen-ai) — Azure AI Foundry evaluation overview
