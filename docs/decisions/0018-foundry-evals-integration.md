---
status: proposed
contact: bentho
date: 2026-02-27
deciders: bentho, markwallace-microsoft, westey-m
consulted: Pratyush Mishra, Shivam Shrivastava, Manni Arora (Centrica eval scenario)
informed: Agent Framework team, Foundry Evals team
---

# Azure AI Foundry Evals Integration

## Context and Problem Statement

Azure AI Foundry provides a rich evaluation service for AI agents — built-in evaluators for agent behavior (task adherence, intent resolution), tool usage (tool call accuracy, tool selection), quality (coherence, fluency, relevance), and safety (violence, self-harm, prohibited actions). Results are viewable in the Foundry portal with dashboards and comparison views.

However, using Foundry Evals with an agent-framework agent today requires significant manual effort. Developers must:

1. Transform agent-framework's `Message`/`Content` types into the OpenAI-style agent message schema that Foundry evaluators expect
2. Map tool definitions from agent-framework's `FunctionTool` format to evaluator-compatible schemas
3. Manually wire up the correct Foundry data source type (`azure_ai_traces`, `jsonl`, `azure_ai_target_completions`, etc.) depending on their scenario
4. Handle App Insights trace ID queries, response ID collection, and eval polling

This friction is especially painful for multi-agent orchestrations (e.g. classifier → workstream agents) where developers need to evaluate individual sub-agents, not just the workflow as a whole — yet all the sub-agent information is already present in the framework's OTel traces and tool registrations.

The goal is to make any agent-framework agent evaluable against Foundry Evals in 3-5 lines of code.

## Decision Drivers

- **Zero-friction evaluation**: Developers should go from "I have an agent" to "I have eval results" with minimal code.
- **Any agent, any provider**: The primary path must work with agents backed by any model provider — OpenAI, Anthropic, Ollama, local models — not just Foundry-registered agents.
- **Agent discovery and tool reuse**: The framework already knows which agents exist and what tools they have. Evals should leverage this automatically, not require developers to re-declare tool definitions in eval config.
- **Foundry-native results**: All paths produce results viewable in the Foundry portal. Results may also be surfaced in DevUI for the local dev inner loop.
- **Progressive disclosure**: Simple scenarios (evaluate what happened) should be near-zero code. Advanced scenarios (red teaming, continuous monitoring) should build on the same primitives.
- **Cross-language parity**: Design must be implementable in both Python and .NET.

## Considered Options

1. **Thin helpers with message converter** — Build a type converter (`AgentEvalConverter`) and lightweight wrapper functions that bridge agent-framework types to Foundry Evals APIs.
2. **Full eval framework** — Build comprehensive eval infrastructure including custom evaluator definitions, scoring profiles, dataset management, and reporting inside agent-framework.

## Decision Outcome

Chosen option: "Thin helpers with message converter", because it delivers the zero-friction developer experience without duplicating Foundry functionality, works with the framework's existing OTel instrumentation, and can be built incrementally starting with the highest-impact path.

### Consequences

- Good, because developers get a 3-5 line path from agent to eval results for any agent
- Good, because no duplication of Foundry's evaluator logic, portal, or dashboards
- Good, because the converter enables interop with any system that consumes the OpenAI agent message format
- Good, because it leverages agent-framework's existing OTel instrumentation for the highest-impact path
- Neutral, because it depends on the Foundry SDK (`azure-ai-projects` / `Azure.AI.Projects`) as a runtime dependency
- Bad, because advanced Foundry features (scheduled evals, continuous eval, red teaming) are only available for Foundry-registered agents and cannot be extended to arbitrary agents through helpers alone

## Pros and Cons of the Options

### Full eval framework

Build comprehensive eval infrastructure including custom evaluator definitions (scoring profiles, LLM-as-judge, code-based evaluators), dataset management, run scheduling, and reporting.

- Good, because it could provide evaluator features that Foundry doesn't support natively (e.g. custom scoring profiles)
- Good, because it addresses feedback from customers who want end-to-end eval config without Foundry
- Bad, because it duplicates functionality that Foundry already provides and will continue to improve
- Bad, because it significantly expands agent-framework's scope and maintenance burden
- Bad, because custom evaluator infrastructure would need to be built and maintained for both Python and .NET
- Bad, because eval results would live in a separate system from Foundry portal, fragmenting the developer's view

### Why not a `.WithEvaluation()` wrapper?

A decorator approach (similar to `OpenTelemetryAgent`) that automatically evaluates every `RunAsync` call was considered as an API shape for the chosen option. However, evaluation is typically a development/CI activity, not a per-request concern. Evaluating every run would be expensive (each eval is a Foundry API call with LLM-as-judge cost) and conflates the eval lifecycle (batch, scheduled, one-time) with the request lifecycle. Standalone helper functions better match how developers actually run evals.

## Integration Paths

The chosen option provides four evaluation paths, matching the ways Foundry can ingest data:

### Path 1: Trace-Based Evaluation ⭐ Highest Impact

Evaluate any agent by pointing Foundry at OTel traces in App Insights. Works with any model provider. Zero changes to agent code.

```python
from agent_framework_azure_ai import Evaluators, evaluate_traces

results = await evaluate_traces(
    response_ids=[response.response_id],
    evaluators=[Evaluators.INTENT_RESOLUTION, Evaluators.TASK_ADHERENCE],
    openai_client=openai_client,
    model_deployment="gpt-4o",
)
results.assert_passed()
print(f"View results: {results.report_url}")
```

### Path 2: Foundry-Managed Evaluation

For agents or models registered in Foundry. Foundry invokes the target, evaluates the output. Enables scheduled evals, red teaming, and CI gates.

```python
from agent_framework_azure_ai import Evaluators, evaluate_foundry_target

results = await evaluate_foundry_target(
    target={"type": "azure_ai_agent", "name": "my-agent"},
    test_queries=["Book a flight to Paris"],
    evaluators=[Evaluators.TASK_COMPLETION, Evaluators.TOOL_CALL_ACCURACY],
    openai_client=openai_client,
    model_deployment="gpt-4o",
)
```

### Path 3: Dataset Evaluation

Run your agent locally against test cases, convert the output to Foundry format, evaluate. The dev inner loop for agents with custom code.

```python
from agent_framework_azure_ai import evaluate_agent

results = await evaluate_agent(
    agent=my_agent,
    queries=["What's the weather in Seattle?"],
    # evaluators default to relevance, coherence, task_adherence;
    # tool_call_accuracy is auto-added when the agent has tools
    openai_client=openai_client,
    model_deployment="gpt-4o",
)
results.assert_passed()  # CI gate — raises AssertionError on failure
```

### Path 4: Continuous Evaluation (Not Yet Available)

One-time setup of a Foundry evaluation rule. Every response is automatically evaluated. Requires Foundry-registered agent. This path depends on the Foundry evaluation rules API, which is not yet integrated.

```python
from agent_framework_azure_ai import setup_continuous_eval

# Not yet available — raises NotImplementedError.
# Use evaluate_traces() to evaluate responses in the meantime.
await setup_continuous_eval(
    agent_name="production-bot",
    evaluators=[Evaluators.VIOLENCE, Evaluators.TASK_ADHERENCE],
    openai_client=openai_client,
    model_deployment="gpt-4o",
)
```

## What To Build

### Evaluators — Discoverable Constants

String constants for all Foundry built-in evaluators, enabling IDE autocomplete and typo prevention:

```python
from agent_framework_azure_ai import Evaluators

evaluators = [Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY]
```

### AgentEvalConverter — Message Format Bridge

Converts agent-framework types into the OpenAI-style agent message schema that Foundry evaluators expect:

| Agent Framework | Foundry Eval Format |
|---|---|
| `Content.function_call` | `tool_call` content type with `call_id` |
| `Content.function_result` | `tool_result` content type |
| `FunctionTool` | `{name, description, parameters}` schema |
| `Message` history split | `query` (user input) + `response` (agent output) |

The converter also extracts tool definitions directly from `agent.tools`, eliminating the need to re-declare them in eval configuration.

### EvalResults — CI-Friendly Results

Rich result type with convenience properties for CI integration:

```python
results = await evaluate_agent(...)
assert results.all_passed          # bool: no failures or errors
print(results.passed)              # int: passing count
print(results.failed)              # int: failure count
print(results.per_evaluator)       # dict: per-evaluator breakdown
results.assert_passed()            # raises AssertionError with details

# Workflow evals — per-agent breakdown
for name, sub in results.sub_results.items():
    print(f"{name}: {sub.passed}/{sub.total}")
```

### Wrapper Functions

Thin async wrappers that handle Foundry API plumbing. All functions accept optional `evaluators` (defaults to relevance + coherence + task_adherence):

| Function | Path | What it does |
|---|---|---|
| `evaluate_response()` | 3/1 | Evaluates response(s) from a completed `agent.run()`. Uses Responses API fast path if available, otherwise falls back to dataset eval with query + conversation |
| `evaluate_agent()` | 3 | Runs agent, converts via `AgentEvalConverter`, submits as JSONL |
| `evaluate_workflow()` | 3 | Evaluates a multi-agent workflow with per-agent breakdown in `sub_results` |
| `evaluate_traces()` | 1 | Evaluates by response IDs or OTel trace IDs from App Insights |
| `evaluate_foundry_target()` | 2 | Constructs target completions data source, submits eval |
| `evaluate_dataset()` | 3 | Evaluates pre-collected data (for when developer runs agent separately) |
| `setup_continuous_eval()` | 4 | Creates Foundry `EvaluationRule` (not yet available) |

### Package Location

All code in the existing Azure integration packages (`agent_framework_azure_ai` for Python, `Microsoft.Agents.AI.AzureAIFoundry` or similar for .NET), since every path requires the Foundry SDK.

## Implementation Order

1. **`Evaluators` constants + `AgentEvalConverter`** — Core building blocks; constants for DX, converter for Path 3
2. **`evaluate_agent()` / `evaluate_dataset()`** — Primary developer path; run agent + evaluate in one call
3. **`evaluate_traces()`** — Highest-impact production path; works with any agent that has OTel
4. **`evaluate_foundry_target()`** — Target completions for registered agents
5. **`setup_continuous_eval()`** — Production monitoring (blocked on Foundry evaluation rules API)
6. **Samples and documentation** — Getting started sample for each path

## Known Gaps

1. **`project_endpoint` convenience parameter**: The current API requires developers to create an `AIProjectClient` and `OpenAI` client externally. A future improvement would accept `project_endpoint: str` directly and handle client creation internally, reducing boilerplate. This is blocked on determining the right client lifecycle semantics.
2. **`model_deployment` defaulting**: Currently required on every call. Could potentially be inferred from the Foundry project configuration.

## Open Questions

1. **Orchestration sub-agent evaluation**: ~~For multi-agent workflows, can Foundry trace-based eval be scoped to specific spans within a trace?~~ **Resolved**: `evaluate_workflow()` extracts per-agent query/response pairs from `WorkflowRunResult` events and evaluates each agent via dataset eval (Path 3). No OTEL/App Insights required. Results are returned with per-agent breakdown in `EvalResults.sub_results`. See [orchestration eval design](../foundry-orchestration-eval.md).
2. **Red teaming non-registered agents**: Foundry red teaming requires invoking the target. A state-machine API where the framework owns the poll→run→submit loop could enable red teaming of any agent, but requires Foundry API support for callback-based flows. See [red team state machine proposal](../foundry-redteam-state-machine.md).
3. **Auto-detection**: Should `evaluate_agent()` auto-detect Foundry-registered agents and route to `azure_ai_target_completions` instead of JSONL?
4. **.NET parity timeline**: What is the priority for .NET implementation relative to Python?

## More Information

- [Integration paths summary](../foundry-evals-paths.md) — Decision tree and quick reference table
- [Red team state machine proposal](../foundry-redteam-state-machine.md) — Sequence diagram for red-teaming non-registered agents
- [Orchestration eval design](../foundry-orchestration-eval.md) — Evaluating sub-agents in multi-agent workflows
- [Foundry Evals documentation](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-approach-gen-ai) — Azure AI Foundry evaluation overview
