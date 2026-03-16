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

### What Developers Need from Agent Evaluation

Agent evaluation needs to work across several dimensions of complexity.

**Single agents and workflows.** At the simplest level, a developer has one agent and wants to know if its responses are good — relevant, coherent, grounded, safe. But agents rarely work alone. In a workflow, multiple agents collaborate: a planner breaks down a task, a researcher gathers information, a writer produces output. Developers need to evaluate both the overall workflow result and the individual agents within it, so they can pinpoint which agent is underperforming without re-running the entire workflow.

**One-shot and multi-turn conversations.** A single query-response pair is the simplest thing to evaluate, but real agents carry multi-turn conversations with tool calls, intermediate reasoning, and follow-up questions. The evaluation data must capture the full conversation trajectory — including tool invocations and their results — because evaluators need this context to assess whether the agent maintained coherence, used its tools correctly, and stayed on task across turns.

**Conversation factoring.** A multi-turn conversation can be factored into "query" and "response" in multiple valid ways, and the choice changes what you're measuring. You might split at the last user message to evaluate "did the agent answer this specific question well?" Or you might treat the first user message as the query and everything after as the response to evaluate "did the entire conversation serve the original request?" Or you might split per-turn to get fine-grained scores at each step. These different factorings can produce meaningfully different scores for the same conversation. Developers need to control this easily, both globally and per-evaluator.

**Multiple providers, mix and match.** Azure AI Foundry offers rich LLM-as-judge evaluators for quality, safety, and tool usage — with portal dashboards and comparison views. But developers also want fast local checks (keyword matching, tool-was-called assertions) for inner-loop development and CI, custom function-based evaluators for domain-specific logic, and potentially third-party evaluation libraries. A developer should be able to run Foundry evaluators alongside local checks on the same data without restructuring their code, and swap providers without rewriting evaluation calls.

**Bring your own evaluator.** Beyond selecting from a menu of built-in evaluators, developers want to write their own evaluation logic — from simple functions that check a response for specific content, to sophisticated LLM-as-judge prompts, to evaluators that compare against ground-truth expected outputs. The barrier to creating a custom evaluator should be as low as writing a function.

**Evaluate without re-running.** Developers often have responses from production logs, previous test runs, or manual testing. They want to evaluate these existing responses without invoking the agent again. The framework already knows the agent's tools and instructions — it should use that context automatically.

## Decision Drivers

- **Zero-friction evaluation**: Developers should go from "I have an agent" to "I have eval results" with minimal code.
- **Provider-agnostic API**: Core evaluation capabilities must not be tied to any specific provider. Provider configuration should be separate from the evaluation call.
- **Lowest concept count**: Introduce the fewest possible new types, abstractions, and APIs for developers to learn.
- **Leverage existing knowledge**: The framework already knows which agents exist, what tools they have, and what conversations occurred. Evals should use this automatically rather than requiring the developer to re-specify it.
- **Foundry-native results**: When using Foundry, results should be viewable in the Foundry portal with dashboards and comparison views.
- **Progressive disclosure**: Simple scenarios should be near-zero code. Advanced scenarios should build on the same primitives.
- **Cross-language parity**: Design must be implementable in both Python and .NET.

## Considered Options

1. **Provider-specific functions** — Build Foundry-specific helper functions (`evaluate_agent()`, etc.) directly in the Azure package. All eval functions take Foundry connection parameters.
2. **Evaluator protocol with core orchestration** — Define a provider-agnostic `Evaluator` protocol in core. Orchestration functions live in core. Providers implement the protocol.
3. **Full eval framework** — Build comprehensive eval infrastructure including custom evaluator definitions, scoring profiles, and reporting inside agent-framework.

## Decision Outcome

Proposed option: "Evaluator protocol with core orchestration", because it delivers the low-friction developer experience, supports multiple providers without API changes, and keeps the concept count low.

### Usage Examples

#### Evaluate an agent

**Python:**

```python
evals = FoundryEvals(
    project_client=client,
    model_deployment="gpt-4o",
    evaluators=[FoundryEvals.RELEVANCE, FoundryEvals.COHERENCE],
)

results = await evaluate_agent(
    agent=my_agent,
    queries=["What's the weather?"],
    evaluators=evals,
)
for r in results:
    r.assert_passed()
```

**C#:**

```csharp
var evals = new FoundryEvaluator(chatConfiguration, FoundryEvals.Relevance, FoundryEvals.Coherence);

AgentEvaluationResults results = await agent.EvaluateAsync(
    new[] { "What's the weather?" },
    evals);

results.AssertAllPassed();
```

`evaluate_agent` returns one `EvalResults` per evaluator. Each result contains per-item scores:

```
# results[0] (FoundryEvals)
EvalResults(status="completed", passed=1, failed=0, total=1)
  items[0]: EvalItemResult(query="What's the weather?", scores={"relevance": 5, "coherence": 5})
```

#### Evaluate a response you already have

**Python:**

```python
query = "What's the weather?"
response = await agent.run([Message("user", [query])])

results = await evaluate_agent(
    agent=agent,
    responses=response,
    queries=[query],
    evaluators=evals,
)
```

**C#:**

```csharp
var query = "What's the weather?";
AgentResponse response = await agent.RunAsync(
    new[] { new ChatMessage(ChatRole.User, query) });

AgentEvaluationResults results = await agent.EvaluateAsync(
    responses: new[] { response },
    queries: new[] { query },
    evals);
```

Same return shape as above — the response is evaluated without re-running the agent.

#### Evaluate with conversation split strategies

By default, evaluators see only the last turn (final user message → final assistant response). For multi-turn conversations, you can control how the conversation is factored for evaluation:

**Python:**

```python
results = await evaluate_agent(
    agent=agent,
    queries=["Plan a 3-day trip to Paris"],
    evaluators=evals,
    conversation_split=ConversationSplit.FULL,      # evaluate entire trajectory
)

# Or per-turn: each user→assistant exchange scored independently
results = await evaluate_agent(
    agent=agent,
    queries=["Plan a 3-day trip to Paris"],
    evaluators=evals,
    conversation_split=ConversationSplit.PER_TURN,
)
```

**C#:**

```csharp
// Full conversation as context
AgentEvaluationResults results = await agent.EvaluateAsync(
    new[] { "Plan a 3-day trip to Paris" },
    evals,
    splitter: ConversationSplitters.Full);

// Per-turn splitting
var items = EvalItem.PerTurnItems(conversation);  // one EvalItem per user turn
var results = await evals.EvaluateAsync(items);
```

With `PER_TURN`, a 3-turn conversation produces 3 scored items:

```
EvalResults(status="completed", passed=3, failed=0, total=3)
  items[0]: query="Plan a 3-day trip to Paris"    scores={"relevance": 5}
  items[1]: query="What about restaurants?"        scores={"relevance": 4}
  items[2]: query="Make it budget-friendly"        scores={"relevance": 5}
```

#### Evaluate a multi-agent workflow

**Python:**

```python
result = await workflow.run("Plan a trip to Paris")
eval_results = await evaluate_workflow(
    workflow=workflow,
    workflow_result=result,
    evaluators=evals,
)

for r in eval_results:
    print(f"  overall: {r.passed}/{r.total}")
    for name, sub in r.sub_results.items():
        print(f"    {name}: {sub.passed}/{sub.total}")
```

**C#:**

```csharp
WorkflowRunResult result = await workflow.RunAsync("Plan a trip to Paris");

IReadOnlyList<AgentEvaluationResults> evalResults = await result.EvaluateAsync(evals);

foreach (var r in evalResults)
{
    Console.WriteLine($"  overall: {r.Passed}/{r.Total}");
    foreach (var (name, sub) in r.SubResults)
        Console.WriteLine($"    {name}: {sub.Passed}/{sub.Total}");
}
```

Workflows return one result per evaluator, with sub-results per agent in the workflow:

```
EvalResults(status="completed", passed=2, failed=0, total=2)
  sub_results:
    "planner":  EvalResults(passed=1, total=1)
    "researcher": EvalResults(passed=1, total=1)
```

#### Mix multiple providers

**Python:**

```python
@function_evaluator
def is_helpful(response: str) -> bool:
    return len(response.split()) > 10

foundry = FoundryEvals(
    project_client=client,
    model_deployment="gpt-4o",
    evaluators=[FoundryEvals.RELEVANCE, FoundryEvals.COHERENCE],
)

results = await evaluate_agent(
    agent=agent,
    queries=queries,
    evaluators=[is_helpful, keyword_check("weather"), foundry],
)
```

**C#:**

```csharp
IReadOnlyList<AgentEvaluationResults> results = await agent.EvaluateAsync(
    queries,
    evaluators: new IAgentEvaluator[]
    {
        new LocalEvaluator(
            EvalChecks.KeywordCheck("weather"),
            FunctionEvaluator.Create("is_helpful", (string r) => r.Split(' ').Length > 10)),
        new FoundryEvaluator(chatConfiguration, FoundryEvals.Relevance, FoundryEvals.Coherence),
    });
```

Multiple evaluators return one result each — `results[0]` is the local evaluator, `results[1]` is Foundry.

#### Custom function evaluators

**Python:**

```python
@function_evaluator
def mentions_city(response: str, expected: str) -> bool:
    return expected.lower() in response.lower()

@function_evaluator
def used_tools(conversation: list, tools: list) -> float:
    # ... scoring logic
    return score

local = LocalEvaluator(mentions_city, used_tools)
```

**C#:**

```csharp
var local = new LocalEvaluator(
    FunctionEvaluator.Create("mentions_city",
        (EvalItem item) => item.Expected != null
            && item.Response.Contains(item.Expected, StringComparison.OrdinalIgnoreCase)),
    FunctionEvaluator.Create("is_concise",
        (string response) => response.Split(' ').Length < 500));
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

The protocol is minimal — just `name` and `evaluate()`.

### Core: EvalItem

Provider-agnostic data format for items to evaluate:

```python
@dataclass
class EvalItem:
    conversation: list[Message]               # Single source of truth
    tools: list[FunctionTool] | None = None   # Agent's available tools
    context: str | None = None
    expected: str | None = None               # Ground-truth for comparison
    response_id: str | None = None            # For Responses API providers
    split_strategy: ConversationSplitter | None = None

    query: str       # property — derived from conversation split
    response: str    # property — derived from conversation split
```

`conversation` is the single source of truth. `query` and `response` are derived properties — splitting the conversation at the last user message (default) and extracting text from each side. Changing the `split_strategy` consistently changes all derived values.

`tools` provides typed `FunctionTool` objects — including MCP tools, which are automatically extracted after agent runs.

`EvalItem.to_dict()` serializes to flat dict: splits conversation, converts each half to OpenAI chat format (`query_messages`/`response_messages`), derives `query`/`response` strings, and serializes `tools` as `tool_definitions`.

### Internal: AgentEvalConverter

Internal class that converts agent-framework types to `EvalItem`. Used by `evaluate_agent()` and `evaluate_workflow()` — not part of the public API:

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
results.items               # list[EvalItemResult]: per-item scores and errors
results.error               # str | None: error details on failure
results.sub_results         # dict: per-agent breakdown (workflow evals)
results.report_url          # str | None: portal link (Foundry)
results.assert_passed()     # raises AssertionError with details
```

### Core: Orchestration Functions

Provider-agnostic functions that extract data and delegate to evaluators:

| Function | What it does |
|---|---|
| `evaluate_agent()` | Runs agent against test queries (or evaluates pre-existing `responses=`), converts to `EvalItem`s, passes to evaluator |
| `evaluate_workflow()` | Extracts per-agent data from `WorkflowRunResult`, evaluates each agent and overall output. Per-agent breakdown in `sub_results` |

### Core: Conversation Split Strategies

Multi-turn conversations must be split into query (input) and response (output) halves for evaluation. How you split determines *what you're evaluating*:

**Last-turn split** — split at the last user message. Everything up to and including it is the query context; the agent's subsequent actions are the response:

```
conversation: user1 → assistant1 → user2 → assistant2(tool) → tool_result → assistant3
query_messages:    [user1, assistant1, user2]
response_messages: [assistant2(tool), tool_result, assistant3]
```

This evaluates: "Given all the context so far, did the agent answer the latest question well?" Best for response quality at a specific point in the conversation.

**Full-conversation split** — the first user message is the query; everything after is the response:

```
query_messages:    [user1]
response_messages: [assistant1, user2, assistant2(tool), tool_result, assistant3]
```

This evaluates: "Given the original request, did the entire conversation trajectory serve the user?" Best for task completion and overall conversation quality.

**Per-turn split** — produces N eval items from an N-turn conversation. Each turn is evaluated with its cumulative context:

```
item 1: query = [user1],                        response = [assistant1]
item 2: query = [user1, assistant1, user2],      response = [assistant2(tool), tool_result, assistant3]
```

This evaluates each response independently. Best for fine-grained analysis and pinpointing where a conversation goes wrong.

These factorings produce different scores for the same conversation. The framework ships all three as built-in strategies, defaulting to last-turn. Developers can also provide a custom splitter — a function (Python) or `IConversationSplitter` implementation (.NET) — and override the strategy at the call site or per evaluator.

### Azure AI: FoundryEvals

`Evaluator` implementation backed by Azure AI Foundry:

```python
class FoundryEvals:
    def __init__(self, *, project_client=None, openai_client=None,
                 model_deployment: str, evaluators=None, ...)
    async def evaluate(self, items, *, eval_name) -> EvalResults
```

**Smart auto-detection in `evaluate()`:**
- Default evaluators: relevance, coherence, task_adherence
- Auto-adds `tool_call_accuracy` when items have tools/`tool_definitions`
- Filters out tool evaluators for items without tools
- Responses API fast path when all items have `response_id` and no tool evaluators

### Azure AI: FoundryEvaluators Constants

```python
from agent_framework_azure_ai import FoundryEvaluators

evaluators = [FoundryEvals.RELEVANCE, FoundryEvals.TOOL_CALL_ACCURACY]
```

Categories: Agent behavior, Tool usage, Quality, Safety.

### Azure AI: Foundry-Specific Functions

| Function | What it does |
|---|---|
| `evaluate_traces()` | Evaluate from stored response IDs or OTel traces |
| `evaluate_foundry_target()` | Evaluate a Foundry-registered agent or deployment |

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
def used_tools(conversation: list, tools: list) -> float:
    # ... scoring logic
    return score

local = LocalEvaluator(is_concise, mentions_city, used_tools)
```

Supported parameters: `query`, `response`, `expected`, `conversation`, `tools`, `context`.
Return types: `bool`, `float` (≥0.5 = pass), `dict` with `score` or `passed` key, or `CheckResult`.

Async functions are handled automatically — `@function_evaluator` detects `async def` and produces the right wrapper.

### Package Location

- Core types and orchestration: `agent_framework._eval`, `agent_framework._local_eval` (Python), `Microsoft.Agents.AI` (.NET)
- Foundry provider: `agent_framework_azure_ai._foundry_evals` (Python), `Microsoft.Agents.AI.AzureAI` (.NET)
- Azure-AI re-exports core types for convenience (Python)

## Known Limitations

1. **Per-agent workflow evals use dataset path**: Workflow sub-agent evaluations always clear `response_id` to force the dataset evaluation path. The Responses API retrieval path doesn't work for agents whose input is another agent's full conversation (the evaluator can't extract a clean user query from the stored response).
2. **Tool evaluators require query + agent**: Tool evaluators need tool definition schemas, which are not available through Responses API response retrieval. When using these evaluators with `evaluate_agent(responses=...)`, provide `queries=` and pass an agent with tool definitions.
3. **`model_deployment` always required**: Could potentially be inferred from the Foundry project configuration.

## Open Questions

1. **Red teaming non-registered agents**: Requires Foundry API support for callback-based flows.
2. **Datasets with expected outputs**: A dataset abstraction for pre-populating `expected` values across eval runs is a natural next step but not yet designed.

## .NET Implementation Design

### Key Difference: MEAI Ecosystem

Unlike Python, the .NET ecosystem already has `Microsoft.Extensions.AI.Evaluation` (v10.3.0) providing:

- `IEvaluator` — per-item evaluation of `(messages, chatResponse) → EvaluationResult`
- `CompositeEvaluator` — combines multiple evaluators
- Quality evaluators — `RelevanceEvaluator`, `CoherenceEvaluator`, `GroundednessEvaluator`
- Safety evaluators — `ContentHarmEvaluator`, `ProtectedMaterialEvaluator`
- Metric types — `NumericMetric`, `BooleanMetric`, `StringMetric`

The .NET integration uses MEAI's `IEvaluator` directly — no new evaluator interface. Our contribution is the **orchestration layer**: extension methods that run agents, extract data, call `IEvaluator` per item, and aggregate results.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Developer Code                                              │
│  agent.EvaluateAsync(queries, evaluator)                     │
│  run.EvaluateAsync(evaluator)                                │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────────────┐
│  Orchestration Layer (Microsoft.Agents.AI)                   │
│  AgentEvaluationExtensions — runs agents, extracts data,     │
│  calls IEvaluator per item, aggregates into                  │
│  AgentEvaluationResults                                      │
└────────────────┬─────────────────────────────────────────────┘
                 │ IEvaluator (MEAI)
                 │
     ┌───────────┼────────────┐
     │           │            │
 ┌───▼───-┐  ┌───▼────┐  ┌────▼──────────┐
 │ MEAI   │  │ Local  │  │ Foundry       │
 │ Quality│  │ Checks │  │ (cloud batch) │
 │ Safety │  │ Lambdas│  │               │
 └────────┘  └────────┘  └───────────────┘
```

All evaluators implement MEAI's `IEvaluator`. The orchestration layer doesn't need to know which kind — it calls `EvaluateAsync(messages, chatResponse)` per item on all of them. `FoundryEvaluator` handles batching internally (buffers items, submits once, returns per-item results).

### .NET Core Types

**No new evaluator interface.** Use MEAI's `IEvaluator` directly.

**`AgentEvaluationResults`** — The only new type. Aggregates per-item MEAI `EvaluationResult`s across a batch of queries:

```csharp
public class AgentEvaluationResults
{
    public string Provider { get; init; }
    public string? ReportUrl { get; init; }

    // Per-item — standard MEAI EvaluationResult, unchanged
    public IReadOnlyList<EvaluationResult> Items { get; init; }

    // Aggregate pass/fail derived from metric interpretations
    public int Passed { get; }
    public int Failed { get; }
    public int Total { get; }
    public bool AllPassed { get; }

    // Workflow: per-agent breakdown
    public IReadOnlyDictionary<string, AgentEvaluationResults>? SubResults { get; init; }

    public void AssertAllPassed(string? message = null);
}
```

### .NET Evaluator Implementations

All implement MEAI's `IEvaluator`:

**`LocalEvaluator`** — Runs lambda checks locally, returns `BooleanMetric` per check:

```csharp
var local = new LocalEvaluator(
    FunctionEvaluator.Create("is_concise",
        (string response) => response.Split().Length < 500),
    EvalChecks.KeywordCheck("weather"),
    EvalChecks.ToolCalledCheck("get_weather"));
```

**MEAI evaluators** — Used directly, no adapter needed:

```csharp
var quality = new CompositeEvaluator(
    new RelevanceEvaluator(),
    new CoherenceEvaluator());
```

**`FoundryEvaluator`** — Implements `IEvaluator` but batches internally. On first call, buffers the item. On the last item (or when explicitly flushed), submits the batch to Foundry and distributes per-item results:

```csharp
var foundry = new FoundryEvaluator(projectClient, "gpt-4o");
```

### .NET Orchestration: Extension Methods

```csharp
public static class AgentEvaluationExtensions
{
    // Evaluate an agent against test queries
    public static Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IEvaluator evaluator,
        ChatConfiguration? chatConfiguration = null,
        CancellationToken cancellationToken = default);

    // Evaluate pre-existing responses (without re-running the agent)
    public static Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        AgentResponse responses,
        IEvaluator evaluator,
        IEnumerable<string>? queries = null,
        ChatConfiguration? chatConfiguration = null,
        CancellationToken cancellationToken = default);

    // Evaluate with multiple evaluators (one result per evaluator)
    public static Task<IReadOnlyList<AgentEvaluationResults>> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IEnumerable<IEvaluator> evaluators,
        ChatConfiguration? chatConfiguration = null,
        CancellationToken cancellationToken = default);

    // Evaluate a workflow run with per-agent breakdown
    public static Task<AgentEvaluationResults> EvaluateAsync(
        this Run run,
        IEvaluator evaluator,
        ChatConfiguration? chatConfiguration = null,
        bool includeOverall = true,
        bool includePerAgent = true,
        CancellationToken cancellationToken = default);
}
```

**Usage:**

```csharp
// MEAI evaluators — just works
var results = await agent.EvaluateAsync(
    queries: ["What's the weather?"],
    evaluator: new RelevanceEvaluator(),
    chatConfiguration: new ChatConfiguration(evalClient));

// Local checks
var results = await agent.EvaluateAsync(
    queries: ["What's the weather?"],
    evaluator: new LocalEvaluator(
        EvalChecks.KeywordCheck("weather")));

// Foundry cloud
var results = await agent.EvaluateAsync(
    queries: ["What's the weather?"],
    evaluator: new FoundryEvaluator(projectClient, "gpt-4o"));

// Evaluate existing response (without re-running the agent)
var response = await agent.RunAsync("What's the weather?");
var results = await agent.EvaluateAsync(
    responses: response,
    queries: ["What's the weather?"],
    evaluator: new FoundryEvaluator(projectClient, "gpt-4o"));

// Mixed — one result per evaluator
var results = await agent.EvaluateAsync(
    queries: ["What's the weather?"],
    evaluators: [
        new LocalEvaluator(EvalChecks.KeywordCheck("weather")),
        new RelevanceEvaluator(),
        new FoundryEvaluator(projectClient, "gpt-4o")
    ],
    chatConfiguration: new ChatConfiguration(evalClient));

// Workflow with per-agent breakdown
Run run = await workflowRunner.RunAsync(workflow, "Plan a trip");
var results = await run.EvaluateAsync(
    evaluator: new FoundryEvaluator(projectClient, "gpt-4o"));
```

### .NET Function Evaluators

Typed factory overloads (C# equivalent of Python's `@function_evaluator`):

```csharp
public static class FunctionEvaluator
{
    public static EvalCheck Create(string name, Func<string, bool> check);           // response only
    public static EvalCheck Create(string name, Func<string, string?, bool> check);  // + expected
    public static EvalCheck Create(string name, Func<EvalItem, bool> check);         // full item
    public static EvalCheck Create(string name, Func<EvalItem, CheckResult> check);  // full control
    public static EvalCheck Create(string name, Func<string, Task<bool>> check);     // async
}
```

`EvalItem` is a lightweight record used only by `FunctionEvaluator` and `LocalEvaluator` to pass context to check functions. It is not part of the `IEvaluator` interface:

```csharp
public record EvalItem(
    string Query, string Response,
    IReadOnlyList<ChatMessage> Conversation,
    IReadOnlyList<AITool>? Tools = null,
    string? Expected = null, string? Context = null);
```

### Workflow Data Extraction (.NET)

`run.EvaluateAsync()` walks `Run.OutgoingEvents` via LINQ:

1. Pair `ExecutorInvokedEvent` / `ExecutorCompletedEvent` by `ExecutorId`
2. Extract `AgentResponseEvent` for per-agent `ChatResponse`
3. Call `evaluator.EvaluateAsync()` per invocation
4. Group by `ExecutorId` for per-agent `SubResults`
5. Use final workflow output for overall eval

### .NET Package Structure

| Package | Contents |
|---------|----------|
| `Microsoft.Agents.AI` | `AgentEvaluationResults`, `LocalEvaluator`, `FunctionEvaluator`, `EvalChecks`, `EvalItem`, `AgentEvaluationExtensions` |
| `Microsoft.Agents.AI.AzureAI` | `FoundryEvaluator`, `Evaluators` constants |

### Python ↔ .NET Mapping

| Python | .NET |
|--------|------|
| `Evaluator` protocol | MEAI `IEvaluator` (no new interface) |
| `EvalItem` dataclass | `EvalItem` record (internal to checks) |
| `EvalResults` | `AgentEvaluationResults` |
| `EvalItemResult` / `EvalScoreResult` | MEAI `EvaluationResult` / `EvaluationMetric` (reused) |
| `LocalEvaluator` | `LocalEvaluator` (implements `IEvaluator`) |
| `@function_evaluator` | `FunctionEvaluator.Create()` overloads |
| `keyword_check()` / `tool_called_check()` | `EvalChecks.KeywordCheck()` / `EvalChecks.ToolCalledCheck()` |
| `FoundryEvals` | `FoundryEvaluator` (implements `IEvaluator`) |
| `evaluate_agent()` | `agent.EvaluateAsync(queries, evaluator)` extension method |
| `evaluate_agent(responses=)` | `agent.EvaluateAsync(responses, evaluator)` extension method |
| `evaluate_workflow()` | `run.EvaluateAsync()` extension method |
| `AgentEvalConverter` | Internal to extension methods |

## More Information

- [Foundry Evals documentation](https://learn.microsoft.com/azure/ai-foundry/concepts/evaluation-approach-gen-ai) — Azure AI Foundry evaluation overview
