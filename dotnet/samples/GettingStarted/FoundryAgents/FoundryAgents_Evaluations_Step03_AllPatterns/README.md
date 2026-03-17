# Evaluation — All Patterns

This sample demonstrates all evaluation patterns available in Agent Framework for .NET:

| Section | Pattern | Description |
|---------|---------|-------------|
| 1 | **Function Evaluators** | Custom checks using C# lambdas via `FunctionEvaluator.Create()` |
| 2 | **Built-in Checks** | `EvalChecks.KeywordCheck()` and `EvalChecks.ToolCalledCheck()` |
| 3 | **MEAI Quality Evaluators** | LLM-based scoring with `RelevanceEvaluator`, `CoherenceEvaluator` |
| 4 | **Foundry Evaluators** | Cloud-based evaluation via `FoundryEvals` |
| 5 | **Mixed Evaluators** | Combining local checks with cloud evaluation in one call |
| 6 | **Pre-existing Responses** | Evaluate saved responses without re-running the agent |

## Prerequisites

- Azure AI Foundry project with a deployed model
- Set environment variables:
  - `AZURE_FOUNDRY_PROJECT_ENDPOINT` — Your Azure AI Foundry project endpoint
  - `AZURE_FOUNDRY_PROJECT_DEPLOYMENT_NAME` — Model deployment name (default: `gpt-4o-mini`)

## Key Types

```csharp
// Custom function evaluators
var check = FunctionEvaluator.Create("name", (string response) => response.Length > 10);

// Built-in checks
var keyword = EvalChecks.KeywordCheck("expected", "keywords");
var toolCheck = EvalChecks.ToolCalledCheck("tool_name");

// Local evaluator runs checks without API calls
var local = new LocalEvaluator(check, keyword, toolCheck);

// MEAI evaluators work directly — no adapter needed
var results = await agent.EvaluateAsync(queries, new RelevanceEvaluator(), chatConfig);

// Foundry evaluator uses Azure AI Foundry cloud evaluation
var foundry = new FoundryEvals(chatConfig, Evaluators.Relevance, Evaluators.Coherence);

// Evaluate an agent
AgentEvaluationResults localResults = await agent.EvaluateAsync(queries, local);
localResults.AssertAllPassed();
```

## Running

```bash
dotnet run --project FoundryAgents_Evaluations_Step03_AllPatterns.csproj
```
