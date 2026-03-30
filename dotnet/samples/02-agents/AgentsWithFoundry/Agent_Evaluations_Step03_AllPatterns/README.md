# Evaluation -- All Patterns

This sample demonstrates all evaluation patterns available in Agent Framework for .NET:

| Section | Pattern | Description |
|---------|---------|-------------|
| 1 | **Function Evaluators** | Custom checks using C# lambdas via `FunctionEvaluator.Create()` |
| 2 | **Built-in Checks** | `EvalChecks.KeywordCheck()` and `EvalChecks.ToolCalledCheck()` |
| 3 | **MEAI Quality Evaluators** | LLM-based scoring with `RelevanceEvaluator`, `CoherenceEvaluator` |
| 4 | **Foundry Evaluators** | Cloud-based evaluation via `FoundryEvals` |
| 5 | **Mixed Evaluators** | Combining local checks with cloud evaluation in one call |
| 6 | **Pre-existing Responses** | Evaluate saved responses without re-running the agent |
| 7 | **Conversation Splits** | LastTurn, Full, PerTurn, and custom splitter strategies |

## Prerequisites

- Azure AI Foundry project with a deployed model
- Set environment variables:
  - `AZURE_AI_PROJECT_ENDPOINT` -- Your Azure AI Foundry project endpoint
  - `AZURE_AI_MODEL_DEPLOYMENT_NAME` -- Model deployment name (default: `gpt-4o-mini`)

## Key Types

```csharp
// Custom function evaluators
var check = FunctionEvaluator.Create("name", (string response) => response.Length > 10);

// Built-in checks
var keyword = EvalChecks.KeywordCheck("expected", "keywords");
var toolCheck = EvalChecks.ToolCalledCheck("tool_name");

// Local evaluator runs checks without API calls
var local = new LocalEvaluator(check, keyword, toolCheck);

// MEAI evaluators work directly -- no adapter needed
var results = await agent.EvaluateAsync(queries, new RelevanceEvaluator(), chatConfig);

// Foundry evaluator uses Azure AI Foundry cloud evaluation
var foundry = new FoundryEvals(chatConfig, FoundryEvals.Relevance, FoundryEvals.Coherence);

// Evaluate an agent
AgentEvaluationResults localResults = await agent.EvaluateAsync(queries, local);
localResults.AssertAllPassed();
```

## Running

```powershell
cd dotnet/samples/02-agents/AgentsWithFoundry/Agent_Evaluations_Step03_AllPatterns
dotnet run
```
