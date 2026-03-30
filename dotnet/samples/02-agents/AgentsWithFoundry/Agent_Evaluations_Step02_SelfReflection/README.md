# Self-Reflection Evaluation with Groundedness Assessment

This sample demonstrates the self-reflection pattern using Agent Framework with `Microsoft.Extensions.AI.Evaluation.Quality` evaluators. The agent iteratively improves its responses based on real groundedness evaluation scores.

For details on the self-reflection approach, see [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (NeurIPS 2023).

## What this sample demonstrates

- Self-reflection loop that improves responses using real `GroundednessEvaluator` scores
- Using `RelevanceEvaluator` and `CoherenceEvaluator` for multi-metric quality assessment
- Combining quality and safety evaluators with `CompositeEvaluator`
- Configuring `ContentSafetyServiceConfiguration` for safety evaluators alongside LLM-based quality evaluators
- Tracking improvement across iterations

## Prerequisites

Before you begin, ensure you have the following prerequisites:

- .NET 10 SDK or later
- Azure AI Foundry project (hub and project created)
- Azure OpenAI deployment (e.g., gpt-4o or gpt-4o-mini)
- Azure CLI installed and authenticated (for Azure credential authentication)

**Note**: This demo uses Azure CLI credentials for authentication. Make sure you're logged in with `az login` and have access to the Azure Foundry resource. For more information, see the [Azure CLI documentation](https://learn.microsoft.com/cli/azure/authenticate-azure-cli-interactively).

### Environment Variables

Set the following environment variables:

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com/api/projects/your-project"
$env:AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4o-mini"
```

## Run the sample

Navigate to the sample directory and run:

```powershell
cd dotnet/samples/02-agents/AgentsWithFoundry/Agent_Evaluations_Step02_SelfReflection
dotnet run
```

## Expected behavior

The sample runs three evaluation scenarios:

### 1. Self-Reflection with Groundedness
- Asks a question with grounding context
- Evaluates response groundedness using `GroundednessEvaluator`
- If score is below 4/5, asks the agent to improve with feedback
- Repeats up to 3 iterations

### 2. Quality Evaluation
- Evaluates a single response with `RelevanceEvaluator`, `CoherenceEvaluator`, and `GroundednessEvaluator`

### 3. Combined Quality + Safety Evaluation
- Runs quality evaluators alongside `ContentHarmEvaluator` and `ProtectedMaterialEvaluator`

## Related Resources

- [Reflexion Paper (NeurIPS 2023)](https://arxiv.org/abs/2303.11366)
- [Microsoft.Extensions.AI.Evaluation Libraries](https://learn.microsoft.com/dotnet/ai/evaluation/libraries)
- [GroundednessEvaluator API Reference](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.evaluation.quality.groundednessevaluator)

## Next Steps

After running self-reflection evaluation:
1. Implement similar patterns for other quality metrics
2. Integrate into CI/CD pipeline for continuous quality assurance
3. Explore the All Patterns sample (Agent_Evaluations_Step03_AllPatterns) for the complete evaluation toolkit
