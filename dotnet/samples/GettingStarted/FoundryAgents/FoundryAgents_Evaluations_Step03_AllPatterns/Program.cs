// Copyright (c) Microsoft. All rights reserved.

// This sample demonstrates all evaluation patterns available in Agent Framework for .NET.
// It covers:
//   1. Function evaluators — custom checks using lambdas
//   2. Built-in checks — keyword and tool-called validation
//   3. MEAI evaluators — LLM-based quality scoring (Relevance, Coherence, Groundedness)
//   4. Foundry evaluators — cloud-based evaluation with Azure AI Foundry
//   5. Mixed evaluators — combining local checks with cloud evaluation
//   6. Pre-existing response evaluation — evaluate responses without re-running the agent
//
// Mirrors the Python sample: evaluate_all_patterns_sample.py

using Azure.AI.Projects;
using Azure.AI.Projects.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.AzureAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;
using Microsoft.Extensions.AI.Evaluation.Safety;

using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using ChatRole = Microsoft.Extensions.AI.ChatRole;
using Evaluators = Microsoft.Agents.AI.AzureAI.Evaluators;

string endpoint = Environment.GetEnvironmentVariable("AZURE_FOUNDRY_PROJECT_ENDPOINT")
    ?? throw new InvalidOperationException("AZURE_FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = Environment.GetEnvironmentVariable("AZURE_FOUNDRY_PROJECT_DEPLOYMENT_NAME") ?? "gpt-4o-mini";

Console.WriteLine("=" + new string('=', 79));
Console.WriteLine("AGENT FRAMEWORK EVALUATION — ALL PATTERNS");
Console.WriteLine("=" + new string('=', 79));
Console.WriteLine();

// Initialize Azure credentials and clients — everything derives from the project endpoint
DefaultAzureCredential credential = new();
AIProjectClient aiProjectClient = new(new Uri(endpoint), credential);

// Get a chat client for LLM-based evaluators from the project client
IChatClient chatClient = aiProjectClient
    .GetProjectOpenAIClient()
    .GetChatClient(deploymentName)
    .AsIChatClient();

ContentSafetyServiceConfiguration safetyConfig = new(
    credential: credential,
    endpoint: new Uri(endpoint));

ChatConfiguration chatConfiguration = safetyConfig.ToChatConfiguration(
    originalChatConfiguration: new ChatConfiguration(chatClient));

// Create test agent
AIAgent agent = await aiProjectClient.CreateAIAgentAsync(
    name: "WeatherAgent",
    model: deploymentName,
    instructions: "You are a helpful weather assistant. Answer questions about weather accurately and concisely.");

Console.WriteLine($"Created agent: {agent.Name}");
Console.WriteLine();

string[] queries = ["What's the weather in Seattle?", "Is it going to rain in New York today?"];

try
{
    // ================================================================
    // Section 1: Function Evaluators
    // ================================================================
    Console.WriteLine("SECTION 1: Function Evaluators");
    Console.WriteLine(new string('-', 60));

    var functionEvaluator = new LocalEvaluator(
        FunctionEvaluator.Create("is_concise",
            (string response) => response.Split(' ').Length < 500),
        FunctionEvaluator.Create("has_content",
            (string response) => response.Length > 10),
        FunctionEvaluator.Create("mentions_location",
            (EvalItem item) => item.Response.Contains("Seattle", StringComparison.OrdinalIgnoreCase)
                || item.Response.Contains("New York", StringComparison.OrdinalIgnoreCase)));

    AgentEvaluationResults functionResults = await agent.EvaluateAsync(
        queries,
        functionEvaluator);

    PrintResults("Function Evaluators", functionResults);

    // ================================================================
    // Section 2: Built-in Checks
    // ================================================================
    Console.WriteLine("SECTION 2: Built-in Checks");
    Console.WriteLine(new string('-', 60));

    var builtinEvaluator = new LocalEvaluator(
        EvalChecks.KeywordCheck("weather"),
        EvalChecks.KeywordCheck(caseSensitive: false, "temperature", "forecast"));

    AgentEvaluationResults builtinResults = await agent.EvaluateAsync(
        queries,
        builtinEvaluator);

    PrintResults("Built-in Checks", builtinResults);

    // ================================================================
    // Section 3: MEAI Quality Evaluators
    // ================================================================
    Console.WriteLine("SECTION 3: MEAI Quality Evaluators");
    Console.WriteLine(new string('-', 60));

    // Pass MEAI evaluators directly — no adapter needed
    AgentEvaluationResults meaiResults = await agent.EvaluateAsync(
        queries,
        new CompositeEvaluator(
            new RelevanceEvaluator(),
            new CoherenceEvaluator()),
        chatConfiguration);

    PrintResults("MEAI Quality", meaiResults);

    // Print per-metric details for MEAI results
    foreach (EvaluationResult itemResult in meaiResults.Items)
    {
        foreach (EvaluationMetric metric in itemResult.Metrics.Values)
        {
            if (metric is NumericMetric n)
            {
                string rating = n.Interpretation?.Rating.ToString() ?? "N/A";
                Console.WriteLine($"  {n.Name,-20} Score: {n.Value:F1}/5  Rating: {rating}");
            }
        }
    }

    Console.WriteLine();

    // ================================================================
    // Section 4: Foundry Evaluators (Cloud-based)
    // ================================================================
    Console.WriteLine("SECTION 4: Foundry Evaluators");
    Console.WriteLine(new string('-', 60));

    var foundryEvaluator = new FoundryEvaluator(
        chatConfiguration,
        Evaluators.Relevance,
        Evaluators.Coherence,
        Evaluators.Groundedness);

    AgentEvaluationResults foundryResults = await agent.EvaluateAsync(
        queries,
        foundryEvaluator);

    PrintResults("Foundry Evaluators", foundryResults);

    // ================================================================
    // Section 5: Mixed Evaluators (Local + Cloud)
    // ================================================================
    Console.WriteLine("SECTION 5: Mixed Evaluators");
    Console.WriteLine(new string('-', 60));

    IReadOnlyList<AgentEvaluationResults> mixedResults = await agent.EvaluateAsync(
        queries,
        evaluators: new IAgentEvaluator[]
        {
            new LocalEvaluator(
                EvalChecks.KeywordCheck("weather"),
                FunctionEvaluator.Create("not_empty", (string r) => r.Length > 0)),
            new FoundryEvaluator(chatConfiguration, Evaluators.Relevance),
        });

    foreach (AgentEvaluationResults result in mixedResults)
    {
        PrintResults($"Mixed - {result.Provider}", result);
    }

    // ================================================================
    // Section 6: Evaluate Pre-existing Responses
    // ================================================================
    Console.WriteLine("SECTION 6: Evaluate Pre-existing Responses");
    Console.WriteLine(new string('-', 60));

    // Get responses first
    var responses = new List<(string Query, AgentResponse Response)>();
    foreach (string query in queries)
    {
        AgentResponse response = await agent.RunAsync(
            new List<ChatMessage> { new(ChatRole.User, query) });
        responses.Add((query, response));
    }

    // Evaluate the saved responses without re-running the agent
    AgentEvaluationResults preExistingResults = await agent.EvaluateAsync(
        responses,
        new LocalEvaluator(
            EvalChecks.KeywordCheck("weather"),
            FunctionEvaluator.Create("response_quality",
                (EvalItem item) => new EvalCheckResult(
                    item.Response.Length > 20,
                    item.Response.Length > 20
                        ? "Response is detailed enough"
                        : "Response is too short",
                    "response_quality"))));

    PrintResults("Pre-existing Responses", preExistingResults);
}
finally
{
    // Cleanup
    await aiProjectClient.Agents.DeleteAgentAsync(agent.Name);
    Console.WriteLine("Cleanup: Agent deleted.");
}

// ============================================================================
// Helper Functions
// ============================================================================

static void PrintResults(string title, AgentEvaluationResults results)
{
    string status = results.AllPassed ? "✓ ALL PASSED" : "✗ SOME FAILED";
    Console.WriteLine($"  [{title}] {status} ({results.Passed}/{results.Total})");

    if (results.SubResults is not null)
    {
        foreach (var (agentId, sub) in results.SubResults)
        {
            string subStatus = sub.AllPassed ? "✓" : "✗";
            Console.WriteLine($"    {subStatus} {agentId}: {sub.Passed}/{sub.Total}");
        }
    }

    Console.WriteLine();
}
