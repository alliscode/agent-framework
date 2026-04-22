// Copyright (c) Microsoft. All rights reserved.

// Concurrent Workflow — Azure Functions Hosting
// Demonstrates the Fan-out/Fan-in pattern in a durable workflow hosted as an Azure Function.

using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.DurableTask;
using Microsoft.Agents.AI.Hosting.AzureFunctions;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Azure.Functions.Worker.Builder;
using Microsoft.Extensions.Hosting;
using WorkflowConcurrency;

string endpoint = Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")
    ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = Environment.GetEnvironmentVariable("FOUNDRY_MODEL")
    ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

AIProjectClient client = new(new Uri(endpoint), new DefaultAzureCredential());

// Define the 4 executors for the workflow
ParseQuestionExecutor parseQuestion = new();
AIAgent physicist = client.AsAIAgent(model: deploymentName, instructions: "You are a physics expert. Be concise (2-3 sentences).", name: "Physicist");
AIAgent chemist = client.AsAIAgent(model: deploymentName, instructions: "You are a chemistry expert. Be concise (2-3 sentences).", name: "Chemist");
AggregatorExecutor aggregator = new();

// Build workflow: ParseQuestion -> [Physicist, Chemist] (parallel) -> Aggregator
Workflow workflow = new WorkflowBuilder(parseQuestion)
    .WithName("ExpertReview")
    .AddFanOutEdge(parseQuestion, [physicist, chemist])
    .AddFanInBarrierEdge([physicist, chemist], aggregator)
    .Build();

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableWorkflows(workflows => workflows.AddWorkflows(workflow))
    .Build();
app.Run();
