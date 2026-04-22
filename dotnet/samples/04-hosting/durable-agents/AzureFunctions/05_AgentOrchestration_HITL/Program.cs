// Copyright (c) Microsoft. All rights reserved.

// Human-in-the-Loop Orchestration — Azure Functions Hosting
// Demonstrates the HITL pattern with content generation and approval workflows,
// hosted as an Azure Function.

#pragma warning disable IDE0002 // Simplify Member Access

using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AzureFunctions;
using Microsoft.Azure.Functions.Worker.Builder;
using Microsoft.Extensions.Hosting;

// Get the Azure AI Foundry endpoint and model from environment variables.
string endpoint = Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")
    ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = Environment.GetEnvironmentVariable("FOUNDRY_MODEL")
    ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

AIProjectClient client = new(new Uri(endpoint), new DefaultAzureCredential());

// Single agent used by the orchestration to demonstrate human-in-the-loop workflow.
const string WriterName = "WriterAgent";
const string WriterInstructions =
    """
    You are a professional content writer who creates high-quality articles on various topics.
    You write engaging, informative, and well-structured content that follows best practices for readability and accuracy.
    """;

AIAgent writerAgent = client.AsAIAgent(model: deploymentName, instructions: WriterInstructions, name: WriterName);

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options => options.AddAIAgent(writerAgent))
    .Build();

app.Run();
