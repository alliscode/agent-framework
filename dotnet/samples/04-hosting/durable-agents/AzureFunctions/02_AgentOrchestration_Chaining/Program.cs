// Copyright (c) Microsoft. All rights reserved.

// Agent Orchestration Chaining — Azure Functions Hosting
// Demonstrates chaining sequential agent invocations within a Durable Task orchestration,
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

// Single agent used by the orchestration to demonstrate sequential calls on the same session.
const string WriterName = "WriterAgent";
const string WriterInstructions =
    """
    You refine short pieces of text. When given an initial sentence you enhance it;
    when given an improved sentence you polish it further.
    """;

AIAgent writerAgent = client.AsAIAgent(model: deploymentName, instructions: WriterInstructions, name: WriterName);

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options => options.AddAIAgent(writerAgent))
    .Build();

app.Run();
