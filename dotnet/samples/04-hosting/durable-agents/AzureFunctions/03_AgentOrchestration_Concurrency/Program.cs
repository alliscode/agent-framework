// Copyright (c) Microsoft. All rights reserved.

// Multi-Agent Concurrent Orchestration — Azure Functions Hosting
// Demonstrates running multiple agents concurrently within a Durable Task orchestration,
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

// Two agents used by the orchestration to demonstrate concurrent execution.
const string PhysicistName = "PhysicistAgent";
const string PhysicistInstructions = "You are an expert in physics. You answer questions from a physics perspective.";

const string ChemistName = "ChemistAgent";
const string ChemistInstructions = "You are an expert in chemistry. You answer questions from a chemistry perspective.";

AIAgent physicistAgent = client.AsAIAgent(model: deploymentName, instructions: PhysicistInstructions, name: PhysicistName);
AIAgent chemistAgent = client.AsAIAgent(model: deploymentName, instructions: ChemistInstructions, name: ChemistName);

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options =>
    {
        options
            .AddAIAgent(physicistAgent)
            .AddAIAgent(chemistAgent);
    })
    .Build();

app.Run();
