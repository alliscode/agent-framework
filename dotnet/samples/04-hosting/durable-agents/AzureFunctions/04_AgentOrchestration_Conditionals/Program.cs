// Copyright (c) Microsoft. All rights reserved.

// Multi-Agent Conditional Orchestration — Azure Functions Hosting
// Demonstrates conditional branching in a Durable Task orchestration with spam detection
// and email response agents, hosted as an Azure Function.

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

// Two agents used by the orchestration to demonstrate conditional logic.
const string SpamDetectionName = "SpamDetectionAgent";
const string SpamDetectionInstructions = "You are a spam detection assistant that identifies spam emails.";

const string EmailAssistantName = "EmailAssistantAgent";
const string EmailAssistantInstructions = "You are an email assistant that helps users draft responses to emails with professionalism.";

AIAgent spamDetectionAgent = client.AsAIAgent(model: deploymentName, instructions: SpamDetectionInstructions, name: SpamDetectionName);

AIAgent emailAssistantAgent = client.AsAIAgent(model: deploymentName, instructions: EmailAssistantInstructions, name: EmailAssistantName);

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options =>
    {
        options
            .AddAIAgent(spamDetectionAgent)
            .AddAIAgent(emailAssistantAgent);
    })
    .Build();

app.Run();
