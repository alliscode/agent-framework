// Copyright (c) Microsoft. All rights reserved.

// This sample demonstrates how to configure AI agents to be accessible as MCP tools.
// When using AddAIAgent and enabling MCP tool triggers, the Functions host will automatically
// generate a remote MCP endpoint for the app at /runtime/webhooks/mcp with a agent-specific
// query tool name.

#pragma warning disable IDE0002 // Simplify Member Access

using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.DurableTask;
using Microsoft.Agents.AI.Hosting.AzureFunctions;
using Microsoft.Azure.Functions.Worker.Builder;
using Microsoft.Extensions.Hosting;

// Get the Azure AI Foundry endpoint and model from environment variables.
string endpoint = Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")
    ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = Environment.GetEnvironmentVariable("FOUNDRY_MODEL")
    ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

AIProjectClient client = new(new Uri(endpoint), new DefaultAzureCredential());

// Define three AI agents we are going to use in this application.
AIAgent agent1 = client.AsAIAgent(model: deploymentName, instructions: "You are good at telling jokes.", name: "Joker");

AIAgent agent2 = client.AsAIAgent(model: deploymentName, instructions: "Check stock prices.", name: "StockAdvisor");

AIAgent agent3 = client.AsAIAgent(model: deploymentName, instructions: "Recommend plants.", name: "PlantAdvisor", description: "Get plant recommendations.");

using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options =>
    {
        options
        .AddAIAgent(agent1)  // Enables HTTP trigger by default.
        .AddAIAgent(agent2, enableHttpTrigger: false, enableMcpToolTrigger: true) // Disable HTTP trigger, enable MCP Tool trigger.
        .AddAIAgent(agent3, agentOptions =>
        {
            agentOptions.McpToolTrigger.IsEnabled = true; // Enable MCP Tool trigger.
        });
    })
    .Build();
app.Run();
