// Copyright (c) Microsoft. All rights reserved.

// Host Your Agent — Azure Functions hosting
//
// This sample shows the .NET hosting pattern:
// - Create an agent with Azure OpenAI
// - Register it with DurableAgents
// - Run with Azure Functions Core Tools (func start)
//
// Prerequisites:
//   - Azure Functions Core Tools
//   - Azure OpenAI resource
//
// Environment variables:
//   AZURE_OPENAI_ENDPOINT
//   AZURE_OPENAI_DEPLOYMENT_NAME (defaults to "gpt-5.4-mini")
//
// Run with: func start
// Then call: POST http://localhost:7071/api/agents/HostedAgent/run

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AzureFunctions;
using Microsoft.Azure.Functions.Worker.Builder;
using Microsoft.Extensions.Hosting;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

// <create_agent>
AIAgent agent = new AzureOpenAIClient(new Uri(endpoint), new DefaultAzureCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(
        instructions: "You are a helpful assistant hosted in Azure Functions.",
        name: "HostedAgent");
// </create_agent>

// <host_agent>
// Configure the function app to host the AI agent.
// This automatically generates HTTP API endpoints for the agent.
using IHost app = FunctionsApplication
    .CreateBuilder(args)
    .ConfigureFunctionsWebApplication()
    .ConfigureDurableAgents(options => options.AddAIAgent(agent, timeToLive: TimeSpan.FromHours(1)))
    .Build();
app.Run();
// </host_agent>
