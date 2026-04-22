// Copyright (c) Microsoft. All rights reserved.

// This sample demonstrates a basic AG-UI server hosting a chat agent for the Blazor web client.

using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AGUI.AspNetCore;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpClient().AddLogging();
builder.Services.AddAGUI();

WebApplication app = builder.Build();

string endpoint = builder.Configuration["FOUNDRY_PROJECT_ENDPOINT"] ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = builder.Configuration["FOUNDRY_MODEL"] ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

// Create the AI agent
AIProjectClient aiProjectClient = new(
    new Uri(endpoint),
    new DefaultAzureCredential());

ChatClientAgent agent = aiProjectClient.AsAIAgent(
    model: deploymentName,
    instructions: "You are a helpful assistant.",
    name: "ChatAssistant");

// Map the AG-UI agent endpoint
app.MapAGUI("/ag-ui", agent);

await app.RunAsync();
