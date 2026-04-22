// Copyright (c) Microsoft. All rights reserved.

// AG-UI Getting Started — Server-side agent with AG-UI protocol
//
// This sample shows how to set up an ASP.NET Core server that exposes
// an AI agent via the AG-UI protocol.

using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AGUI.AspNetCore;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpClient().AddLogging();
builder.Services.AddAGUI();

WebApplication app = builder.Build();

string endpoint = builder.Configuration["FOUNDRY_PROJECT_ENDPOINT"]
    ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = builder.Configuration["FOUNDRY_MODEL"]
    ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

// Create the AI agent
AIProjectClient aiProjectClient = new(new Uri(endpoint), new DefaultAzureCredential());

AIAgent agent = aiProjectClient.AsAIAgent(
    model: deploymentName,
    name: "AGUIAssistant",
    instructions: "You are a helpful assistant.");

// Map the AG-UI agent endpoint
app.MapAGUI("/", agent);

await app.RunAsync();
