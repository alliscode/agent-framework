// Copyright (c) Microsoft. All rights reserved.

// AG-UI Server — hosts an AI agent with tools exposed via the AG-UI protocol.
// Demonstrates streaming agent responses to AG-UI compatible clients.

using System.ComponentModel;
using AGUIServer;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Hosting.AGUI.AspNetCore;
using Microsoft.Extensions.AI;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpClient().AddLogging();
builder.Services.ConfigureHttpJsonOptions(options => options.SerializerOptions.TypeInfoResolverChain.Add(AGUIServerSerializerContext.Default));
builder.Services.AddAGUI();

string endpoint = builder.Configuration["FOUNDRY_PROJECT_ENDPOINT"] ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = builder.Configuration["FOUNDRY_MODEL"] ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

const string AgentName = "AGUIAssistant";

// Create the AI agent with tools
var agent = new AIProjectClient(
        new Uri(endpoint),
        new DefaultAzureCredential())
    .AsAIAgent(
        model: deploymentName,
        instructions: "You are a helpful assistant.",
        name: AgentName,
        tools: [
            AIFunctionFactory.Create(
                () => DateTimeOffset.UtcNow,
                name: "get_current_time",
                description: "Get the current UTC time."
            ),
            AIFunctionFactory.Create(
                ([Description("The weather forecast request")]ServerWeatherForecastRequest request) => {
                    return new ServerWeatherForecastResponse()
                    {
                        Summary = "Sunny",
                        TemperatureC = 25,
                        Date = request.Date
                    };
                },
                name: "get_server_weather_forecast",
                description: "Gets the forecast for a specific location and date",
                AGUIServerSerializerContext.Default.Options)
        ]);

// Register the agent with the host and configure it to use an in-memory session store
// so that conversation state is maintained across requests. In production, you may want to use a persistent session store.
builder
    .AddAIAgent(AgentName, (_, _) => agent)
    .WithInMemorySessionStore();

WebApplication app = builder.Build();

// Map the AG-UI agent endpoint
app.MapAGUI(AgentName, "/");

await app.RunAsync();
