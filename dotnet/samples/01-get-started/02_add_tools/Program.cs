// Copyright (c) Microsoft. All rights reserved.

// Add Tools — Give your agent a function tool
//
// This sample shows how to define a function tool and wire it into
// an agent so the model can call it.

using System.ComponentModel;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

// <define_tool>
// Define a function tool the agent can call.
// NOTE: Tool approval is not required in this sample for brevity.
// In production, consider requiring user confirmation before tool execution.
[Description("Get the weather for a given location.")]
static string GetWeather([Description("The location to get the weather for.")] string location)
{
    string[] conditions = ["sunny", "cloudy", "rainy", "stormy"];
    return $"The weather in {location} is {conditions[Random.Shared.Next(conditions.Length)]} with a high of {Random.Shared.Next(10, 31)}°C.";
}
// </define_tool>

// <create_agent_with_tools>
AIAgent agent = new AzureOpenAIClient(
    new Uri(endpoint),
    new DefaultAzureCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(
        instructions: "You are a helpful weather agent. Use the GetWeather tool to answer questions.",
        tools: [AIFunctionFactory.Create(GetWeather)]);
// </create_agent_with_tools>

// <run_agent>
Console.WriteLine(await agent.RunAsync("What's the weather like in Seattle?"));
// </run_agent>
