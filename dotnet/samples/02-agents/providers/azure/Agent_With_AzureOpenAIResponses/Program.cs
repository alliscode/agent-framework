// Copyright (c) Microsoft. All rights reserved.

// Azure OpenAI Responses — Agent using the Responses API
//
// This sample shows how to create an AI agent using the Azure OpenAI
// Responses API.

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Responses;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

AIAgent agent = new AzureOpenAIClient(
    new Uri(endpoint),
    new DefaultAzureCredential())
     .GetResponsesClient()
     .AsAIAgent(model: deploymentName, instructions: "You are good at telling jokes.", name: "Joker");

// Invoke the agent and output the text result.
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));

// Create a responses based agent with "store"=false.
// This means that chat history is managed locally by Agent Framework
// instead of being stored in the service (default).
AIAgent agentStoreFalse = new AzureOpenAIClient(
    new Uri(endpoint),
    new DefaultAzureCredential())
     .GetResponsesClient()
     .AsIChatClientWithStoredOutputDisabled(model: deploymentName)
     .AsAIAgent(instructions: "You are good at telling jokes.", name: "Joker");

// Invoke the agent and output the text result.
Console.WriteLine(await agentStoreFalse.RunAsync("Tell me a joke about a pirate."));
