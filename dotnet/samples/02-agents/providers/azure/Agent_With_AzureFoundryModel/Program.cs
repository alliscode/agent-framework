// Copyright (c) Microsoft. All rights reserved.

// Azure Foundry Model — Agent using Azure AI Foundry model client
//
// This sample shows how to create an AI agent using the Azure AI
// Foundry model client directly.

using System.ClientModel;
using System.ClientModel.Primitives;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var apiKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");
var model = Environment.GetEnvironmentVariable("AZURE_AI_MODEL_DEPLOYMENT_NAME") ?? "Phi-4-mini-instruct";

// Since we are using the OpenAI Client SDK, we need to override the default endpoint to point to Microsoft Foundry.
var clientOptions = new OpenAIClientOptions() { Endpoint = new Uri(endpoint) };

// Create the OpenAI client with either an API key or Azure CLI credential.
OpenAIClient client = string.IsNullOrWhiteSpace(apiKey)
    ? new OpenAIClient(new BearerTokenPolicy(new DefaultAzureCredential(), "https://ai.azure.com/.default"), clientOptions)
    : new OpenAIClient(new ApiKeyCredential(apiKey), clientOptions);

AIAgent agent = client
    .GetChatClient(model)
    .AsAIAgent(instructions: "You are good at telling jokes.", name: "Joker");

// Invoke the agent and output the text result.
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));
