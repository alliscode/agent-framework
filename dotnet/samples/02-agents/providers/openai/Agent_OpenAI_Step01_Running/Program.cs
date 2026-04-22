// Copyright (c) Microsoft. All rights reserved.

// Running with OpenAI — Basic agent using OpenAI
//
// This sample shows how to create and run a simple AI agent using
// OpenAI as the backend provider.

using System.ClientModel;
using Microsoft.Agents.AI;
using OpenAI.Responses;

var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new InvalidOperationException("OPENAI_API_KEY is not set.");
var model = Environment.GetEnvironmentVariable("OPENAI_CHAT_MODEL_NAME") ?? "gpt-5.4-mini";

AIAgent agent =
    new ResponsesClient(new ApiKeyCredential(apiKey))
    .AsAIAgent(model: model, instructions: "You are good at telling jokes.", name: "Joker");

// Once you have the agent, you can invoke it like any other AIAgent.
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));
