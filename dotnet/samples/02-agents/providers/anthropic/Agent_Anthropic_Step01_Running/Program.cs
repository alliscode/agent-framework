// Copyright (c) Microsoft. All rights reserved.

// Running with Anthropic — Basic agent using Anthropic Claude
//
// This sample shows how to create and run a simple AI agent using
// Anthropic as the backend provider.

using Anthropic;
using Anthropic.Core;
using Microsoft.Agents.AI;

var apiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY") ?? throw new InvalidOperationException("ANTHROPIC_API_KEY is not set.");
var model = Environment.GetEnvironmentVariable("ANTHROPIC_CHAT_MODEL_NAME") ?? "claude-haiku-4-5";

AIAgent agent =
    new AnthropicClient(new ClientOptions { ApiKey = apiKey })
    .AsAIAgent(model: model, instructions: "You are good at telling jokes.", name: "Joker");

// Invoke the agent and output the text result.
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));
