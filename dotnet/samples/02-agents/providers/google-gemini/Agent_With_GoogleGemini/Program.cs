// Copyright (c) Microsoft. All rights reserved.

// Google Gemini — Agent using Google Gemini as backend
//
// This sample shows how to create an AI agent using Google Gemini
// as the backend provider.

using Google.GenAI;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Mscc.GenerativeAI.Microsoft;

const string JokerInstructions = "You are good at telling jokes.";
const string JokerName = "JokerAgent";

string apiKey = Environment.GetEnvironmentVariable("GOOGLE_GENAI_API_KEY") ?? throw new InvalidOperationException("Please set the GOOGLE_GENAI_API_KEY environment variable.");
string model = Environment.GetEnvironmentVariable("GOOGLE_GENAI_MODEL") ?? "gemini-2.5-flash";

// Using a Google GenAI IChatClient implementation

ChatClientAgent agentGenAI = new(
    new Client(vertexAI: false, apiKey: apiKey).AsIChatClient(model),
    name: JokerName,
    instructions: JokerInstructions);

AgentResponse response = await agentGenAI.RunAsync("Tell me a joke about a pirate.");
Console.WriteLine($"Google GenAI client based agent response:\n{response}");

// Using a community driven Mscc.GenerativeAI.Microsoft package

ChatClientAgent agentCommunity = new(
    new GeminiChatClient(apiKey: apiKey, model: model),
    name: JokerName,
    instructions: JokerInstructions);

response = await agentCommunity.RunAsync("Tell me a joke about a pirate.");
Console.WriteLine($"Community client based agent response:\n{response}");
