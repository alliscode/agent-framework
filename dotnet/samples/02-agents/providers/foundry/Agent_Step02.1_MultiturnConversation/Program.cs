// Copyright (c) Microsoft. All rights reserved.

// Multi-Turn Conversation — Preserve context across multiple runs
//
// This sample shows how to create a multi-turn conversation agent using
// sessions. Context is preserved via response ID chaining in the session.

using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;

string endpoint = Environment.GetEnvironmentVariable("AZURE_AI_PROJECT_ENDPOINT") ?? throw new InvalidOperationException("AZURE_AI_PROJECT_ENDPOINT is not set.");
string deploymentName = Environment.GetEnvironmentVariable("AZURE_AI_MODEL_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

AIAgent agent = new AIProjectClient(new Uri(endpoint), new DefaultAzureCredential())
    .AsAIAgent(deploymentName, instructions: "You are good at telling jokes.", name: "JokerAgent");

// Create a session to maintain context across multiple runs.
AgentSession session = await agent.CreateSessionAsync();

// First turn
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate.", session));

// Second turn — the agent remembers the first turn via the session.
Console.WriteLine(await agent.RunAsync("Now add some emojis to the joke and tell it in the voice of a pirate's parrot.", session));
