// Copyright (c) Microsoft. All rights reserved.

// Hello Agent — Simplest possible agent
//
// This sample creates a minimal agent using Azure OpenAI, and runs it in
// both non-streaming and streaming modes.

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

// <create_agent>
AIAgent agent = new AzureOpenAIClient(
    new Uri(endpoint),
    new DefaultAzureCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(
        instructions: "You are a friendly assistant. Keep your answers brief.",
        name: "HelloAgent");
// </create_agent>

// <run_agent>
// Non-streaming: get the complete response at once
Console.WriteLine(await agent.RunAsync("What is the capital of France?"));
// </run_agent>

// <run_agent_streaming>
// Streaming: receive tokens as they are generated
Console.Write("Agent (streaming): ");
await foreach (var chunk in agent.RunStreamingAsync("Tell me a one-sentence fun fact."))
{
    Console.Write(chunk);
}
Console.WriteLine();
// </run_agent_streaming>
