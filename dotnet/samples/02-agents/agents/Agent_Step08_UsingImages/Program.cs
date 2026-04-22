// Copyright (c) Microsoft. All rights reserved.

// Using Images — Multimodal input with an AI agent
//
// This sample shows how to send image content to an AI agent
// for vision-based analysis.

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Extensions.AI;
using OpenAI.Chat;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = System.Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

var agent = new AzureOpenAIClient(new Uri(endpoint), new DefaultAzureCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(
        name: "VisionAgent",
        instructions: "You are a helpful agent that can analyze images");

ChatMessage message = new(ChatRole.User, [
    new TextContent("What do you see in this image?"),
    await DataContent.LoadFromAsync("Assets/walkway.jpg"),
]);

var session = await agent.CreateSessionAsync();

await foreach (var update in agent.RunStreamingAsync(message, session))
{
    Console.WriteLine(update);
}
