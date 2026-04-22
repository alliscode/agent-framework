// Copyright (c) Microsoft. All rights reserved.

// OpenAI from Response Client — Create agent from OpenAI Response client
//
// This sample shows how to create an AI agent from an OpenAI
// Response client instance.

using OpenAI;
using OpenAI.Responses;
using OpenAIResponseClientSample;

var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new InvalidOperationException("OPENAI_API_KEY is not set.");
var model = Environment.GetEnvironmentVariable("OPENAI_CHAT_MODEL_NAME") ?? "gpt-5.4-mini";

// Create a ResponsesClient directly from OpenAIClient
ResponsesClient responseClient = new OpenAIClient(apiKey).GetResponsesClient();

// Create an agent directly from the ResponsesClient using OpenAIResponseClientAgent
OpenAIResponseClientAgent agent = new(responseClient, instructions: "You are good at telling jokes.", name: "Joker", model: model);

ResponseItem userMessage = ResponseItem.CreateUserMessageItem("Tell me a joke about a pirate.");

// Invoke the agent and output the text result.
ResponseResult response = await agent.RunAsync([userMessage]);
Console.WriteLine(response.GetOutputText());

// Invoke the agent with streaming support.
IAsyncEnumerable<StreamingResponseUpdate> responseUpdates = agent.RunStreamingAsync([userMessage]);
await foreach (StreamingResponseUpdate responseUpdate in responseUpdates)
{
    if (responseUpdate is StreamingResponseOutputTextDeltaUpdate textUpdate)
    {
        Console.WriteLine(textUpdate.Delta);
    }
}
