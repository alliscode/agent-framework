// Copyright (c) Microsoft. All rights reserved.

// Agents in a Workflow — Use AI agents as workflow executors
//
// This sample wires three translation agents (French → Spanish → English) into a
// sequential workflow. Instead of simple text-processing executors, each node is an
// AI-powered agent that translates its input to a target language, showing how agents
// integrate seamlessly into workflow pipelines.
//
// Prerequisites:
// - An Azure OpenAI chat completion deployment must be configured.

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

namespace WorkflowAgentsInWorkflowsSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Set up the Azure OpenAI client
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
        var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";
        var chatClient = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential()).GetChatClient(deploymentName).AsIChatClient();

        // Step 2: Create the translation agents
        AIAgent frenchAgent = GetTranslationAgent("French", chatClient);
        AIAgent spanishAgent = GetTranslationAgent("Spanish", chatClient);
        AIAgent englishAgent = GetTranslationAgent("English", chatClient);

        // Step 3: Build the workflow — sequential translation chain
        var workflow = new WorkflowBuilder(frenchAgent)
            .AddEdge(frenchAgent, spanishAgent)
            .AddEdge(spanishAgent, englishAgent)
            .Build();

        // Step 4: Execute the workflow and stream results
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, new ChatMessage(ChatRole.User, "Hello World!"));

        // Send the turn token to trigger agents. Agents cache incoming messages
        // and only start processing when they receive a TurnToken.
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is AgentResponseUpdateEvent executorComplete)
            {
                Console.WriteLine($"{executorComplete.ExecutorId}: {executorComplete.Data}");
            }
            else if (evt is WorkflowErrorEvent workflowError)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Error.WriteLine(workflowError.Exception?.ToString() ?? "Unknown workflow error occurred.");
                Console.ResetColor();
            }
            else if (evt is ExecutorFailedEvent executorFailed)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Error.WriteLine($"Executor '{executorFailed.ExecutorId}' failed with {(executorFailed.Data == null ? "unknown error" : $"exception {executorFailed.Data}")}.");
                Console.ResetColor();
            }
        }
    }

    /// <summary>Creates a translation agent for the specified target language.</summary>
    private static ChatClientAgent GetTranslationAgent(string targetLanguage, IChatClient chatClient) =>
        new(chatClient, $"You are a translation assistant that translates the provided text to {targetLanguage}.");
}
