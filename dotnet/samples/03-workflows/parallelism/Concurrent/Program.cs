// Copyright (c) Microsoft. All rights reserved.

// Concurrent Fan-Out / Fan-In — Multiple agents process the same input in parallel
//
// A dispatcher fans out the same question to a Physicist agent and a Chemist agent.
// Both answer independently and in parallel, then an aggregator fans in their
// responses and combines them into a single consolidated report.
//
// Prerequisites:
// - An Azure OpenAI chat completion deployment must be configured.

using System.Text;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

namespace WorkflowConcurrentSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Set up the Azure AI Project client
        var endpoint = Environment.GetEnvironmentVariable("AZURE_AI_PROJECT_ENDPOINT")
            ?? throw new InvalidOperationException("AZURE_AI_PROJECT_ENDPOINT is not set.");
        var deploymentName = Environment.GetEnvironmentVariable("AZURE_AI_MODEL_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";
        var chatClient = new AIProjectClient(new Uri(endpoint), new AzureCliCredential())
                            .ProjectOpenAIClient.GetChatClient(deploymentName).AsIChatClient();

        // Step 2: Create the expert agents and helper executors
        var physicist = new ChatClientAgent(
            chatClient,
            name: "Physicist",
            instructions: "You are an expert in physics. You answer questions from a physics perspective."
        ).BindAsExecutor(new AIAgentHostOptions { ForwardIncomingMessages = false });

        var chemist = new ChatClientAgent(
            chatClient,
            name: "Chemist",
            instructions: "You are an expert in chemistry. You answer questions from a chemistry perspective."
        ).BindAsExecutor(new AIAgentHostOptions { ForwardIncomingMessages = false });

        var startExecutor = new ConcurrentStartExecutor();
        var aggregationExecutor = new ConcurrentAggregationExecutor();

        // Step 3: Build the workflow with fan-out/fan-in pattern
        var workflow = new WorkflowBuilder(startExecutor)
            .AddFanOutEdge(startExecutor, [physicist, chemist])
            .AddFanInBarrierEdge([physicist, chemist], aggregationExecutor)
            .WithOutputFrom(aggregationExecutor)
            .Build();

        // Step 4: Execute the workflow in streaming mode
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, input: "What is temperature?");
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case WorkflowOutputEvent workflowOutput:
                    Console.WriteLine($"Workflow completed with results:\n{workflowOutput.Data}");
                    break;

                case WorkflowErrorEvent workflowError:
                    WriteError(workflowError.Exception?.ToString() ?? "Unknown workflow error occurred");
                    break;

                case ExecutorFailedEvent executorFailed:
                    WriteError($"Executor '{executorFailed.ExecutorId}' failed with {(
                        executorFailed.Data == null ? "unknown error" : $"exception {executorFailed.Data}"
                        )}.");
                    break;
            }
        }

        void WriteError(string error)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Write(error);
            Console.ResetColor();
        }
    }
}

// Executor: broadcasts the user's question and a turn token to all connected agents.
[SendsMessage(typeof(ChatMessage))]
[SendsMessage(typeof(TurnToken))]
internal sealed partial class ConcurrentStartExecutor() :
    Executor("ConcurrentStartExecutor")
{
    [MessageHandler]
    public async ValueTask HandleAsync(string message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        // Broadcast the message to all connected agents
        await context.SendMessageAsync(new ChatMessage(ChatRole.User, message), cancellationToken: cancellationToken);
        // Broadcast the turn token to kick off the agents
        await context.SendMessageAsync(new TurnToken(emitEvents: false), cancellationToken: cancellationToken);
    }
}

// Executor: collects responses from all agents and yields a combined report.
[YieldsOutput(typeof(string))]
internal sealed partial class ConcurrentAggregationExecutor() :
    Executor<List<ChatMessage>>("ConcurrentAggregationExecutor")
{
    private readonly List<ChatMessage> _messages = [];

    public override async ValueTask HandleAsync(List<ChatMessage> message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        this._messages.AddRange(message);
    }

    protected override ValueTask OnMessageDeliveryFinishedAsync(IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        StringBuilder resultBuilder = new();
        foreach (ChatMessage m in this._messages)
        {
            resultBuilder.AppendLine($"{m.AuthorName}: {m.Text}");
            resultBuilder.AppendLine();
        }

        this._messages.Clear();

        return context.YieldOutputAsync(resultBuilder.ToString(), cancellationToken);
    }
}
