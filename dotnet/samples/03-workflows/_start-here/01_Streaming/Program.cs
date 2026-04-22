// Copyright (c) Microsoft. All rights reserved.

// Streaming Workflow — Observe executor results in real time
//
// This sample streams events back in real time as each executor finishes.
// The workflow logic is identical to a non-streaming run (uppercase → reverse),
// but here we watch intermediate results as they happen rather than waiting for
// the entire workflow to complete.
//
// Prerequisites:
// - No external services required.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowStreamingSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Create the executors
        UppercaseExecutor uppercase = new();
        ReverseTextExecutor reverse = new();

        // Step 2: Build the workflow by connecting executors sequentially
        WorkflowBuilder builder = new(uppercase);
        builder.AddEdge(uppercase, reverse).WithOutputFrom(reverse);
        var workflow = builder.Build();

        // Step 3: Execute the workflow in streaming mode
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, input: "Hello, World!");
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is ExecutorCompletedEvent executorCompleted)
            {
                Console.WriteLine($"{executorCompleted.ExecutorId}: {executorCompleted.Data}");
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
}

// First executor: converts input text to uppercase.
// The return value is forwarded as a message along edges to subsequent executors.
internal sealed class UppercaseExecutor() : Executor<string, string>("UppercaseExecutor")
{
    public override ValueTask<string> HandleAsync(string message, IWorkflowContext context, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(message.ToUpperInvariant());
}

// Second executor: reverses the input text.
// Because output is not suppressed, the result is yielded as workflow output.
internal sealed class ReverseTextExecutor() : Executor<string, string>("ReverseTextExecutor")
{
    public override ValueTask<string> HandleAsync(string message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        return ValueTask.FromResult(string.Concat(message.Reverse()));
    }
}
