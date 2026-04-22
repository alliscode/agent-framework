// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowExecutorsAndEdgesSample;

/// <summary>
/// First Workflow — Chain executors with edges
///
/// This sample builds a minimal workflow with two steps:
/// 1. Convert text to uppercase (lambda-bound executor)
/// 2. Reverse the text (class-based executor)
///
/// No external services are required.
/// For input "Hello, World!", the workflow produces "!DLROW ,OLLEH".
/// </summary>
public static class Program
{
    private static async Task Main()
    {
        // <create_workflow>
        // Step 1: An executor that converts text to uppercase
        Func<string, string> uppercaseFunc = s => s.ToUpperInvariant();
        var uppercase = uppercaseFunc.BindAsExecutor("UppercaseExecutor");

        // Step 2: A class-based executor that reverses the string
        ReverseTextExecutor reverse = new();

        // Build the workflow: uppercase → reverse_text
        WorkflowBuilder builder = new(uppercase);
        builder.AddEdge(uppercase, reverse).WithOutputFrom(reverse);
        var workflow = builder.Build();
        // </create_workflow>

        // <run_workflow>
        // Execute the workflow with input data
        await using Run run = await InProcessExecution.RunAsync(workflow, "Hello, World!");
        foreach (WorkflowEvent evt in run.NewEvents)
        {
            if (evt is ExecutorCompletedEvent executorComplete)
            {
                Console.WriteLine($"{executorComplete.ExecutorId}: {executorComplete.Data}");
            }
        }
        // </run_workflow>
    }
}

/// <summary>
/// Step 2 executor: reverses the input text.
/// </summary>
internal sealed class ReverseTextExecutor() : Executor<string, string>("ReverseTextExecutor")
{
    public override ValueTask<string> HandleAsync(string message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        return ValueTask.FromResult(string.Concat(message.Reverse()));
    }
}
