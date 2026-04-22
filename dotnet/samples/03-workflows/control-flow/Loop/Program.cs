// Copyright (c) Microsoft. All rights reserved.

// Simple Loop — Number guessing game with feedback loop
//
// A number guessing game using a workflow with looping behavior. Two executors
// are connected in a feedback loop:
// 1. GuessNumberExecutor makes a guess (binary search).
// 2. JudgeExecutor evaluates the guess and provides Above/Below/Match feedback.
// The workflow continues until the correct number is guessed.
//
// Prerequisites:
// - No external services required.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowLoopSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Create the executors
        GuessNumberExecutor guessNumberExecutor = new("GuessNumber", 1, 100);
        JudgeExecutor judgeExecutor = new("Judge", 42);

        // Step 2: Build the workflow — connect executors in a feedback loop
        var workflow = new WorkflowBuilder(guessNumberExecutor)
            .AddEdge(guessNumberExecutor, judgeExecutor)
            .AddEdge(judgeExecutor, guessNumberExecutor)
            .WithOutputFrom(judgeExecutor)
            .Build();

        // Step 3: Execute the workflow
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, NumberSignal.Init);
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent outputEvent)
            {
                Console.WriteLine($"Result: {outputEvent}");
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

// Signals used for communication between GuessNumberExecutor and JudgeExecutor.
internal enum NumberSignal
{
    Init,
    Above,
    Below,
}

// Executor: makes a binary-search guess based on the current known bounds.
[SendsMessage(typeof(int))]
internal sealed class GuessNumberExecutor : Executor<NumberSignal>
{
    public int LowerBound { get; private set; }
    public int UpperBound { get; private set; }

    public GuessNumberExecutor(string id, int lowerBound, int upperBound) : base(id)
    {
        this.LowerBound = lowerBound;
        this.UpperBound = upperBound;
    }

    private int NextGuess => (this.LowerBound + this.UpperBound) / 2;

    public override async ValueTask HandleAsync(NumberSignal message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        switch (message)
        {
            case NumberSignal.Init:
                await context.SendMessageAsync(this.NextGuess, cancellationToken: cancellationToken);
                break;
            case NumberSignal.Above:
                this.UpperBound = this.NextGuess - 1;
                await context.SendMessageAsync(this.NextGuess, cancellationToken: cancellationToken);
                break;
            case NumberSignal.Below:
                this.LowerBound = this.NextGuess + 1;
                await context.SendMessageAsync(this.NextGuess, cancellationToken: cancellationToken);
                break;
        }
    }
}

// Executor: judges the guess and provides feedback or yields the final result.
[SendsMessage(typeof(NumberSignal))]
[YieldsOutput(typeof(string))]
internal sealed class JudgeExecutor : Executor<int>
{
    private readonly int _targetNumber;
    private int _tries;

    public JudgeExecutor(string id, int targetNumber) : base(id)
    {
        this._targetNumber = targetNumber;
    }

    public override async ValueTask HandleAsync(int message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        this._tries++;
        if (message == this._targetNumber)
        {
            await context.YieldOutputAsync($"{this._targetNumber} found in {this._tries} tries!", cancellationToken);
        }
        else if (message < this._targetNumber)
        {
            await context.SendMessageAsync(NumberSignal.Below, cancellationToken: cancellationToken);
        }
        else
        {
            await context.SendMessageAsync(NumberSignal.Above, cancellationToken: cancellationToken);
        }
    }
}
