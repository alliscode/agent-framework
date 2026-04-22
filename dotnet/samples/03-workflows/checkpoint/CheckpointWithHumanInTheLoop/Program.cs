// Copyright (c) Microsoft. All rights reserved.

// Checkpoint + Human-in-the-Loop — Pause, checkpoint, and resume with human input
//
// A number guessing game where the human provides guesses based on feedback.
// Workflow state is checkpointed at the end of each super step, allowing it
// to be restored and resumed later. Each RequestPort request/response cycle
// takes two super steps (request out, response in), creating two checkpoints
// per human interaction.
//
// Prerequisites:
// - Familiarity with HumanInTheLoopBasic and CheckpointAndResume samples.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowCheckpointWithHumanInTheLoopSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Create the workflow and checkpoint manager
        var workflow = WorkflowFactory.BuildWorkflow();
        var checkpointManager = CheckpointManager.Default;
        var checkpoints = new List<CheckpointInfo>();

        // Step 2: Execute the workflow and collect checkpoints
        await using StreamingRun checkpointedRun = await InProcessExecution
            .RunStreamingAsync(workflow, new SignalWithNumber(NumberSignal.Init), checkpointManager)
            ;
        await foreach (WorkflowEvent evt in checkpointedRun.WatchStreamAsync())
        {
            switch (evt)
            {
                case RequestInfoEvent requestInputEvt:
                    // Handle `RequestInfoEvent` from the workflow
                    ExternalResponse response = HandleExternalRequest(requestInputEvt.Request);
                    await checkpointedRun.SendResponseAsync(response);
                    break;
                case ExecutorCompletedEvent executorCompletedEvt:
                    Console.WriteLine($"* Executor {executorCompletedEvt.ExecutorId} completed.");
                    break;
                case SuperStepCompletedEvent superStepCompletedEvt:
                    // Checkpoints are automatically created at the end of each super step when a
                    // checkpoint manager is provided. You can store the checkpoint info for later use.
                    CheckpointInfo? checkpoint = superStepCompletedEvt.CompletionInfo!.Checkpoint;
                    if (checkpoint is not null)
                    {
                        checkpoints.Add(checkpoint);
                        Console.WriteLine($"** Checkpoint created at step {checkpoints.Count}.");
                    }
                    break;
                case WorkflowOutputEvent workflowOutputEvt:
                    Console.WriteLine($"Workflow completed with result: {workflowOutputEvt.Data}");
                    break;
                case WorkflowErrorEvent workflowError:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine(workflowError.Exception?.ToString() ?? "Unknown workflow error occurred.");
                    Console.ResetColor();
                    break;
                case ExecutorFailedEvent executorFailed:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Executor '{executorFailed.ExecutorId}' failed with {(executorFailed.Data == null ? "unknown error" : $"exception {executorFailed.Data}")}.");
                    Console.ResetColor();
                    break;
            }
        }

        if (checkpoints.Count == 0)
        {
            throw new InvalidOperationException("No checkpoints were created during the workflow execution.");
        }
        Console.WriteLine($"Number of checkpoints created: {checkpoints.Count}");

        // Step 3: Restore from a checkpoint and resume execution
        const int CheckpointIndex = 1;
        Console.WriteLine($"\n\nRestoring from the {CheckpointIndex + 1}th checkpoint.");
        CheckpointInfo savedCheckpoint = checkpoints[CheckpointIndex];
        // Note that we are restoring the state directly to the same run instance.
        await checkpointedRun.RestoreCheckpointAsync(savedCheckpoint, CancellationToken.None);
        await foreach (WorkflowEvent evt in checkpointedRun.WatchStreamAsync())
        {
            switch (evt)
            {
                case RequestInfoEvent requestInputEvt:
                    // Handle `RequestInfoEvent` from the workflow
                    ExternalResponse response = HandleExternalRequest(requestInputEvt.Request);
                    await checkpointedRun.SendResponseAsync(response);
                    break;
                case ExecutorCompletedEvent executorCompletedEvt:
                    Console.WriteLine($"* Executor {executorCompletedEvt.ExecutorId} completed.");
                    break;
                case WorkflowOutputEvent workflowOutputEvt:
                    Console.WriteLine($"Workflow completed with result: {workflowOutputEvt.Data}");
                    break;
                case WorkflowErrorEvent workflowError:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine(workflowError.Exception?.ToString() ?? "Unknown workflow error occurred.");
                    Console.ResetColor();
                    break;
                case ExecutorFailedEvent executorFailed:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Executor '{executorFailed.ExecutorId}' failed with {(executorFailed.Data == null ? "unknown error" : $"exception {executorFailed.Data}")}.");
                    Console.ResetColor();
                    break;
            }
        }
    }

    private static ExternalResponse HandleExternalRequest(ExternalRequest request)
    {
        if (request.TryGetDataAs<SignalWithNumber>(out var signal))
        {
            switch (signal.Signal)
            {
                case NumberSignal.Init:
                    int initialGuess = ReadIntegerFromConsole("Please provide your initial guess: ");
                    return request.CreateResponse(initialGuess);
                case NumberSignal.Above:
                    int lowerGuess = ReadIntegerFromConsole($"You previously guessed {signal.Number} too large. Please provide a new guess: ");
                    return request.CreateResponse(lowerGuess);
                case NumberSignal.Below:
                    int higherGuess = ReadIntegerFromConsole($"You previously guessed {signal.Number} too small. Please provide a new guess: ");
                    return request.CreateResponse(higherGuess);
            }
        }

        throw new NotSupportedException($"Request {request.PortInfo.RequestType} is not supported");
    }

    private static int ReadIntegerFromConsole(string prompt)
    {
        while (true)
        {
            Console.Write(prompt);
            string? input = Console.ReadLine();
            if (int.TryParse(input, out int value))
            {
                return value;
            }
            Console.WriteLine("Invalid input. Please enter a valid integer.");
        }
    }
}
