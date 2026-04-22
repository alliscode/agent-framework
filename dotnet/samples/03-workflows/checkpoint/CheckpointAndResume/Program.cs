// Copyright (c) Microsoft. All rights reserved.

// Checkpoint and Resume — Save and restore workflow state
//
// This sample shows how to checkpoint a long-running workflow and resume from
// a saved point. Key concepts:
// - Super Steps: A workflow executes in stages. Each super step runs one or more
//   executors and completes when all finish.
// - Checkpoints: Automatically saved at the end of each super step when a
//   checkpoint manager is provided.
// - Resume: Restore a checkpoint and continue execution from that saved state.
//
// Prerequisites:
// - No external services required.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowCheckpointAndResumeSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Create the workflow and checkpoint manager
        var workflow = WorkflowFactory.BuildWorkflow();
        var checkpointManager = CheckpointManager.Default;
        var checkpoints = new List<CheckpointInfo>();

        // Step 2: Execute the workflow and collect checkpoints
        await using StreamingRun checkpointedRun = await InProcessExecution.RunStreamingAsync(workflow, NumberSignal.Init, checkpointManager);
        await foreach (WorkflowEvent evt in checkpointedRun.WatchStreamAsync())
        {
            switch (evt)
            {
                case ExecutorCompletedEvent executorCompletedEvt:
                    Console.WriteLine($"* Executor {executorCompletedEvt.ExecutorId} completed.");
                    break;

                case SuperStepCompletedEvent superStepCompletedEvt:
                {
                    // Checkpoints are automatically created at the end of each super step when a
                    // checkpoint manager is provided. You can store the checkpoint info for later use.
                    CheckpointInfo? checkpoint = superStepCompletedEvt.CompletionInfo!.Checkpoint;
                    if (checkpoint is not null)
                    {
                        checkpoints.Add(checkpoint);
                        Console.WriteLine($"** Checkpoint created at step {checkpoints.Count}.");
                    }

                    break;
                }

                case WorkflowOutputEvent outputEvent:
                    Console.WriteLine($"Workflow completed with result: {outputEvent.Data}");
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
        const int CheckpointIndex = 5;
        Console.WriteLine($"\n\nRestoring from the {CheckpointIndex + 1}th checkpoint.");
        CheckpointInfo savedCheckpoint = checkpoints[CheckpointIndex];
        // Note that we are restoring the state directly to the same run instance.
        await checkpointedRun.RestoreCheckpointAsync(savedCheckpoint, CancellationToken.None);
        await foreach (WorkflowEvent evt in checkpointedRun.WatchStreamAsync())
        {
            switch (evt)
            {
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
}
