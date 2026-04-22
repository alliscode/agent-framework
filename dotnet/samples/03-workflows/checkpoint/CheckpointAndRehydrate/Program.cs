// Copyright (c) Microsoft. All rights reserved.

// Checkpoint and Rehydrate — Resume a workflow on a fresh instance
//
// Similar to CheckpointAndResume, but instead of restoring the same run,
// this sample rehydrates a completely new workflow instance from a saved
// checkpoint. This simulates recovery across process restarts.
//
// Key concepts:
// - Super Steps: Workflow executes in stages; checkpoints are saved after each.
// - Rehydration: Create a new workflow and resume from a saved checkpoint.
//
// Prerequisites:
// - No external services required.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowCheckpointAndRehydrateSample;

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
            .RunStreamingAsync(workflow, NumberSignal.Init, checkpointManager);

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

        // Step 3: Rehydrate a new workflow instance from a saved checkpoint
        var newWorkflow = WorkflowFactory.BuildWorkflow();
        const int CheckpointIndex = 5;
        Console.WriteLine($"\n\nHydrating a new workflow instance from the {CheckpointIndex + 1}th checkpoint.");
        CheckpointInfo savedCheckpoint = checkpoints[CheckpointIndex];

        await using StreamingRun newCheckpointedRun =
            await InProcessExecution.ResumeStreamingAsync(newWorkflow, savedCheckpoint, checkpointManager);

        await foreach (WorkflowEvent evt in newCheckpointedRun.WatchStreamAsync())
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
