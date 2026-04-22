// Copyright (c) Microsoft. All rights reserved.

// Human-in-the-Loop Basic — RequestPort for external interaction
//
// This sample introduces RequestPort and ExternalRequest to enable human-in-the-loop
// workflows. A request port acts like an executor: when it receives a message, it
// emits a RequestInfoEvent to the external world, then waits for a response.
//
// The sample implements a number guessing game where the human guesses a target
// number and the workflow provides Above/Below/Match feedback.
//
// Prerequisites:
// - No external services required.

using Microsoft.Agents.AI.Workflows;

namespace WorkflowHumanInTheLoopBasicSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Create the workflow
        var workflow = WorkflowFactory.BuildWorkflow();

        // Step 2: Execute the workflow and handle human interaction
        await using StreamingRun handle = await InProcessExecution.RunStreamingAsync(workflow, NumberSignal.Init);
        await foreach (WorkflowEvent evt in handle.WatchStreamAsync())
        {
            switch (evt)
            {
                case RequestInfoEvent requestInputEvt:
                    // Handle `RequestInfoEvent` from the workflow
                    ExternalResponse response = HandleExternalRequest(requestInputEvt.Request);
                    await handle.SendResponseAsync(response);
                    break;

                case WorkflowOutputEvent outputEvt:
                    // The workflow has yielded output
                    Console.WriteLine($"Workflow completed with result: {outputEvt.Data}");
                    return;

                case WorkflowErrorEvent workflowError:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine(workflowError.Exception?.ToString() ?? "Unknown workflow error occurred.");
                    Console.ResetColor();
                    return;

                case ExecutorFailedEvent executorFailed:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.WriteLine($"Executor '{executorFailed.ExecutorId}' failed with {(executorFailed.Data == null ? "unknown error" : $"exception {executorFailed.Data}")}.");
                    Console.ResetColor();
                    return;
            }
        }
    }

    private static ExternalResponse HandleExternalRequest(ExternalRequest request)
    {
        if (request.TryGetDataAs<NumberSignal>(out var signal))
        {
            switch (signal)
            {
                case NumberSignal.Init:
                    int initialGuess = ReadIntegerFromConsole("Please provide your initial guess: ");
                    return request.CreateResponse(initialGuess);
                case NumberSignal.Above:
                    int lowerGuess = ReadIntegerFromConsole("You previously guessed too large. Please provide a new guess: ");
                    return request.CreateResponse(lowerGuess);
                case NumberSignal.Below:
                    int higherGuess = ReadIntegerFromConsole("You previously guessed too small. Please provide a new guess: ");
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
