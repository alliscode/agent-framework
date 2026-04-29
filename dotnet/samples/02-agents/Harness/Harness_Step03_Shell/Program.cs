// Copyright (c) Microsoft. All rights reserved.

// This sample demonstrates how to use the new Microsoft.Agents.AI.Tools.Shell
// LocalShellTool inside the Harness stack: a ChatClientAgent equipped with
// TodoProvider, AgentModeProvider, FileMemoryProvider, and the harness
// ToolApprovalAgent. The shell tool is the first non-network capability the
// agent has, so the harness's approval flow becomes the security boundary.
//
// Special commands:
//   /todos  — Display the current todo list without invoking the agent.
//   exit    — End the session.

#pragma warning disable OPENAI001 // Suppress experimental API warnings for Responses API usage.
#pragma warning disable MAAI001  // Suppress experimental API warnings for Agents AI experiments.

using System.ClientModel.Primitives;
using Azure.Identity;
using Harness.Shared.Console;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Compaction;
using Microsoft.Agents.AI.Tools.Shell;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;

var endpoint = Environment.GetEnvironmentVariable("AZURE_FOUNDRY_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_FOUNDRY_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_AI_MODEL_DEPLOYMENT_NAME") ?? "gpt-5.4";

const int MaxContextWindowTokens = 1_050_000;
const int MaxOutputTokens = 128_000;

// Build the shell tool. Approval-in-the-loop is the security boundary, so
// every call will surface as a ToolApprovalRequest in the harness console.
var shell = new LocalShellTool(
    timeout: TimeSpan.FromSeconds(30),
    maxOutputBytes: 32 * 1024,
    onCommand: cmd => Console.Error.WriteLine($"[shell] running: {cmd}"));

var instructions =
    """
    You are an assistant with access to a local shell on the user's machine.
    The user reviews and approves every shell command you propose.

    ## Mandatory planning workflow

    For every new substantive user request, your behavior is determined by the mode you are in.
    If you are in plan mode, follow *Plan Mode*. If you are in execute mode, follow *Execute Mode*.

    *Plan Mode*

    1. Analyze the request.
    2. Ask clarifying questions where needed; offer numbered options when you have specific choices in mind.
    3. Create one or more todo items.
    4. Write the plan to a file using the FileMemory_* tools so it survives compaction.
    5. Present the plan to the user.
    6. Ask for approval to switch to execute mode.
    7. When approval is granted, switch to execute mode (using the `AgentMode_Set` tool).

    *Execute Mode*

    1. If you don't have a plan yet, create one. (Skip if you came from plan mode.)
    2. Work autonomously — propose shell commands, observe their output, and continue.
    3. Mark tasks completed as you finish them.
    4. When proposing shell commands, prefer:
       - non-destructive read-only commands first (ls, cat, grep, git status, git log).
       - explicit absolute paths over wildcards.
       - one command per call so each is reviewable.
    5. Never propose `rm -rf`, `dd`, or other destructive commands without explicit user instruction.

    ## Shell-tool conventions

    - The tool runs on the user's actual machine. Treat any output you read as authoritative.
    - You do **not** have a persistent shell — `cd` does not carry across calls. Use absolute paths.
    - Stdout, stderr, and exit code are returned together. Always check the exit code before claiming success.
    - If a command times out the result includes `[command timed out]`.

    ## File memory

    Use the FileMemory_* tools to store plans, intermediate findings, and final reports so they survive
    compaction.
    """;

var compactionStrategy = new ContextWindowCompactionStrategy(
    maxContextWindowTokens: MaxContextWindowTokens,
    maxOutputTokens: MaxOutputTokens);

AIAgent agent =
    new OpenAIClient(
        new BearerTokenPolicy(new DefaultAzureCredential(), "https://ai.azure.com/.default"),
        new OpenAIClientOptions()
        {
            Endpoint = new Uri(endpoint),
            RetryPolicy = new ClientRetryPolicy(3),
        })
    .GetResponsesClient()
    .AsIChatClientWithStoredOutputDisabled(deploymentName)
    .AsBuilder()
    .UseFunctionInvocation()
    .UsePerServiceCallChatHistoryPersistence()
    .UseAIContextProviders(new CompactionProvider(compactionStrategy))
    .BuildAIAgent(
        new ChatClientAgentOptions
        {
            Name = "ShellAgent",
            Description = "An assistant that can run local shell commands with explicit user approval.",
            UseProvidedChatClientAsIs = true,
            RequirePerServiceCallChatHistoryPersistence = true,
            ChatHistoryProvider = new InMemoryChatHistoryProvider(
                new InMemoryChatHistoryProviderOptions
                {
                    ChatReducer = compactionStrategy.AsChatReducer(),
                }),
            AIContextProviders =
            [
                new TodoProvider(),
                new AgentModeProvider(),
                new FileMemoryProvider(
                    new FileSystemAgentFileStore(Path.Combine(AppContext.BaseDirectory, "agent-files")),
                    (_) => new FileMemoryState() { WorkingFolder = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss") + "_" + Guid.NewGuid().ToString() }),
            ],
            ChatOptions = new ChatOptions
            {
                Instructions = instructions,
                Tools =
                [
                    shell.AsAIFunction(name: "run_shell"),
                ],
                MaxOutputTokens = MaxOutputTokens,
            },
        })
    .AsBuilder()
    .UseToolApproval()
    .Build();

await HarnessConsole.RunAgentAsync(
    agent,
    title: "Shell Assistant",
    userPrompt: "Tell me what you'd like to do on your machine. Every shell command will be presented for your approval.",
    maxContextWindowTokens: MaxContextWindowTokens,
    maxOutputTokens: MaxOutputTokens);
