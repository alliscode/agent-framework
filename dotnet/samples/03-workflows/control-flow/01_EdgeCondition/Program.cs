// Copyright (c) Microsoft. All rights reserved.

// Conditional Edge Routing — Route emails based on spam detection
//
// A minimal decision workflow that classifies inbound emails as spam or not spam,
// then routes to the appropriate handler using boolean edge conditions.
//
// Flow:
// 1. Spam Detection Agent reads the email and returns a DetectionResult.
// 2. If not spam → Email Assistant Agent drafts a reply → Send Email Executor.
// 3. If spam → Handle Spam Executor marks it as spam.
//
// Key concepts:
// - Boolean edge conditions that inspect an executor's output for routing.
// - Structured JSON outputs with Pydantic-style models for robust parsing.
// - Shared state to persist email content between executors.
//
// Prerequisites:
// - An Azure OpenAI chat completion deployment that supports structured outputs.

using System.Text.Json;
using System.Text.Json.Serialization;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

namespace WorkflowEdgeConditionSample;

public static class Program
{
    private static async Task Main()
    {
        // Step 1: Set up the Azure OpenAI client
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
        var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";
        var chatClient = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential()).GetChatClient(deploymentName).AsIChatClient();

        // Step 2: Create agents and executors
        AIAgent spamDetectionAgent = GetSpamDetectionAgent(chatClient);
        AIAgent emailAssistantAgent = GetEmailAssistantAgent(chatClient);

        var spamDetectionExecutor = new SpamDetectionExecutor(spamDetectionAgent);
        var emailAssistantExecutor = new EmailAssistantExecutor(emailAssistantAgent);
        var sendEmailExecutor = new SendEmailExecutor();
        var handleSpamExecutor = new HandleSpamExecutor();

        // Step 3: Build the workflow with conditional edges
        var workflow = new WorkflowBuilder(spamDetectionExecutor)
            .AddEdge(spamDetectionExecutor, emailAssistantExecutor, condition: GetCondition(expectedResult: false))
            .AddEdge(emailAssistantExecutor, sendEmailExecutor)
            .AddEdge(spamDetectionExecutor, handleSpamExecutor, condition: GetCondition(expectedResult: true))
            .WithOutputFrom(handleSpamExecutor, sendEmailExecutor)
            .Build();

        // Step 4: Read the email input
        string email = Resources.Read("spam.txt");

        // Step 5: Execute the workflow
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, new ChatMessage(ChatRole.User, email));
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent outputEvent)
            {
                Console.WriteLine($"{outputEvent}");
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

    /// <summary>Creates a routing condition based on the expected spam detection result.</summary>
    private static Func<object?, bool> GetCondition(bool expectedResult) =>
        detectionResult => detectionResult is DetectionResult result && result.IsSpam == expectedResult;

    /// <summary>Creates a spam detection agent.</summary>
    private static ChatClientAgent GetSpamDetectionAgent(IChatClient chatClient) =>
        new(chatClient, new ChatClientAgentOptions()
        {
            ChatOptions = new()
            {
                Instructions = "You are a spam detection assistant that identifies spam emails.",
                ResponseFormat = ChatResponseFormat.ForJsonSchema<DetectionResult>()
            }
        });

    /// <summary>Creates an email assistant agent.</summary>
    private static ChatClientAgent GetEmailAssistantAgent(IChatClient chatClient) =>
        new(chatClient, new ChatClientAgentOptions()
        {
            ChatOptions = new()
            {
                Instructions = "You are an email assistant that helps users draft responses to emails with professionalism.",
                ResponseFormat = ChatResponseFormat.ForJsonSchema<EmailResponse>()
            }
        });
}

// Shared state scope for email data between executors.
internal static class EmailStateConstants
{
    public const string EmailStateScope = "EmailState";
}

// Structured result from the spam detection agent.
public sealed class DetectionResult
{
    [JsonPropertyName("is_spam")]
    public bool IsSpam { get; set; }

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    // Email ID is generated by the executor not the agent
    [JsonIgnore]
    public string EmailId { get; set; } = string.Empty;
}

// In-memory record of the email content stored in shared state.
internal sealed class Email
{
    [JsonPropertyName("email_id")]
    public string EmailId { get; set; } = string.Empty;

    [JsonPropertyName("email_content")]
    public string EmailContent { get; set; } = string.Empty;
}

// Executor: invokes the spam detection agent and stores the email in shared state.
internal sealed class SpamDetectionExecutor : Executor<ChatMessage, DetectionResult>
{
    private readonly AIAgent _spamDetectionAgent;

    public SpamDetectionExecutor(AIAgent spamDetectionAgent) : base("SpamDetectionExecutor")
    {
        this._spamDetectionAgent = spamDetectionAgent;
    }

    public override async ValueTask<DetectionResult> HandleAsync(ChatMessage message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        // Generate a random email ID and store the email content to the shared state
        var newEmail = new Email
        {
            EmailId = Guid.NewGuid().ToString("N"),
            EmailContent = message.Text
        };
        await context.QueueStateUpdateAsync(newEmail.EmailId, newEmail, scopeName: EmailStateConstants.EmailStateScope, cancellationToken);

        // Invoke the agent
        var response = await this._spamDetectionAgent.RunAsync(message, cancellationToken: cancellationToken);
        var detectionResult = JsonSerializer.Deserialize<DetectionResult>(response.Text);

        detectionResult!.EmailId = newEmail.EmailId;

        return detectionResult;
    }
}

// Structured response from the email assistant agent.
public sealed class EmailResponse
{
    [JsonPropertyName("response")]
    public string Response { get; set; } = string.Empty;
}

// Executor: drafts a professional reply for non-spam emails.
internal sealed class EmailAssistantExecutor : Executor<DetectionResult, EmailResponse>
{
    private readonly AIAgent _emailAssistantAgent;

    public EmailAssistantExecutor(AIAgent emailAssistantAgent) : base("EmailAssistantExecutor")
    {
        this._emailAssistantAgent = emailAssistantAgent;
    }

    public override async ValueTask<EmailResponse> HandleAsync(DetectionResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (message.IsSpam)
        {
            throw new InvalidOperationException("This executor should only handle non-spam messages.");
        }

        // Retrieve the email content from the shared state
        var email = await context.ReadStateAsync<Email>(message.EmailId, scopeName: EmailStateConstants.EmailStateScope, cancellationToken)
            ?? throw new InvalidOperationException("Email not found.");

        // Invoke the agent
        var response = await this._emailAssistantAgent.RunAsync(email.EmailContent, cancellationToken: cancellationToken);
        var emailResponse = JsonSerializer.Deserialize<EmailResponse>(response.Text);

        return emailResponse!;
    }
}

// Terminal executor: simulates sending the drafted reply.
[YieldsOutput(typeof(string))]
internal sealed class SendEmailExecutor() : Executor<EmailResponse>("SendEmailExecutor")
{
    public override async ValueTask HandleAsync(EmailResponse message, IWorkflowContext context, CancellationToken cancellationToken = default) =>
        await context.YieldOutputAsync($"Email sent: {message.Response}", cancellationToken);
}

/// <summary>
/// Executor that handles spam messages.
/// </summary>
[YieldsOutput(typeof(string))]
internal sealed class HandleSpamExecutor() : Executor<DetectionResult>("HandleSpamExecutor")
{
    /// <summary>
    /// Simulate the handling of a spam message.
    /// </summary>
    public override async ValueTask HandleAsync(DetectionResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (message.IsSpam)
        {
            await context.YieldOutputAsync($"Email marked as spam: {message.Reason}", cancellationToken);
        }
        else
        {
            throw new InvalidOperationException("This executor should only handle spam messages.");
        }
    }
}
