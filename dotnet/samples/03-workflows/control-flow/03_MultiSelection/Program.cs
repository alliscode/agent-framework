// Copyright (c) Microsoft. All rights reserved.

// Multi-Selection Routing — Fan out to multiple branches based on analysis
//
// Extending the switch-case pattern, this workflow can trigger multiple executors
// simultaneously based on data characteristics:
// - Legitimate emails: Email Assistant (always) + Email Summary (if long)
// - Spam emails: Handle Spam executor only
// - Uncertain emails: Handle Uncertain executor only
// - Database logging for both short emails and summarized long emails
//
// This pattern is useful for workflows needing parallel processing based on
// data characteristics, such as triggering different analytics pipelines.
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

namespace WorkflowMultiSelectionSample;

public static class Program
{
    private const int LongEmailThreshold = 100;

    private static async Task Main()
    {
        // Step 1: Set up the Azure OpenAI client
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
        var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";
        var chatClient = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential()).GetChatClient(deploymentName).AsIChatClient();

        // Step 2: Create agents and executors
        AIAgent emailAnalysisAgent = GetEmailAnalysisAgent(chatClient);
        AIAgent emailAssistantAgent = GetEmailAssistantAgent(chatClient);
        AIAgent emailSummaryAgent = GetEmailSummaryAgent(chatClient);

        var emailAnalysisExecutor = new EmailAnalysisExecutor(emailAnalysisAgent);
        var emailAssistantExecutor = new EmailAssistantExecutor(emailAssistantAgent);
        var emailSummaryExecutor = new EmailSummaryExecutor(emailSummaryAgent);
        var sendEmailExecutor = new SendEmailExecutor();
        var handleSpamExecutor = new HandleSpamExecutor();
        var handleUncertainExecutor = new HandleUncertainExecutor();
        var databaseAccessExecutor = new DatabaseAccessExecutor();

        // Step 3: Build the workflow with multi-selection fan-out
        WorkflowBuilder builder = new(emailAnalysisExecutor);
        builder.AddFanOutEdge(
            emailAnalysisExecutor,
            [
                handleSpamExecutor,
                emailAssistantExecutor,
                emailSummaryExecutor,
                handleUncertainExecutor,
            ],
            GetTargetAssigner()
        )
        .AddEdge(emailAssistantExecutor, sendEmailExecutor)
        // Save analysis to database: short emails go directly, long emails go via summary
        .AddEdge<AnalysisResult>(
            emailAnalysisExecutor,
            databaseAccessExecutor,
            condition: analysisResult => analysisResult?.EmailLength <= LongEmailThreshold)
        .AddEdge(emailSummaryExecutor, databaseAccessExecutor)
        .WithOutputFrom(handleUncertainExecutor, handleSpamExecutor, sendEmailExecutor);

        var workflow = builder.Build();

        // Step 4: Read the email input
        string email = Resources.Read("email.txt");

        // Step 5: Execute the workflow
        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, new ChatMessage(ChatRole.User, email));
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            if (evt is WorkflowOutputEvent outputEvent)
            {
                Console.WriteLine($"{outputEvent}");
            }
            else if (evt is DatabaseEvent databaseEvent)
            {
                Console.WriteLine($"{databaseEvent}");
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

    /// <summary>Creates a target assigner for multi-selection routing based on spam decision.</summary>
    private static Func<AnalysisResult?, int, IEnumerable<int>> GetTargetAssigner()
    {
        return (analysisResult, targetCount) =>
        {
            if (analysisResult is not null)
            {
                if (analysisResult.spamDecision == SpamDecision.Spam)
                {
                    return [0]; // Route to spam handler
                }
                else if (analysisResult.spamDecision == SpamDecision.NotSpam)
                {
                    List<int> targets = [1]; // Route to the email assistant

                    if (analysisResult.EmailLength > LongEmailThreshold)
                    {
                        targets.Add(2); // Route to the email summarizer too
                    }

                    return targets;
                }
                else
                {
                    return [3];
                }
            }
            throw new InvalidOperationException("Invalid analysis result.");
        };
    }

    /// <summary>Creates an email analysis agent.</summary>
    private static ChatClientAgent GetEmailAnalysisAgent(IChatClient chatClient) =>
        new(chatClient, new ChatClientAgentOptions()
        {
            ChatOptions = new()
            {
                Instructions = "You are a spam detection assistant that identifies spam emails.",
                ResponseFormat = ChatResponseFormat.ForJsonSchema<AnalysisResult>()
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

    /// <summary>Creates an email summary agent.</summary>
    private static ChatClientAgent GetEmailSummaryAgent(IChatClient chatClient) =>
        new(chatClient, new ChatClientAgentOptions()
        {
            ChatOptions = new()
            {
                Instructions = "You are an assistant that helps users summarize emails.",
                ResponseFormat = ChatResponseFormat.ForJsonSchema<EmailSummary>()
            }
        });
}

internal static class EmailStateConstants
{
    public const string EmailStateScope = "EmailState";
}

// Three-way spam decision enum.
public enum SpamDecision
{
    NotSpam,
    Spam,
    Uncertain
}

// Structured result from the email analysis agent.
public sealed class AnalysisResult
{
    [JsonPropertyName("spam_decision")]
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public SpamDecision spamDecision { get; set; }

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    [JsonIgnore]
    public int EmailLength { get; set; }

    [JsonIgnore]
    public string EmailSummary { get; set; } = string.Empty;

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

// Executor: analyzes emails with the AI agent and stores them in shared state.
internal sealed class EmailAnalysisExecutor : Executor<ChatMessage, AnalysisResult>
{
    private readonly AIAgent _emailAnalysisAgent;

    public EmailAnalysisExecutor(AIAgent emailAnalysisAgent) : base("EmailAnalysisExecutor")
    {
        this._emailAnalysisAgent = emailAnalysisAgent;
    }

    public override async ValueTask<AnalysisResult> HandleAsync(ChatMessage message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        // Generate a random email ID and store the email content
        var newEmail = new Email
        {
            EmailId = Guid.NewGuid().ToString("N"),
            EmailContent = message.Text
        };
        await context.QueueStateUpdateAsync(newEmail.EmailId, newEmail, scopeName: EmailStateConstants.EmailStateScope, cancellationToken);

        // Invoke the agent
        var response = await this._emailAnalysisAgent.RunAsync(message, cancellationToken: cancellationToken);
        var AnalysisResult = JsonSerializer.Deserialize<AnalysisResult>(response.Text);

        AnalysisResult!.EmailId = newEmail.EmailId;
        AnalysisResult!.EmailLength = newEmail.EmailContent.Length;

        return AnalysisResult;
    }
}

// Structured response from the email assistant agent.
public sealed class EmailResponse
{
    [JsonPropertyName("response")]
    public string Response { get; set; } = string.Empty;
}

// Executor: drafts a professional reply for non-spam emails.
internal sealed class EmailAssistantExecutor : Executor<AnalysisResult, EmailResponse>
{
    private readonly AIAgent _emailAssistantAgent;

    public EmailAssistantExecutor(AIAgent emailAssistantAgent) : base("EmailAssistantExecutor")
    {
        this._emailAssistantAgent = emailAssistantAgent;
    }

    public override async ValueTask<EmailResponse> HandleAsync(AnalysisResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (message.spamDecision == SpamDecision.Spam)
        {
            throw new InvalidOperationException("This executor should only handle non-spam messages.");
        }

        // Retrieve the email content from the context
        var email = await context.ReadStateAsync<Email>(message.EmailId, scopeName: EmailStateConstants.EmailStateScope, cancellationToken);

        // Invoke the agent
        var response = await this._emailAssistantAgent.RunAsync(email!.EmailContent, cancellationToken: cancellationToken);
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

// Terminal executor: marks the email as spam.
[YieldsOutput(typeof(string))]
internal sealed class HandleSpamExecutor() : Executor<AnalysisResult>("HandleSpamExecutor")
{
    public override async ValueTask HandleAsync(AnalysisResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (message.spamDecision == SpamDecision.Spam)
        {
            await context.YieldOutputAsync($"Email marked as spam: {message.Reason}", cancellationToken);
        }
        else
        {
            throw new InvalidOperationException("This executor should only handle spam messages.");
        }
    }
}

// Terminal executor: flags uncertain emails for human review.
[YieldsOutput(typeof(string))]
internal sealed class HandleUncertainExecutor() : Executor<AnalysisResult>("HandleUncertainExecutor")
{
    public override async ValueTask HandleAsync(AnalysisResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        if (message.spamDecision == SpamDecision.Uncertain)
        {
            var email = await context.ReadStateAsync<Email>(message.EmailId, scopeName: EmailStateConstants.EmailStateScope, cancellationToken);
            await context.YieldOutputAsync($"Email marked as uncertain: {message.Reason}. Email content: {email?.EmailContent}", cancellationToken);
        }
        else
        {
            throw new InvalidOperationException("This executor should only handle uncertain spam decisions.");
        }
    }
}

// Structured response from the email summary agent.
public sealed class EmailSummary
{
    [JsonPropertyName("summary")]
    public string Summary { get; set; } = string.Empty;
}

// Executor: summarizes long emails using the AI agent.
internal sealed class EmailSummaryExecutor : Executor<AnalysisResult, AnalysisResult>
{
    private readonly AIAgent _emailSummaryAgent;

    public EmailSummaryExecutor(AIAgent emailSummaryAgent) : base("EmailSummaryExecutor")
    {
        this._emailSummaryAgent = emailSummaryAgent;
    }

    public override async ValueTask<AnalysisResult> HandleAsync(AnalysisResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        // Read the email content from the shared states
        var email = await context.ReadStateAsync<Email>(message.EmailId, scopeName: EmailStateConstants.EmailStateScope, cancellationToken);

        // Invoke the agent
        var response = await this._emailSummaryAgent.RunAsync(email!.EmailContent, cancellationToken: cancellationToken);
        var emailSummary = JsonSerializer.Deserialize<EmailSummary>(response.Text);
        message.EmailSummary = emailSummary!.Summary;

        return message;
    }
}

// Custom workflow event for database operations.
internal sealed class DatabaseEvent(string message) : WorkflowEvent(message) { }

// Executor: simulates persisting the analysis result to a database.
internal sealed class DatabaseAccessExecutor() : Executor<AnalysisResult>("DatabaseAccessExecutor")
{
    public override async ValueTask HandleAsync(AnalysisResult message, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        // 1. Save the email content
        await context.ReadStateAsync<Email>(message.EmailId, scopeName: EmailStateConstants.EmailStateScope, cancellationToken);
        await Task.Delay(100, cancellationToken); // Simulate database access delay

        // 2. Save the analysis result
        await Task.Delay(100, cancellationToken); // Simulate database access delay

        // Not using the `WorkflowCompletedEvent` because this is not the end of the workflow.
        // The end of the workflow is signaled by the `SendEmailExecutor` or the `HandleUnknownExecutor`.
        await context.AddEventAsync(new DatabaseEvent($"Email {message.EmailId} saved to database."), cancellationToken);
    }
}
