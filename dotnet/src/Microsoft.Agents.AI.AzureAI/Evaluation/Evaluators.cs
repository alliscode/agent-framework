// Copyright (c) Microsoft. All rights reserved.

namespace Microsoft.Agents.AI.AzureAI;

/// <summary>
/// Constants for Foundry built-in evaluator names.
/// </summary>
/// <remarks>
/// Use these instead of raw strings for IDE autocomplete and typo prevention.
/// </remarks>
public static class Evaluators
{
    // Agent behavior

    /// <summary>Evaluates whether the agent correctly resolves user intent.</summary>
    public const string IntentResolution = "intent_resolution";

    /// <summary>Evaluates whether the agent adheres to its task instructions.</summary>
    public const string TaskAdherence = "task_adherence";

    /// <summary>Evaluates whether the agent completes the requested task.</summary>
    public const string TaskCompletion = "task_completion";

    /// <summary>Evaluates the efficiency of the agent's navigation to complete the task.</summary>
    public const string TaskNavigationEfficiency = "task_navigation_efficiency";

    // Tool usage

    /// <summary>Evaluates the accuracy of tool calls made by the agent.</summary>
    public const string ToolCallAccuracy = "tool_call_accuracy";

    /// <summary>Evaluates whether the agent selects the correct tools.</summary>
    public const string ToolSelection = "tool_selection";

    /// <summary>Evaluates the accuracy of inputs provided to tools.</summary>
    public const string ToolInputAccuracy = "tool_input_accuracy";

    /// <summary>Evaluates how well the agent uses tool outputs.</summary>
    public const string ToolOutputUtilization = "tool_output_utilization";

    /// <summary>Evaluates whether tool calls succeed.</summary>
    public const string ToolCallSuccess = "tool_call_success";

    // Quality

    /// <summary>Evaluates the coherence of the response.</summary>
    public const string Coherence = "coherence";

    /// <summary>Evaluates the fluency of the response.</summary>
    public const string Fluency = "fluency";

    /// <summary>Evaluates the relevance of the response to the query.</summary>
    public const string Relevance = "relevance";

    /// <summary>Evaluates whether the response is grounded in the provided context.</summary>
    public const string Groundedness = "groundedness";

    /// <summary>Evaluates the completeness of the response.</summary>
    public const string ResponseCompleteness = "response_completeness";

    /// <summary>Evaluates the similarity between the response and the expected output.</summary>
    public const string Similarity = "similarity";

    // Safety

    /// <summary>Evaluates the response for violent content.</summary>
    public const string Violence = "violence";

    /// <summary>Evaluates the response for sexual content.</summary>
    public const string Sexual = "sexual";

    /// <summary>Evaluates the response for self-harm content.</summary>
    public const string SelfHarm = "self_harm";

    /// <summary>Evaluates the response for hate or unfairness.</summary>
    public const string HateUnfairness = "hate_unfairness";
}
