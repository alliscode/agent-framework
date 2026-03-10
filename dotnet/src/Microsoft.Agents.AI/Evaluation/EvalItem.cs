// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Provider-agnostic data for a single evaluation item.
/// </summary>
public sealed class EvalItem
{
    /// <summary>
    /// Initializes a new instance of the <see cref="EvalItem"/> class.
    /// </summary>
    /// <param name="query">The user query.</param>
    /// <param name="response">The agent response text.</param>
    /// <param name="conversation">The full conversation as <see cref="ChatMessage"/> list.</param>
    public EvalItem(string query, string response, IReadOnlyList<ChatMessage> conversation)
    {
        Query = query;
        Response = response;
        Conversation = conversation;
    }

    /// <summary>Gets the user query.</summary>
    public string Query { get; }

    /// <summary>Gets the agent response text.</summary>
    public string Response { get; }

    /// <summary>Gets the full conversation history.</summary>
    public IReadOnlyList<ChatMessage> Conversation { get; }

    /// <summary>Gets or sets the tools available to the agent.</summary>
    public IReadOnlyList<AITool>? Tools { get; set; }

    /// <summary>Gets or sets grounding context for evaluation.</summary>
    public string? Context { get; set; }

    /// <summary>Gets or sets the expected output for ground-truth comparison.</summary>
    public string? Expected { get; set; }

    /// <summary>Gets or sets the raw chat response for MEAI evaluators.</summary>
    public ChatResponse? RawResponse { get; set; }
}
