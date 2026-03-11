// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
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
        this.Query = query;
        this.Response = response;
        this.Conversation = conversation;
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

    /// <summary>
    /// Gets or sets the split strategy for this item.
    /// </summary>
    /// <remarks>
    /// When set by orchestration functions (e.g. <c>EvaluateAsync(conversationSplit: ...)</c>),
    /// this takes precedence over evaluator-level defaults.  Priority order:
    /// explicit <see cref="Split(ConversationSplit?)"/> argument &gt;
    /// <see cref="SplitStrategy"/> &gt; <see cref="ConversationSplit.LastTurn"/>.
    /// </remarks>
    public ConversationSplit? SplitStrategy { get; set; }

    /// <summary>
    /// Splits the conversation into query messages and response messages.
    /// </summary>
    /// <param name="split">
    /// The split strategy to use. When <c>null</c>, uses <see cref="SplitStrategy"/>
    /// if set, otherwise <see cref="ConversationSplit.LastTurn"/>.
    /// </param>
    /// <returns>A tuple of (query messages, response messages).</returns>
    public (IReadOnlyList<ChatMessage> QueryMessages, IReadOnlyList<ChatMessage> ResponseMessages) Split(
        ConversationSplit? split = null)
    {
        var effective = split ?? this.SplitStrategy ?? ConversationSplit.LastTurn;

        return effective switch
        {
            ConversationSplit.Full => SplitFull(this.Conversation),
            _ => SplitLastTurn(this.Conversation),
        };
    }

    /// <summary>
    /// Splits the conversation using a custom splitter.
    /// </summary>
    /// <param name="splitter">The custom splitter implementation.</param>
    /// <returns>A tuple of (query messages, response messages).</returns>
    public (IReadOnlyList<ChatMessage> QueryMessages, IReadOnlyList<ChatMessage> ResponseMessages) Split(
        IConversationSplitter splitter)
    {
        if (splitter is null)
        {
            throw new ArgumentNullException(nameof(splitter));
        }

        return splitter.Split(this.Conversation);
    }

    /// <summary>
    /// Splits a conversation at the last user message.
    /// </summary>
    /// <remarks>
    /// This static method is useful as a fallback in custom <see cref="IConversationSplitter"/>
    /// implementations when the custom split pattern is not found.
    /// </remarks>
    /// <param name="conversation">The conversation to split.</param>
    /// <returns>A tuple of (query messages, response messages).</returns>
    public static (IReadOnlyList<ChatMessage> QueryMessages, IReadOnlyList<ChatMessage> ResponseMessages) SplitLastTurn(
        IReadOnlyList<ChatMessage> conversation)
    {
        int lastUserIdx = -1;
        for (int i = 0; i < conversation.Count; i++)
        {
            if (conversation[i].Role == ChatRole.User)
            {
                lastUserIdx = i;
            }
        }

        if (lastUserIdx >= 0)
        {
            return (
                conversation.Take(lastUserIdx + 1).ToList(),
                conversation.Skip(lastUserIdx + 1).ToList());
        }

        return (Array.Empty<ChatMessage>(), conversation.ToList());
    }

    /// <summary>
    /// Splits a multi-turn conversation into one <see cref="EvalItem"/> per user turn.
    /// </summary>
    /// <remarks>
    /// Each user message starts a new turn. The resulting item has cumulative context:
    /// query messages contain the full conversation up to and including that user message,
    /// and the response is everything up to the next user message.
    /// </remarks>
    /// <param name="conversation">The full conversation to split.</param>
    /// <param name="tools">Optional tools available to the agent.</param>
    /// <param name="context">Optional grounding context.</param>
    /// <returns>A list of eval items, one per user turn.</returns>
    public static IReadOnlyList<EvalItem> PerTurnItems(
        IReadOnlyList<ChatMessage> conversation,
        IReadOnlyList<AITool>? tools = null,
        string? context = null)
    {
        var items = new List<EvalItem>();
        var userIndices = new List<int>();

        for (int i = 0; i < conversation.Count; i++)
        {
            if (conversation[i].Role == ChatRole.User)
            {
                userIndices.Add(i);
            }
        }

        for (int t = 0; t < userIndices.Count; t++)
        {
            int userIdx = userIndices[t];
            int nextBoundary = t + 1 < userIndices.Count
                ? userIndices[t + 1]
                : conversation.Count;

            var queryMessages = conversation.Take(userIdx + 1).ToList();
            var responseMessages = conversation.Skip(userIdx + 1).Take(nextBoundary - userIdx - 1).ToList();

            var query = conversation[userIdx].Text ?? string.Empty;
            var responseText = string.Join(
                " ",
                responseMessages
                    .Where(m => m.Role == ChatRole.Assistant && !string.IsNullOrEmpty(m.Text))
                    .Select(m => m.Text));

            var fullSlice = conversation.Take(nextBoundary).ToList();
            var item = new EvalItem(query, responseText, fullSlice)
            {
                Tools = tools,
                Context = context,
            };

            items.Add(item);
        }

        return items;
    }

    private static (IReadOnlyList<ChatMessage>, IReadOnlyList<ChatMessage>) SplitFull(
        IReadOnlyList<ChatMessage> conversation)
    {
        int firstUserIdx = -1;
        for (int i = 0; i < conversation.Count; i++)
        {
            if (conversation[i].Role == ChatRole.User)
            {
                firstUserIdx = i;
                break;
            }
        }

        if (firstUserIdx >= 0)
        {
            return (
                conversation.Take(firstUserIdx + 1).ToList(),
                conversation.Skip(firstUserIdx + 1).ToList());
        }

        return (Array.Empty<ChatMessage>(), conversation.ToList());
    }
}
