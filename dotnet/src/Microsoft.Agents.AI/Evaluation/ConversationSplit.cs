// Copyright (c) Microsoft. All rights reserved.

namespace Microsoft.Agents.AI;

/// <summary>
/// Built-in strategies for splitting a conversation into query/response halves.
/// </summary>
/// <remarks>
/// Different splits evaluate different aspects of agent behavior:
/// <list type="bullet">
///   <item><see cref="LastTurn"/>: Evaluates whether the agent answered the <em>latest</em> question well.</item>
///   <item><see cref="Full"/>: Evaluates whether the <em>whole conversation trajectory</em> served the original request.</item>
/// </list>
/// For custom splits (e.g. split before a memory-retrieval tool call), implement
/// <see cref="IConversationSplitter"/> instead.
/// </remarks>
public enum ConversationSplit
{
    /// <summary>
    /// Split at the last user message. Everything up to and including that message
    /// is the query; everything after is the response. Default strategy.
    /// </summary>
    LastTurn = 0,

    /// <summary>
    /// The first user message (and any preceding system messages) is the query;
    /// the entire remainder of the conversation is the response.
    /// Evaluates overall conversation trajectory.
    /// </summary>
    Full = 1,
}
