// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Custom strategy for splitting a conversation into query and response halves.
/// </summary>
/// <remarks>
/// Implement this interface to define domain-specific splitting logic, such as
/// splitting before a memory-retrieval tool call to evaluate recall quality.
/// For built-in strategies, use <see cref="ConversationSplit"/> instead.
/// </remarks>
public interface IConversationSplitter
{
    /// <summary>
    /// Splits a conversation into query messages and response messages.
    /// </summary>
    /// <param name="conversation">The full conversation to split.</param>
    /// <returns>A tuple of (query messages, response messages).</returns>
    (IReadOnlyList<ChatMessage> QueryMessages, IReadOnlyList<ChatMessage> ResponseMessages) Split(
        IReadOnlyList<ChatMessage> conversation);
}
