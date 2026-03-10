// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Built-in check functions for common evaluation patterns.
/// </summary>
public static class EvalChecks
{
    /// <summary>
    /// Creates a check that verifies the response contains all specified keywords.
    /// </summary>
    /// <param name="keywords">Keywords that must appear in the response.</param>
    /// <returns>An <see cref="EvalCheck"/> delegate.</returns>
    public static EvalCheck KeywordCheck(params string[] keywords)
    {
        return KeywordCheck(caseSensitive: false, keywords);
    }

    /// <summary>
    /// Creates a check that verifies the response contains all specified keywords.
    /// </summary>
    /// <param name="caseSensitive">Whether the comparison is case-sensitive.</param>
    /// <param name="keywords">Keywords that must appear in the response.</param>
    /// <returns>An <see cref="EvalCheck"/> delegate.</returns>
    public static EvalCheck KeywordCheck(bool caseSensitive, params string[] keywords)
    {
        return (EvalItem item) =>
        {
            var comparison = caseSensitive
                ? StringComparison.Ordinal
                : StringComparison.OrdinalIgnoreCase;

            var missing = keywords
                .Where(kw => !item.Response.Contains(kw, comparison))
                .ToList();

            var passed = missing.Count == 0;
            var reason = passed
                ? $"All keywords found: {string.Join(", ", keywords)}"
                : $"Missing keywords: {string.Join(", ", missing)}";

            return new CheckResult(passed, reason, "keyword_check");
        };
    }

    /// <summary>
    /// Creates a check that verifies specific tools were called in the conversation.
    /// </summary>
    /// <param name="toolNames">Tool names that must appear in the conversation.</param>
    /// <returns>An <see cref="EvalCheck"/> delegate.</returns>
    public static EvalCheck ToolCalledCheck(params string[] toolNames)
    {
        return (EvalItem item) =>
        {
            var calledTools = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var message in item.Conversation)
            {
                foreach (var content in message.Contents)
                {
                    if (content is FunctionCallContent functionCall)
                    {
                        calledTools.Add(functionCall.Name);
                    }
                }
            }

            var missing = toolNames
                .Where(t => !calledTools.Contains(t))
                .ToList();

            var passed = missing.Count == 0;
            var reason = passed
                ? $"All tools called: {string.Join(", ", toolNames)}"
                : $"Missing tool calls: {string.Join(", ", missing)}";

            return new CheckResult(passed, reason, "tool_called_check");
        };
    }
}
