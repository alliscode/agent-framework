// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Microsoft.Agents.AI;

/// <summary>
/// Extension methods for evaluating agents, responses, and workflow runs.
/// </summary>
public static partial class AgentEvaluationExtensions
{
    /// <summary>
    /// Evaluates an agent by running it against test queries and scoring the responses.
    /// </summary>
    /// <param name="agent">The agent to evaluate.</param>
    /// <param name="queries">Test queries to send to the agent.</param>
    /// <param name="evaluator">The evaluator to score responses.</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation results.</returns>
    public static async Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IAgentEvaluator evaluator,
        string evalName = "Agent Framework Eval",
        CancellationToken cancellationToken = default)
    {
        var items = new List<EvalItem>();

        foreach (var query in queries)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var messages = new List<ChatMessage>
            {
                new(ChatRole.User, query),
            };

            var response = await agent.RunAsync(messages, cancellationToken: cancellationToken).ConfigureAwait(false);
            var item = BuildEvalItem(query, response, messages, agent);
            items.Add(item);
        }

        return await evaluator.EvaluateAsync(items, evalName, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Evaluates an agent by running it against test queries with multiple evaluators.
    /// </summary>
    /// <param name="agent">The agent to evaluate.</param>
    /// <param name="queries">Test queries to send to the agent.</param>
    /// <param name="evaluators">The evaluators to score responses.</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>One result per evaluator.</returns>
    public static async Task<IReadOnlyList<AgentEvaluationResults>> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IEnumerable<IAgentEvaluator> evaluators,
        string evalName = "Agent Framework Eval",
        CancellationToken cancellationToken = default)
    {
        // Run agent once, evaluate with all evaluators
        var items = new List<EvalItem>();

        foreach (var query in queries)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var messages = new List<ChatMessage>
            {
                new(ChatRole.User, query),
            };

            var response = await agent.RunAsync(messages, cancellationToken: cancellationToken).ConfigureAwait(false);
            var item = BuildEvalItem(query, response, messages, agent);
            items.Add(item);
        }

        var results = new List<AgentEvaluationResults>();
        foreach (var evaluator in evaluators)
        {
            var result = await evaluator.EvaluateAsync(items, evalName, cancellationToken).ConfigureAwait(false);
            results.Add(result);
        }

        return results;
    }

    /// <summary>
    /// Evaluates pre-existing agent responses without re-running the agent.
    /// </summary>
    /// <param name="agent">The agent (used for tool definitions).</param>
    /// <param name="responses">Pre-existing query/response pairs.</param>
    /// <param name="evaluator">The evaluator to score responses.</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation results.</returns>
    public static async Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<(string Query, AgentResponse Response)> responses,
        IAgentEvaluator evaluator,
        string evalName = "Agent Framework Eval",
        CancellationToken cancellationToken = default)
    {
        var items = new List<EvalItem>();

        foreach (var (query, response) in responses)
        {
            var messages = new List<ChatMessage>
            {
                new(ChatRole.User, query),
            };
            messages.AddRange(response.Messages);

            var item = BuildEvalItem(query, response, messages, agent);
            items.Add(item);
        }

        return await evaluator.EvaluateAsync(items, evalName, cancellationToken).ConfigureAwait(false);
    }

    internal static EvalItem BuildEvalItem(
        string query,
        AgentResponse response,
        List<ChatMessage> messages,
        AIAgent agent)
    {
        // Add response messages to conversation
        foreach (var msg in response.Messages)
        {
            if (!messages.Contains(msg))
            {
                messages.Add(msg);
            }
        }

        return new EvalItem(query, response.Text, messages)
        {
            RawResponse = new ChatResponse(response.Messages.LastOrDefault()
                ?? new ChatMessage(ChatRole.Assistant, response.Text)),
        };
    }
}
