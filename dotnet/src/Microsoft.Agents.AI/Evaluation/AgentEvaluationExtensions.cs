// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;

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
    /// <param name="splitter">
    /// Optional conversation splitter to apply to all items.
    /// Use <see cref="ConversationSplitters.LastTurn"/>, <see cref="ConversationSplitters.Full"/>,
    /// or a custom <see cref="IConversationSplitter"/> implementation.
    /// </param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation results.</returns>
    public static async Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IAgentEvaluator evaluator,
        string evalName = "Agent Framework Eval",
        IConversationSplitter? splitter = null,
        CancellationToken cancellationToken = default)
    {
        var items = await RunAgentForEvalAsync(agent, queries, splitter, cancellationToken).ConfigureAwait(false);
        return await evaluator.EvaluateAsync(items, evalName, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Evaluates an agent using an MEAI evaluator directly.
    /// </summary>
    /// <param name="agent">The agent to evaluate.</param>
    /// <param name="queries">Test queries to send to the agent.</param>
    /// <param name="evaluator">The MEAI evaluator (e.g., <c>RelevanceEvaluator</c>, <c>CompositeEvaluator</c>).</param>
    /// <param name="chatConfiguration">Chat configuration for the MEAI evaluator (includes the judge model).</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="splitter">
    /// Optional conversation splitter to apply to all items.
    /// Use <see cref="ConversationSplitters.LastTurn"/>, <see cref="ConversationSplitters.Full"/>,
    /// or a custom <see cref="IConversationSplitter"/> implementation.
    /// </param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation results.</returns>
    public static async Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IEvaluator evaluator,
        ChatConfiguration chatConfiguration,
        string evalName = "Agent Framework Eval",
        IConversationSplitter? splitter = null,
        CancellationToken cancellationToken = default)
    {
        var wrapped = new MeaiEvaluatorAdapter(evaluator, chatConfiguration);
        return await agent.EvaluateAsync(queries, wrapped, evalName, splitter, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Evaluates an agent by running it against test queries with multiple evaluators.
    /// </summary>
    /// <param name="agent">The agent to evaluate.</param>
    /// <param name="queries">Test queries to send to the agent.</param>
    /// <param name="evaluators">The evaluators to score responses.</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="splitter">
    /// Optional conversation splitter to apply to all items.
    /// Use <see cref="ConversationSplitters.LastTurn"/>, <see cref="ConversationSplitters.Full"/>,
    /// or a custom <see cref="IConversationSplitter"/> implementation.
    /// </param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>One result per evaluator.</returns>
    public static async Task<IReadOnlyList<AgentEvaluationResults>> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<string> queries,
        IEnumerable<IAgentEvaluator> evaluators,
        string evalName = "Agent Framework Eval",
        IConversationSplitter? splitter = null,
        CancellationToken cancellationToken = default)
    {
        var items = await RunAgentForEvalAsync(agent, queries, splitter, cancellationToken).ConfigureAwait(false);

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
    /// <param name="responses">Pre-existing agent responses.</param>
    /// <param name="queries">The queries that produced each response (must match count).</param>
    /// <param name="evaluator">The evaluator to score responses.</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation results.</returns>
    public static async Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<AgentResponse> responses,
        IEnumerable<string> queries,
        IAgentEvaluator evaluator,
        string evalName = "Agent Framework Eval",
        CancellationToken cancellationToken = default)
    {
        var items = BuildItemsFromResponses(agent, responses, queries);
        return await evaluator.EvaluateAsync(items, evalName, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Evaluates pre-existing agent responses using an MEAI evaluator directly.
    /// </summary>
    /// <param name="agent">The agent (used for tool definitions).</param>
    /// <param name="responses">Pre-existing agent responses.</param>
    /// <param name="queries">The queries that produced each response (must match count).</param>
    /// <param name="evaluator">The MEAI evaluator.</param>
    /// <param name="chatConfiguration">Chat configuration for the MEAI evaluator.</param>
    /// <param name="evalName">Display name for this evaluation run.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Evaluation results.</returns>
    public static async Task<AgentEvaluationResults> EvaluateAsync(
        this AIAgent agent,
        IEnumerable<AgentResponse> responses,
        IEnumerable<string> queries,
        IEvaluator evaluator,
        ChatConfiguration chatConfiguration,
        string evalName = "Agent Framework Eval",
        CancellationToken cancellationToken = default)
    {
        var wrapped = new MeaiEvaluatorAdapter(evaluator, chatConfiguration);
        return await agent.EvaluateAsync(responses, queries, wrapped, evalName, cancellationToken).ConfigureAwait(false);
    }

    private static List<EvalItem> BuildItemsFromResponses(
        AIAgent agent,
        IEnumerable<AgentResponse> responses,
        IEnumerable<string> queries)
    {
        var responseList = responses.ToList();
        var queryList = queries.ToList();

        if (responseList.Count != queryList.Count)
        {
            throw new ArgumentException(
                $"Got {queryList.Count} queries but {responseList.Count} responses. Counts must match.");
        }

        var items = new List<EvalItem>();
        for (int i = 0; i < responseList.Count; i++)
        {
            var query = queryList[i];
            var response = responseList[i];

            var messages = new List<ChatMessage>
            {
                new(ChatRole.User, query),
            };
            messages.AddRange(response.Messages);

            var item = BuildEvalItem(query, response, messages, agent);
            items.Add(item);
        }

        return items;
    }

    private static async Task<List<EvalItem>> RunAgentForEvalAsync(
        AIAgent agent,
        IEnumerable<string> queries,
        IConversationSplitter? splitter,
        CancellationToken cancellationToken)
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
            item.Splitter = splitter;
            items.Add(item);
        }

        return items;
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
