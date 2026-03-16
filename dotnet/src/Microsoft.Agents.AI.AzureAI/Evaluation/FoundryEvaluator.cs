// Copyright (c) Microsoft. All rights reserved.

using System.Linq;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;
using Microsoft.Extensions.AI.Evaluation.Safety;

namespace Microsoft.Agents.AI.AzureAI;

/// <summary>
/// Evaluator that uses Azure AI Foundry evaluators via MEAI evaluation packages.
/// </summary>
/// <remarks>
/// Maps Foundry evaluator names (e.g., "relevance", "coherence") to the corresponding
/// MEAI evaluators. When the Azure.AI.Projects .NET SDK adds native evaluation API
/// support, this class will be updated to use it for full parity with the Python
/// <c>FoundryEvals</c> class.
/// </remarks>
public sealed class FoundryEvaluator : IAgentEvaluator
{
    private readonly ChatConfiguration _chatConfiguration;
    private readonly string[] _evaluatorNames;
    private readonly IConversationSplitter? _splitter;

    /// <summary>
    /// Initializes a new instance of the <see cref="FoundryEvaluator"/> class.
    /// </summary>
    /// <param name="chatConfiguration">Chat configuration for the LLM-based evaluators.</param>
    /// <param name="evaluators">
    /// Names of evaluators to use (e.g., "relevance", "coherence").
    /// When empty, defaults to relevance and coherence.
    /// </param>
    public FoundryEvaluator(ChatConfiguration chatConfiguration, params string[] evaluators)
        : this(chatConfiguration, splitter: null, evaluators)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FoundryEvaluator"/> class with a default splitter.
    /// </summary>
    /// <param name="chatConfiguration">Chat configuration for the LLM-based evaluators.</param>
    /// <param name="splitter">
    /// Default conversation splitter for multi-turn conversations. Overridden by
    /// <see cref="EvalItem.Splitter"/> when set on individual items.
    /// Use <see cref="ConversationSplitters.LastTurn"/>, <see cref="ConversationSplitters.Full"/>,
    /// or a custom <see cref="IConversationSplitter"/> implementation.
    /// </param>
    /// <param name="evaluators">
    /// Names of evaluators to use (e.g., "relevance", "coherence").
    /// When empty, defaults to relevance and coherence.
    /// </param>
    public FoundryEvaluator(ChatConfiguration chatConfiguration, IConversationSplitter? splitter, params string[] evaluators)
    {
        this._chatConfiguration = chatConfiguration;
        this._splitter = splitter;
        this._evaluatorNames = evaluators.Length > 0
            ? evaluators
            : [FoundryEvals.Relevance, FoundryEvals.Coherence];
    }

    /// <inheritdoc />
    public string Name => "FoundryEvaluator";

    /// <inheritdoc />
    public async Task<AgentEvaluationResults> EvaluateAsync(
        IReadOnlyList<EvalItem> items,
        string evalName = "Foundry Eval",
        CancellationToken cancellationToken = default)
    {
        var meaiEvaluators = BuildEvaluators(this._evaluatorNames);
        var composite = new CompositeEvaluator(meaiEvaluators.ToArray());

        var results = new List<EvaluationResult>(items.Count);

        foreach (var item in items)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Resolve splitter: item-level > evaluator-level > LastTurn default
            var effectiveSplitter = item.Splitter ?? this._splitter;
            var (queryMessages, _) = item.Split(effectiveSplitter);
            var messages = queryMessages.ToList();

            var chatResponse = item.RawResponse
                ?? new ChatResponse(new ChatMessage(ChatRole.Assistant, item.Response));

            var additionalContext = new List<EvaluationContext>();

            if (item.Context is not null)
            {
                additionalContext.Add(new GroundednessEvaluatorContext(item.Context));
            }

            var result = await composite.EvaluateAsync(
                messages,
                chatResponse,
                this._chatConfiguration,
                additionalContext: additionalContext.Count > 0 ? additionalContext : null,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            results.Add(result);
        }

        return new AgentEvaluationResults(this.Name, results);
    }

    private static List<IEvaluator> BuildEvaluators(string[] names)
    {
        var evaluators = new List<IEvaluator>();

        foreach (var name in names)
        {
            var evaluator = name switch
            {
                FoundryEvals.Relevance => new RelevanceEvaluator(),
                FoundryEvals.Coherence => new CoherenceEvaluator(),
                FoundryEvals.Groundedness => new GroundednessEvaluator(),
                FoundryEvals.Fluency => (IEvaluator)new FluencyEvaluator(),

                // Safety evaluators
                FoundryEvals.Violence or
                FoundryEvals.Sexual or
                FoundryEvals.SelfHarm or
                FoundryEvals.HateUnfairness => new ContentHarmEvaluator(),

                // Agent evaluators not yet available in MEAI — log warning and skip
                _ => null,
            };

            if (evaluator is not null)
            {
                evaluators.Add(evaluator);
            }
        }

        return evaluators;
    }
}
