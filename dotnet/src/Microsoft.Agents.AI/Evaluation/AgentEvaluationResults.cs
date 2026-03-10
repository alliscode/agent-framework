// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.AI.Evaluation;

namespace Microsoft.Agents.AI;

/// <summary>
/// Aggregate evaluation results across multiple items.
/// </summary>
public sealed class AgentEvaluationResults
{
    private readonly List<EvaluationResult> _items;

    /// <summary>
    /// Initializes a new instance of the <see cref="AgentEvaluationResults"/> class.
    /// </summary>
    /// <param name="provider">Name of the evaluation provider.</param>
    /// <param name="items">Per-item MEAI evaluation results.</param>
    public AgentEvaluationResults(string provider, IEnumerable<EvaluationResult> items)
    {
        Provider = provider;
        _items = new List<EvaluationResult>(items);
    }

    /// <summary>Gets the evaluation provider name.</summary>
    public string Provider { get; }

    /// <summary>Gets the portal URL for viewing results (Foundry only).</summary>
    public Uri? ReportUrl { get; set; }

    /// <summary>Gets the per-item MEAI evaluation results.</summary>
    public IReadOnlyList<EvaluationResult> Items => _items;

    /// <summary>Gets per-agent results for workflow evaluations.</summary>
    public IReadOnlyDictionary<string, AgentEvaluationResults>? SubResults { get; set; }

    /// <summary>Gets the number of items that passed.</summary>
    public int Passed => _items.Count(ItemPassed);

    /// <summary>Gets the number of items that failed.</summary>
    public int Failed => _items.Count(i => !ItemPassed(i));

    /// <summary>Gets the total number of items evaluated.</summary>
    public int Total => _items.Count;

    /// <summary>Gets whether all items passed.</summary>
    public bool AllPassed
    {
        get
        {
            if (SubResults is not null)
            {
                return SubResults.Values.All(s => s.AllPassed);
            }

            return Total > 0 && Failed == 0;
        }
    }

    /// <summary>
    /// Asserts that all items passed. Throws <see cref="InvalidOperationException"/> on failure.
    /// </summary>
    /// <param name="message">Optional custom failure message.</param>
    /// <exception cref="InvalidOperationException">Thrown when any items failed.</exception>
    public void AssertAllPassed(string? message = null)
    {
        if (!AllPassed)
        {
            var detail = message ?? $"{Provider}: {Passed} passed, {Failed} failed out of {Total}.";
            if (ReportUrl is not null)
            {
                detail += $" See {ReportUrl} for details.";
            }

            if (SubResults is not null)
            {
                var failedAgents = SubResults
                    .Where(kvp => !kvp.Value.AllPassed)
                    .Select(kvp => kvp.Key);
                detail += $" Failed agents: {string.Join(", ", failedAgents)}.";
            }

            throw new InvalidOperationException(detail);
        }
    }

    private static bool ItemPassed(EvaluationResult result)
    {
        foreach (var metric in result.Metrics.Values)
        {
            if (metric.Interpretation?.Failed == true)
            {
                return false;
            }

            if (metric is NumericMetric numeric && numeric.Value.HasValue)
            {
                if (numeric.Value.Value < 3.0)
                {
                    return false;
                }
            }
            else if (metric is BooleanMetric boolean && boolean.Value.HasValue)
            {
                if (!boolean.Value.Value)
                {
                    return false;
                }
            }
        }

        return result.Metrics.Count > 0;
    }
}
