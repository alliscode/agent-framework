// Copyright (c) Microsoft. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;

namespace Microsoft.Agents.AI.UnitTests;

/// <summary>
/// Tests for the evaluation types: <see cref="LocalEvaluator"/>, <see cref="FunctionEvaluator"/>,
/// <see cref="EvalChecks"/>, and <see cref="AgentEvaluationResults"/>.
/// </summary>
public sealed class EvaluationTests
{
    private static EvalItem CreateItem(
        string query = "What is the weather?",
        string response = "The weather in Seattle is sunny and 72°F.",
        IReadOnlyList<ChatMessage>? conversation = null)
    {
        conversation ??= new List<ChatMessage>
        {
            new(ChatRole.User, query),
            new(ChatRole.Assistant, response),
        };

        return new EvalItem(query, response, conversation);
    }

    // ---------------------------------------------------------------
    // EvalItem tests
    // ---------------------------------------------------------------

    [Fact]
    public void EvalItem_Constructor_SetsProperties()
    {
        // Arrange & Act
        var item = CreateItem();

        // Assert
        Assert.Equal("What is the weather?", item.Query);
        Assert.Equal("The weather in Seattle is sunny and 72°F.", item.Response);
        Assert.Equal(2, item.Conversation.Count);
        Assert.Null(item.Expected);
        Assert.Null(item.Context);
        Assert.Null(item.Tools);
    }

    [Fact]
    public void EvalItem_OptionalProperties_CanBeSet()
    {
        // Arrange & Act
        var item = CreateItem();
        item.Expected = "sunny";
        item.Context = "Weather data for Seattle";

        // Assert
        Assert.Equal("sunny", item.Expected);
        Assert.Equal("Weather data for Seattle", item.Context);
    }

    // ---------------------------------------------------------------
    // LocalEvaluator tests
    // ---------------------------------------------------------------

    [Fact]
    public async Task LocalEvaluator_WithPassingCheck_ReturnsPassedResultAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            FunctionEvaluator.Create("always_pass", (string _) => true));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.Equal("LocalEvaluator", results.Provider);
        Assert.Equal(1, results.Total);
        Assert.Equal(1, results.Passed);
        Assert.Equal(0, results.Failed);
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task LocalEvaluator_WithFailingCheck_ReturnsFailedResultAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            FunctionEvaluator.Create("always_fail", (string _) => false));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.Equal(1, results.Total);
        Assert.Equal(0, results.Passed);
        Assert.Equal(1, results.Failed);
        Assert.False(results.AllPassed);
    }

    [Fact]
    public async Task LocalEvaluator_WithMultipleChecks_AllChecksRunAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            FunctionEvaluator.Create("check1", (string _) => true),
            FunctionEvaluator.Create("check2", (string _) => true));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.Equal(1, results.Total);
        Assert.True(results.AllPassed);
        var itemResult = results.Items[0];
        Assert.Equal(2, itemResult.Metrics.Count);
        Assert.True(itemResult.Metrics.ContainsKey("check1"));
        Assert.True(itemResult.Metrics.ContainsKey("check2"));
    }

    [Fact]
    public async Task LocalEvaluator_WithMultipleItems_EvaluatesAllAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.KeywordCheck("weather"));

        var items = new List<EvalItem>
        {
            CreateItem(response: "The weather is sunny."),
            CreateItem(response: "I don't know about that topic."),
        };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.Equal(2, results.Total);
        Assert.Equal(1, results.Passed);
        Assert.Equal(1, results.Failed);
    }

    // ---------------------------------------------------------------
    // FunctionEvaluator tests
    // ---------------------------------------------------------------

    [Fact]
    public async Task FunctionEvaluator_ResponseOnly_PassesResponseAsync()
    {
        // Arrange
        var check = FunctionEvaluator.Create("length_check",
            (string response) => response.Length > 10);

        var evaluator = new LocalEvaluator(check);
        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task FunctionEvaluator_WithExpected_PassesExpectedAsync()
    {
        // Arrange
        var check = FunctionEvaluator.Create("contains_expected",
            (string response, string? expected) =>
                expected != null && response.Contains(expected, StringComparison.OrdinalIgnoreCase));

        var evaluator = new LocalEvaluator(check);
        var item = CreateItem();
        item.Expected = "sunny";
        var items = new List<EvalItem> { item };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task FunctionEvaluator_FullItem_AccessesAllFieldsAsync()
    {
        // Arrange
        var check = FunctionEvaluator.Create("full_check",
            (EvalItem item) => item.Query.Contains("weather", StringComparison.OrdinalIgnoreCase)
                && item.Response.Length > 0);

        var evaluator = new LocalEvaluator(check);
        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task FunctionEvaluator_WithCheckResult_ReturnsCustomReasonAsync()
    {
        // Arrange
        var check = FunctionEvaluator.Create("custom_check",
            (EvalItem item) => new EvalCheckResult(true, "Custom reason", "custom_check"));

        var evaluator = new LocalEvaluator(check);
        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.True(results.AllPassed);
        var metric = results.Items[0].Get<BooleanMetric>("custom_check");
        Assert.Equal("Custom reason", metric.Reason);
    }

    // ---------------------------------------------------------------
    // EvalChecks tests
    // ---------------------------------------------------------------

    [Fact]
    public async Task KeywordCheck_AllKeywordsPresent_PassesAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.KeywordCheck("weather", "sunny"));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task KeywordCheck_MissingKeyword_FailsAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.KeywordCheck("snow"));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.False(results.AllPassed);
    }

    [Fact]
    public async Task KeywordCheck_CaseInsensitiveByDefault_PassesAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.KeywordCheck("WEATHER", "SUNNY"));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task KeywordCheck_CaseSensitive_FailsOnWrongCaseAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.KeywordCheck(caseSensitive: true, "WEATHER"));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.False(results.AllPassed);
    }

    [Fact]
    public async Task ToolCalledCheck_ToolPresent_PassesAsync()
    {
        // Arrange
        var conversation = new List<ChatMessage>
        {
            new(ChatRole.User, "What is the weather?"),
            new(ChatRole.Assistant, new List<AIContent>
            {
                new FunctionCallContent("call1", "get_weather", new Dictionary<string, object?> { ["city"] = "Seattle" }),
            }),
            new(ChatRole.Tool, new List<AIContent>
            {
                new FunctionResultContent("call1", "72°F and sunny"),
            }),
            new(ChatRole.Assistant, "The weather is sunny and 72°F."),
        };

        var item = CreateItem(conversation: conversation);
        var evaluator = new LocalEvaluator(
            EvalChecks.ToolCalledCheck("get_weather"));

        // Act
        var results = await evaluator.EvaluateAsync(new List<EvalItem> { item });

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public async Task ToolCalledCheck_ToolMissing_FailsAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.ToolCalledCheck("get_weather"));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.False(results.AllPassed);
    }

    // ---------------------------------------------------------------
    // AgentEvaluationResults tests
    // ---------------------------------------------------------------

    [Fact]
    public void AgentEvaluationResults_AllPassed_WhenAllMetricsGood()
    {
        // Arrange
        var evalResult = new EvaluationResult();
        evalResult.Metrics["check"] = new BooleanMetric("check", true)
        {
            Interpretation = new EvaluationMetricInterpretation
            {
                Rating = EvaluationRating.Good,
                Failed = false,
            },
        };

        // Act
        var results = new AgentEvaluationResults("test", new[] { evalResult });

        // Assert
        Assert.True(results.AllPassed);
        Assert.Equal(1, results.Passed);
        Assert.Equal(0, results.Failed);
    }

    [Fact]
    public void AgentEvaluationResults_NotAllPassed_WhenMetricFailed()
    {
        // Arrange
        var evalResult = new EvaluationResult();
        evalResult.Metrics["check"] = new BooleanMetric("check", false)
        {
            Interpretation = new EvaluationMetricInterpretation
            {
                Rating = EvaluationRating.Unacceptable,
                Failed = true,
            },
        };

        // Act
        var results = new AgentEvaluationResults("test", new[] { evalResult });

        // Assert
        Assert.False(results.AllPassed);
        Assert.Equal(0, results.Passed);
        Assert.Equal(1, results.Failed);
    }

    [Fact]
    public void AssertAllPassed_ThrowsOnFailure()
    {
        // Arrange
        var evalResult = new EvaluationResult();
        evalResult.Metrics["check"] = new BooleanMetric("check", false)
        {
            Interpretation = new EvaluationMetricInterpretation
            {
                Rating = EvaluationRating.Unacceptable,
                Failed = true,
            },
        };

        var results = new AgentEvaluationResults("test", new[] { evalResult });

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => results.AssertAllPassed());
        Assert.Contains("0 passed", ex.Message);
        Assert.Contains("1 failed", ex.Message);
    }

    [Fact]
    public void AssertAllPassed_DoesNotThrowOnSuccess()
    {
        // Arrange
        var evalResult = new EvaluationResult();
        evalResult.Metrics["check"] = new BooleanMetric("check", true)
        {
            Interpretation = new EvaluationMetricInterpretation
            {
                Rating = EvaluationRating.Good,
                Failed = false,
            },
        };

        var results = new AgentEvaluationResults("test", new[] { evalResult });

        // Act & Assert (no exception)
        results.AssertAllPassed();
    }

    [Fact]
    public void AgentEvaluationResults_NumericMetric_HighScorePasses()
    {
        // Arrange
        var evalResult = new EvaluationResult();
        evalResult.Metrics["relevance"] = new NumericMetric("relevance", 4.5);

        // Act
        var results = new AgentEvaluationResults("test", new[] { evalResult });

        // Assert
        Assert.True(results.AllPassed);
    }

    [Fact]
    public void AgentEvaluationResults_NumericMetric_LowScoreFails()
    {
        // Arrange
        var evalResult = new EvaluationResult();
        evalResult.Metrics["relevance"] = new NumericMetric("relevance", 2.0);

        // Act
        var results = new AgentEvaluationResults("test", new[] { evalResult });

        // Assert
        Assert.False(results.AllPassed);
    }

    [Fact]
    public void AgentEvaluationResults_SubResults_AllPassedChecksChildren()
    {
        // Arrange
        var passResult = new EvaluationResult();
        passResult.Metrics["check"] = new BooleanMetric("check", true)
        {
            Interpretation = new EvaluationMetricInterpretation
            {
                Rating = EvaluationRating.Good,
                Failed = false,
            },
        };

        var failResult = new EvaluationResult();
        failResult.Metrics["check"] = new BooleanMetric("check", false)
        {
            Interpretation = new EvaluationMetricInterpretation
            {
                Rating = EvaluationRating.Unacceptable,
                Failed = true,
            },
        };

        var results = new AgentEvaluationResults("test", Array.Empty<EvaluationResult>())
        {
            SubResults = new Dictionary<string, AgentEvaluationResults>
            {
                ["agent1"] = new("test", new[] { passResult }),
                ["agent2"] = new("test", new[] { failResult }),
            },
        };

        // Assert
        Assert.False(results.AllPassed);
    }

    // ---------------------------------------------------------------
    // Mixed evaluator tests
    // ---------------------------------------------------------------

    [Fact]
    public async Task LocalEvaluator_MixedChecks_ReportsCorrectCountsAsync()
    {
        // Arrange
        var evaluator = new LocalEvaluator(
            EvalChecks.KeywordCheck("weather"),
            EvalChecks.KeywordCheck("snow"),
            FunctionEvaluator.Create("is_long", (string r) => r.Length > 5));

        var items = new List<EvalItem> { CreateItem() };

        // Act
        var results = await evaluator.EvaluateAsync(items);

        // Assert
        Assert.Equal(1, results.Total);

        // One item with 3 checks: "weather" passes, "snow" fails, "is_long" passes
        // The item has one failed metric so it should count as failed
        Assert.Equal(0, results.Passed);
        Assert.Equal(1, results.Failed);
    }

    // ---------------------------------------------------------------
    // Conversation Split tests
    // ---------------------------------------------------------------

    private static List<ChatMessage> CreateMultiTurnConversation()
    {
        return new List<ChatMessage>
        {
            new(ChatRole.User, "What's the weather in Seattle?"),
            new(ChatRole.Assistant, "Seattle is 62°F and cloudy."),
            new(ChatRole.User, "And Paris?"),
            new(ChatRole.Assistant, "Paris is 68°F and partly sunny."),
            new(ChatRole.User, "Compare them."),
            new(ChatRole.Assistant, "Seattle is cooler; Paris is warmer and sunnier."),
        };
    }

    [Fact]
    public void Split_LastTurn_SplitsAtLastUserMessage()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();
        var item = new EvalItem("Compare them.", "Seattle is cooler; Paris is warmer and sunnier.", conversation);

        // Act
        var (query, response) = item.Split(ConversationSplit.LastTurn);

        // Assert — query includes everything up to and including "Compare them."
        Assert.Equal(5, query.Count);
        Assert.Equal(ChatRole.User, query[query.Count - 1].Role);
        Assert.Contains("Compare", query[query.Count - 1].Text);

        // Response is the final assistant message
        Assert.Single(response);
        Assert.Equal(ChatRole.Assistant, response[0].Role);
    }

    [Fact]
    public void Split_Full_SplitsAtFirstUserMessage()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();
        var item = new EvalItem("What's the weather in Seattle?", "Full trajectory", conversation);

        // Act
        var (query, response) = item.Split(ConversationSplit.Full);

        // Assert — query is just the first user message
        Assert.Single(query);
        Assert.Contains("Seattle", query[0].Text);

        // Response is everything after
        Assert.Equal(5, response.Count);
    }

    [Fact]
    public void Split_Full_IncludesSystemMessagesInQuery()
    {
        // Arrange
        var conversation = new List<ChatMessage>
        {
            new(ChatRole.System, "You are a weather assistant."),
            new(ChatRole.User, "What's the weather?"),
            new(ChatRole.Assistant, "It's sunny."),
        };

        var item = new EvalItem("What's the weather?", "It's sunny.", conversation);

        // Act
        var (query, response) = item.Split(ConversationSplit.Full);

        // Assert — system message + first user message
        Assert.Equal(2, query.Count);
        Assert.Equal(ChatRole.System, query[0].Role);
        Assert.Equal(ChatRole.User, query[1].Role);
        Assert.Single(response);
    }

    [Fact]
    public void Split_DefaultIsLastTurn()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();
        var item = new EvalItem("Compare them.", "response", conversation);

        // Act — no split specified
        var (query, response) = item.Split();

        // Assert — same as LastTurn
        Assert.Equal(5, query.Count);
        Assert.Single(response);
    }

    [Fact]
    public void Split_SplitStrategyProperty_UsedWhenNoExplicitSplit()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();
        var item = new EvalItem("query", "response", conversation)
        {
            SplitStrategy = ConversationSplit.Full,
        };

        // Act — no explicit split, should use SplitStrategy
        var (query, response) = item.Split();

        // Assert — Full split
        Assert.Single(query);
        Assert.Equal(5, response.Count);
    }

    [Fact]
    public void Split_ExplicitSplit_OverridesSplitStrategy()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();
        var item = new EvalItem("query", "response", conversation)
        {
            SplitStrategy = ConversationSplit.Full,
        };

        // Act — explicit LastTurn overrides Full
        var (query, response) = item.Split(ConversationSplit.LastTurn);

        // Assert — LastTurn behavior
        Assert.Equal(5, query.Count);
        Assert.Single(response);
    }

    [Fact]
    public void Split_WithToolMessages_PreservesToolPairs()
    {
        // Arrange
        var conversation = new List<ChatMessage>
        {
            new(ChatRole.User, "What's the weather?"),
            new(ChatRole.Assistant, new List<AIContent>
            {
                new FunctionCallContent("c1", "get_weather", new Dictionary<string, object?> { ["city"] = "Seattle" }),
            }),
            new(ChatRole.Tool, new List<AIContent>
            {
                new FunctionResultContent("c1", "62°F, cloudy"),
            }),
            new(ChatRole.Assistant, "Seattle is 62°F and cloudy."),
            new(ChatRole.User, "Thanks!"),
            new(ChatRole.Assistant, "You're welcome!"),
        };

        var item = new EvalItem("Thanks!", "You're welcome!", conversation);

        // Act
        var (query, response) = item.Split(ConversationSplit.LastTurn);

        // Assert — tool messages stay in query context
        Assert.Equal(5, query.Count);
        Assert.Equal(ChatRole.Tool, query[2].Role);
        Assert.Single(response);
    }

    [Fact]
    public void SplitLastTurn_Static_CanBeUsedAsCustomFallback()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();

        // Act
        var (query, response) = EvalItem.SplitLastTurn(conversation);

        // Assert
        Assert.Equal(5, query.Count);
        Assert.Single(response);
    }

    // ---------------------------------------------------------------
    // PerTurnItems tests
    // ---------------------------------------------------------------

    [Fact]
    public void PerTurnItems_SplitsMultiTurnConversation()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();

        // Act
        var items = EvalItem.PerTurnItems(conversation);

        // Assert — 3 user messages = 3 items
        Assert.Equal(3, items.Count);

        // First turn: "What's the weather in Seattle?"
        Assert.Contains("Seattle", items[0].Query);
        Assert.Contains("62°F", items[0].Response);
        Assert.Equal(2, items[0].Conversation.Count);

        // Second turn: "And Paris?"
        Assert.Contains("Paris", items[1].Query);
        Assert.Contains("68°F", items[1].Response);
        Assert.Equal(4, items[1].Conversation.Count);

        // Third turn: "Compare them."
        Assert.Contains("Compare", items[2].Query);
        Assert.Contains("cooler", items[2].Response);
        Assert.Equal(6, items[2].Conversation.Count);
    }

    [Fact]
    public void PerTurnItems_PropagatesToolsAndContext()
    {
        // Arrange
        var conversation = CreateMultiTurnConversation();

        // Act
        var items = EvalItem.PerTurnItems(
            conversation,
            context: "Weather database");

        // Assert
        Assert.All(items, item => Assert.Equal("Weather database", item.Context));
    }

    [Fact]
    public void PerTurnItems_SingleTurn_ReturnsOneItem()
    {
        // Arrange
        var conversation = new List<ChatMessage>
        {
            new(ChatRole.User, "Hello"),
            new(ChatRole.Assistant, "Hi there!"),
        };

        // Act
        var items = EvalItem.PerTurnItems(conversation);

        // Assert
        Assert.Single(items);
        Assert.Equal("Hello", items[0].Query);
        Assert.Equal("Hi there!", items[0].Response);
    }

    // ---------------------------------------------------------------
    // Custom IConversationSplitter tests
    // ---------------------------------------------------------------

    [Fact]
    public void Split_CustomSplitter_IsUsed()
    {
        // Arrange — splitter that splits before a tool call message
        var conversation = new List<ChatMessage>
        {
            new(ChatRole.User, "Remember this"),
            new(ChatRole.Assistant, "Storing..."),
            new(ChatRole.User, "What did I say?"),
            new(ChatRole.Assistant, new List<AIContent>
            {
                new FunctionCallContent("c1", "retrieve_memory"),
            }),
            new(ChatRole.Tool, new List<AIContent>
            {
                new FunctionResultContent("c1", "You said: Remember this"),
            }),
            new(ChatRole.Assistant, "You said 'Remember this'."),
        };

        var splitter = new MemorySplitter();
        var item = new EvalItem("What did I say?", "You said 'Remember this'.", conversation);

        // Act
        var (query, response) = item.Split(splitter);

        // Assert — split before the tool call
        Assert.Equal(3, query.Count);
        Assert.Equal(3, response.Count);
    }

    private sealed class MemorySplitter : IConversationSplitter
    {
        public (IReadOnlyList<ChatMessage> QueryMessages, IReadOnlyList<ChatMessage> ResponseMessages) Split(
            IReadOnlyList<ChatMessage> conversation)
        {
            for (int i = 0; i < conversation.Count; i++)
            {
                var msg = conversation[i];
                if (msg.Role == ChatRole.Assistant && msg.Contents != null)
                {
                    foreach (var content in msg.Contents)
                    {
                        if (content is FunctionCallContent fc && fc.Name == "retrieve_memory")
                        {
                            return (
                                conversation.Take(i).ToList(),
                                conversation.Skip(i).ToList());
                        }
                    }
                }
            }

            // Fallback to last-turn split
            return EvalItem.SplitLastTurn(conversation);
        }
    }
}
