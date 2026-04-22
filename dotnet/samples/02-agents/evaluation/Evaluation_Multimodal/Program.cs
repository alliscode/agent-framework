// Copyright (c) Microsoft. All rights reserved.

// Evaluate Multimodal Conversations — Verify image content in eval pipeline
//
// Demonstrates that the evaluation pipeline preserves multimodal content:
// 1. Build EvalItems with image content in conversations
// 2. Use HasImageContent() to verify images flow through the eval pipeline
// 3. Combine with other checks like NonEmpty()
//
// No Azure credentials needed — this sample builds EvalItems locally.

using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

// Simulate a vision agent conversation where the user sends an image.
// Just pass the conversation — query/response are derived automatically.
// For cloud-based quality evaluation of multimodal conversations, see the
// 05-end-to-end/Evaluation samples (FoundryQuality, ConversationSplits).
EvalItem imageItem = new(
    conversation:
    [
        new(ChatRole.User,
        [
            new TextContent("What do you see in this image?"),
            new UriContent(new Uri("https://example.com/mountain.png"), "image/png"),
        ]),
        new(ChatRole.Assistant, "The image shows a mountain landscape with snow-capped peaks."),
    ]);

// Simulate a text-only conversation (no image).
EvalItem textItem = new(
    query: "Tell me about mountains.",
    response: "Mountains are large landforms that rise above the surrounding terrain.");

// HasImageContent() passes when the conversation contains an image, fails otherwise.
// This lets you verify that your vision agent actually received the image.
LocalEvaluator evaluator = new(
    EvalChecks.HasImageContent(),
    EvalChecks.NonEmpty());

AgentEvaluationResults results = await evaluator.EvaluateAsync([imageItem, textItem]);

Console.WriteLine($"Evaluation: {results.Passed}/{results.Total} passed");
Console.WriteLine();

Console.WriteLine($"Image conversation: has_image_content = {imageItem.HasImageContent}");  // true
Console.WriteLine($"Text conversation:  has_image_content = {textItem.HasImageContent}");   // false
Console.WriteLine();

for (int i = 0; i < results.Items.Count; i++)
{
    Console.WriteLine($"Item {i + 1}: {results.InputItems![i].Query}");
    foreach (var metric in results.Items[i].Metrics)
    {
        string status = metric.Value.Interpretation?.Failed == true ? "FAIL" : "PASS";
        Console.WriteLine($"  [{status}] {metric.Key}: {metric.Value.Interpretation?.Reason}");
    }

    Console.WriteLine();
}
