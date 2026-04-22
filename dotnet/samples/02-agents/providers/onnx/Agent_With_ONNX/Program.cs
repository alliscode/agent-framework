// Copyright (c) Microsoft. All rights reserved.

// ONNX — Agent using an ONNX runtime model
//
// This sample shows how to create an AI agent using an ONNX runtime
// model for local inference.

using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntimeGenAI;

// E.g. C:\repos\Phi-4-mini-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4
var modelPath = Environment.GetEnvironmentVariable("ONNX_MODEL_PATH") ?? throw new InvalidOperationException("ONNX_MODEL_PATH is not set.");

// Get a chat client for ONNX and use it to construct an AIAgent.
using OnnxRuntimeGenAIChatClient chatClient = new(modelPath);
AIAgent agent = chatClient.AsAIAgent(instructions: "You are good at telling jokes.", name: "Joker");

// Invoke the agent and output the text result.
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));
