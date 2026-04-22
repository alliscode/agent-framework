// Copyright (c) Microsoft. All rights reserved.

// Agent with MCP Server Tools — Connect to an external MCP server
//
// This sample demonstrates how to create an AI agent that uses tools
// from an MCP (Model Context Protocol) server. It connects to a GitHub
// MCP server via stdio, discovers available tools, and wires them into
// the agent so the model can call them.

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

// --- Connect to the MCP server ---
await using var mcpClient = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "MCPServer",
    Command = "npx",
    Arguments = ["-y", "--verbose", "@modelcontextprotocol/server-github"],
}));

// Discover tools exposed by the MCP server
var mcpTools = await mcpClient.ListToolsAsync().ConfigureAwait(false);

// --- Create the agent with MCP tools ---
AIAgent agent = new AzureOpenAIClient(
    new Uri(endpoint),
    new DefaultAzureCredential())
     .GetChatClient(deploymentName)
     .AsAIAgent(instructions: "You answer questions related to GitHub repositories only.", tools: [.. mcpTools.Cast<AITool>()]);

// --- Run the agent ---
Console.WriteLine(await agent.RunAsync("Summarize the last four commits to the microsoft/semantic-kernel repository?"));
