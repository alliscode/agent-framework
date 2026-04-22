// Copyright (c) Microsoft. All rights reserved.

// Foundry Agent with Hosted MCP Tools — Server-side MCP tool execution
//
// This sample shows how to use Hosted MCP Tools with Microsoft Foundry Agents.
// The Foundry service invokes MCP tools server-side — the Agent Framework does
// not call them directly. Two modes are demonstrated:
// 1. Auto-approval — tools execute without user confirmation
// 2. Required approval — human-in-the-loop approval before each tool call

using Azure.AI.Projects;
using Azure.AI.Projects.Agents;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Responses;

// --- Configuration ---
var endpoint = Environment.GetEnvironmentVariable("AZURE_AI_PROJECT_ENDPOINT") ?? throw new InvalidOperationException("AZURE_AI_PROJECT_ENDPOINT is not set.");
var model = Environment.GetEnvironmentVariable("AZURE_AI_MODEL_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

var aiProjectClient = new AIProjectClient(new Uri(endpoint), new DefaultAzureCredential());

// --- 1. MCP Tool with Auto Approval ---

// Create an MCP tool definition that the agent can use.
// In this case we allow the tool to always be called without approval.
var mcpTool = ResponseTool.CreateMcpTool(
    serverLabel: "microsoft_learn",
    serverUri: new Uri("https://learn.microsoft.com/api/mcp"),
    toolCallApprovalPolicy: new McpToolCallApprovalPolicy(GlobalMcpToolCallApprovalPolicy.NeverRequireApproval));

// Create a server side agent with the mcp tool, and expose it as an AIAgent.
ProjectsAgentVersion agentVersion = await aiProjectClient.AgentAdministrationClient.CreateAgentVersionAsync(
    "MicrosoftLearnAgent",
    new ProjectsAgentVersionCreationOptions(
        new DeclarativeAgentDefinition(model: model)
        {
            Instructions = "You answer questions by searching the Microsoft Learn content only.",
            Tools = { mcpTool }
        }));

AIAgent agent = aiProjectClient.AsAIAgent(agentVersion);

// You can then invoke the agent like any other AIAgent.
AgentSession session = await agent.CreateSessionAsync();
Console.WriteLine(await agent.RunAsync("Please summarize the Azure AI Agent documentation related to MCP Tool calling?", session));

// Cleanup for sample purposes.
aiProjectClient.AgentAdministrationClient.DeleteAgent(agent.Name);

// --- 2. MCP Tool with Approval Required ---

// Create an MCP tool definition that the agent can use.
// In this case we require approval before the tool can be called.
var mcpToolWithApproval = ResponseTool.CreateMcpTool(
    serverLabel: "microsoft_learn",
    serverUri: new Uri("https://learn.microsoft.com/api/mcp"),
    allowedTools: new McpToolFilter() { ToolNames = { "microsoft_docs_search" } },
    toolCallApprovalPolicy: new McpToolCallApprovalPolicy(GlobalMcpToolCallApprovalPolicy.AlwaysRequireApproval));

// Create an agent with the MCP tool that requires approval.
ProjectsAgentVersion agentVersionWithApproval = await aiProjectClient.AgentAdministrationClient.CreateAgentVersionAsync(
    "MicrosoftLearnAgentWithApproval",
    new ProjectsAgentVersionCreationOptions(
        new DeclarativeAgentDefinition(model: model)
        {
            Instructions = "You answer questions by searching the Microsoft Learn content only.",
            Tools = { mcpToolWithApproval }
        }));

AIAgent agentWithRequiredApproval = aiProjectClient.AsAIAgent(agentVersionWithApproval);

// You can then invoke the agent like any other AIAgent.
// For simplicity, we are assuming here that only mcp tool approvals are pending.
AgentSession sessionWithRequiredApproval = await agentWithRequiredApproval.CreateSessionAsync();
AgentResponse response = await agentWithRequiredApproval.RunAsync("Please summarize the Azure AI Agent documentation related to MCP Tool calling?", sessionWithRequiredApproval);
List<ToolApprovalRequestContent> approvalRequests = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();

while (approvalRequests.Count > 0)
{
    // Ask the user to approve each MCP call request.
    List<ChatMessage> userInputResponses = approvalRequests
        .ConvertAll(approvalRequest =>
        {
            McpServerToolCallContent mcpToolCall = (McpServerToolCallContent)approvalRequest.ToolCall!;
            Console.WriteLine($"""
                The agent would like to invoke the following MCP Tool, please reply Y to approve.
                ServerName: {mcpToolCall.ServerName}
                Name: {mcpToolCall.Name}
                Arguments: {string.Join(", ", mcpToolCall.Arguments?.Select(x => $"{x.Key}: {x.Value}") ?? [])}
                """);
            return new ChatMessage(ChatRole.User, [approvalRequest.CreateResponse(Console.ReadLine()?.Equals("Y", StringComparison.OrdinalIgnoreCase) ?? false)]);
        });

    // Pass the user input responses back to the agent for further processing.
    response = await agentWithRequiredApproval.RunAsync(userInputResponses, sessionWithRequiredApproval);

    approvalRequests = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();
}

Console.WriteLine($"\nAgent: {response}");
