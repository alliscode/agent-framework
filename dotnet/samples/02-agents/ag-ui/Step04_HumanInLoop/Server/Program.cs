// Copyright (c) Microsoft. All rights reserved.

// AG-UI Human in the Loop — Server with approval workflows
//
// This sample shows how to implement human-in-the-loop approval
// workflows via the AG-UI protocol.

using System.ComponentModel;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting.AGUI.AspNetCore;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.AspNetCore.HttpLogging;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using ServerFunctionApproval;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

builder.Services.AddHttpLogging(logging =>
{
    logging.LoggingFields = HttpLoggingFields.RequestPropertiesAndHeaders | HttpLoggingFields.RequestBody
        | HttpLoggingFields.ResponsePropertiesAndHeaders | HttpLoggingFields.ResponseBody;
    logging.RequestBodyLogLimit = int.MaxValue;
    logging.ResponseBodyLogLimit = int.MaxValue;
});

builder.Services.AddHttpClient().AddLogging();
builder.Services.ConfigureHttpJsonOptions(options =>
    options.SerializerOptions.TypeInfoResolverChain.Add(ApprovalJsonContext.Default));
builder.Services.AddAGUI();

WebApplication app = builder.Build();

app.UseHttpLogging();

string endpoint = builder.Configuration["FOUNDRY_PROJECT_ENDPOINT"]
    ?? throw new InvalidOperationException("FOUNDRY_PROJECT_ENDPOINT is not set.");
string deploymentName = builder.Configuration["FOUNDRY_MODEL"]
    ?? throw new InvalidOperationException("FOUNDRY_MODEL is not set.");

// Define approval-required tool
[Description("Approve the expense report.")]
static string ApproveExpenseReport(string expenseReportId)
{
    return $"Expense report {expenseReportId} approved";
}

// Get JsonSerializerOptions
var jsonOptions = app.Services.GetRequiredService<IOptions<JsonOptions>>().Value;

// Create approval-required tool
#pragma warning disable MEAI001 // Type is for evaluation purposes only
AITool[] tools = [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(ApproveExpenseReport))];
#pragma warning restore MEAI001

// Create base agent
AIProjectClient aiProjectClient = new(new Uri(endpoint), new DefaultAzureCredential());

ChatClientAgent baseAgent = aiProjectClient.AsAIAgent(
    model: deploymentName,
    name: "AGUIAssistant",
    instructions: "You are a helpful assistant in charge of approving expenses",
    tools: tools);

// Wrap with ServerFunctionApprovalAgent
var agent = new ServerFunctionApprovalAgent(baseAgent, jsonOptions.SerializerOptions);

app.MapAGUI("/", agent);
await app.RunAsync();
