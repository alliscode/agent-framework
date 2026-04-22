# Samples Structure & Design Choices — .NET

> This file documents the structure and conventions of the .NET samples so that
> agents (AI or human) can maintain them without rediscovering decisions.

## Directory layout

```
dotnet/samples/
├── 01-get-started/                    # Progressive tutorial (steps 01–06)
│   ├── 01_hello_agent/                # Create and run your first agent
│   ├── 02_add_tools/                  # Add function tools
│   ├── 03_multi_turn/                 # Multi-turn conversations with AgentSession
│   ├── 04_memory/                     # Agent memory with AIContextProvider
│   ├── 05_first_workflow/             # Build a workflow with executors and edges
│   └── 06_host_your_agent/            # Host your agent via Azure Functions
├── 02-agents/                         # Deep-dive concept samples
│   ├── agents/                        # Core agent patterns (tools, structured output,
│   │                                  #   conversations, middleware, plugins, MCP, etc.)
│   ├── providers/                     # Provider-specific samples, organized by provider
│   │   ├── a2a/                       #   Agent-to-Agent protocol provider
│   │   ├── anthropic/                 #   Anthropic Claude models
│   │   ├── azure/                     #   Azure OpenAI, Foundry Model, AI Project, etc.
│   │   ├── custom/                    #   Custom AIAgent implementations
│   │   ├── foundry/                   #   Microsoft Foundry (FoundryAgent + AsAIAgent)
│   │   ├── github-copilot/            #   GitHub Copilot SDK
│   │   ├── google-gemini/             #   Google Gemini
│   │   ├── ollama/                    #   Ollama local models
│   │   ├── onnx/                      #   ONNX Runtime
│   │   └── openai/                    #   OpenAI direct (ChatCompletion + Responses)
│   ├── observability/                 # OpenTelemetry integration
│   ├── skills/                        # Agent skills patterns
│   ├── context-providers/             # Memory providers (chat history, Mem0, Foundry)
│   ├── rag/                           # RAG patterns (text, vector store, Foundry)
│   ├── ag-ui/                         # AG-UI protocol samples
│   ├── declarative/                   # Declarative agent definitions
│   ├── devui/                         # DevUI samples
│   ├── evaluation/                    # Evaluation samples
│   └── mcp/                           # MCP server/client patterns
├── 03-workflows/                      # Workflow patterns
│   ├── _start-here/                   # Introductory workflow samples
│   ├── agents/                        # Agents in workflows
│   ├── checkpoint/                    # Checkpointing & resume
│   ├── control-flow/                  # Conditional routing & loops
│   ├── declarative/                   # YAML-based workflows
│   ├── evaluation/                    # Workflow evaluation
│   ├── human-in-the-loop/             # HITL patterns
│   ├── observability/                 # Workflow telemetry
│   ├── orchestrations/                # Orchestration patterns
│   ├── parallelism/                   # Concurrent execution
│   ├── resources/                     # Shared resources
│   ├── state-management/              # State isolation
│   └── visualization/                 # Workflow visualization
├── 04-hosting/                        # Deployment & hosting
│   ├── a2a/                           # Agent-to-Agent protocol
│   ├── durable-agents/                # Durable task framework agents
│   │   ├── AzureFunctions/            #   Azure Functions hosting
│   │   └── ConsoleApps/               #   Console app hosting
│   ├── durable-workflows/             # Durable task framework workflows
│   │   ├── AzureFunctions/            #   Azure Functions hosting
│   │   └── ConsoleApps/               #   Console app hosting
│   └── foundry-hosted-agents/         # Hosted Foundry agent scenarios
├── 05-end-to-end/                     # Complete applications
│   ├── a2a-client-server/             # A2A client/server demo
│   ├── ag-ui-client-server/           # AG-UI client/server demo
│   ├── ag-ui-web-chat/                # AG-UI web chat
│   ├── agent-web-chat/                # Aspire-based web chat
│   ├── aspnet-agent-authorization/    # Auth client/server
│   ├── devui-aspire-integration/      # DevUI/Aspire integration
│   ├── evaluation/                    # Evaluation demos
│   ├── m365-agent/                    # Microsoft 365 agent
│   └── purview-agent/                 # Purview integration
```

## Design principles

1. **Progressive complexity**: Sections 01→05 build from "hello world" to
   production. Within 01-get-started, projects are numbered 01–06 and each step
   adds exactly one concept.

2. **One concept per project** in 01-get-started. Each step is a standalone
   C# project with a single `Program.cs` file.

3. **Workflows preserved**: 03-workflows/ keeps the upstream folder names
   intact. Do not rename or restructure workflow samples.

4. **Per-project structure**: Each sample is a separate .csproj. Shared build
   configuration is inherited from `Directory.Build.props`.

## Default provider

All canonical samples (01-get-started) use **Azure OpenAI** via `AzureOpenAIClient`
with `DefaultAzureCredential`:

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

// WARNING: DefaultAzureCredential is convenient for development but requires careful consideration in production.
// In production, consider using a specific credential (e.g., ManagedIdentityCredential) to avoid
// latency issues, unintended credential probing, and potential security risks from fallback mechanisms.
AIAgent agent = new AzureOpenAIClient(new Uri(endpoint), new DefaultAzureCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(instructions: "...", name: "...");
```

Environment variables:
- `AZURE_OPENAI_ENDPOINT` — Your Azure OpenAI endpoint
- `AZURE_OPENAI_DEPLOYMENT_NAME` — Model deployment name (defaults to `gpt-5.4-mini`)

For authentication, run `az login` before running samples.

## Snippet tags for docs integration

Samples embed named snippet regions for future `:::code` integration:

```csharp
// <snippet_name>
code here
// </snippet_name>
```

## Building and running

All samples use project references to the framework source. To build and run:

```bash
cd dotnet/samples/01-get-started/01_hello_agent
dotnet run
```

## Current API notes

- `AIAgent` is the primary agent abstraction (created via `ChatClient.AsAIAgent(...)`)
- `AgentSession` manages multi-turn conversation state
- `AIContextProvider` injects memory and context
- Prefer `client.GetChatClient(deployment).AsAIAgent(...)` extension method pattern
- Azure Functions hosting uses `ConfigureDurableAgents(options => options.AddAIAgent(agent))`
- Workflows use `WorkflowBuilder` with `Executor<TIn, TOut>` and edge connections
