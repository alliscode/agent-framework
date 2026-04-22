# Sample Guidelines

Samples are extremely important for developers to get started with Agent Framework. We strive to provide a wide range of samples that demonstrate the capabilities of Agent Framework with consistency and quality. This document outlines the guidelines for creating .NET samples.

## File Structure

Every sample project should follow this structure:

1. Copyright header: `// Copyright (c) Microsoft. All rights reserved.`
2. Sample summary comment block explaining what the sample demonstrates
3. Required `using` statements
4. Environment variable setup (if needed)
5. Agent/tool definitions
6. Main execution flow
7. Supporting classes (if any)

When modifying samples, update associated README files in the same or parent folders.

## Project Structure

Each sample is a standalone .NET console application with its own `.csproj` file. Samples inherit shared build configuration from `Directory.Build.props`.

```
concept-folder/
├── README.md              # Overview and sample list
├── SampleName/
│   ├── SampleName.csproj  # Project file
│   ├── Program.cs         # Main sample code
│   └── README.md          # Sample-specific README (optional)
```

### Naming Conventions

- **Concept folders** use kebab-case: `context-providers/`, `human-in-the-loop/`
- **Sample project folders** use PascalCase with step prefixes where applicable: `Agent_Step01_Basics/`
- These conventions match the cross-language standard while remaining idiomatic for .NET

## Azure Credentials

**Always use `DefaultAzureCredential`** in samples. This provides the broadest compatibility across development environments (Visual Studio, VS Code, Azure CLI, managed identity).

```csharp
using Azure.Identity;

var credential = new DefaultAzureCredential();
```

## Environment Variables

### Basic / Getting Started Samples

For getting started samples (`01-get-started/`), use environment variables with sensible defaults:

```csharp
// Copyright (c) Microsoft. All rights reserved.

// Sample summary explaining what this demonstrates.

using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5.4-mini";

AIAgent agent = new AzureOpenAIClient(new Uri(endpoint), new DefaultAzureCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(instructions: "You are a helpful assistant.", name: "MyAgent");

Console.WriteLine(await agent.RunAsync("Hello!"));
```

### Advanced Samples

For advanced samples, use the `SampleEnvironment` helper which reads from environment variables and `.env` files, and prompts the user interactively if values are missing:

```csharp
var endpoint = SampleEnvironment.Get("AZURE_OPENAI_ENDPOINT");
var model = SampleEnvironment.Get("AZURE_OPENAI_MODEL", "gpt-4o");
```

### Default Provider for Samples

Unless a sample is specifically demonstrating a particular provider (OpenAI direct, Anthropic, Ollama, etc.), use **Azure OpenAI** as the default provider. This is the recommended path for Azure AI Foundry deployments.

Provider-specific samples belong in `02-agents/providers/<provider>/` and should use the provider's native client.

## Doc Snippet Tags

Use XML comment tags to mark sections that may be referenced by documentation:

```csharp
// <create_agent>
AIAgent agent = new AzureOpenAIClient(...)
    .GetChatClient(deploymentName)
    .AsAIAgent(instructions: "...", name: "...");
// </create_agent>

// <run_agent>
Console.WriteLine(await agent.RunAsync("Hello!"));
// </run_agent>
```

These tags should match their Python equivalents (`# <create_agent>`, `# </create_agent>`) for cross-language documentation parity.

## General Guidelines

- **Clear and Concise**: Samples should demonstrate a specific set of features. The fewer concepts per sample, the better.
- **Consistent Structure**: All samples should follow the same patterns for setup, execution, and output.
- **Incremental Complexity**: Samples should start simple and gradually increase in complexity within each concept folder.
- **Documentation**: Samples should be well-documented with comments explaining the "what" and "why".

### Clear and Concise

Try not to include too many concepts in a single sample. If you find yourself including too many concepts, consider breaking the sample into multiple samples. For example, separate non-streaming and streaming modes into distinct demonstrations.

### Consistent Structure

Samples follow a consistent directory layout:

```
dotnet/samples/
├── 01-get-started/          # Progressive tutorial (steps 01–06)
├── 02-agents/               # Deep-dive concept samples
│   ├── agents/              # Core agent patterns
│   ├── providers/           # Provider-specific (azure/, openai/, anthropic/, etc.)
│   ├── observability/       # OpenTelemetry integration
│   ├── skills/              # Agent skills
│   ├── mcp/                 # Model Context Protocol
│   ├── declarative/         # YAML-based agents
│   └── ...                  # Additional concept folders
├── 03-workflows/            # Workflow patterns
├── 04-hosting/              # Deployment & hosting
└── 05-end-to-end/           # Complete applications
```

### Incremental Complexity

Make sure samples within a concept folder are incremental in complexity. Getting started samples should build on each other, and concept samples should build on the getting started samples.

### Documentation

Over-document the samples. This includes:

1. A **README.md** in each concept folder explaining the purpose and listing samples
2. A **summary comment block** at the top of each Program.cs:

    ```csharp
    // Multi-Turn Conversations — Use AgentSession to maintain context
    //
    // This sample shows how to keep conversation history across multiple calls
    // by reusing the same session object.
    ```

3. **Section comments** to explain the purpose of each code block:

    ```csharp
    // Create a session to maintain conversation history
    AgentSession session = await agent.CreateSessionAsync();

    // First turn
    Console.WriteLine(await agent.RunAsync("My name is Alice.", session));

    // Second turn — the agent should remember the user's name
    Console.WriteLine(await agent.RunAsync("What do you remember about me?", session));
    ```

4. A README with **expected output** if applicable

## Cross-Language Alignment

.NET samples should align with Python samples where possible:

- **Same concept folders** and naming conventions (kebab-case directories)
- **Same demonstrated concepts** and progression within each folder
- **Same comment structure** — descriptive header, section markers, doc snippet tags
- **Same prompts and scenarios** — use matching questions/inputs where the concept allows
- **.NET idioms preserved** — use C# patterns (top-level statements, `async/await`, attributes) rather than mimicking Python syntax
