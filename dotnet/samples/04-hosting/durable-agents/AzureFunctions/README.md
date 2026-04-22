# Azure Functions Samples

This directory contains samples for Azure Functions.

- **[01_SingleAgent](01_SingleAgent)**: A sample that demonstrates how to host a single conversational agent in an Azure Functions app and invoke it directly over HTTP.
- **[02_AgentOrchestration_Chaining](02_AgentOrchestration_Chaining)**: A sample that demonstrates how to host a single conversational agent in an Azure Functions app and invoke it using a durable orchestration.
- **[03_AgentOrchestration_Concurrency](03_AgentOrchestration_Concurrency)**: A sample that demonstrates how to host multiple agents in an Azure Functions app and run them concurrently using a durable orchestration.
- **[04_AgentOrchestration_Conditionals](04_AgentOrchestration_Conditionals)**: A sample that demonstrates how to host multiple agents in an Azure Functions app and run them sequentially using a durable orchestration with conditionals.
- **[05_AgentOrchestration_HITL](05_AgentOrchestration_HITL)**: A sample that demonstrates how to implement a human-in-the-loop workflow using durable orchestration, including external event handling for human approval.
- **[06_LongRunningTools](06_LongRunningTools)**: A sample that demonstrates how agents can start and interact with durable orchestrations from tool calls to enable long-running tool scenarios.
- **[07_AgentAsMcpTool](07_AgentAsMcpTool)**: A sample that demonstrates how to configure durable AI agents to be accessible as Model Context Protocol (MCP) tools.
- **[08_ReliableStreaming](08_ReliableStreaming)**: A sample that demonstrates how to implement reliable streaming for durable agents using Redis Streams, enabling clients to disconnect and reconnect without losing messages.

## Running the Samples

These samples are designed to be run locally in a cloned repository.

### Prerequisites

The following prerequisites are required to run the samples:

- [.NET 10.0 SDK or later](https://dotnet.microsoft.com/download/dotnet)
- [Azure Functions Core Tools](https://learn.microsoft.com/azure/azure-functions/functions-run-local) (version 4.x or later)
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed and authenticated (`az login`)
- [Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/) project with a deployed model (gpt-5.4-mini or better is recommended)
- [Durable Task Scheduler](https://learn.microsoft.com/azure/azure-functions/durable/durable-task-scheduler/develop-with-durable-task-scheduler) (local emulator or Azure-hosted)
- [Docker](https://docs.docker.com/get-docker/) installed if running the Durable Task Scheduler emulator locally

### Authentication

These samples use `DefaultAzureCredential` to authenticate with Azure AI Foundry. Ensure you are logged in with the Azure CLI (`az login`) and have the appropriate role assignments on your Azure AI Foundry project.

> **Note:** API key authentication is not supported with Azure AI Foundry. Use `DefaultAzureCredential` (via `az login`, managed identity, or other supported credential types) instead.

### Start Durable Task Scheduler

Most samples use the Durable Task Scheduler (DTS) to support hosted agents and durable orchestrations. DTS also allows you to view the status of orchestrations and their inputs and outputs from a web UI.

To run the Durable Task Scheduler locally, you can use the following `docker` command:

```bash
docker run -d --name dts-emulator -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest
```

The DTS dashboard will be available at `http://localhost:8080`.

### Start the Azure Storage Emulator

All Function apps require an Azure Storage account to store functions-specific state. You can use the Azure Storage Emulator to run a local instance of the Azure Storage service.

You can run the Azure Storage emulator locally as a standalone process or via a Docker container.

#### Docker

```bash
docker run -d --name storage-emulator -p 10000:10000 -p 10001:10001 -p 10002:10002 mcr.microsoft.com/azure-storage/azurite
```

#### Standalone

```bash
npm install -g azurite
azurite
```

### Environment Configuration

Each sample has its own `local.settings.json` file that contains the environment variables for the sample. You'll need to update the `local.settings.json` file with the correct values for your Azure AI Foundry project.

```json
{
  "Values": {
    "FOUNDRY_PROJECT_ENDPOINT": "https://your-project.services.ai.azure.com/",
    "FOUNDRY_MODEL": "your-deployment-name"
  }
}
```

Alternatively, you can set the environment variables in the command line.

### Bash (Linux/macOS/WSL)

```bash
export FOUNDRY_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com/"
export FOUNDRY_MODEL="your-deployment-name"
```

### PowerShell

```powershell
$env:FOUNDRY_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com/"
$env:FOUNDRY_MODEL="your-deployment-name"
```

These environment variables, when set, will override the values in the `local.settings.json` file, making it convenient to test the sample without having to update the `local.settings.json` file.

### Start the Azure Functions app

Navigate to the sample directory and start the Azure Functions app:

```bash
cd dotnet/samples/04-hosting/durable-agents/AzureFunctions/01_SingleAgent
func start
```

The Azure Functions app will be available at `http://localhost:7071`.

### Test the Azure Functions app

The README.md file in each sample directory contains instructions for testing the sample. Each sample also includes a `demo.http` file that can be used to test the sample from the command line. These files can be opened in VS Code with the [REST Client](https://marketplace.visualstudio.com/items?itemName=humao.rest-client) extension or in the Visual Studio IDE.

### Viewing the sample output

The Azure Functions app logs are displayed in the terminal where you ran `func start`. This is where most agent output will be displayed. You can adjust logging levels in the `host.json` file as needed.

You can also see the state of agents and orchestrations in the DTS dashboard.
