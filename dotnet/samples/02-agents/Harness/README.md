# Harness Agent Samples

Samples demonstrating the [Harness AIContextProviders](../../../src/Microsoft.Agents.AI/Harness/) — reusable providers that add planning, task management, and mode tracking to any `ChatClientAgent`.

## Samples

| Sample | Description |
| --- | --- |
| [Harness_Step01_Research](./Harness_Step01_Research/README.md) | Using a ChatClientAgent with TodoProvider and AgentModeProvider for research, showcasing planning mode and todo management |
| [Harness_Step02_Research_WithSubAgents](./Harness_Step02_Research_WithSubAgents/README.md) | Using SubAgentsProvider to delegate stock price lookups to a web-search sub-agent concurrently |
| [Harness_Step03_Shell](./Harness_Step03_Shell/README.md) | Wiring `LocalShellTool` (Microsoft.Agents.AI.Tools.Shell) into the harness with approval-in-the-loop as the security boundary |
