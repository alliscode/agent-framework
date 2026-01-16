# Agent Harness Samples

The Agent Harness provides workflow-based runtime infrastructure for durable, observable, and controllable agent execution.

## What is the Agent Harness?

The Agent Harness is a specialized workflow pattern that wraps an agent with infrastructure for:

- **Turn-based execution control** - Configure maximum turns to prevent runaway agents
- **Transcript tracking** - Full observability into each turn, tool call, and response
- **Repair mechanisms** - Automatic repair of execution invariants (e.g., dangling tool calls)
- **Layered stop conditions** - Hard stops (max turns), agent signals (done), and error handling
- **Checkpointing** - Durable execution with pause/resume support

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Harness Workflow                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌───────────────┐   ┌────────────────┐         │
│  │  Repair  │ → │  AgentTurn    │ → │  StopDecision  │         │
│  │ Executor │   │   Executor    │   │    Executor    │         │
│  └──────────┘   └───────────────┘   └────────────────┘         │
│       ↑                                     │                   │
│       └────────── (loop if continue) ───────┘                   │
├─────────────────────────────────────────────────────────────────┤
│  Harness State: transcript, turn_count, status, stop_reason     │
└─────────────────────────────────────────────────────────────────┘
```

## Samples

### 1. `agent_harness_basics.py`
**Start here!** Demonstrates:
- Simple `AgentHarness` API for quick usage
- `HarnessWorkflowBuilder` for more control
- Turn limits and stop conditions
- Transcript tracking

### 2. `agent_harness_checkpoint.py`
Shows durability features:
- Checkpointing after each turn
- Inspecting checkpoint contents
- Durability across workflow instances

### 3. `agent_harness_custom_executors.py`
Advanced usage:
- Adding custom executors to the harness loop
- Building custom harness workflows
- Integrating logging and progress reporting

## Quick Start

```python
import asyncio

from agent_framework._harness import AgentHarness
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


async def main() -> None:
    # Create an agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions="You are a helpful assistant.",
        name="assistant",
    )

    # Wrap in harness with max 10 turns
    harness = AgentHarness(agent, max_turns=10)

    # Run
    async for event in harness.run_stream("Solve this task..."):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
```

## Key APIs

### `AgentHarness`
Simple wrapper for running an agent with harness infrastructure.

```text
harness = AgentHarness(agent, max_turns=20)
result = await harness.run("query")
```

### `HarnessWorkflowBuilder`
Builder for creating harness workflows with full control.

```text
builder = HarnessWorkflowBuilder(agent, max_turns=20, checkpoint_storage=storage)
workflow = builder.build()
result = await workflow.run(RepairTrigger(), **builder.get_harness_kwargs())
```

### `HarnessResult`
Result from harness execution containing:
- `status`: HarnessStatus (RUNNING, DONE, FAILED, STALLED)
- `reason`: StopReason (why execution stopped)
- `transcript`: List of HarnessEvent (full execution history)
- `turn_count`: Number of turns executed

## Prerequisites

- Azure OpenAI configured with required environment variables
- Authentication via `azure-identity` (`az login`)
- Python 3.10+
