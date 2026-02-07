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

---

## Production Setup Guide

The following configuration has been validated through extensive experimentation
comparing output quality across different settings and models.

### Recommended Configuration

```python
from agent_framework._harness import AgentHarness, get_task_complete_tool
from agent_framework._harness._compaction import (
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
)

harness = AgentHarness(
    agent,
    # --- Turn management ---
    max_turns=50,                          # Allow thorough multi-turn work
    enable_stall_detection=True,           # Detect when agent stops making progress
    stall_threshold=3,                     # 3 consecutive no-progress turns = stall
    enable_continuation_prompts=True,      # Nudge agent to continue if it pauses
    max_continuation_prompts=2,            # Max 2 nudges before accepting done

    # --- Quality enforcement ---
    enable_work_items=True,                # Work item planning + self-critique loop

    # --- Context management ---
    enable_compaction=True,                # Full compaction pipeline
    compaction_store=InMemoryCompactionStore(),
    artifact_store=InMemoryArtifactStore(),
    summary_cache=InMemorySummaryCache(max_entries=100),
    max_input_tokens=100_000,              # Token budget (model limit minus headroom)
    soft_threshold_percent=0.85,           # Trigger compaction at 85% of budget

    # --- Environment awareness ---
    sandbox_path=str(sandbox_dir),         # Injects cwd, OS, directory listing into system prompt
)
```

### What Gets Injected Automatically

When configured as above, the harness automatically injects:

| Layer | Content | Survives Compaction? |
|-------|---------|---------------------|
| **System prompt** | Environment context, tool strategy, work item guidance, task completion rules | ✅ Yes |
| **First-turn messages** | Planning prompt, tool strategy details, work item protocol | ❌ Can be compacted |
| **Per-turn messages** | Current work item state, incomplete-items reminders | ❌ Per-turn |
| **Tools** | `task_complete`, `work_item_add/update/list`, `work_item_set_artifact/flag_revision` | ✅ Always available |

### Agent Instructions

Keep agent instructions **minimal** — the harness context providers handle detailed
guidance. Verbose instructions duplicate and can conflict with the injected guidance.

```python
instructions = """You are a capable AI coding assistant with access to a local workspace.
You can read and write files, list directories, and run shell commands.
When asked to investigate code, be thorough — read every relevant source file
before drawing conclusions or writing deliverables.
"""
```

### Model Selection

Model choice is the single biggest quality lever:

| Model | Deliverable Quality | Exploration Accuracy |
|-------|-------------------|---------------------|
| **gpt-4.1** (recommended) | 8.5KB, specific class refs, correct focus | Found target code immediately |
| gpt-4o | 3-3.8KB, generic summaries | Often documented wrong packages |
| Claude Sonnet 4.6 (reference) | 29KB, exceptional depth | Systematic multi-pass |

Set via environment variable:
```bash
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4.1
```

### Interactive Entry Points

| Script | Best For | Usage |
|--------|---------|-------|
| `harness_repl.py` | Development, interactive exploration | `uv run python harness_repl.py --sandbox /path` |
| `devui_harness.py` | Visual debugging via web UI | `uv run python devui_harness.py --sandbox /path` |
| `harness_test_runner.py` | Benchmarking, regression testing | `uv run python harness_test_runner.py` |

### Further Reading

- `docs/agent-harness-architecture.md` — Full harness architecture
- `docs/copilot-cli-architecture.md` — GitHub Copilot CLI comparison
- `docs/harness-parity-plan.md` — Multi-phase improvement plan with experiment results
