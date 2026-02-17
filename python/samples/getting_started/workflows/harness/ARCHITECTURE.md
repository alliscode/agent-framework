# Agent Harness: Architecture Deep-Dive

> A comprehensive technical reference for the Agent Harness — the execution control
> plane that turns a bare chat-completion agent into a reliable, long-running
> autonomous system.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Developer Usage](#2-developer-usage)
3. [Key Concepts Deep-Dive](#3-key-concepts-deep-dive)
   - [3.1 The Executor Loop](#31-the-executor-loop)
   - [3.2 Context Compaction](#32-context-compaction)
   - [3.3 Work Item Management](#33-work-item-management)
   - [3.4 Stop Decision & Completion Control](#34-stop-decision--completion-control)
   - [3.5 Task Contracts & Verification](#35-task-contracts--verification)
   - [3.6 Continuation Prompts & Stall Detection](#36-continuation-prompts--stall-detection)
   - [3.7 JIT Instructions](#37-jit-instructions)
   - [3.8 Hooks & Middleware](#38-hooks--middleware)
   - [3.9 Context Providers & System Prompt Construction](#39-context-providers--system-prompt-construction)
   - [3.10 Sub-Agent Delegation](#310-sub-agent-delegation)
   - [3.11 Transcript & Observability](#311-transcript--observability)
   - [3.12 Repair & Invariant Maintenance](#312-repair--invariant-maintenance)

---

## 1. Architecture Overview

### The Problem

LLM chat-completion APIs are stateless, single-turn request/response interfaces. Building
an autonomous agent on top of them means the developer has to solve, for every agent:

- **When to stop.** The model can hallucinate completion, spin in circles, or simply forget.
- **What to remember.** Context windows are finite; long tasks overflow them.
- **How to verify.** "The agent said it's done" is not the same as "it actually did the work."
- **How to recover.** Tool calls can fail, the model can produce unparseable output, or
  the process can crash mid-turn.

The Agent Harness provides a **control plane** that wraps any
`AgentProtocol`-compatible agent and solves all four of these problems through a
composable pipeline of executors.

### Three-Layer Model

The harness separates concerns into three conceptual layers:

```
┌─────────────────────────────────────────────────┐
│                   HARNESS                        │
│  Execution loop, stop decisions, compaction,     │
│  repair, transcript tracking, observability      │
│  ─────────────────────────────────────────────── │
│  Always present. The "operating system" for the  │
│  agent. Developers configure but rarely modify.  │
├─────────────────────────────────────────────────┤
│                  ENVIRONMENT                     │
│  Tools (file I/O, shell, APIs), sandbox path,    │
│  sub-agent delegation                            │
│  ─────────────────────────────────────────────── │
│  Pluggable. Different environments for different │
│  use cases (coding, research, data analysis).    │
├─────────────────────────────────────────────────┤
│                    PERSONA                       │
│  System instructions, task contract, work item   │
│  templates, JIT rules                            │
│  ─────────────────────────────────────────────── │
│  Configurable per-task. Shapes what the agent    │
│  does, not how the infrastructure works.         │
└─────────────────────────────────────────────────┘
```

### The Executor Loop

At runtime the harness is a workflow of four executors connected in a cycle:

```
                    ┌──────────────┐
 RepairTrigger ────►│    Repair    │ ◄─── RepairTrigger (loop back)
                    └──────┬───────┘
                           │ RepairComplete
                           ▼
                    ┌──────────────┐
                    │  Compaction  │  (optional — only when enabled)
                    └──────┬───────┘
                           │ CompactionComplete (extends RepairComplete)
                           ▼
                    ┌──────────────┐
                    │  Agent Turn  │  ◄─── The actual LLM call
                    └──────┬───────┘
                           │ TurnComplete
                           ▼
                    ┌──────────────┐
                    │ Stop Decision│ ───► HarnessResult  (terminal)
                    └──────┬───────┘
                           │ RepairTrigger (continue)
                           └───────────────────────────────┘
```

Each executor communicates through typed messages (`RepairTrigger`, `RepairComplete`,
`TurnComplete`, `HarnessResult`) and reads/writes shared state via `WorkflowContext`.
The loop repeats until the `StopDecisionExecutor` decides to terminate and yields a
final `HarnessResult`.

### Shared State

All executors coordinate through a `SharedState` dictionary accessed via
`ctx.get_shared_state(key)` / `ctx.set_shared_state(key, value)`. Key entries include:

| Key | Type | Owner |
|-----|------|-------|
| `harness.transcript` | `list[HarnessEvent]` | All executors append |
| `harness.turn_count` | `int` | AgentTurnExecutor increments |
| `harness.max_turns` | `int` | RepairExecutor initializes |
| `harness.status` | `str` | StopDecisionExecutor sets |
| `harness.stop_reason` | `dict` | StopDecisionExecutor sets |
| `harness.token_budget` | `dict` | AgentTurnExecutor updates |
| `harness.compaction_plan` | `dict` | CompactionExecutor sets |
| `harness.work_item_ledger` | `dict` | AgentTurnExecutor syncs |
| `harness.task_contract` | `dict` | RepairExecutor initializes |
| `harness.coverage_ledger` | `dict` | StopDecisionExecutor checks |
| `harness.progress_tracker` | `dict` | StopDecisionExecutor updates |

### File Map

```
_harness/
├── __init__.py                  # Public API exports (~115 symbols)
├── _harness_builder.py          # AgentHarness + HarnessWorkflowBuilder
├── _state.py                    # HarnessResult, HarnessStatus, StopReason, events
├── _constants.py                # SharedState keys, default config values
├── _agent_turn_executor.py      # Runs agent, manages cache, applies compaction
├── _stop_decision_executor.py   # 4-layer stop strategy
├── _repair_executor.py          # Fixes dangling tool calls, initializes state
├── _compaction_executor.py      # Detects pressure, orchestrates compaction
├── _context_pressure.py         # TokenBudget v1 (SharedState), estimation
├── _context_providers.py        # System prompt injection (env + guidance)
├── _task_contract.py            # TaskContract, CoverageLedger, predicates
├── _contract_verifier.py        # Evaluates predicates against evidence
├── _work_items.py               # WorkItemLedger, tools, artifact validation
├── _done_tool.py                # work_complete() tool + deprecated aliases
├── _hooks.py                    # Pre-tool, post-tool, agent-stop hooks
├── _jit_instructions.py         # Conditional per-turn guidance injection
├── _renderers.py                # Terminal / markdown event renderers
├── _sub_agents.py               # Explore, task, document sub-agent factories
└── _compaction/                 # Context compaction subsystem
    ├── __init__.py              # Subsystem exports
    ├── _types.py                # CompactionPlan, SpanReference, action records
    ├── _strategies.py           # Clear, Externalize, Summarize, Drop + Coordinator
    ├── _tokenizer.py            # Provider-aware token counting (tiktoken)
    ├── _durability.py           # Tool result durability classification
    ├── _store.py                # CompactionStore, ArtifactStore, SummaryCache
    ├── _summary.py              # StructuredSummary with drift resistance
    ├── _summarizer.py           # ChatClientSummarizer (LLM-backed)
    ├── _adapters.py             # Bridge: executor cache → compaction interface
    ├── _events.py               # CompactionMetrics, event emitter protocol
    ├── _turn_context.py         # Rehydration loop prevention
    └── _durability.py           # ToolDurability enum + default policies
```

---

## 2. Developer Usage

### Minimal Setup (Interactive Mode)

The simplest way to use the harness is through `AgentHarness`:

```python
from agent_framework._harness import AgentHarness

# Create your agent with environment tools only — the harness auto-injects
# work_complete, work item tools, and sub-agent tools at runtime.
agent = chat_client.create_agent(
    instructions="You are a helpful assistant.",
    tools=my_tools,
)

# Wrap in harness
harness = AgentHarness(agent, max_turns=10)

# Run to completion
result = await harness.run("Build a REST API client in Python")
print(result.status)       # HarnessStatus.DONE | STALLED | FAILED
print(result.turn_count)   # Number of turns used
```

> **Note:** The harness automatically injects `work_complete()`, work item management
> tools (when `task_list` is provided), and sub-agent tools (when configured) into
> every agent turn via `run_kwargs["tools"]`. You do **not** need to add these to the
> agent's tool list yourself. Only provide your environment-specific tools (file I/O,
> shell, APIs) when creating the agent.

### Streaming Events

```python
async for event in harness.run_stream("Implement a binary search"):
    if isinstance(event, AgentRunUpdateEvent):
        # Real-time agent text/tool updates
        print(event.data.text, end="", flush=True)
    elif isinstance(event, HarnessLifecycleEvent):
        # Harness control events (compaction, turn changes)
        print(f"[{event.event_type}]")
    elif isinstance(event, WorkflowOutputEvent):
        # Final HarnessResult
        result = event.data
```

### Production Configuration

For production autonomous tasks, enable all subsystems:

```python
from agent_framework._harness import AgentHarness
from agent_framework._harness._work_items import WorkItemTaskList
from agent_framework._harness._compaction import (
    ChatClientSummarizer,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
)

task_list = WorkItemTaskList()
harness = AgentHarness(
    agent,
    max_turns=50,
    # Stall detection
    enable_stall_detection=True,
    stall_threshold=3,
    # Continuation prompts
    enable_continuation_prompts=True,
    max_continuation_prompts=2,
    # Work item self-tracking
    task_list=task_list,
    # Context compaction
    enable_compaction=True,
    compaction_store=InMemoryCompactionStore(),
    artifact_store=InMemoryArtifactStore(),
    summary_cache=InMemorySummaryCache(max_entries=100),
    summarizer=ChatClientSummarizer(chat_client),
    max_input_tokens=128_000,
    soft_threshold_percent=0.85,
    # Environment context
    sandbox_path=str(workspace_dir),
)
```

### Advanced: Custom Workflow

For full control, use `HarnessWorkflowBuilder`:

```python
from agent_framework._harness import HarnessWorkflowBuilder

builder = HarnessWorkflowBuilder(
    agent,
    max_turns=20,
    task_contract=my_contract,
    enable_stall_detection=True,
)
workflow = builder.build()
result = await workflow.run("Do the thing", **builder.get_harness_kwargs())
```

### Conversation History

The harness accumulates conversation history across multiple `run()` / `run_stream()`
calls on the same instance. Each call appends to the internal message cache. To reset,
create a new `AgentHarness` instance.

---

## 3. Key Concepts Deep-Dive

---

### 3.1 The Executor Loop

#### What It Does

The executor loop is the heartbeat of the harness. It decomposes the problem of
"run an agent until it's done" into four independent, testable phases that execute
in a strict cycle: **Repair → (Compaction) → Agent Turn → Stop Decision → repeat**.

Each phase is implemented as an `Executor` — a workflow node that receives a typed
message, does its work, and sends a typed message to the next phase. The workflow
runtime handles scheduling, checkpointing, and event routing.

#### Benefits

- **Separation of concerns:** Each executor owns one responsibility. The agent turn
  executor never decides when to stop. The stop decision executor never touches the
  LLM. The repair executor never runs tools. This makes each piece independently
  testable.
- **Composability:** Executors can be added, removed, or replaced. Compaction is
  optional — without it, the loop is repair → turn → stop. Custom executors (logging,
  progress reporting) can be inserted at any edge.
- **Checkpointability:** Because all inter-executor state flows through `SharedState`,
  the entire harness can be checkpointed and restored at any executor boundary. This
  enables durable execution across process restarts.
- **Observability:** Each executor emits `HarnessLifecycleEvent`s that stream to
  callers in real-time, enabling DevUI overlays, progress bars, and structured logging.

#### What Happens Without It

Without the executor loop, developers must write their own `while True` loop with
ad-hoc checks scattered throughout. Stop logic gets tangled with tool execution.
Error recovery becomes a maze of try/catch blocks. Checkpointing is impossible
because state isn't centralized.

#### Technical Deep-Dive

**Construction (`_harness_builder.py`):**

`HarnessWorkflowBuilder.build()` registers four executor factories with a
`WorkflowBuilder` and wires their edges:

```python
builder.add_edge("repair", "compaction")      # or "repair" → "agent_turn" if no compaction
builder.add_edge("compaction", "agent_turn")
builder.add_edge("agent_turn", "stop_decision")
builder.add_edge("stop_decision", "repair")   # the loop-back
builder.set_start_executor("repair")
```

The `max_iterations` is set to `max_turns × supersteps_per_turn × 2` to ensure the
workflow runtime doesn't terminate the loop prematurely.

**Message Flow:**

```
RepairTrigger ──► RepairExecutor ──► RepairComplete
RepairComplete ──► CompactionExecutor ──► CompactionComplete (extends RepairComplete)
RepairComplete/CompactionComplete ──► AgentTurnExecutor ──► TurnComplete
TurnComplete ──► StopDecisionExecutor ──► RepairTrigger (loop) | HarnessResult (stop)
```

`TurnComplete` carries the agent's completion signal:
- `agent_done: bool` — whether the agent signaled completion
- `called_work_complete: bool` — whether `work_complete()` was explicitly called
- `error: str | None` — any error from the turn

**AgentHarness convenience class:**

`AgentHarness` wraps `HarnessWorkflowBuilder` for callers who don't need custom
executor wiring. It lazily builds the workflow on the first `run()` call and provides
`run_stream()` for event streaming.

---

### 3.2 Context Compaction

#### What It Does

Context compaction keeps the LLM's input within its context window budget by
progressively compressing older parts of the conversation. It uses an
**immutable-log + compaction-plan + renderer** architecture: the full message cache
is never mutated — instead, a `CompactionPlan` describes how to transform it into a
smaller view for the next LLM call.

#### Benefits

- **Enables long-running tasks.** Without compaction, a 50-turn coding task with
  `read_file` results would overflow a 128K context window in under 15 turns.
- **Preserves important context.** The strategy ladder applies increasingly aggressive
  compression, starting with clearing ephemeral tool results and ending with dropping
  content entirely — so the most important information is kept longest.
- **Full audit trail.** Because the original cache is never mutated, the complete
  conversation history remains available for debugging, evaluation, and replay.
- **Avoids information loss spirals.** Structured summaries resist drift across
  compaction cycles by organizing facts into semantic categories (decisions,
  open items, tool outcomes) rather than raw text.

#### What Happens Without It

After ~10-15 turns of active tool use, the context window fills up. The LLM either
receives truncated input (losing early context) or the API returns a token limit error.
In either case, the agent loses track of what it was doing, what files it read, and
what decisions it made.

#### Technical Deep-Dive

**Token Budget (`_compaction/_tokenizer.py`, `_context_pressure.py`):**

Two `TokenBudget` classes coexist:
- **v1** (`_context_pressure.py`): Simple dataclass stored in SharedState. Uses a rough
  `1 token ≈ 4 chars` estimate. Drives the CompactionExecutor's pressure detection.
- **v2** (`_compaction/_tokenizer.py`): Provider-aware, tiktoken-backed. Accounts for
  system prompt overhead, tool schema tokens, and safety buffers. Used by strategies
  for precise token counting.

Budget thresholds:
- **Soft threshold (default 80%):** Triggers proactive compaction during the normal loop.
- **Blocking threshold (95%):** Triggers emergency compaction; the turn cannot proceed
  until context is reduced.

After every agent turn, `AgentTurnExecutor._update_token_budget()` updates the budget
from the API's `usage_details.input_token_count` — this is the authoritative count.

**The CompactionPlan (`_compaction/_types.py`):**

A `CompactionPlan` is a serializable data structure that maps message IDs to actions:

```python
class CompactionPlan:
    thread_id: str
    version: int
    actions: dict[str, CompactionAction]    # message_id → INCLUDE/CLEAR/SUMMARIZE/EXTERNALIZE/DROP
    clear_records: list[ClearRecord]
    summarization_records: list[SummarizationRecord]
    externalization_records: list[ExternalizationRecord]
    drop_records: list[DropRecord]
```

The plan is pure data — it does not hold references to messages or stores. It can be
serialized to SharedState, checkpointed, and rehydrated without any external dependencies.

**Strategy Ladder (`_compaction/_strategies.py`):**

Four strategies are applied in order from least to most aggressive:

| # | Strategy | Aggressiveness | What It Does | Content Preserved? |
|---|----------|---------------|--------------|-------------------|
| 1 | **Clear** | 1 (Lowest) | Replace tool results with 1-line placeholders | Key fields only (via durability policy) |
| 2 | **Externalize** | 2 | Write large non-replayable content to `ArtifactStore`, keep pointer + summary | Full content in store |
| 3 | **Summarize** | 3 | LLM-compress message spans into `StructuredSummary` | Semantic summary only |
| 4 | **Drop** | 4 (Highest) | Remove cleared content entirely from prompt | No — content is gone |

> **Note:** `CompactionCoordinator` sorts strategies by their `aggressiveness` property
> at construction time, so the order they're appended doesn't matter — runtime order
> is always determined by the aggressiveness values above.

The `CompactionCoordinator` runs strategies sequentially, checking after each whether
the budget is satisfied. It stops as soon as enough tokens are freed.

**Tool Durability (`_compaction/_durability.py`):**

Not all tool results are equal. A `read_file` result can be re-read (REPLAYABLE), but a
`web_search` result cannot be reproduced (NON_REPLAYABLE). The durability system
classifies tool results so ClearStrategy knows what's safe to compress:

| Durability | Meaning | Clear Behavior |
|-----------|---------|---------------|
| EPHEMERAL | Can be dropped freely (e.g., `list_directory`) | Clear immediately |
| ANCHORING | Important but compressible (preserve key fields) | Clear with preserved fields |
| REPLAYABLE | Can be re-executed to reproduce (e.g., `read_file`) | Clear; note tool + args |
| NON_REPLAYABLE | Cannot be reproduced (e.g., API calls) | Externalize first, then clear |

Default policies are defined for common tools in `DEFAULT_DURABILITY_POLICIES`.

**Structured Summaries (`_compaction/_summary.py`):**

When `SummarizeStrategy` compresses a span, it produces a `StructuredSummary`:

```python
class StructuredSummary:
    facts: list[str]              # Verified facts discovered
    decisions: list[Decision]      # Decisions made (with rationale)
    open_items: list[OpenItem]     # TODOs still pending
    tool_outcomes: list[ToolOutcome]  # What tools returned
    artifact_refs: list[ArtifactReference]  # Pointers to externalized content
    current_task: str              # What the agent is currently doing
    current_plan: str              # Agent's stated plan
```

This structure resists **summary drift** — the tendency for iterative summarization
to gradually lose critical details. By organizing information into semantic categories,
each compaction cycle can merge structured data rather than re-summarizing prose.

**Adapter Layer (`_compaction/_adapters.py`):**

The compaction subsystem operates on `ChatMessageStore` / `AgentThread` protocols, but
`AgentTurnExecutor` keeps messages in a plain list (`self._cache`). `CacheMessageStore`
and `CacheThreadAdapter` bridge these two worlds without modifying the executor's cache
structure. `CacheMessageStore.list_messages()` auto-assigns stable `cache-{uuid}` IDs
to messages that lack them.

**Applying the Plan (`_agent_turn_executor.py`):**

When a `CompactionPlan` exists, `AgentTurnExecutor._apply_compaction_plan()` transforms
the cache into the view the LLM will see:

1. For each message, look up the action in the plan.
2. `INCLUDE` → pass through unchanged.
3. `CLEAR` → replace with a 1-line placeholder from the `ClearRecord`.
4. `SUMMARIZE` → replace the entire span with the `StructuredSummary` rendered as text.
5. `EXTERNALIZE` → replace with an artifact reference and brief summary.
6. `DROP` → omit entirely.

The full cache is never mutated — only the view passed to `agent.run()` changes.

**Fallback: Direct Clear (`_agent_turn_executor.py`):**

If the full compaction pipeline produces no proposals (e.g., all content is already
compacted), `_apply_direct_clear()` falls back to clearing the oldest tool results
in the cache. It targets freeing 50% of the current budget to avoid re-triggering
compaction on the very next turn.

---

### 3.3 Work Item Management

#### What It Does

Work item management gives the agent a structured self-tracking system. The agent
decomposes its task into subtasks (work items), tracks their status, stores
artifacts (deliverables), and flags items for revision. This turns unstructured
"I'll do the thing" into an auditable, verifiable plan.

#### Benefits

- **Decomposition discipline.** Agents that plan in work items produce better results
  because the decomposition forces them to think about scope before diving in.
- **Progress visibility.** The harness can track `completion_percentage` at any time
  and display it in the UI.
- **Artifact capture.** Each work item can hold an artifact (file content, analysis
  report, etc.) classified by role: DELIVERABLE, WORKING, or CONTROL.
- **Self-critique loop.** The `work_item_flag_revision` tool lets the agent say "this
  isn't good enough" and cycle back — without external intervention.
- **Stop verification.** The `StopDecisionExecutor` can verify that all work items
  are complete before accepting the agent's completion signal.

#### What Happens Without It

The agent operates as a stream-of-consciousness executor. There's no way to know what
subtasks it plans to do, which are done, or whether it forgot something. The harness
can only check "did the agent call work_complete?" — not "did it actually finish all
the pieces?"

#### Technical Deep-Dive

**WorkItemLedger (`_work_items.py`):**

The core data structure is `WorkItemLedger` — an ordered dictionary of `WorkItem`
objects keyed by ID:

```python
class WorkItem:
    id: str
    title: str
    status: WorkItemStatus        # PENDING | IN_PROGRESS | DONE | SKIPPED
    priority: WorkItemPriority    # HIGH | MEDIUM | LOW
    notes: str
    artifact: str | None          # Stored deliverable content
    artifact_role: ArtifactRole   # DELIVERABLE | WORKING | CONTROL
    requires_revision: bool
    revision_of: str | None       # Links to original item
```

**Agent-Facing Tools:**

`WorkItemTaskList.get_tools()` returns five `@ai_function` tools:

1. **`work_item_add(id, title, priority)`** — Create a new subtask.
2. **`work_item_update(id, status, notes)`** — Update status/notes.
3. **`work_item_list()`** — View all items and their status.
4. **`work_item_set_artifact(id, content, role)`** — Store deliverable content.
5. **`work_item_flag_revision(id, reason)`** — Mark an item as needing rework.

These tools are injected alongside the agent's environment tools so the LLM uses them
naturally during execution.

**Artifact Validation (`validate_artifact_content()`):**

When the agent stores an artifact via `work_item_set_artifact`, the content is validated
to detect **narration contamination** — when the agent wraps the actual deliverable in
meta-commentary ("Here's the file I created:", "I'll now store this artifact:"). The
validator checks the first and last 5 lines for boundary narration patterns and
auto-strips them, returning a `ArtifactContaminationLevel` (CLEAN, LIGHT, HEAVY).

**Event Middleware (`WorkItemEventMiddleware`):**

`WorkItemEventMiddleware` is a `FunctionMiddleware` that intercepts work item tool calls
and queues `HarnessLifecycleEvent`s for real-time UI updates. When
`work_item_set_artifact` is called, a `deliverables_updated` event is emitted with the
item ID, title, and role.

**Ledger Sync:**

After each agent turn, `AgentTurnExecutor._sync_work_item_ledger()` serializes the
ledger to SharedState (`HARNESS_WORK_ITEM_LEDGER_KEY`), making it available to the
`StopDecisionExecutor` for completion verification.

---

### 3.4 Stop Decision & Completion Control

#### What It Does

The `StopDecisionExecutor` implements a layered strategy for deciding whether to
terminate the harness loop or continue for another turn. It is the single arbiter
of "are we done?" — no other executor can terminate the loop.

#### Benefits

- **Prevents premature termination.** The agent saying "I'm done" is necessary but
  not sufficient — the harness verifies contracts, work items, and progress before
  accepting.
- **Prevents infinite loops.** Hard stops (max turns, errors) and stall detection
  guarantee the harness always terminates.
- **Extensible via hooks.** The `agent_stop` hook lets external code block termination
  with a reason (e.g., "you forgot to run the tests").
- **Explicit completion signal.** The `work_complete()` tool creates an unambiguous
  "I'm done" signal, avoiding ambiguity from the model simply stopping output.

#### What Happens Without It

The harness would have to guess whether the agent is done based on whether it made tool
calls. This fails in both directions: the agent might stop making tool calls because
it's confused (false positive), or it might call `work_complete` but forget a critical
subtask (false negative).

#### Technical Deep-Dive

**Four-Layer Strategy:**

The layers are evaluated in order; the first match wins:

**Layer 1 — Hard Stops (workflow-owned):**
- Agent turn raised an error → `FAILED`
- `turn_count >= max_turns` → `DONE` with reason "max_turns"

**Layer 2 — Stall Detection:**
- Compute `ProgressFingerprint` from transcript length + ledger state + work items
- If fingerprint unchanged for `stall_threshold` consecutive turns → `STALLED`

**Layer 3 — Verified Completion:**
When `turn_result.agent_done == True`:
1. Verify `work_complete()` was called (with retry: up to `work_complete_max_retries`
   times, then accept anyway to prevent infinite loops).
2. Verify task contract satisfied (if enabled) — delegates to `ContractVerifier`.
3. Verify all work items complete (if enabled) — checks `WorkItemLedger.is_all_complete()`.
4. Run `agent_stop` hooks — any hook returning `"block"` rejects the stop.
5. If all pass → `DONE`.

**Layer 4 — Default Continuation:**
- If no stop condition matched → send `RepairTrigger` to loop back.

**The `work_complete()` Tool (`_done_tool.py`):**

```python
@ai_function(name="work_complete", approval_mode="never_require")
def work_complete(summary: str) -> str:
    """Indicate that all requested work is done and no further actions remain."""
    return f"Work recorded as complete: {summary}"
```

This tool is auto-injected by the harness. When the agent calls it, `AgentTurnExecutor`
detects the call in the response and sets `called_work_complete=True` on the
`TurnComplete` message.

**work_complete Retry Logic:**

If `require_work_complete=True` (default) and the agent signals done without calling
`work_complete()`, the stop decision executor rejects the completion and sends a
`RepairTrigger` with guidance. After `work_complete_max_retries` (default 3) rejections,
it accepts the stop anyway to prevent infinite loops.

---

### 3.5 Task Contracts & Verification

#### What It Does

Task contracts define formal completion criteria — what files must exist, what
text must appear in the output, what predicates must be satisfied. The
`ContractVerifier` evaluates these predicates against evidence (tool results,
artifacts, transcript) and updates a `CoverageLedger` that tracks which
requirements are met.

#### Benefits

- **Objective completion criteria.** Instead of trusting the agent's self-assessment,
  the harness checks verifiable conditions ("does `output.py` exist?", "does the
  test output contain 'PASSED'?").
- **Gap reporting.** When the contract isn't satisfied, a `GapReport` tells the agent
  exactly what's missing — enabling targeted remediation instead of vague retries.
- **Automation mode.** Task contracts enable fully unattended execution where the
  harness loop continues until all requirements are provably met.
- **Evaluation.** Contracts provide a structured signal for benchmark evaluation —
  "3 of 5 requirements met" is more useful than "the agent said done."

#### What Happens Without It

The harness falls back to **interactive mode**: the agent's `work_complete()` call is
trusted as the sole completion signal. This works for interactive use but is insufficient
for automation where you need guarantees.

#### Technical Deep-Dive

**TaskContract (`_task_contract.py`):**

```python
class TaskContract:
    id: str
    goal: str                                # Human-readable goal description
    required_outputs: list[RequiredOutput]    # What must be produced
    questions: list[UserQuestion]             # Clarifications to ask if needed
    acceptability: AcceptabilityCriteria      # How strict to be
```

Each `RequiredOutput` has a `Predicate` that defines how to verify it:

| PredicateType | What It Checks |
|--------------|---------------|
| `FILE_EXISTS` | A file exists at the specified path |
| `CONTAINS_TEXT` | A regex pattern appears in transcript or output |
| `TOOL_RESULT_SUCCESS` | A specific tool call succeeded |
| `JSON_SCHEMA_VALID` | Output matches a JSON schema |
| `CUSTOM` | Arbitrary Python callable |
| `ALWAYS_TRUE` | Soft requirement (model judgment) |

**CoverageLedger:**

Tracks status per requirement:
```python
class RequirementCoverage:
    requirement_id: str
    status: RequirementStatus    # UNMET | MET | PARTIALLY_MET | SKIPPED
    evidence: list[Evidence]     # Proof of satisfaction
    notes: str
```

**ContractVerifier (`_contract_verifier.py`):**

`ContractVerifier.verify_contract()` evaluates all predicates, updates the ledger with
evidence, and returns a `ContractVerificationResult` with `satisfied: bool` and
`unmet_requirements: list[str]`.

**GapReport:**

When the contract is not satisfied, `GapReport.from_contract_and_ledger()` generates a
structured report of unmet requirements and unanswered questions, which is injected
into the agent's next turn to guide remediation.

**ProgressFingerprint:**

`ProgressFingerprint.compute()` hashes the transcript length, ledger state, and work
item status into a single string. The `ProgressTracker` stores fingerprints per turn
and detects stalls when N consecutive fingerprints are identical.

---

### 3.6 Continuation Prompts & Stall Detection

#### What It Does

**Continuation prompts** handle the case where the agent stops generating output without
calling `work_complete()` or making tool calls — a common failure mode where the model
simply "gives up" mid-task. The harness injects a nudge message to get the agent back
on track.

**Stall detection** catches the higher-level failure mode where the agent is making
tool calls but not making progress — reading the same files repeatedly, producing
identical outputs, or looping without advancing.

#### Benefits

- **Recovers from attention failures.** LLMs sometimes stop mid-task because of
  attention drift, not because they're done. A well-crafted continuation prompt
  recovers the majority of these cases.
- **Prevents infinite spinning.** Stall detection catches circular behavior that
  continuation prompts can't fix, providing a clean exit with a `STALLED` status.
- **Configurable strictness.** `stall_threshold` controls how many stagnant turns
  are tolerated. `max_continuation_prompts` caps the number of nudges.

#### What Happens Without Continuation Prompts

The agent stops, the harness interprets silence as "done," and the task terminates
prematurely — often with work half-finished. The user gets an incomplete result with
no indication that the agent simply lost focus.

#### What Happens Without Stall Detection

The harness runs for all `max_turns` even when the agent is clearly stuck, wasting
tokens and time. The user has to manually inspect the transcript to realize the agent
was going in circles.

#### Technical Deep-Dive

**Continuation Prompt Injection (`_agent_turn_executor.py`):**

When the agent's response has no tool calls and `work_complete()` wasn't called:

1. Check `continuation_count < max_continuation_prompts`.
2. Inject the continuation prompt as a user message into the cache.
3. Set `agent_done=False` on `TurnComplete` so the loop continues.
4. Increment `continuation_count` in SharedState.

The default continuation prompt is carefully worded to prevent the agent from repeating
itself:

```
You stopped without calling work_complete or making tool calls.
IMPORTANT: Do NOT repeat, rephrase, or summarize anything you already said.
Choose exactly one action:
1. If your task is complete → call work_complete now.
2. If work remains → make the next tool call immediately.
3. If something failed → try a different approach.
Do not narrate. Do not restate your findings. Just act.
```

**Stall Detection (`_stop_decision_executor.py`):**

After each turn, `StopDecisionExecutor._check_stall()`:
1. Computes a `ProgressFingerprint` from:
   - Transcript event count
   - Coverage ledger state (if contracts enabled)
   - Work item completion percentage (if work items enabled)
2. Appends fingerprint to `ProgressTracker`.
3. Calls `tracker.is_stalled()` which returns `True` if the last `stall_threshold`
   fingerprints are identical.

---

### 3.7 JIT Instructions

#### What It Does

JIT (Just-In-Time) instructions are conditional rules that inject guidance into the
agent's message stream based on execution state. They fire when specific patterns are
detected — too many reads without writes, turn limit approaching, all planning with
no execution — and provide targeted course-correction.

#### Benefits

- **Adaptive guidance.** Rather than front-loading all possible instructions into the
  system prompt (wasting tokens), JIT instructions fire only when relevant.
- **Pattern detection.** Catches common anti-patterns (serial reads, excessive planning)
  that waste turns, and nudges the agent toward better behavior.
- **One-shot rules.** Some instructions fire only once (e.g., "start producing
  deliverables") to avoid nagging the agent on every turn.
- **Extensible.** Developers can add custom `JitInstruction` rules for domain-specific
  patterns.

#### What Happens Without It

The agent receives the same static instructions regardless of how it's behaving. If
it falls into an anti-pattern (e.g., reading files one-at-a-time when it could batch),
nothing corrects it until the stall detector eventually fires — which may be too late.

#### Technical Deep-Dive

**JitInstruction (`_jit_instructions.py`):**

```python
class JitInstruction:
    id: str                              # Unique identifier
    instruction: str | Callable          # Text or dynamic generator
    condition: Callable[[JitContext], bool]  # When to fire
    once: bool                           # Fire once or every matching turn
```

**JitContext:**

A snapshot of execution state passed to each rule's condition:

```python
class JitContext:
    turn: int
    max_turns: int
    tool_usage: dict[str, int]           # Tool name → call count
    work_items_complete: int
    work_items_total: int
    compaction_count: int
```

**Built-in Rules (7):**

1. **No reads after 3 turns** — "Start reading code before making assumptions."
2. **Serial reads detected** — "Use parallel tool calls to read multiple files."
3. **Many reads, no deliverable** — "Start producing output."
4. **Turn limit approaching** — "You have N turns left. Prioritize completion."
5. **All planning, no execution** — "Stop planning and start doing."
6. **Post-compaction guidance** — "Context was compacted. Check work items for state."
7. **Repeated compaction warning** — "Complete one work item at a time to stay within budget."

**Evaluation Flow:**

`AgentTurnExecutor._inject_jit_instructions()` builds a `JitContext`, passes it to
`JitInstructionProcessor.evaluate()`, and appends any triggered instructions as a
user message in the cache. The `_fired` set tracks which `once`-rules have already
fired.

---

### 3.8 Hooks & Middleware

#### What It Does

The hooks system provides three interception points where external code can observe or
control harness execution: **pre-tool** (before a tool runs), **post-tool** (after it
returns), and **agent-stop** (when the agent tries to finish).

#### Benefits

- **Safety gates.** A `pre_tool` hook can deny dangerous operations (e.g., blocking
  `rm -rf /`), returning a denial reason to the agent instead.
- **Audit logging.** `post_tool` hooks capture every tool invocation for compliance.
- **Quality enforcement.** An `agent_stop` hook can inspect the work and block
  completion ("You haven't run the tests yet").
- **Non-invasive.** Hooks are callbacks — they don't require subclassing executors or
  modifying the agent.

#### What Happens Without It

The only control points are the executor parameters. To add custom validation at
tool-call time, you'd need to wrap every tool function individually or subclass
`AgentTurnExecutor`.

#### Technical Deep-Dive

**HarnessHooks (`_hooks.py`):**

```python
class HarnessHooks:
    pre_tool: list[PreToolHook]       # Called before tool execution
    post_tool: list[PostToolHook]     # Called after tool execution
    agent_stop: list[AgentStopHook]   # Called when agent signals done
```

**HarnessToolMiddleware:**

A `FunctionMiddleware` implementation that intercepts tool invocations. On each tool
call:
1. Runs all `pre_tool` hooks. If any returns `ToolHookResult(decision="deny")`, the
   tool call is short-circuited and the denial reason is returned to the agent.
2. Calls `next()` to execute the actual tool.
3. Runs all `post_tool` hooks with the result.

**Agent-Stop Hooks:**

When `StopDecisionExecutor` evaluates a `TurnComplete` with `agent_done=True`, it
calls `_run_agent_stop_hooks()`. Each hook receives an `AgentStopEvent` (turn count,
tool usage stats, whether `work_complete` was called) and returns an `AgentStopResult`.
If any returns `decision="block"`, the stop is rejected and the loop continues.

---

### 3.9 Context Providers & System Prompt Construction

#### What It Does

Context providers inject persistent guidance into the agent's system prompt — information
about the environment (working directory, OS, files present) and harness guidance
(response style, work completion instructions, sub-agent usage). This content is
protected from compaction because it's injected at the system prompt level, not as
conversation messages.

#### Benefits

- **Environment awareness.** The agent knows its sandbox path, OS, and initial directory
  contents without needing a tool call.
- **Consistent guidance.** Response style instructions and tool usage best practices are
  always present regardless of how much conversation history has been compacted.
- **Composable.** Multiple providers can be stacked via `AggregateContextProvider`,
  preserving any user-configured providers alongside harness providers.

#### What Happens Without It

The agent operates blind — it doesn't know its working directory, OS, or what tools
are available. It wastes early turns discovering basic environmental facts.

#### Technical Deep-Dive

**EnvironmentContextProvider (`_context_providers.py`):**

Injected when `sandbox_path` is provided. Returns a `Context` containing:
- Current working directory path
- Operating system
- Directory listing (first `max_entries` files, 2 levels deep)

**HarnessGuidanceProvider:**

Assembles up to three guidance sections based on configuration flags:
- `RESPONSE_STYLE_GUIDANCE` — How to format responses, avoid narration.
- `WORK_COMPLETION_INSTRUCTIONS` — How to use work items, when to call `work_complete`.
- `SUB_AGENT_GUIDANCE` — When and how to delegate to sub-agents.

**Wiring (`_harness_builder.py`):**

`_wire_context_providers()` appends harness providers to the agent's `context_provider`
property. If the agent already has a `context_provider`, it wraps everything in an
`AggregateContextProvider` to preserve the original.

---

### 3.10 Sub-Agent Delegation

#### What It Does

Sub-agent delegation creates specialized `ChatAgent` tools that the main agent can
invoke for specific types of work: fast codebase exploration, command execution,
and deep documentation writing.

#### Benefits

- **Specialization.** Each sub-agent has a focused system prompt optimized for its
  task type. The exploration sub-agent emphasizes speed and parallelism; the document
  sub-agent emphasizes depth and accuracy.
- **Context isolation.** Sub-agents run in their own context window, so their
  intermediate steps don't consume the main agent's context budget.
- **Parallel execution.** The main agent can invoke sub-agents as regular tool calls,
  potentially in parallel with other tool calls.

#### What Happens Without It

The main agent must handle all tasks in its single context window. Deep file exploration
consumes many turns and fills the context. Documentation generation interleaves with
coding work, making both harder to track.

#### Technical Deep-Dive

**Factory Functions (`_sub_agents.py`):**

| Factory | Sub-Agent Purpose | Key System Prompt Features |
|---------|------------------|--------------------------|
| `create_explore_tool()` | Codebase Q&A | Answers < 300 words, maximizes parallel tool calls |
| `create_task_tool()` | Command execution | Brief success / verbose error reporting |
| `create_document_tool()` | Technical writing | Deep reading, thorough analysis, ASCII diagrams |

Each factory:
1. Creates a `ChatAgent` wrapping the provided `ChatClientProtocol`.
2. Equips it with the same environment tools as the main agent.
3. Wraps the agent's `run()` method as an `@ai_function` tool.
4. Returns the tool for injection into the main agent.

---

### 3.11 Transcript & Observability

#### What It Does

The harness maintains a complete transcript of execution events in SharedState and
streams real-time lifecycle events to callers.

#### Benefits

- **Debugging.** The transcript contains every turn start, agent response, tool call,
  repair action, stall detection, and stop decision — providing a complete execution
  trace.
- **Evaluation.** Structured `HarnessResult` with `status`, `reason`, `turn_count`,
  and `deliverables` enables automated benchmarking.
- **Real-time UI.** `HarnessLifecycleEvent`s stream through the workflow event system,
  enabling progress bars, compaction notifications, and deliverable updates in DevUI.
- **OpenTelemetry.** The harness is instrumented with spans for agent calls, compaction,
  and turn execution, enabling distributed tracing.

#### What Happens Without It

The developer receives only the final agent response with no insight into how the
harness arrived there. Debugging requires adding ad-hoc logging to every component.

#### Technical Deep-Dive

**HarnessEvent (`_state.py`):**

Transcript entries recorded in SharedState:

```python
class HarnessEvent:
    event_type: str    # turn_start, agent_response, tool_call, repair, stall_detected, ...
    data: dict         # Event-specific payload
    event_id: str      # UUID
    timestamp: str     # ISO 8601
```

**HarnessLifecycleEvent (`_state.py`):**

Real-time streaming events via `ctx.add_event()`:

```python
class HarnessLifecycleEvent(WorkflowEvent):
    event_type: str        # harness_started, turn_started, turn_completed,
                           # continuation_prompt, stall_detected, context_pressure,
                           # compaction_started, compaction_completed,
                           # deliverables_updated, work_item_changed,
                           # harness_completed
    turn_number: int
    max_turns: int
    data: dict
```

**HarnessResult (`_state.py`):**

Final output yielded when the loop terminates:

```python
class HarnessResult:
    status: HarnessStatus              # DONE | STALLED | FAILED
    reason: StopReason | None          # Why we stopped (kind, message, details)
    transcript: list[HarnessEvent]     # Full event transcript
    turn_count: int
    deliverables: list[dict[str, Any]] # Work item artifacts marked as DELIVERABLE
                                       # Each dict: {item_id, title, content}
```

**Renderers (`_renderers.py`):**

The `HarnessRenderer` protocol and its implementations (`MarkdownRenderer`,
`PassthroughRenderer`) transform lifecycle events into formatted text for different
display targets. `render_stream()` wraps `harness.run_stream()` and injects synthetic
text events from renderer callbacks.

**OpenTelemetry Integration:**

`AgentTurnExecutor` creates spans for:
- `harness.agent_call` — each LLM invocation with turn number, cache size, token counts
- `harness.compaction` — compaction runs with tokens before/after/freed and duration

---

### 3.12 Repair & Invariant Maintenance

#### What It Does

The `RepairExecutor` runs before every agent turn to fix transcript invariants and
initialize harness state. Its primary job is handling **dangling tool calls** — tool
invocations that were started but never completed (e.g., due to a process crash or
timeout).

#### Benefits

- **Crash recovery.** If the harness is checkpointed and restored after a mid-turn
  crash, dangling tool calls would confuse the LLM. The repair executor inserts
  synthetic error results so the conversation remains coherent.
- **One-time initialization.** The repair executor loads harness configuration from
  workflow kwargs and initializes all SharedState keys on the first turn, providing
  a clean separation between configuration and execution.
- **Invariant guarantee.** Every agent turn is preceded by a repair pass, ensuring
  the transcript is always in a valid state regardless of what happened before.

#### What Happens Without It

After a crash or timeout, the LLM receives a conversation with tool calls that have
no results. Some models handle this gracefully; others hallucinate tool results or
get confused and loop. Without repair, durable execution across process restarts is
unreliable.

#### Technical Deep-Dive

**RepairExecutor (`_repair_executor.py`):**

On each invocation (receiving `RepairTrigger`):

1. **First run only:** Load configuration from `workflow_kwargs`:
   - `max_turns`, `initial_message`, `task_contract`
   - Initialize SharedState: transcript (empty list), turn count (0), status (RUNNING),
     empty pending tool calls, token budget, coverage ledger, progress tracker.
   - Emit `harness_started` lifecycle event.

2. **Every run:** Load `HARNESS_PENDING_TOOL_CALLS_KEY` from SharedState:
   - For each `PendingToolCall`, insert a synthetic repair event into the transcript
     with details of which tool was interrupted.
   - Clear the pending tool calls list.

3. Send `RepairComplete(repairs_made=count)` to the next executor.

**PendingToolCall Tracking:**

`AgentTurnExecutor` records pending tool calls in SharedState before tool execution
begins. If the process crashes mid-execution, the next `RepairExecutor` invocation
finds them and generates cleanup events.

---

## Summary

The Agent Harness transforms a stateless chat-completion agent into a durable,
self-monitoring, long-running autonomous system through twelve interlocking subsystems:

| # | Subsystem | One-Line Purpose |
|---|-----------|-----------------|
| 1 | Executor Loop | Four-phase cycle: repair → compact → turn → decide |
| 2 | Context Compaction | Keep conversation within token budget via strategy ladder |
| 3 | Work Items | Agent self-decomposes tasks, tracks status, stores artifacts |
| 4 | Stop Decision | Four-layer strategy ensures proper termination |
| 5 | Task Contracts | Formal, predicate-based completion verification |
| 6 | Continuation/Stall | Recover from attention failures; detect circular behavior |
| 7 | JIT Instructions | Conditional guidance based on execution patterns |
| 8 | Hooks & Middleware | Intercept tools and stop decisions for safety/quality |
| 9 | Context Providers | Inject environment and guidance into system prompts |
| 10 | Sub-Agents | Delegate specialized work to focused child agents |
| 11 | Transcript | Complete event log + real-time streaming + OTEL |
| 12 | Repair | Fix invariants and handle crash recovery |

Each subsystem is independently toggleable, testable, and replaceable — the harness
scales from a 5-line "just run the agent" to a fully instrumented production pipeline.
