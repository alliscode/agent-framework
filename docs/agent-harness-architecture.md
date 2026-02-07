# Agent Harness Internal Architecture

> Analysis of the Agent Harness framework in `python/packages/core/agent_framework/_harness/`.
> This document mirrors the structure of `docs/copilot-cli-architecture.md` for easy comparison.

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [System Prompt Architecture](#system-prompt-architecture)
3. [Agentic Loop — Workflow-Based](#agentic-loop--workflow-based)
4. [Quality Enforcement Mechanisms](#quality-enforcement-mechanisms)
5. [Compaction / Context Management](#compaction--context-management)
6. [Token Counting](#token-counting)
7. [Sub-Agent System](#sub-agent-system)
8. [Custom Instructions & Skills](#custom-instructions--skills)
9. [Hooks System](#hooks-system)
10. [Tool Permission & Filtering](#tool-permission--filtering)
11. [JIT (Just-In-Time) Instructions](#jit-just-in-time-instructions)
12. [Feature Flags](#feature-flags)
13. [Comparison with Copilot CLI](#comparison-with-copilot-cli)

---

## Package Structure

```
python/packages/core/agent_framework/_harness/
├── __init__.py                    # 424 lines — massive public API surface (229 exports)
├── _agent_turn_executor.py        # ~1100 lines — core turn execution, message injection, cache
├── _compaction_executor.py        # 395 lines — compaction orchestrator (plan pipeline)
├── _constants.py                  # 41 lines — SharedState keys, default values
├── _context_pressure.py           # ~500 lines — v1 token budget, legacy strategies
├── _context_pressure_executor.py  # ~200 lines — legacy context pressure executor
├── _contract_verifier.py          # ~150 lines — task contract verification
├── _done_tool.py                  # 37 lines — task_complete tool definition
├── _harness_builder.py            # 518 lines — AgentHarness + HarnessWorkflowBuilder
├── _renderers.py                  # ~200 lines — output rendering (markdown, passthrough)
├── _repair_executor.py            # 207 lines — transcript invariant repair
├── _state.py                      # 270 lines — state types (events, status, messages)
├── _stop_decision_executor.py     # 489 lines — layered stop condition evaluation
├── _task_contract.py              # ~400 lines — formal task contracts (Phase 3)
├── _work_items.py                 # ~900 lines — work item tracking, artifact validation
├── _compaction/                   # Production compaction subsystem (Phase 9)
│   ├── __init__.py
│   ├── _durability.py             # Tool durability policies
│   ├── _events.py                 # Compaction event types
│   ├── _rehydration.py            # Content rehydration from artifacts
│   ├── _renderer.py               # Prompt rendering with compaction
│   ├── _store.py                  # Storage protocols (InMemory, pluggable)
│   ├── _strategies.py             # Clear, Summarize, Externalize, Drop strategies
│   ├── _summary.py                # LLM-based summarization
│   ├── _tokenizer.py              # TiktokenTokenizer, TokenBudgetV2
│   ├── _turn_context.py           # Turn-level context for compaction
│   └── _types.py                  # Core compaction types (Plan, Proposal, etc.)
├── AGENT_HARNESS.md               # Design documentation
├── CONTEXT_COMPACTION_DESIGN.md   # Compaction design doc
└── COMPACTION_TOKEN_COUNTING_ISSUES.md  # Token counting bug analysis

Samples/Tests:
python/samples/getting_started/workflows/harness/
├── devui_harness.py               # Reference implementation for DevUI integration
├── harness_test_runner.py         # Automated test runner for experiments
├── harness_repl.py                # Interactive REPL-style harness runner
├── coding_tools.py                # File/directory/command tools for coding tasks
├── evaluate_baseline.py           # Baseline evaluation
├── evaluate_harness_output.py     # Output evaluation
└── harness_coding_test.py         # Coding integration test
```

---

## System Prompt Architecture

The harness does NOT construct a system prompt itself. The system prompt comes from the `agent` passed to the harness (typically a `ChatAgent` with `instructions`). The harness injects supplementary messages into the conversation cache as `role: "user"` messages.

### Injected Messages (Turn 1)

On the first turn, the harness injects up to 3 user messages into the cache before the user's actual message:

**1. Tool Strategy Guidance** (always injected):

```python
TOOL_STRATEGY_GUIDANCE = """
TOOL STRATEGY GUIDE:
Use these patterns for effective investigation and task execution.

DISCOVERY — find what's relevant before reading:
- run_command('find . -name "*.py" -path "*/target_dir/*"') to locate files
- run_command('grep -r "pattern" path/ --include="*.py" -l') to find files containing a term
- run_command('grep -rn "class\\|def " file.py | head -40') for file structure overview

PROGRESSIVE EXPLORATION — go broad then deep:
1. list_directory('.') to understand top-level layout
2. list_directory on promising subdirectories
3. Keep drilling into subdirectories until you reach actual source files.
4. Use grep/find to narrow down which files matter most
5. read_file on each relevant file — one at a time, in full

THOROUGH READING — depth matters more than speed:
- Read EVERY relevant source file individually.
- If a directory has 5-15 source files, plan to read each one.
- After reading, note specific class names, methods, and patterns.

DELIVERABLE CREATION — only after thorough investigation:
- Do NOT write deliverables until you have read all relevant source files.
- Use write_file to produce deliverable documents as real files.
- Reference specific class names, method signatures you found by reading.
"""
```

**2. Work Item Guidance** (when `enable_work_items=True`):

A massive ~3KB prompt covering:
- Tool usage reference (work_item_add, work_item_update, work_item_list, etc.)
- Artifact & revision protocol
- 5-step workflow: Plan → Execute → Audit → Revise → Verify
- Artifact content rules (no narrative contamination)
- Artifact roles (deliverable, working, control)
- Control artifact JSON format
- Anti-double-emission rules

**3. Planning Prompt** (when `enable_work_items=True`):

```python
PLANNING_PROMPT = """
Before taking any actions, create a detailed plan using work_item_add
for each step you will take. Your plan should be specific and actionable:
- BAD work item: 'Analyze the core modules' (vague, unverifiable)
- GOOD work item: 'Read and analyze each file in the target directory'

Each work item should involve concrete tool usage. A work item that can
be completed without using any tools is too vague — break it down further.

Do not mark a work item as done until you have actually used tools to
complete it and stored meaningful artifacts from the work.
"""
```

### Injected Messages (Turns 2+)

**Work Item State** (when work items are incomplete):

```
Work items (N remaining):
  [ ] [item-1] Title of pending item
  [~] [item-2] Title of in-progress item [NEEDS REVISION]

Your tool usage so far: read_file: 3, list_directory: 2, run_command: 1
NOTE: You have only read 3 file(s). For thorough code investigation,
aim to read 10-20+ relevant source files.

Continue working on the next incomplete item.
```

**Work Item Reminder** (when the stop decision rejected for `work_items_incomplete`):

A formatted reminder from `format_work_item_reminder()` showing which items need attention.

### What's NOT in the Prompt

Unlike Copilot CLI, the harness does NOT inject:
- Environment context (cwd, OS, directory listing) — must be in agent instructions
- Tool-specific usage instructions (bash modes, grep syntax) — must be in agent instructions
- Code change guidelines, style rules — must be in agent instructions
- Self-documentation, tips and tricks — must be in agent instructions
- Version information — not tracked

---

## Agentic Loop — Workflow-Based

### Architecture: Pregel-Style Superstep Workflow

Unlike Copilot CLI's imperative two-level loop, the harness uses a **declarative workflow graph** based on the framework's Pregel-inspired execution engine. Each "executor" is a node in the graph, and messages flow between them.

```
┌──────────────────────────────────────────────────────────┐
│                    Workflow Graph                         │
│                                                          │
│  ┌──────────┐    ┌────────────┐    ┌──────────────┐     │
│  │  Repair   │───▶│ Compaction │───▶│  Agent Turn  │     │
│  │ Executor  │    │  Executor  │    │  Executor    │     │
│  └──────────┘    └────────────┘    └──────┬───────┘     │
│       ▲                                    │             │
│       │                                    ▼             │
│       │                            ┌──────────────┐     │
│       └────────────────────────────│ Stop Decision │     │
│            RepairTrigger           │  Executor     │     │
│                                    └──────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Message Flow Per Turn

```
1. RepairTrigger
   → RepairExecutor: Fix dangling tool calls, init state on first run
   → Emits: RepairComplete(repairs_made=N)

2. RepairComplete
   → CompactionExecutor: Check token budget, signal if compaction needed
   → Emits: CompactionComplete(compaction_needed=True/False)

3. CompactionComplete (or RepairComplete if no compaction)
   → AgentTurnExecutor: Run one agent turn
     a. Increment turn count
     b. Inject messages (tool guidance, work items, planning)
     c. Apply direct cache clearing if compaction needed
     d. Run agent (streaming or non-streaming)
     e. Check for task_complete tool call
     f. Handle continuation prompts if agent stops without tool calls
     g. Update token budget
   → Emits: TurnComplete(agent_done=True/False, error=None)

4. TurnComplete
   → StopDecisionExecutor: Evaluate stop conditions
     Layer 1: Hard stops (error, max turns)
     Layer 2: Stall detection (fingerprint comparison)
     Layer 3: Agent done signal + optional verification
       3a: Contract verification
       3b: Work item verification
     Continue: Send RepairTrigger to restart loop
     Stop: Yield HarnessResult and terminate
```

### Inner Turn: AgentTurnExecutor

```python
async def run_turn(self, trigger: RepairComplete, ctx: WorkflowContext):
    turn_count += 1

    # First turn: inject guidance messages
    if turn_count == 1:
        inject_tool_strategy_guidance()  # Always
        if work_items_enabled:
            inject_work_item_guidance()   # Work item tool reference
            inject_planning_prompt()      # "Plan before executing"

    # Later turns: inject work item state
    if turn_count > 1 and work_items_enabled:
        inject_work_item_state()  # Show incomplete items + tool usage stats

    # Work item reminder if previous stop rejected
    if work_items_enabled:
        maybe_inject_work_item_reminder()  # "These items still need work"

    # Apply compaction if signaled
    if compaction_needed:
        apply_direct_clear()  # Replace old tool results with placeholders

    # Run the agent (one turn = one LLM call with tool execution)
    response = await agent.run(messages, thread=thread, tools=tools, middleware=middleware)

    # Cache management
    cache.extend(response.messages)

    # Determine completion
    if response has task_complete call:
        agent_done = True
    elif response has other tool calls:
        agent_done = False
    else:
        # No tool calls — agent wants to stop
        if continuation_prompts_enabled and count < max:
            inject continuation prompt
            agent_done = False
        else:
            agent_done = True  # Accept stop

    # Update token budget for compaction
    update_token_budget(response)

    # Signal to stop decision
    send_message(TurnComplete(agent_done=agent_done))
```

**Key difference from Copilot CLI**: The agent.run() call handles the ENTIRE inner loop (LLM call → tool execution → repeat until no tool calls). The harness only sees the final result. There is no per-tool-call interception within a single turn.

### Outer Loop: StopDecisionExecutor

```python
async def evaluate(self, turn_result: TurnComplete, ctx: WorkflowContext):
    # Layer 1: Hard stops
    if turn_result.error:
        stop(FAILED)
        return

    if turn_count >= max_turns:
        stop(DONE, "max_turns")
        return

    # Layer 2: Stall detection
    if enable_stall_detection:
        fingerprint = compute_fingerprint(transcript_length, work_items)
        tracker.add_fingerprint(fingerprint)
        if tracker.is_stalled():
            stop(STALLED)
            return

    # Layer 3: Agent signals done
    if turn_result.agent_done:
        # 3a: Contract verification
        if enable_contract_verification:
            satisfied, gaps = verify_contract()
            if not satisfied:
                send_message(RepairTrigger())  # Continue
                return

        # 3b: Work item verification
        if enable_work_item_verification:
            all_complete = verify_work_items()  # ledger.is_all_complete()
            if not all_complete:
                send_message(RepairTrigger())  # Continue
                return

        # All checks passed
        stop(DONE, "agent_done")
        return

    # No stop condition — continue
    send_message(RepairTrigger())
```

---

## Quality Enforcement Mechanisms

### 1. `task_complete` Tool

Defined in `_done_tool.py`:

```python
@ai_function(name="task_complete", approval_mode="never_require")
def task_complete(
    summary: Annotated[str, "Brief summary of what was accomplished"],
) -> str:
    """Signal that the current task is complete.

    Call this tool when you have finished the user's request and have no more
    actions to take. Provide a brief summary of what was accomplished.
    """
    return f"Task marked complete: {summary}"
```

**Important**: Unlike Copilot CLI, this tool is NOT automatically added. It must be included in the agent's tool set. The harness detects its usage by scanning response messages for `FunctionCallContent` with `name == "task_complete"`.

When detected, `agent_done=True` is set immediately, bypassing continuation prompts.

### 2. Continuation Prompts

When the agent stops making tool calls (no more `FunctionCallContent` in response) without calling `task_complete`, the harness injects a continuation prompt:

```python
DEFAULT_CONTINUATION_PROMPT = "Continue if there's more to do, or just say 'done' if finished."
```

Configuration:
- `enable_continuation_prompts`: bool (default: True)
- `max_continuation_prompts`: int (default: 2)
- `continuation_prompt`: str (customizable)

After `max_continuation_prompts` are exhausted, the harness accepts `agent_done=True`.

**Comparison with Copilot CLI**: Copilot's AUTOPILOT_MODE continuation nudge is much more assertive:
> "You have not yet marked the task as complete using the task_complete tool. If you were planning, stop planning and start implementing."

Our default is passive: "Continue if there's more to do, or just say 'done' if finished."

### 3. Work Item Verification (Stop Decision Layer 3b)

When `enable_work_items=True`, the `StopDecisionExecutor` checks `ledger.is_all_complete()` before accepting the agent's done signal.

```python
if turn_result.agent_done and enable_work_item_verification:
    work_items_complete = verify_work_items()
    if not work_items_complete:
        # Reject done signal, continue execution
        send_message(RepairTrigger())
        return
```

This causes the loop to continue, and `AgentTurnExecutor` will inject a work item reminder on the next turn explaining which items are still incomplete.

**Key limitation**: The check is purely `is_all_complete()` — it only verifies that all items have status `DONE` or `SKIPPED`. It does NOT verify:
- Quality or depth of the work
- Whether enough files were read
- Whether deliverables were produced
- Whether artifacts are substantive

If the agent marks all items as `DONE` after shallow work, this gate passes.

### 4. Contract Verification (Stop Decision Layer 3a)

When a `TaskContract` is provided, the `ContractVerifier` checks requirements:

```python
contract = TaskContract.simple(
    "Fix the bug",
    "Identify root cause",
    "Implement fix",
    "Tests pass",
)
```

The verifier scans the transcript for evidence that each requirement was addressed. If gaps are found, a `GapReport` is generated and execution continues.

**Current state**: This is designed for automation mode but rarely used in interactive mode.

### 5. Stall Detection

Tracks `ProgressFingerprint` across turns. If the fingerprint (based on transcript length, work item statuses) doesn't change for `stall_threshold` turns (default: 3), the harness stops with `STALLED` status.

### 6. Max Turns

Hard limit on total turns (default: 50). Stops with `DONE` status and `max_turns` reason.

---

## Compaction / Context Management

### Two Systems (v1 and v2)

**v1: ContextPressureExecutor** (legacy, Phase 2)
- Simple strategy-based system
- Strategies: `ClearToolResultsStrategy`, `CompactConversationStrategy`, `DropOldestStrategy`
- Direct transcript modification
- Enabled via `enable_context_pressure=True`

**v2: CompactionExecutor** (production, Phase 9)
- Plan Pipeline architecture
- Storage-protocol based (injectable for different environments)
- Strategy ladder: Clear → Summarize → Externalize → Drop
- `CompactionPlan` flows through SharedState
- Enabled via `enable_compaction=True`

### CompactionExecutor Flow

```python
async def check_compaction(self, trigger: RepairComplete, ctx: WorkflowContext):
    # 1. Load token budget from SharedState
    budget = get_or_create_budget()  # TokenBudget v1

    # 2. Read current_estimate (written by AgentTurnExecutor after each turn)
    current_tokens = budget.current_estimate

    # 3. Check threshold
    if not budget.is_under_pressure:
        # Under 85% of max — no action needed
        send_message(CompactionComplete(compaction_needed=False))
        return

    # 4. Signal that compaction is needed
    # NOTE: CompactionExecutor does NOT have access to the message cache.
    # It only signals. AgentTurnExecutor does the actual clearing.
    send_message(CompactionComplete(compaction_needed=True))
```

### Direct Cache Clearing (AgentTurnExecutor)

When `CompactionComplete.compaction_needed=True`, the AgentTurnExecutor applies:

```python
def _apply_direct_clear(self, current_turn, preserve_recent_turns=2):
    """Replace old FunctionResultContent with short placeholders."""
    cutoff = len(cache) - preserve_recent_turns

    for i in range(cutoff):
        for content in msg.contents:
            if isinstance(content, FunctionResultContent) and len(str(content.result)) > 100:
                content.result = "[Tool result cleared to save context]"
```

This is a lightweight analog of Copilot CLI's full compaction pipeline.

### Plan Pipeline (Not Fully Connected)

The `compact_thread()` method on `CompactionExecutor` exists for full LLM-based compaction:

```python
async def compact_thread(self, thread, current_plan, budget, turn_number):
    # Calculate tokens to free (over threshold + 10% headroom)
    tokens_to_free = tokens_over + (max_input_tokens * 0.1)

    # Run CompactionCoordinator with strategy ladder
    result = await coordinator.compact(
        thread, current_plan, budget, tokenizer,
        tokens_to_free=tokens_to_free, turn_context=TurnContext(turn_number)
    )
    return result.plan
```

**Current state**: This method exists but is NOT called from the workflow loop. The CompactionExecutor signals need, and AgentTurnExecutor applies direct clearing only. Full LLM-based summarization compaction is not wired up.

### Strategy Ladder

When fully connected, strategies execute in order:

| Strategy | Dependencies | Action |
|----------|-------------|--------|
| ClearStrategy | None | Replace tool results with placeholders |
| SummarizeStrategy | Summarizer (LLM) | LLM-compress older spans |
| ExternalizeStrategy | ArtifactStore + Summarizer | Store large content to artifact store |
| DropStrategy | None (last resort) | Remove content entirely |

### TokenBudget (v1 — Used for Inter-Executor Communication)

```python
@dataclass
class TokenBudget:
    max_input_tokens: int = 100000       # Default ceiling
    soft_threshold_percent: float = 0.85  # 85% triggers compaction
    current_estimate: int = 0             # Updated by AgentTurnExecutor

    @property
    def is_under_pressure(self) -> bool:
        return self.current_estimate >= self.soft_threshold

    @property
    def soft_threshold(self) -> int:
        return int(self.max_input_tokens * self.soft_threshold_percent)
```

### TokenBudget (v2 — Compaction-Internal)

Located in `_compaction/_tokenizer.py`, this richer budget tracks:
- `system_prompt_tokens`
- `tool_schema_tokens`
- `safety_buffer_tokens`
- `max_input_tokens`
- Method: `is_under_pressure(current_tokens) -> bool`
- Method: `tokens_over_threshold(current_tokens) -> int`

---

## Token Counting

### Primary: API Usage Details

When available, uses `response.usage_details.input_token_count` from the LLM API response:

```python
if response and response.usage_details and response.usage_details.input_token_count:
    current_tokens = response.usage_details.input_token_count
    counting_method = "api_usage"
```

This is the most accurate method — it includes system prompt, tool schemas, all formatting overhead. This approach was implemented based on our compaction token counting issues analysis (recommended as "Approach C").

### Fallback: Tiktoken Content Counting

When API usage is unavailable, counts tokens from cache contents:

```python
elif self._tokenizer is not None:
    for msg in self._cache:
        for content in msg.contents:
            if isinstance(content, TextContent):
                current_tokens += tokenizer.count_tokens(content.text)
            elif isinstance(content, FunctionCallContent):
                current_tokens += tokenizer.count_tokens(content.name)
                current_tokens += tokenizer.count_tokens(str(content.arguments))
            elif isinstance(content, FunctionResultContent):
                current_tokens += tokenizer.count_tokens(str(content.result))
        current_tokens += 4  # Per-message overhead
    counting_method = "tokenizer"
```

### TiktokenTokenizer (from `_compaction/_tokenizer.py`)

```python
class TiktokenTokenizer(ProviderAwareTokenizer):
    """Production tokenizer using tiktoken for accurate counts."""

    def __init__(self, model_name: str = "gpt-4o"):
        self._encoding = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))
```

### Comparison with Copilot CLI

| Aspect | Copilot CLI | Our Harness |
|--------|-------------|-------------|
| Primary method | tiktoken on wire-format messages | API `usage_details.input_token_count` |
| Fallback | None (tiktoken is always available) | tiktoken on content objects |
| Per-message overhead | 3-4 tokens + role | 4 tokens flat |
| Tool call counting | name + arguments separately | name + arguments separately |
| Tool schema counting | `OIe(toolDefs, model)` dedicated function | Not counted separately |
| System prompt counting | Included in `fR()` | Only via API usage (primary) |
| Scale factor | Per-model `zMe(model)` | None (raw tiktoken count) |
| Format | Counts from OpenAI wire format | Counts from framework content objects |

---

## Sub-Agent System

### Not Implemented

The harness does NOT have a sub-agent system. There is no equivalent of Copilot CLI's explore/task/code-review/general-purpose agent delegation.

The harness runs a single agent (`AgentProtocol`) for the entire session. Any sub-agent behavior must be implemented within the agent itself (e.g., a `ChatAgent` that has a `task` tool calling another agent).

---

## Custom Instructions & Skills

### Not Implemented

The harness does NOT load instructions from files like `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, etc.

All custom instructions must be provided via:
1. The agent's `instructions` parameter (system prompt)
2. Additional tools added to the agent
3. Work item guidance (injected by the harness when `enable_work_items=True`)

---

## Hooks System

### Not Implemented

The harness does NOT have a hooks system like Copilot CLI's `preToolUse`, `postToolUse`, `agentStop`, `sessionStart`, `sessionEnd`, etc.

The closest equivalents are:
- **WorkItemEventMiddleware**: A `FunctionMiddleware` that intercepts work item tool calls to capture state changes. This is not a general-purpose hook — it only tracks work item operations.
- **Workflow events**: `HarnessLifecycleEvent` objects emitted via `ctx.add_event()` for DevUI display. These are output-only (no interception/modification).

### Lifecycle Events Emitted

```python
# Event types emitted via ctx.add_event(HarnessLifecycleEvent):
"harness_started"        # Once at initialization
"turn_started"           # Before each agent turn
"turn_completed"         # After each agent turn
"continuation_prompt"    # When continuation prompt injected
"stall_detected"         # When stall detection triggers
"context_pressure"       # When compaction signals pressure
"deliverables_updated"   # When work item deliverables change
"harness_completed"      # Once at termination
"work_item_changed"      # When work item status changes
```

---

## Tool Permission & Filtering

### Not Implemented

The harness does NOT filter, deny, or modify tool calls. All tool permissions are handled by the underlying agent framework's approval system (`approval_mode` on `@ai_function`).

The `task_complete` tool is defined with `approval_mode="never_require"` to ensure it always executes without user confirmation.

Work item tools are injected at runtime via `run_kwargs["tools"]` when `enable_work_items=True`, but there is no filtering or gating mechanism.

---

## JIT (Just-In-Time) Instructions

### Partial Implementation

The harness injects messages at specific points, but does NOT have a general-purpose JIT instruction system like Copilot CLI's `JitInstructionsProcessor`.

Current injections:

| When | What | Condition |
|------|------|-----------|
| Turn 1 | Tool strategy guidance | Always |
| Turn 1 | Work item guidance | `enable_work_items=True` |
| Turn 1 | Planning prompt | `enable_work_items=True` |
| Turns 2+ | Work item state | Work items incomplete |
| After stop rejection | Work item reminder | Last stop was `work_items_incomplete` |
| Agent stops without tools | Continuation prompt | `enable_continuation_prompts=True` and count < max |

**Not implemented**:
- Time-based injections (no timeout tracking)
- Git-change-based injections (no file change monitoring)
- Conditional injection based on execution state beyond work items
- Configurable injection rules

---

## Feature Flags

### Not Implemented

The harness does NOT have a feature flag system. All behavior is controlled via constructor parameters:

```python
AgentHarness(
    agent,
    max_turns=50,                        # Hard turn limit
    enable_stall_detection=True,         # Stall detection on/off
    stall_threshold=3,                   # Turns before stall
    enable_continuation_prompts=True,    # Continuation prompts on/off
    max_continuation_prompts=2,          # Max continuations
    enable_work_items=True,              # Work item system on/off
    enable_compaction=True,              # Compaction on/off
    max_input_tokens=100000,             # Token budget ceiling
    soft_threshold_percent=0.85,         # Compaction trigger threshold
    enable_context_pressure=False,       # Legacy v1 (mutually exclusive with compaction)
    task_contract=None,                  # Formal contract verification
)
```

---

## Comparison with Copilot CLI

### Architecture

| Aspect | Copilot CLI | Our Harness |
|--------|-------------|-------------|
| Loop model | Imperative 2-level loop (inner: turns, outer: session) | Declarative workflow graph (Pregel supersteps) |
| Inner loop | `getCompletionWithTools` — turn-by-turn with per-tool interception | `agent.run()` — entire turn delegated to agent framework |
| Outer loop | `runAgenticLoop` with `agentStop` hook | `StopDecisionExecutor` with layered checks |
| Communication | Direct method calls + event yields | Message passing between executors via SharedState |
| State management | In-memory on session object | SharedState (serializable, supports checkpointing) |

### Quality Enforcement

| Mechanism | Copilot CLI | Our Harness |
|-----------|-------------|-------------|
| Explicit completion tool | `task_complete` (AUTOPILOT only, auto-added) | `task_complete` (must be manually added to agent) |
| Post-stop hook | `agentStop` hook → can block with reason | `StopDecisionExecutor` → checks contracts + work items |
| Pre-tool hook | `preToolUse` → can deny/modify | Not implemented |
| Continuation nudge | Assertive: "You have not yet marked the task as complete..." | Passive: "Continue if there's more to do..." |
| Work item system | Simple SQLite todos table | Rich: WorkItemLedger, priorities, artifacts, revisions, contamination detection |
| Minimum effort check | None (relies on prompt quality) | Read-count nudge when < 10 reads (advisory only) |
| Contract verification | None | TaskContract + ContractVerifier (transcript scanning) |
| Stall detection | None (relies on agentStop hook) | ProgressFingerprint tracking |

### Compaction

| Aspect | Copilot CLI | Our Harness |
|--------|-------------|-------------|
| Architecture | `CompactionProcessor` as preRequest processor | `CompactionExecutor` as workflow node + direct clearing in `AgentTurnExecutor` |
| Trigger | 80% background, 95% blocking | 85% threshold (configurable) |
| Background compaction | Yes — runs async, continues processing | No — synchronous check before each turn |
| Full LLM compaction | Yes — calls LLM to summarize | Designed but not wired up (compact_thread exists) |
| Direct clearing | Not separate — part of compaction pipeline | Yes — `_apply_direct_clear()` replaces old tool results |
| Preserves across compaction | Original user messages, plan content, todos | Recent N messages preserved (preserve_recent_turns=2) |

### Token Counting

| Aspect | Copilot CLI | Our Harness |
|--------|-------------|-------------|
| Primary | tiktoken on wire-format messages | API `usage_details.input_token_count` |
| Includes system prompt | Yes (counted in fR) | Yes (via API usage) |
| Includes tool schemas | Yes (OIe function) | Yes (via API usage) |
| Fallback | None needed | tiktoken on content objects |
| Scale factors | Per-model zMe() | None |

### Prompt Architecture

| Aspect | Copilot CLI | Our Harness |
|--------|-------------|-------------|
| System prompt construction | Rich template composition (identity + code_change + guidelines + env + tools) | Delegated to agent — harness doesn't touch system prompt |
| Environment context | Auto-injected (cwd, OS, dir listing) | Must be in agent instructions |
| Tool instructions | Per-tool XML blocks in system prompt | Not injected (agent must define) |
| Custom instructions | AGENTS.md, CLAUDE.md, .github/copilot-instructions.md auto-loaded | None — must be in agent instructions |
| Guidance injection | Part of system prompt (static) | User messages injected into cache (dynamic, per-turn) |

### Missing from Our Harness (Compared to Copilot CLI)

1. **agentStop hook**: No way for external code to inspect session state and block completion. Our stop decision is rule-based (work items, contracts) not extensible via hooks.

2. **preToolUse / postToolUse hooks**: No per-tool-call interception. Cannot deny, modify, or add context to individual tool calls.

3. **Background compaction**: Compaction is synchronous and only does direct clearing. No async LLM summarization while agent continues working.

4. **Sub-agent delegation**: No explore/task/code-review agents. Single agent handles everything.

5. **Custom instructions loading**: No auto-discovery of AGENTS.md, CLAUDE.md, etc.

6. **JIT instruction system**: No time-based or condition-based dynamic instruction injection beyond work items.

7. **Assertive continuation nudge**: Our continuation prompt is too passive — it suggests the agent is nearly done. Copilot's says "you aren't done, keep working."

8. **Environment context auto-injection**: Agent must manually include cwd, OS, etc. in its own instructions.

### Strengths of Our Harness (Compared to Copilot CLI)

1. **Rich work item system**: Full ledger with priorities, artifacts, revisions, contamination detection. More structured than Copilot's SQLite todos.

2. **Formal contract verification**: TaskContract system for automation mode with gap reporting.

3. **Serializable state**: All state flows through SharedState, supporting checkpointing and durability.

4. **Declarative workflow**: Graph-based execution enables easy addition of new executor nodes.

5. **Artifact validation**: Contamination detection for narrative vs. deliverable content.

6. **Multiple compaction strategies**: Strategy ladder design (clear/summarize/externalize/drop) vs. Copilot's single LLM summarization.

7. **Stall detection**: Automatic detection of spinning agents via progress fingerprinting.
