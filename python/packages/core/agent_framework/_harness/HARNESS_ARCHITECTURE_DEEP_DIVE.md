# Agent Harness Deep Dive

This document explains how `python/packages/core/agent_framework/_harness` currently works end-to-end, what the design gets right, and where it can be improved.

## 1. What the harness is

The harness is a workflow-driven outer control loop around an `AgentProtocol` implementation:

- It runs turns durably using workflow state and checkpoint hooks.
- It applies pre-turn repair and optional compaction.
- It decides whether to continue or stop using layered policies.
- It tracks a transcript and emits lifecycle events for streaming/DevUI.

Primary entrypoints:

- `AgentHarness` (`_harness_builder.py`): convenience API (`run`, `run_stream`).
- `HarnessWorkflowBuilder` (`_harness_builder.py`): explicit workflow construction.

## 2. Runtime architecture

### 2.1 Workflow topology

Built in `HarnessWorkflowBuilder.build()`:

- With compaction: `repair -> compaction -> agent_turn -> stop_decision -> repair`
- Without compaction: `repair -> agent_turn -> stop_decision -> repair`

Executors:

- `RepairExecutor` (`_repair_executor.py`)
- `CompactionExecutor` (`_compaction_executor.py`) optional
- `AgentTurnExecutor` (`_agent_turn_executor.py`)
- `StopDecisionExecutor` (`_stop_decision_executor.py`)

### 2.2 Shared state contract

SharedState keys in `_constants.py` model the harness runtime contract:

- control: `harness.turn_count`, `harness.max_turns`, `harness.status`, `harness.stop_reason`
- observability: `harness.transcript`
- compaction: `harness.token_budget`, `harness.compaction_plan`, `harness.compaction_metrics`
- progress/quality: `harness.task_contract`, `harness.coverage_ledger`, `harness.progress_tracker`
- work management: `harness.work_item_ledger`, `harness.work_complete_retry_count`
- run bootstrap: `harness.initial_message`

## 3. Execution lifecycle (what actually happens)

### 3.1 Repair phase

`RepairExecutor.repair()`:

- Initializes harness state on first run from `harness.config`.
- Emits `harness_started` lifecycle event.
- Repairs any pending tool calls by appending synthetic `repair` transcript events.
- Sends `RepairComplete` to continue.

Important note: pending tool calls are initialized and cleared, but no current producer in harness writes them during agent turns.

### 3.2 Compaction check phase (optional)

`CompactionExecutor.check_compaction()`:

- Loads `TokenBudget` from shared state.
- Reads `budget.current_estimate` (written by `AgentTurnExecutor`).
- If under threshold: passthrough.
- If over threshold: emits `compaction_started`, stores metrics, sends `CompactionComplete(compaction_needed=True, blocking=...)`.

Key detail: actual compaction operations are not executed here. This executor currently acts as pressure detector/orchestrator.

### 3.3 Agent turn phase

`AgentTurnExecutor.run_turn()` is the heaviest component:

- Increments turn count.
- Injects initial user message on turn 1.
- Optionally injects work-item reminders/state and JIT instructions.
- If compaction signaled, runs full compaction pipeline against internal cache.
- Calls `agent.run()` or `agent.run_stream()` with dynamic tools/middleware.
- Updates token budget from cache + usage details overhead approximation.
- Decides turn-level completion signal:
  - done if `work_complete` was called
  - continue if other tool calls exist
  - otherwise continuation prompt loop (bounded) before accepting done
- Emits lifecycle/transcript events and sends `TurnComplete`.

Internal model:

- Canonical conversation for this executor is an in-memory `_cache` list.
- Checkpoint save/restore serializes this cache.
- Compaction flattening mutates this cache into compacted view.

### 3.4 Stop decision phase

`StopDecisionExecutor.evaluate()` layered logic:

1. hard stop on agent error
2. hard stop on max turns
3. optional stall detection via `ProgressFingerprint`
4. if agent signaled done:
   - enforce `work_complete` (with retry cap)
   - optional contract verification
   - optional work-item completion verification
   - optional `agent_stop` hooks
   - then stop as done
5. else continue loop

On stop, it emits final lifecycle event and yields `HarnessResult`.

## 4. Supporting subsystems

### 4.1 Work item system

`_work_items.py` provides:

- ledger model + statuses/priorities/roles
- runtime tools (`work_item_add`, `work_item_update`, `work_item_list`, `work_item_set_artifact`, `work_item_flag_revision`)
- artifact contamination cleaning and control-artifact validation
- middleware to emit work-item change events

This enables explicit planning and self-critique loops, and supports deliverable extraction in final result.

### 4.2 Contract system

`_task_contract.py` + `_contract_verifier.py` provide:

- `TaskContract` with predicates and required outputs
- `CoverageLedger` and gap reporting
- contract verification path in `StopDecisionExecutor`

### 4.3 Hooks system

`_hooks.py` adds extension points:

- pre-tool deny
- post-tool observe
- agent-stop block/allow

### 4.4 Context providers and guidance

`_context_providers.py` wires environment and harness guidance into system prompt context provider pipeline.

### 4.5 Compaction package

`_compaction/` contains:

- plan/action model (`_types.py`)
- strategy ladder (`_strategies.py`)
- tokenizer and token budget (`_tokenizer.py`)
- summary and storage abstractions
- cache adapters (`_adapters.py`)

The current harness implementation uses this package through `AgentTurnExecutor` flattening workflow.

## 5. Design strengths

- Clear separation of control-plane stages via workflow executors.
- Good extension surface: hooks, context providers, injectable compaction components.
- Rich observability model (`HarnessLifecycleEvent` + transcript events).
- Useful self-management constructs (work items + control artifact invariants).
- Practical stopping model combines hard limits and semantic checks.
- Compaction strategy abstractions are strong and reusable.

## 6. High-impact issues and architectural gaps

## 6.1 Contract evidence model is not fully wired

`ContractVerifier` expects evidence patterns such as `tool_result` and response text fields, but harness transcript currently records only coarse `agent_response` metadata (counts/flags) and does not emit `tool_result` events in current turn executor flow.

Impact:

- Some predicate types (especially `TOOL_RESULT_SUCCESS`, detailed text checks) can fail or be noisy even when task was actually completed.

## 6.2 Repair invariant is partially dead

`RepairExecutor` repairs `HARNESS_PENDING_TOOL_CALLS_KEY`, but no active writer updates this key during normal turn execution.

Impact:

- A meaningful crash-repair mechanism exists in shape but not in active runtime path.

## 6.3 Compaction orchestration is split across two executors with mixed ownership

- `CompactionExecutor` detects pressure.
- `AgentTurnExecutor` performs actual compaction/flattening.

This works, but blurs boundaries and makes lifecycle reasoning harder.

Impact:

- Debugging and ownership clarity suffer.
- Compaction behavior depends on turn executor internals rather than a single compaction control point.

## 6.4 Doc/design drift around compaction

Historical docs in the folder describe architectures that no longer exactly match runtime behavior (plan persistence vs flatten-in-cache flow, executor roles, token accounting history).

Impact:

- New contributors can make wrong assumptions quickly.

## 6.5 Guidance duplication and token overhead

Guidance exists both as context provider content and first-turn injected system messages in some modes.

Impact:

- Increased context pressure.
- Higher chance of prompt bloat and behavioral over-steering.

## 6.6 Turn executor is too large

`_agent_turn_executor.py` has accumulated many responsibilities: compaction, prompt policy, middleware composition, work-item orchestration, token accounting, continuation policy, eventing, checkpoint logic.

Impact:

- Harder to test thoroughly.
- Harder to evolve without regressions.

## 6.7 Work-complete enforcement can still terminate on retries exhausted

`StopDecisionExecutor` accepts done when work-complete retries are exhausted, even if signal discipline was never corrected.

Impact:

- Correctness tradeoff favors liveness; acceptable for interactive mode, riskier for automation.

## 7. Recommended improvements (prioritized)

### P0: Correctness and contract reliability

1. Emit structured tool-call/result transcript events from agent turn path.
2. Record richer final assistant content summaries for predicate checks (sanitized/size-bounded).
3. Add tests for each predicate type against real harness transcript shape.

### P0: Repair path completion

1. Instrument pending tool-call writes before invocation and clear on result.
2. Validate repair behavior under forced interruption/checkpoint restore tests.

### P1: Compaction ownership cleanup

1. Choose one owner for compaction execution.
2. Preferred: keep `CompactionExecutor` as orchestration owner; move full compaction call there and keep `AgentTurnExecutor` focused on message execution.
3. Alternatively, formalize current model and downgrade `CompactionExecutor` to explicit pressure signaler in code/docs.

### P1: Executor decomposition

Extract from `AgentTurnExecutor`:

1. `TurnPromptAssembler` (initial/jit/reminder/work-item-state injection).
2. `TurnCompactionService` (plan load/apply/flatten + fallback clear).
3. `TurnTokenBudgetService` (counting + overhead reconciliation).
4. `TurnToolingService` (tool/middleware composition).

This can be pure refactor with no behavior change, then incrementally harden tests.

### P1: Guidance rationalization

1. Keep durable policy guidance in context providers.
2. Remove duplicate first-turn message injection for same policies.
3. Keep only dynamic, state-dependent injections in turn executor.

### P2: Stop policy modes

Introduce explicit stop modes:

- `interactive`: current liveness-biased behavior
- `strict_automation`: no retries-exhausted auto-accept; must satisfy explicit completion gate

### P2: Documentation consolidation

1. Add one “source of truth” runtime architecture doc (this file can be that basis).
2. Mark older design docs as historical where they differ from implementation.

## 8. Suggested implementation sequence

1. Land transcript/tool-result instrumentation + tests.
2. Land pending-tool-call lifecycle instrumentation + interruption tests.
3. Decompose `AgentTurnExecutor` by extraction (no behavior changes).
4. Normalize compaction ownership and update docs.
5. Add strict automation stop mode.

## 9. Bottom line

The harness is already a strong control-plane foundation: durable outer loop, extensibility points, and advanced context/work tracking. The biggest improvements now are not new features; they are alignment and hardening:

- align transcript evidence with contract verification,
- complete currently partial repair invariants,
- simplify component ownership around compaction,
- and split the turn executor into focused units.

Doing these will materially improve reliability, maintainability, and operator trust without changing the harness’s core architectural direction.
