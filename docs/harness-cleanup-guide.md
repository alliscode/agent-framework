# Harness Cleanup Guide

> **Goal**: Reduce the harness to its minimum valuable size by removing dead code,
> consolidating duplicates, and cleaning up legacy systems — without regressing
> runtime performance.
>
> **Context**: The harness has been iterated on for weeks across 13+ phases. Some
> earlier systems (context pressure, rehydration, prompt renderer) have been
> superseded by newer implementations but never removed. This guide identifies
> every cleanup opportunity and provides safe execution order.

---

## Current Size

The harness is **~14,400 lines** across 32 files, plus **~6,800 lines** of tests
across 12 test files. The target is to remove roughly **3,000–4,000 lines** of
dead or legacy code.

| Category | Files | Lines | Status |
|----------|-------|-------|--------|
| Core runtime | 14 files | ~6,400 | Active — keep |
| Compaction subsystem | 12 files | ~5,200 | Partially active — prune |
| Legacy context pressure | 2 files | ~970 | Superseded — remove |
| Renderers | 1 file | ~334 | UI utility — keep but trim exports |
| Task contracts | 2 files | ~1,100 | Active — keep |
| Tests | 12 files | ~6,800 | Fix + prune matching dead code |

---

## Cleanup Tasks

### 1. Remove Legacy Context Pressure System (~970 lines) — ✅ COMPLETE

**What**: The Phase 2 context pressure system has been fully superseded by the
Phase 9 CompactionExecutor. The legacy code is only used when
`enable_context_pressure=True`, which no sample or production code uses.

**Summary of changes made**:
- Deleted `_context_pressure_executor.py` (464 lines) — legacy executor and
  `ContextPressureComplete` message type.
- Stripped `_context_pressure.py` down to only `TokenBudget`, `estimate_tokens()`,
  and `estimate_transcript_tokens()` — removed `ContextEditKind`, `TranscriptRange`,
  `ExternalizeEdit`, `ClearEdit`, `CompactEdit`, `DropEdit`, `ContextEditPlan`,
  `ContextPressureStrategy` protocol, `ClearToolResultsStrategy`,
  `CompactConversationStrategy`, `DropOldestStrategy`, and `get_default_strategies()`.
- Updated `_harness_builder.py`: Removed `enable_context_pressure` parameter from
  both `HarnessWorkflowBuilder` and `AgentHarness`, removed the
  `ContextPressureExecutor` registration branch and edge wiring, simplified
  supersteps-per-turn calculation.
- Updated `_constants.py`: Removed `HARNESS_CONTEXT_EDIT_HISTORY_KEY`.
- Updated `__init__.py`: Removed all legacy context pressure imports and exports
  (`ClearEdit`, `CompactEdit`, `DropEdit`, `ExternalizeEdit`, `ContextEdit`,
  `ContextEditKind`, `ContextEditPlan`, `ContextPressureComplete`,
  `ContextPressureExecutor`, `ContextPressureStrategy`, `ClearToolResultsStrategy`,
  `CompactConversationStrategy`, `DropOldestStrategy`, `TranscriptRange`,
  `get_default_strategies`, `HARNESS_CONTEXT_EDIT_HISTORY_KEY`). Kept `TokenBudget`,
  `estimate_tokens`, `estimate_transcript_tokens`.
- Updated `test_context_pressure.py`: Removed 14 legacy tests (strategy applicability,
  edit type defaults, executor workflow tests). Kept 6 tests for `TokenBudget` and
  estimation functions.
- All 466 remaining harness tests pass. Ruff lint clean.

---

### 2. Remove Unfinished Rehydration System (~613 lines) — ✅ COMPLETE

**What**: The rehydration system (`_compaction/_rehydration.py`) is architectural
scaffolding that was never wired into the runtime. It defines
`RehydrationInterceptor`, `RehydrationState`, `RehydrationConfig`, etc. but
nothing in the executor chain instantiates or calls any of it.

**Summary of changes made**:
- Deleted `_compaction/_rehydration.py` (613 lines) — entire file containing
  `RehydrationInterceptor`, `RehydrationState`, `RehydrationEvent`,
  `RehydrationBlockedEvent`, `ToolCall`, and `create_rehydration_interceptor`.
- Updated `_compaction/__init__.py`: Removed all rehydration imports
  (`RehydrationBlockedEvent`, `RehydrationEvent`, `RehydrationInterceptor`,
  `RehydrationState`, `ToolCall`, `create_rehydration_interceptor`) and the
  `RehydrationBlockedEventV2` alias. Removed `RehydrationConfig` and
  `RehydrationResult` re-exports from `_turn_context`. Removed all corresponding
  `__all__` entries.
- Updated `_harness/__init__.py`: Removed all rehydration imports and `__all__`
  entries (`RehydrationBlockedEvent`, `RehydrationConfig`, `RehydrationEvent`,
  `RehydrationInterceptor`, `RehydrationResult`, `RehydrationState`, `ToolCall`,
  `create_rehydration_interceptor`).
- Updated `_compaction/_tokenizer.py`: Kept `rehydration_reserve_tokens` and
  `available_for_rehydration` fields (they affect budget math) but added comments
  noting rehydration is not yet implemented.
- Updated `tests/harness/test_compaction.py`: Removed `TestRehydrationInterceptor`,
  `TestRehydrationState`, and `TestToolCall` test classes (6 tests). Removed
  rehydration imports. Updated module docstring.
- Kept `_turn_context.py` unchanged — `TurnContext`, `RehydrationConfig`, and
  `RehydrationResult` remain as they are used by compaction strategies.
- All 460 remaining harness tests pass. Ruff lint clean.

---

### 3. Remove Unfinished PromptRenderer (~565 lines) — ✅ COMPLETE

**What**: `_compaction/_renderer.py` defines a `PromptRenderer` that applies a
`CompactionPlan` to produce a rendered prompt. This was the intended architecture
but the actual runtime uses `AgentTurnExecutor._apply_compaction_plan()` instead.
`PromptRenderer` is never instantiated.

**Summary of changes made**:
- Deleted `_compaction/_renderer.py` (565 lines) — entire file containing
  `PromptRenderer`, `RenderedPrompt`, `render_summary_text()`,
  `render_externalization_text()`, and `COMPACTION_RENDER_FORMAT_VERSION`.
- Relocated `ArtifactStore` protocol, `ArtifactMetadata`, and `SecurityContext`
  to `_compaction/_store.py` — these were the only types from `_renderer.py`
  with runtime usage (referenced by `_strategies.py` `ExternalizeStrategy`
  and `_compaction_executor.py`).
- Updated `_compaction/_strategies.py`: Changed two `from ._renderer import`
  statements to `from ._store import` for `ArtifactStore` and `ArtifactMetadata`.
- Updated `_compaction/__init__.py`: Removed renderer import block and all
  `__all__` entries for `PromptRenderer`, `RenderedPrompt`,
  `COMPACTION_RENDER_FORMAT_VERSION`, `render_summary_text`,
  `render_externalization_text`. Added `ArtifactMetadata`, `ArtifactStore`,
  `SecurityContext` to the `_store` import block. Updated docstring.
- Updated `_harness/__init__.py`: Removed `COMPACTION_RENDER_FORMAT_VERSION`,
  `PromptRenderer`, `RenderedPrompt`, `render_externalization_text`,
  `render_summary_text` from imports and `__all__`.
- Updated `_compaction_executor.py`: Removed stale `PromptRenderer` reference
  in docstring.
- All 460 harness tests pass. Ruff lint clean.

---

### 4. Trim Compaction Event System (~700 lines → ~200 lines) — ✅ COMPLETE

**What**: `_compaction/_events.py` defines 12 event subclasses, metrics
collectors, and emitter protocols. None of these are used outside the _harness
package — they're internal telemetry infrastructure. The `CompactionCoordinator`
doesn't actually emit most of these events. Only the base `CompactionEvent` and
a couple of emitters have any runtime usage.

**Summary of changes made**:
- Trimmed `_compaction/_events.py` from 700 lines to ~195 lines. Removed all 12
  unused event subclasses (`CompactionCheckStartedEvent`, `CompactionCompletedEvent`,
  `ProposalGeneratedEvent`, `ProposalRejectedEvent`, `ContentClearedEvent`,
  `ContentSummarizedEvent`, `ContentExternalizedEvent`, `ContentDroppedEvent`,
  `ContentRehydratedEvent`, `RehydrationBlockedEvent`, `CompactionErrorEvent`,
  `VersionConflictEvent`). Removed `MetricsCollector`, `LoggingEventEmitter`,
  and `CompositeEventEmitter` classes — none were instantiated in runtime code.
- Removed unused `CompactionEventType` enum members: `CONTENT_REHYDRATED`,
  `REHYDRATION_BLOCKED`, `VERSION_CONFLICT` (rehydration was removed in Phase 2;
  version conflicts were part of removed store complexity).
- Kept: `CompactionEventType` (core enum), `CompactionEvent` (base class),
  `CompactionMetrics` (aggregated statistics), `CompactionEventEmitter` (protocol).
- Updated `_compaction/__init__.py`: Removed 14 dead event imports and corresponding
  `__all__` entries.
- Updated `_harness/__init__.py`: Removed 14 dead event imports and corresponding
  `__all__` entries.
- Updated `tests/harness/test_compaction.py`: Removed `TestMetricsCollector` class
  (2 tests) and `MetricsCollector` import. Updated module docstring. Kept
  `TestCompactionMetrics` (3 tests) for the retained `CompactionMetrics` class.
- Removed `logging` import from `_events.py` (no longer needed without emitter
  implementations).
- All 458 remaining harness tests pass. Ruff lint clean.

---

### 5. Consolidate Duplicate TokenBudget Classes

**What**: There are two `TokenBudget` classes serving different purposes:
- **V1** (`_context_pressure.py`): Simple SharedState signaling
  (`current_estimate`, `is_under_pressure`, `is_blocking`)
- **V2** (`_compaction/_tokenizer.py`): Detailed overhead accounting
  (`system_prompt_tokens`, `tool_schema_tokens`, `available_for_messages`)

Both are stored in SharedState under `HARNESS_TOKEN_BUDGET_KEY`.

**Action**: This is lower priority and higher risk than the other cleanups.
Evaluate whether V1 can be removed by having CompactionExecutor work directly
with V2 and AgentTurnExecutor read V2 from SharedState. The main challenge is
that V1 is serialized/deserialized via `to_dict()`/`from_dict()` throughout
the codebase.

**If consolidating**:
- Keep V2 as the single `TokenBudget`
- Add V1's `is_under_pressure` and `is_blocking` properties to V2
- Update `CompactionExecutor` and `AgentTurnExecutor` to use V2 only
- Update all `to_dict()`/`from_dict()` calls

**If not consolidating now**: At minimum, rename to avoid confusion:
- V1 → `PressureBudget` (signaling only)
- V2 → `TokenBudget` (detailed accounting)

---

### 6. Clean Up `__init__.py` Exports (~481 lines → ~250 lines)

**What**: `_harness/__init__.py` exports **~220 symbols** in `__all__`. Many are
internal implementation details that no external consumer uses. Same for
`_compaction/__init__.py` with ~80 symbols.

**Action**: After completing tasks 1–4, rebuild `__all__` to only export:
- Types that samples actually import
- Types that a downstream consumer would reasonably need
- The `AgentHarness` convenience class and its direct dependencies

**Remove from `__all__`** (internal implementation details):
- All compaction event types (internal telemetry)
- All compaction store/strategy internals
- `SpanReference`, `CacheEntry`, `CacheMessageStore`, `CacheThreadAdapter`
- `CompactionTransaction`, `CompactionProposal`, `CompactionAction`
- `SecurityContext`, `OpenItem`, `Decision`
- All `ToolDurability*` types (internal to compaction)
- `PendingToolCall`, `ToolCall`, `ToolOutcome`

**Keep in `__all__`** (public API):
- `AgentHarness`, `HarnessWorkflowBuilder`, `AgentTurnExecutor`
- `CompactionExecutor`, `CompactionComplete`
- `StopDecisionExecutor`, `RepairExecutor`
- `TokenBudget` (whichever survives consolidation)
- `HarnessHooks`, `AgentStopEvent`, `AgentStopResult`
- `JitInstruction`, `JitContext`, `JitInstructionProcessor`
- `WorkItem*` types, `work_complete`, `get_work_complete_tool`
- `TaskContract`, `ContractVerifier` (if task contracts are a public feature)
- `InMemoryCompactionStore`, `InMemoryArtifactStore`, `InMemorySummaryCache`
- `Summarizer`, `ChatClientSummarizer`
- `EnvironmentContextProvider`, `HarnessGuidanceProvider`
- Utilities: `estimate_tokens`, `render_stream`

---

### 7. Fix the 7 Failing JIT Instruction Tests — ✅ COMPLETE

**What**: `test_jit_instructions.py` has 7 test failures because tests reference
old instruction IDs that were renamed:
- `no_reads_after_5_turns` → actual ID is `no_reads_after_3_turns`
- `no_writes_after_reads` → actual ID is `no_deliverable_after_many_reads`

**Files to fix**:
- `tests/harness/test_jit_instructions.py`: Update instruction IDs in test
  methods and any threshold values that changed (e.g., turn >= 5 → turn >= 3)

**Validation**: All 480 tests should pass after this fix (currently 473 pass, 7 fail).

**Summary of changes made**:
Updated `tests/harness/test_jit_instructions.py` in class `TestDefaultInstructions`:
- Renamed 3 `no_reads_after_5_turns` tests → `no_reads_after_3_turns`, updated
  turn thresholds from 5/4 → 3/2 to match the actual condition (`turn >= 3`).
- Renamed 4 `no_writes_after_reads` tests → `no_deliverable_after_many_reads`,
  updated turn thresholds to match the actual condition (`turn >= 8`, `read_file >= 5`,
  `write_file == 0`, `work_items_complete < work_items_total`), and added
  `work_items_total=3` to contexts that need the condition to fire.
- All 480 harness tests now pass. Ruff lint clean.

---

### 8. Remove Unused Compaction Store Complexity (~697 lines → ~150 lines) — ✅ COMPLETE

**What**: `_compaction/_store.py` defines `CompactionStore` protocol,
`InMemoryCompactionStore`, `CompactionTransaction`, `SecurityContext`, and
elaborate versioning/locking infrastructure. In practice, only
`InMemoryCompactionStore` is ever used, and the versioning/locking is never
exercised because the harness runs single-threaded.

**Summary of changes made**:
- Removed `CompactionTransaction` class (~76 lines) — transactional wrapper with
  optimistic version tracking. Never used in runtime code (only `CompactionExecutor`
  instantiates `InMemoryCompactionStore` but uses SharedState for actual plan
  persistence, never calling store methods).
- Removed `SecurityContext` class (~31 lines) — access control context that was
  only referenced as a type hint in the `ArtifactStore` protocol's `retrieve`
  method. Never instantiated anywhere.
- Simplified `CompactionStore` protocol: replaced version-based optimistic locking
  API (`get_current_plan` returning `(plan, version)`, `commit_plan` with
  `expected_version`) with simple `get_plan`/`save_plan`/`delete_plan` methods.
- Simplified `InMemoryCompactionStore`: removed `threading.Lock`, removed version
  tracking and conflict detection logic. Now a plain dict-backed store.
- Simplified `InMemorySummaryCache`: removed `threading.Lock` wrapping all
  operations (single-threaded runtime).
- Simplified `InMemoryArtifactStore`: removed `threading.Lock` wrapping all
  operations.
- Simplified `ArtifactStore` protocol: replaced `SecurityContext | None` parameter
  in `retrieve` with `Any | None`.
- Updated `_compaction/__init__.py`: removed `CompactionTransaction` and
  `SecurityContext` imports and `__all__` entries.
- Updated `_harness/__init__.py`: removed `CompactionTransaction` and
  `SecurityContext` imports and `__all__` entries.
- Updated `tests/harness/test_compaction.py`: removed `TestCompactionTransaction`
  class (3 tests), removed `CompactionTransaction` import. Rewrote
  `TestInMemoryCompactionStore` to use simplified `get_plan`/`save_plan` API
  (replaced version conflict test with overwrite test).
- File reduced from 838 lines to 624 lines (~214 lines removed).
- All 455 remaining harness tests pass. Ruff lint clean.

---

## Execution Order

Do these in order — each step should be independently committable and testable:

1. **Fix JIT tests** (task 7) — gets to green baseline first
2. **Remove context pressure legacy** (task 1) — biggest win, lowest risk
3. **Remove rehydration** (task 2) — clean removal, no runtime impact
4. **Remove PromptRenderer** (task 3) — clean removal, no runtime impact
5. **Trim compaction events** (task 4) — moderate effort
6. **Simplify compaction store** (task 8) — moderate effort
7. **Clean up exports** (task 6) — depends on tasks 1–5
8. **Consolidate TokenBudget** (task 5) — highest risk, do last

After each step:
- Run `uv run ruff check agent_framework/_harness/ tests/harness/`
- Run `uv run pytest tests/harness/ -v`
- Run the REPL with: *"Investigate this repo and find the python based workflow
  engine. Research the code and create a detailed architectural design."*
- Verify compaction fires and agent produces accurate output

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Harness source files | 32 | ~26 |
| Harness source lines | ~14,400 | ~10,000 |
| `__all__` exports | ~220 | ~80 |
| Test files | 12 | 12 |
| Test lines | ~6,800 | ~5,500 |
| Passing tests | 473/480 | ~440/440 (all green) |

---

## What NOT to Remove

These may look like candidates but are actively used:

- **`_renderers.py`**: Used by `devui_harness.py` for stream rendering. It's a
  UI utility, not core runtime, but it's wired and functional.
- **`_contract_verifier.py`** + **`_task_contract.py`**: Used by
  `StopDecisionExecutor` when a `TaskContract` is provided. Active runtime code.
- **`_compaction/_durability.py`**: Defines `ToolDurability` enum and
  `ToolResultEnvelope` used by compaction strategies. Active.
- **`_compaction/_turn_context.py`**: Small (158 lines) but used by strategies.
- **`_compaction/_summary.py`**: `StructuredSummary` used by summarizer. Active.
- **`_compaction/_adapters.py`**: Small (70 lines), bridges cache to strategies. Active.
- **`_sub_agents.py`**: Creates explore/run_task/document tools. Active.
