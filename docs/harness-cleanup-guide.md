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

### 1. Remove Legacy Context Pressure System (~970 lines)

**What**: The Phase 2 context pressure system has been fully superseded by the
Phase 9 CompactionExecutor. The legacy code is only used when
`enable_context_pressure=True`, which no sample or production code uses.

**Files to remove**:
- `_context_pressure_executor.py` (464 lines) — legacy executor
- Three strategy classes in `_context_pressure.py`:
  - `ClearToolResultsStrategy` (~50 lines)
  - `CompactConversationStrategy` (~50 lines)
  - `DropOldestStrategy` (~50 lines)
  - `get_default_strategies()` function
  - `ContextEditKind`, `TranscriptRange`, `ExternalizeEdit`, `ClearEdit`,
    `CompactEdit`, `DropEdit`, `ContextEditPlan` dataclasses

**Keep in `_context_pressure.py`**:
- `TokenBudget` class — still used by CompactionExecutor for SharedState signaling
- `estimate_tokens()` and `estimate_transcript_tokens()` — utility functions

**Files to update**:
- `_harness_builder.py`: Remove `enable_context_pressure` parameter and the
  `ContextPressureExecutor` registration branch (lines ~326–365)
- `_harness_builder.py` (`AgentHarness`): Remove `enable_context_pressure` param
- `__init__.py`: Remove all context pressure exports except `TokenBudget`,
  `estimate_tokens`, `estimate_transcript_tokens`
- `_constants.py`: Remove `HARNESS_CONTEXT_EDIT_HISTORY_KEY` if only used by
  context pressure

**Tests to update**:
- `test_context_pressure.py` (567 lines): Remove tests for the three legacy
  strategies and `ContextPressureExecutor`. Keep tests for `TokenBudget` and
  the estimation functions.

**Validation**: Run `uv run pytest tests/harness/ -v` — count should drop by
~15-20 tests but all remaining tests must pass. Then run the REPL with the
standard test prompt to verify compaction still works.

---

### 2. Remove Unfinished Rehydration System (~613 lines)

**What**: The rehydration system (`_compaction/_rehydration.py`) is architectural
scaffolding that was never wired into the runtime. It defines
`RehydrationInterceptor`, `RehydrationState`, `RehydrationConfig`, etc. but
nothing in the executor chain instantiates or calls any of it.

**Files to remove**:
- `_compaction/_rehydration.py` (613 lines) — entire file

**Files to update**:
- `_compaction/__init__.py`: Remove all rehydration imports and `__all__` entries:
  `RehydrationBlockedEvent`, `RehydrationConfig`, `RehydrationEvent`,
  `RehydrationInterceptor`, `RehydrationResult`, `RehydrationState`,
  `create_rehydration_interceptor`
- `_harness/__init__.py`: Remove same rehydration entries from `__all__`
- `_compaction/_tokenizer.py`: The `TokenBudget` class has
  `rehydration_reserve_tokens` and `available_for_rehydration` — keep these
  fields (they affect budget math) but add a comment noting rehydration is
  not yet implemented

**Tests to update**:
- `test_compaction.py`: Remove any rehydration-specific tests (search for
  "rehydration" in test names/assertions). Keep tests for types that happen
  to mention rehydration in passing.

**Validation**: Run tests, verify compaction REPL still works.

---

### 3. Remove Unfinished PromptRenderer (~565 lines)

**What**: `_compaction/_renderer.py` defines a `PromptRenderer` that applies a
`CompactionPlan` to produce a rendered prompt. This was the intended architecture
but the actual runtime uses `AgentTurnExecutor._apply_compaction_plan()` instead.
`PromptRenderer` is never instantiated.

**Files to remove**:
- `_compaction/_renderer.py` (565 lines) — entire file

**Files to update**:
- `_compaction/__init__.py`: Remove `PromptRenderer`, `RenderedPrompt` imports
  and `__all__` entries. Also remove `COMPACTION_RENDER_FORMAT_VERSION` if only
  used by the renderer.
- `_harness/__init__.py`: Remove same entries from `__all__`

**Validation**: Run tests, verify compaction REPL still works.

---

### 4. Trim Compaction Event System (~700 lines → ~200 lines)

**What**: `_compaction/_events.py` defines 12 event subclasses, metrics
collectors, and emitter protocols. None of these are used outside the _harness
package — they're internal telemetry infrastructure. The `CompactionCoordinator`
doesn't actually emit most of these events. Only the base `CompactionEvent` and
a couple of emitters have any runtime usage.

**Action**: Don't delete the file, but audit each class:
- **Keep**: `CompactionEvent`, `CompactionEventType`, `CompactionMetrics`
  (if used by coordinator)
- **Remove if unused by coordinator**: Individual event subclasses
  (`CompactionCheckStartedEvent`, `ProposalGeneratedEvent`,
  `ContentClearedEvent`, etc.) — check if `CompactionCoordinator.compact()`
  actually creates instances of these
- **Remove from `__all__` exports**: All event types not used externally

**How to verify**: Search `_compaction/_strategies.py` and
`_compaction/_types.py` for imports from `_events`. Keep whatever is actually
instantiated; remove the rest.

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

### 7. Fix the 7 Failing JIT Instruction Tests

**What**: `test_jit_instructions.py` has 7 test failures because tests reference
old instruction IDs that were renamed:
- `no_reads_after_5_turns` → actual ID is `no_reads_after_3_turns`
- `no_writes_after_reads` → actual ID is `no_deliverable_after_many_reads`

**Files to fix**:
- `tests/harness/test_jit_instructions.py`: Update instruction IDs in test
  methods and any threshold values that changed (e.g., turn >= 5 → turn >= 3)

**Validation**: All 480 tests should pass after this fix (currently 473 pass, 7 fail).

---

### 8. Remove Unused Compaction Store Complexity (~697 lines → ~150 lines)

**What**: `_compaction/_store.py` defines `CompactionStore` protocol,
`InMemoryCompactionStore`, `CompactionTransaction`, `SecurityContext`, and
elaborate versioning/locking infrastructure. In practice, only
`InMemoryCompactionStore` is ever used, and the versioning/locking is never
exercised because the harness runs single-threaded.

**Action**: Simplify `_store.py`:
- Keep: `CompactionStore` protocol, `InMemoryCompactionStore`
- Remove: `CompactionTransaction`, `SecurityContext`, file-system store stubs,
  elaborate version conflict handling
- Simplify `InMemoryCompactionStore` to just store/load without transaction
  overhead

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
