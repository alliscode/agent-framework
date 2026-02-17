# Harness Workflow Refactor Design

## Context

This design addresses two goals:

1. Fix **compaction ownership ambiguity** (`CompactionExecutor` signals pressure while `AgentTurnExecutor` performs most compaction work).
2. Make harness structure **cleaner, easier to reason about, and safer to evolve** (with explicit regression controls).

Related analysis: `python/packages/core/agent_framework/_harness/HARNESS_ARCHITECTURE_DEEP_DIVE.md`.

## Problem Statement

Current shape is functional but structurally mixed:

- `CompactionExecutor` is mainly a pressure detector.
- `AgentTurnExecutor` owns turn execution plus compaction implementation details plus guidance injection plus token accounting.
- Cross-cutting logic is concentrated in a single large executor (`_agent_turn_executor.py`).

This increases risk for behavior regressions and makes ownership unclear for future changes.

## Design Principles

1. **Behavior-preserving first**: refactor internals before changing control flow.
2. **Single-responsibility runtime units**: each executor/service should have one primary concern.
3. **Framework-reusable runtime units**: extracted units should be designed so they can be adopted outside this harness (as composable `agent-framework` features), with clear interfaces and minimal harness-specific coupling.
4. **Feature-flagged migration** for any topology/ownership shift.
5. **Observable parity**: old vs new paths produce comparable metrics and lifecycle events.
6. **No data model churn without payoff**: avoid costly shared-state payload changes until needed.

## Non-Goals

- No rewrite of task contracts, work item tools, or hook APIs.
- No abrupt breaking changes to lifecycle event types.
- No immediate conversion to many tiny executors if overhead and complexity outweigh benefits.

## Target Architecture (End State)

### Executors (conceptual)

1. `RepairExecutor`
- Initialize/repair invariants.

2. `CompactionExecutor` (**single compaction owner**)
- Decide when compaction runs.
- Execute compaction strategy pipeline.
- Publish compaction outcome + compacted prompt view metadata.

3. `TurnPreparationExecutor` (optional split)
- Inject dynamic turn-time context (JIT, work-item state/reminders, continuation nudges if needed before call).

4. `AgentInvocationExecutor` (core run)
- Run agent streaming/non-streaming with tools + middleware.

5. `TurnPostprocessExecutor` (optional split)
- Determine `agent_done`, update token budget, write transcript/lifecycle turn summary.

6. `StopDecisionExecutor`
- Stop/continue decision policy.

### Flow

`repair -> compaction -> prep -> invoke -> postprocess -> stop -> repair`

For first migration phases, keep current external flow and extract internal services first.

## Reusability Impact

The new principle changes implementation details in important ways.

## 1. Extraction target changes: reusable first, harness adapter second

For each extracted responsibility, define:

- a framework-level interface (stable, typed, harness-agnostic),
- then a thin harness adapter that maps harness state/events into that interface.

This avoids creating helper classes that are only usable from `_harness`.

## 2. Module placement recommendations

Use this placement bias during refactor:

- Keep in `_harness` (policy/local orchestration):
  - `RepairExecutor`, `StopDecisionExecutor`, harness-specific policy profiles
  - harness lifecycle event mapping and UI-specific event shaping

- Candidate shared runtime modules (outside `_harness` once stable):
  - `TurnBuffer` contracts + implementations
  - token budget/accounting service contracts
  - transcript/event append utility contracts
  - generic turn preparation pipeline primitives

Practical landing strategy:

1. Land initially under `_harness` with neutral names and no harness-key coupling.
2. Add adoption in one additional workflow/harness.
3. Promote to shared runtime package after proving reuse.

## 3. Contract discipline updates

Treat new units as public-ish contracts from day one:

- explicit request/response dataclasses (no ad-hoc dict payloads),
- clear error semantics and fallback contracts,
- minimal dependency on shared-state key strings,
- no direct dependency on DevUI/lifecycle event enums.

## 4. Testing updates for reusable units

In addition to harness integration tests:

- add unit contract tests that run without harness workflow runtime,
- add compatibility tests ensuring harness adapters preserve existing behavior,
- add at least one cross-consumer test (second consumer path) before promotion.

## 5. Documentation updates for reusable units

Each extracted reusable unit should ship with:

- API contract doc,
- “how to integrate” snippet,
- known invariants and failure modes,
- migration notes if replacing an older harness-local path.

## Migration Strategy (Low Regression)

## Implementation Approach Recommendation

Use **phase-by-phase delivery**, not a single all-at-once merge.

Reason:

- compaction ownership is a control-flow concern with high regression risk,
- extracting reusable runtime units is easier to verify in isolation,
- shadow mode only has value if there is a stable baseline phase to compare against.

Recommended execution model:

1. One owning engineer (or agent) for architecture integration and acceptance gates.
2. Parallel sub-workstreams by unit/phase where safe (service extraction, test hardening, telemetry).
3. Merge in phase order with required gate pass at each boundary.

This keeps velocity high while preserving debuggability and rollback safety.

## Phase 0: Lock behavior with test/invariant expansion

Before structural changes:

- Add/strengthen tests around:
  - turn completion semantics (`work_complete`, continuation count, tool-call detection)
  - compaction lifecycle events and thresholds
  - token budget update contracts
  - stop decision outcomes in edge cases
- Add invariant assertions and metrics:
  - monotonic turn count
  - consistent transcript append behavior
  - compaction event pairs (`started`/`completed`)

Exit criteria:

- baseline tests green
- baseline metrics snapshot captured

## Phase 1: Internal extraction (no topology change)

Keep existing executor graph. Refactor `AgentTurnExecutor` into internal collaborators:

- `TurnPromptAssembler`
  - initial message injection
  - work-item state/reminders
  - JIT instruction injection
  - continuation prompt creation

- `TurnCompactionService`
  - plan load/apply
  - full compaction execution
  - fallback direct clear

- `TurnTokenBudgetService`
  - cache token counting
  - usage-overhead reconciliation
  - budget persistence

- `TurnToolingService`
  - runtime tool composition
  - middleware composition

No behavior changes expected; this is pure organization.

Reusability requirement for this phase:

- define interfaces and DTOs as if they will be shared,
- keep harness specifics in adapter classes.

Exit criteria:

- no user-visible behavior changes
- code ownership clearer
- same tests and event outputs

Phase 1 status (completed on February 13, 2026):

- Extracted and wired internal runtime services in `python/packages/core/agent_framework/_harness/_turn_services.py`:
  - `TurnToolingService`
  - `TurnInvocationService`
  - `TurnInitialMessageService`
  - `TurnJitInstructionService`
  - `TurnGuidanceService`
  - `TurnPostprocessService`
  - `TurnTokenBudgetService`
  - `TurnBudgetSyncService`
  - `TurnSystemMessageService`
  - `TurnCheckpointService`
  - `TurnWorkItemPromptService`
  - `TurnPromptAssembler`
  - `TurnCompactionService`
  - `TurnCompactionTelemetryService`
  - `TurnCompactionViewService`
  - `TurnOwnerModeService`
  - `TurnCompactionEngine`
  - `TurnPreambleService`
  - `TurnContinuationService`
  - `TurnCompletionSignalService`
  - `TurnErrorHandlingService`
  - `TurnLoggingService`
  - `TurnExecutionCoordinator`
  - `TurnEventWriter`
  - `TurnOutcomeEvaluator`
  - `TurnWorkItemStateService`
- `AgentTurnExecutor` now acts as orchestration glue while behavior remains parity-locked by:
  - `python/packages/core/tests/harness/test_harness_phase0_benchmark.py`
  - `python/packages/core/tests/harness/test_compaction_pipeline.py`
  - `python/packages/core/tests/harness/test_compaction.py`
  - `python/packages/core/tests/harness/test_turn_services.py`

Phase 1 incremental status (continued February 13, 2026):

- Added `TurnInvocationService` to centralize agent invocation responsibilities:
  - non-streaming invocation + run-event emission,
  - streaming invocation + update-event emission/response assembly,
  - response append to turn buffer + work-item change emission.
- Rewired `AgentTurnExecutor` to delegate invocation execution to this service,
  removing duplicated inline streaming/non-streaming execution branches.
- Added direct `TurnInvocationService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnPostprocessService` to centralize turn finalization responsibilities:
  - transcript `agent_response` append,
  - optional work-item ledger sync,
  - outcome evaluation and token-budget update,
  - owner-mode shared-buffer publish,
  - `turn_completed` lifecycle emission payload shaping.
- Rewired `AgentTurnExecutor` to delegate completion/finalization to this service,
  further reducing inline post-agent branching in `run_turn`.
- Added direct `TurnPostprocessService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnCompactionViewService` to centralize turn-side compaction plan state/view behavior:
  - compaction plan load from shared state,
  - plan application over current turn buffer,
  - owner-mode aware message-view selection for invocation.
- Rewired `AgentTurnExecutor` and `TurnCompactionEngine` to consume this service
  for turn-side compaction plan load/apply semantics.
- Added direct `TurnCompactionViewService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnPreambleService` to centralize turn-start orchestration:
  - turn counter increment/persistence,
  - max-turn resolution with default fallback,
  - `turn_started` lifecycle emission,
  - `turn_start` transcript event append.
- Rewired `AgentTurnExecutor` to delegate turn preamble setup to this service,
  making `run_turn` flow cleaner (`preamble -> prep/compact -> invoke -> postprocess`).
- Added direct `TurnPreambleService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnCompletionSignalService` to centralize `TurnComplete` signaling
  semantics (success/pending/error) from turn orchestration.
- Rewired `AgentTurnExecutor` to delegate completion message sends to this service
  for response-none, normal-complete, and error paths.
- Added direct `TurnCompletionSignalService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnErrorHandlingService` to centralize error-path behavior:
  - transcript error append (`agent_response` with error payload),
  - completion error signaling (`TurnComplete.error`).
- Rewired `AgentTurnExecutor` exception path to delegate to this service,
  reducing inline error-branch logic in `run_turn`.
- Added direct `TurnErrorHandlingService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnLoggingService` to centralize turn-level logging decisions
  (start, compaction, invocation timing, outcome, error).
- Rewired `AgentTurnExecutor` to delegate top-level logging branches to this
  service, reducing orchestration noise in `run_turn`.
- Added `TurnOwnerModeService` to centralize turn-side compaction owner-mode
  resolution from shared state with normalized fallback semantics.
- Rewired `AgentTurnExecutor` and `TurnCompactionService` to use this service
  (removing duplicated owner-mode key reads on the turn path).
- Added direct `TurnOwnerModeService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnCompactionTelemetryService` to centralize turn-side compaction
  lifecycle/shadow telemetry emission:
  - `compaction_completed` payload/event shaping,
  - shadow `context_pressure.shadow_compare` payload/event shaping.
- Rewired `TurnCompactionService` to delegate telemetry emission to this service,
  separating compaction decision execution from event payload logic.
- Added direct `TurnCompactionTelemetryService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnExecutionCoordinator` to centralize turn sequence orchestration:
  - `preamble -> shared-buffer sync -> prompt prep -> compaction -> invoke ->
    postprocess -> completion signal` plus error-path delegation.
- Rewired `AgentTurnExecutor.run_turn` to delegate orchestration to this
  coordinator, leaving executor as composition/adaptation glue.
- Expanded `TurnTokenBudgetService` with read access (`get_budget_estimate`) and
  rewired turn-side compaction/postprocess budget reads through this service,
  removing remaining inline budget-state read logic from `AgentTurnExecutor`.
- Added `TurnContinuationService` to centralize continuation prompt behavior:
  - continuation-count state reads/writes,
  - prompt construction with open work-item context,
  - transcript/lifecycle continuation event emission.
- Rewired `AgentTurnExecutor`/`TurnOutcomeEvaluator` continuation callbacks to
  use this service, removing continuation state/event mutation logic from executor.
- Added direct `TurnContinuationService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnInitialMessageService` to centralize initial-message hydration
  from shared state (including string-to-`ChatMessage` normalization).
- Added `TurnBudgetSyncService` to centralize turn-side budget update+log flow,
  and rewired compaction/postprocess budget update callbacks through this service.
- Added `TurnJitInstructionService` to centralize JIT context derivation and
  instruction injection into the turn buffer.
- Added `TurnWorkItemPromptService` to centralize turn-time work-item prompt
  behavior:
  - tool-usage extraction from turn buffer,
  - incomplete-work-item state prompt injection,
  - stop-decision-driven reminder prompt injection.
- Rewired `AgentTurnExecutor` prompt/budget/JIT paths to use these services,
  removing more state/logic helpers from executor internals.
- Added direct service coverage in
  `python/packages/core/tests/harness/test_turn_services.py` for:
  - `TurnInitialMessageService`,
  - `TurnBudgetSyncService`,
  - `TurnJitInstructionService`,
  - `TurnWorkItemPromptService`.
- Extracted large static prompt templates into
  `python/packages/core/agent_framework/_harness/_turn_prompt_catalog.py`
  and introduced `TurnGuidanceService` for guidance prompt injection.
- Rewired `TurnPromptAssembler` callback wiring in `AgentTurnExecutor` to use
  `TurnGuidanceService`, removing in-executor prompt constants/methods.
- Rewired `HarnessGuidanceProvider` in
  `python/packages/core/agent_framework/_harness/_context_providers.py` to use
  prompt catalog constants directly (instead of executor class constants).
- Added direct `TurnGuidanceService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.
- Added `TurnSystemMessageService` to centralize system-message append behavior
  for turn-time injections.
- Added `TurnCheckpointService` to centralize turn-buffer checkpoint
  save/restore and initial-message seeding.
- Rewired `AgentTurnExecutor` continuation/work-item invariant prompts to use
  `TurnSystemMessageService`, and checkpoint lifecycle hooks to use
  `TurnCheckpointService`.
- Added direct service coverage in
  `python/packages/core/tests/harness/test_turn_services.py` for:
  - `TurnSystemMessageService`,
  - `TurnCheckpointService`.
- Expanded `AgentTurnExecutor` dependency-injection seams for extracted turn services:
  - `TurnInitialMessageService`,
  - `TurnGuidanceService`,
  - `TurnSystemMessageService`,
  - `TurnCheckpointService`,
  - `TurnWorkItemPromptService`,
  - `TurnBudgetSyncService`,
  - `TurnJitInstructionService`,
  - `TurnContinuationService`,
  - `TurnExecutionCoordinator`.
- Added DI seam coverage in
  `python/packages/core/tests/harness/test_turn_services.py` proving injected:
  - checkpoint service is used by checkpoint/initial-message hooks,
  - initial/guidance/JIT services are used by turn preparation callbacks.

## Phase 2: Introduce compaction ownership mode flag

Add config:

- `compaction_owner_mode: "agent_turn" | "compaction_executor" | "shadow"`
- default is `"compaction_executor"` (with `"agent_turn"` retained as rollback mode).

Modes:

- `agent_turn`: rollback compatibility path.
- `shadow`: `CompactionExecutor` runs candidate logic but does not drive runtime mutation; emits comparison telemetry.
- `compaction_executor`: new owner path active.

This enables safe parallel validation before switching ownership.

Phase 2 status (in progress, updated February 13, 2026):

- Added `compaction_owner_mode` plumbing through builder/harness config with validation and default now `"compaction_executor"`.
- Extracted shared owner-mode contract helpers into
  `python/packages/core/agent_framework/_harness/_compaction_owner_mode.py`
  and rewired builder/repair/turn/state paths to use normalized mode handling,
  eliminating duplicated owner-mode tuple checks and fallback logic.
- Persisted owner mode in shared state during repair initialization.
- Added shadow-mode candidate decision snapshot from `CompactionExecutor` (non-mutating).
- Added shadow compare telemetry from turn-time compaction path, including divergence flag between candidate and actual effective compaction behavior.
- Added non-mutating candidate compaction simulation in `CompactionExecutor` (shadow mode only)
  using shared turn-buffer snapshots, including candidate strategies/tokens metadata.
- Active runtime ownership is `"compaction_executor"` by default, with canary gates/telemetry.

## Phase 3: Shared turn buffer abstraction

Main blocker for single ownership today: only `AgentTurnExecutor` owns in-memory `_cache`.

Introduce abstraction:

- `TurnBuffer` interface:
  - `load_messages()`
  - `save_messages()`
  - `append_messages()`
  - `snapshot_id/version()`

Implementations:

- `ExecutorLocalTurnBuffer` (current path, no data movement).
- `SharedStateTurnBuffer` (serialized via existing conversation encode/decode utility).

In shadow mode, wire `CompactionExecutor` against `SharedStateTurnBuffer` while `AgentTurnExecutor` still uses local buffer; compare resulting compacted views + token deltas.

Reusability requirement for this phase:

- `TurnBuffer` contract should not reference harness key names or event types,
- serialization strategy should be pluggable so non-harness runtimes can adopt it.

Exit criteria:

- parity metrics acceptable
- no major perf regressions

Phase 3 status (started, updated February 13, 2026):

- Added `TurnBuffer` protocol and `ExecutorLocalTurnBuffer` implementation in
  `python/packages/core/agent_framework/_harness/_turn_buffer.py`.
- Wired `AgentTurnExecutor` to own an `ExecutorLocalTurnBuffer` and use it for
  response appends, prompt injections, checkpoint restore, and initial message replacement.
- Wired `TurnCompactionEngine` to consume `TurnBuffer` instead of a captured list
  reference, removing stale-cache reference risk after cache replacement.
- Added unit coverage in `python/packages/core/tests/harness/test_turn_services.py`
  for local turn buffer mutation/version behavior.
- Added `SharedStateTurnBuffer` in `python/packages/core/agent_framework/_harness/_turn_buffer.py`
  using existing conversation encode/decode utilities for checkpoint-safe payloads.
- Wired shadow-mode buffer snapshot publication from `AgentTurnExecutor` and included
  buffer parity fields in shadow compare telemetry (`local` vs `shared` message_count/version).
- Added `TurnBufferSyncService` in `python/packages/core/agent_framework/_harness/_turn_services.py`
  to centralize owner/shadow turn-buffer synchronization semantics:
  - owner-mode shared snapshot adoption,
  - shadow-mode snapshot publication,
  - owner-mode end-of-turn snapshot publication.
- Rewired `AgentTurnExecutor` to delegate turn-buffer sync concerns to this service,
  reducing per-turn mode-check branching in the executor.
- Added direct `TurnBufferSyncService` coverage in
  `python/packages/core/tests/harness/test_turn_services.py`.

## Phase 4: Activate compaction executor ownership

Switch flag default (controlled rollout):

- canary: selected tests/sample harness runs
- then broader default

Behavior:

- `CompactionExecutor` performs full strategy ladder and writes compacted view/plan artifacts.
- `AgentInvocationExecutor` consumes prepared prompt view and no longer executes compaction logic.

Keep fallback:

- if compaction path errors, degrade gracefully to previous turn view + continue (with telemetry).

Phase 4 status (canary started, updated February 13, 2026):

- Added `compaction_executor` canary branch in `CompactionExecutor`:
  - attempts owner-mode compaction from shared turn-buffer snapshot,
  - stores resulting compaction plan + updated budget when successful,
  - applies compaction plan to the shared turn-buffer snapshot and republishes
    the compacted snapshot for next-turn adoption,
  - includes owner-mode direct-clear fallback when strategy ladder yields empty plan
    under recent-message protection windows (bootstrap-safe owner-path progress),
  - emits owner-path `compaction_completed` telemetry and sets `CompactionComplete(compaction_needed=False)`.
- Added guarded fallback behavior:
  - if snapshot is missing/invalid, plan is empty, or owner-path compaction fails,
    it falls back to existing `agent_turn` ownership signaling (`compaction_needed=True`).
- Added fallback reason telemetry (`owner_fallback_reason`) on fallback-path lifecycle events
  to make canary diagnosis explicit.
- Added `AgentTurnExecutor` shared-buffer adoption/publish hooks for `compaction_executor` mode:
  - adopts newer shared snapshot before turn prep,
  - publishes local snapshot at end of turn for next compaction cycle,
  - skips plan-application prompt-view mutation when `compaction_executor` owns compaction.
- Extracted shared compaction rendering/mutation helpers into
  `python/packages/core/agent_framework/_harness/_compaction_view.py` and rewired:
  - `AgentTurnExecutor` plan view rendering path,
  - `CompactionExecutor` owner-mode plan/direct-clear mutation paths.
  This removed duplicate compaction view logic and centralized tool-pairing invariants.
- Extracted owner/fallback telemetry payload shaping into
  `python/packages/core/agent_framework/_harness/_compaction_telemetry.py`
  and rewired `CompactionExecutor` to use typed payload builders for:
  - owner-path metrics + `compaction_completed`,
  - fallback `compaction_started`,
  - fallback `context_pressure`,
  reducing inline dict drift risk during rollout.
- Replaced ad-hoc owner-attempt dict plumbing with typed `OwnerCompactionResult`
  in `CompactionExecutor`, including test override updates in compaction pipeline tests.
- Canonicalized owner fallback reasons with typed `OwnerFallbackReason` enum while
  preserving stable string telemetry values (`owner_fallback_reason`) for dashboards/tests.
- Extracted owner-path execution internals into dedicated
  `python/packages/core/agent_framework/_harness/_compaction_owner.py`
  (`CompactionOwnerService`), and changed `CompactionExecutor` owner path to
  orchestration/delegation only.
- Extracted shadow-candidate logic into
  `python/packages/core/agent_framework/_harness/_compaction_shadow.py`
  (`CompactionShadowService`) and delegated shadow publish/simulation from
  `CompactionExecutor`.
- Extracted compaction shared-state key plumbing into
  `python/packages/core/agent_framework/_harness/_compaction_state.py`
  (`CompactionStateStore`) so `CompactionExecutor` reads/writes budget/plan/metrics
  through typed accessors rather than inline key handling.
- Added direct contract tests for extracted compaction services in
  `python/packages/core/tests/harness/test_compaction_services.py`:
  - `CompactionOwnerService`,
  - `CompactionShadowService`,
  - `CompactionStateStore`.
- Added `CompactionPolicy` in
  `python/packages/core/agent_framework/_harness/_compaction_policy.py`
  and moved core threshold/owner-path/plan-presence flow decisions behind
  this policy abstraction, with direct policy coverage in compaction service tests.
- Added `CompactionLifecycleEmitter` in
  `python/packages/core/agent_framework/_harness/_compaction_lifecycle.py`
  and delegated compaction lifecycle event emission (`compaction_completed`,
  `compaction_started`, `context_pressure`) from `CompactionExecutor`.
- Added shared compaction helpers in
  `python/packages/core/agent_framework/_harness/_compaction_helpers.py`
  for pressure recovery math and strategy-summary normalization, and rewired
  owner/shadow/executor call paths to remove duplicated helper logic.
- Simplified owner-path delegation contract in `CompactionExecutor` by removing
  obsolete owner-attempt parameters (`plan`, `version`) after service extraction,
  further reducing orchestration-path coupling.
- Added optional dependency-injection seams on `CompactionExecutor` for
  `CompactionStateStore`, `CompactionOwnerService`, `CompactionShadowService`,
  `CompactionPolicy`, and `CompactionLifecycleEmitter`, with integration coverage
  via injected-policy behavior test in compaction pipeline tests.
- Expanded DI seam coverage in compaction pipeline tests to include:
  - injected owner-service behavior,
  - injected thread-service delegation path.
- Expanded DI seam coverage further to include:
  - injected shadow-service behavior,
  - injected state-store behavior.
- Added DI seam coverage for injected lifecycle emitter behavior:
  - fallback path (`compaction_started` + `context_pressure`),
  - owner path (`compaction_completed` via owner-completed emission).
- Extracted thread-access compaction execution into
  `python/packages/core/agent_framework/_harness/_compaction_thread.py`
  (`CompactionThreadService`) and delegated `CompactionExecutor.compact_thread`
  to this service, with direct service tests added.
- Refactored owner-path attempt handling in `CompactionExecutor` into a focused
  private orchestration helper to reduce `check_compaction` branch complexity
  and keep main handler flow linear/readable.
- Reduced turn-time ownership-mode coupling in `AgentTurnExecutor` by resolving
  `compaction_owner_mode` once per turn and threading it through compaction view
  selection and shared turn-buffer sync/adopt/publish paths.

Phase 4 status (promotion gates added, updated February 17, 2026):

- Added owner-fallback gate policy in
  `python/packages/core/agent_framework/_harness/_compaction_policy.py`:
  - bootstrap fallback window for shared-buffer bootstrap reasons,
  - always-allowed safety fallback for true owner-path execution failures,
  - explicit gate-violation signal when non-allowed fallback occurs.
- Wired gate policy through `CompactionExecutor` with rollout controls:
  - `owner_bootstrap_fallback_turn_limit`,
  - `enforce_owner_fallback_gate`.
- Extended compaction lifecycle + metrics payloads with owner-gate fields:
  - `owner_fallback_allowed`,
  - `owner_fallback_gate_violation`.
- Preserved injected-policy backward compatibility by gracefully defaulting gate
  behavior when older policy stubs do not implement new gate methods.
- Added coverage in compaction tests for:
  - fallback gate policy decisions,
  - owner fallback gate-violation telemetry in pipeline flow.

## Phase 5: Optional executor decomposition

After compaction ownership is stable, decide if splitting `AgentTurnExecutor` is worthwhile.

Recommended split threshold:

- proceed only if maintainability gain > runtime coordination overhead.

Minimum practical split:

- `AgentInvocationExecutor`
- `TurnPostprocessExecutor`

Keep `TurnPreparationExecutor` internal unless complexity demands externalization.

Phase 5 status (incremental, updated February 17, 2026):

- Added typed handoff contracts in
  `python/packages/core/agent_framework/_harness/_turn_services.py`:
  - `PreparedTurnContext`,
  - `InvocationResult`,
  - `CompactionDecision`.
- Rewired `TurnExecutionCoordinator`/`TurnInvocationService` to use these
  contracts instead of ad-hoc owner/streaming/usage argument plumbing.
- This preserves behavior while reducing implicit coupling between prep,
  invocation, and postprocess stages.

## Additional Structural Improvements

## 1. Message contract normalization between executors

Define explicit dataclasses for handoffs:

- `CompactionDecision` (pressure status, blocking flag, strategy summary)
- `PreparedTurnContext` (message view id, prep metadata)
- `InvocationResult` (response metadata + done signals + usage)

Reduces implicit coupling via shared-state keys and ad-hoc dicts.

## 2. Shared-state schema module

Consolidate harness shared-state read/write into a typed access layer:

- `HarnessStateStore` with typed getters/setters
- central serialization/deserialization logic

Benefits:

- fewer key mismatches
- easier migration of state fields
- cleaner tests

Progress update (February 17, 2026):

- Added `HarnessStateStore` in
  `python/packages/core/agent_framework/_harness/_state_store.py` as a typed
  shared-state accessor seam.
- Migrated `StopDecisionExecutor` to this store for high-churn state reads/writes
  (turn counts, retries, transcript append/read, completion status updates),
  reducing direct key-handling spread.
- Extended migration to full harness workflow paths:
  - `RepairExecutor`,
  - turn runtime services in `_turn_services.py`,
  - compaction state/owner/shadow services
    (`_compaction_state.py`, `_compaction_owner.py`, `_compaction_shadow.py`).
- After this migration, harness-key shared-state access now routes through
  `HarnessStateStore` (with `TurnBuffer` intentionally retaining its own generic
  shared-state boundary for pluggable buffer keys/serialization).

## 3. Transcript/event writer utility

Unify repeated `_append_event` patterns and lifecycle emissions into one utility.

Benefits:

- consistency
- easier auditing of event completeness

## 4. Stop policy profiles

Introduce explicit profile:

- `interactive` vs `strict_automation`

Clarifies liveness-vs-correctness tradeoffs and makes behavior easier to reason about in production.

Progress update (February 17, 2026):

- Added stop-policy profile model in
  `python/packages/core/agent_framework/_harness/_stop_policy.py` with:
  - `interactive` (current liveness-biased default),
  - `strict_automation` (no retries-exhausted auto-accept for `work_complete`).
- Wired profile selection through `HarnessWorkflowBuilder`/`AgentHarness`
  (`stop_policy_profile`) and persisted normalized profile to shared state during
  repair initialization.
- Extended `StopDecisionExecutor` with
  `accept_done_after_retries_exhausted` so strict profile behavior is explicit
  and testable.

## 5. Reusable runtime package boundary

Before final rollout, explicitly classify extracted units:

- `harness-only`: remains in `_harness`,
- `share-candidate`: move after second adopter validates value,
- `framework-core`: stable and documented for general use.

This prevents accidental long-term coupling of broadly useful units to harness internals.

## Regression Prevention Plan

## Test Strategy

1. Unit tests
- each extracted service class
- message contract mapping
- buffer adapters and serialization fidelity

2. Integration tests
- full harness loop in each compaction owner mode
- streaming and non-streaming parity
- checkpoint save/restore mid-run

3. Scenario benchmark test (phase gate)
- Add a benchmark-style E2E test that configures a harness with the same "full" shape as `python/samples/getting_started/workflows/harness/harness_repl.py` (compaction, work items, hooks, continuation prompts, etc.).
- Use one fixed challenging multi-turn request designed to trigger long-run behaviors (tool usage, retries/recovery, context growth, compaction, completion checks).
- Score and assert on stability dimensions important to long-running harnesses:
  - stop correctness (`done`/`failed`/`stalled` as expected),
  - turn efficiency (no runaway looping),
  - compaction correctness (triggering and no conversation corruption),
  - completion integrity (deliverables/work item/contract outcomes),
  - state/event invariants (transcript/lifecycle consistency, checkpoint safety).
- Persist a baseline snapshot (metrics + critical events + final assertions) and compare across refactor phases.
- Treat this test as a refactor phase gate: no phase advances without passing baseline parity (or explicitly approved deltas).

Current implementation status:

- Phase 0 gate test added at `python/packages/core/tests/harness/test_harness_phase0_benchmark.py`.
- The scenario now includes large tool result payloads so compaction effectiveness is actually exercised (not only compaction signaling).
- Added shadow parity gate in the same module (`test_phase0_harness_shadow_parity_gate`) to enforce
  `shadow_compare` presence, shared/local turn-buffer parity, and bounded divergence rate.
- Added canary owner-mode gate (`test_phase0_harness_compaction_executor_canary_gate`) for
  `compaction_owner_mode="compaction_executor"` with explicit assertions on ownership-path/fallback
  signaling plus core harness stability invariants.
  - tightened to require owner-path application and constrain fallback to bootstrap-only
    in the deterministic benchmark scenario.

4. Differential (shadow) tests
- same seeded inputs through old/new compaction ownership
- compare:
  - stop reason
  - turn count
  - compaction frequency
  - token estimates
  - final deliverables presence

5. Fault-injection tests
- compaction strategy exceptions
- summarizer failure
- shared-state decode errors

Validation checkpoint (February 13, 2026):

- Full harness refactor validation slice is green:
  - `python/packages/core/tests/harness/test_turn_services.py`
  - `python/packages/core/tests/harness/test_harness.py`
  - `python/packages/core/tests/harness/test_harness_phase0_benchmark.py`
  - `python/packages/core/tests/harness/test_compaction_pipeline.py`
  - `python/packages/core/tests/harness/test_compaction.py`

## Runtime Guardrails

- Feature flags with safe defaults.
- Error budgets/alerts for:
  - compaction failures per run
  - fallback activations
  - divergence in shadow mode beyond threshold
- Structured telemetry for each phase transition.

## Rollout Plan

1. Land Phase 1 extraction behind no new behavior.
2. Add Phase 2 mode flag + shadow instrumentation.
3. Run shadow in CI and dogfood environments.
4. Enable `compaction_executor` in canary.
5. Promote to default after stability window.

Rollback:

- single config switch back to `agent_turn`.

## Suggested Work Breakdown (Concrete)

1. Introduce service classes + refactor `AgentTurnExecutor` (pure extraction) with reusable contracts.
2. Add `compaction_owner_mode` plumbing in builder/config.
3. Add `TurnBuffer` abstraction and initial `ExecutorLocalTurnBuffer`.
4. Add `SharedStateTurnBuffer` and shadow mode comparisons.
5. Move compaction execution path into `CompactionExecutor` for active mode.
6. Remove dead/duplicated compaction branches in turn executor once stable.
7. Classify extracted units (`harness-only` vs `share-candidate`) and promote validated units.
8. Optional executor split (`invoke`/`postprocess`) if still justified.

## Expected Outcome

After this plan:

- Compaction has a single clear owner (with fallback path).
- Turn execution code is significantly easier to understand and maintain.
- Workflow structure is cleaner without high-risk big-bang rewrites.
- Migration risk is controlled via flags, shadowing, and parity telemetry.
