# Context Compaction Implementation Progress

**Last Updated**: 2026-01-22
**Branch**: `harness-poc`
**Status**: All Phases Complete (1-9)

## Overview

This document tracks the implementation progress of the production-quality context compaction system for the Agent Harness. The system is based on an **immutable-log + compaction-plan + renderer** architecture.

### Key Principles
1. **AgentThread is never mutated** - it's the append-only source of truth
2. **CompactionPlan is pure data** describing what spans are compacted
3. **PromptRenderer** takes Thread + Plan to produce the actual model request

## File Structure

```
python/packages/core/agent_framework/_harness/_compaction/
├── __init__.py              # Package exports (all types exported)
├── _types.py                # Core types: SpanReference, CompactionPlan, records
├── _summary.py              # StructuredSummary and related types
├── _durability.py           # ToolDurability, ToolResultEnvelope, DeterminismMetadata
├── _turn_context.py         # TurnContext, RehydrationConfig, RehydrationResult
├── _tokenizer.py            # ProviderAwareTokenizer, TiktokenTokenizer, TokenBudget
├── _renderer.py             # PromptRenderer, ArtifactStore protocol, RenderedPrompt
├── _strategies.py           # CompactionStrategy, ClearStrategy, SummarizeStrategy, etc.
├── _rehydration.py          # RehydrationInterceptor, ToolCall, RehydrationState
├── _store.py                # CompactionStore, SummaryCache, InMemory implementations
├── _events.py               # CompactionEvent types, MetricsCollector, EventEmitters
├── CONTEXT_COMPACTION_DESIGN.md  # Full design document
└── IMPLEMENTATION_PROGRESS.md    # This file
```

## Completed Phases

### Phase 1: Core Data Structures ✅

**File**: `_types.py`

| Type | Description | Status |
|------|-------------|--------|
| `SpanReference` | Self-sufficient reference to message span (stores explicit message_ids list) | ✅ |
| `CompactionAction` | Enum: INCLUDE, CLEAR, SUMMARIZE, EXTERNALIZE, DROP | ✅ |
| `COMPACTION_PRECEDENCE` | Precedence ordering (DROP > EXTERNALIZE > SUMMARIZE > CLEAR > INCLUDE) | ✅ |
| `ExternalizationRecord` | Record of externalized content with artifact_id | ✅ |
| `SummarizationRecord` | Record of summarized content with summary | ✅ |
| `ClearRecord` | Record of cleared content with preserved_fields | ✅ |
| `DropRecord` | Record of dropped content with reason | ✅ |
| `CompactionPlan` | Complete plan with normalized action map | ✅ |

**Key Implementation Details**:
- `SpanReference.message_ids` is an explicit list, not start/end indices
- `CompactionPlan.__post_init__()` builds normalized action map
- `CompactionPlan._build_action_map()` handles precedence and overlap detection
- `CompactionPlan.rebuild_action_map()` public method for updating after modifications
- `CompactionPlan.get_action(message_id)` returns `(action, record)` tuple

### Phase 2: Provider-Aware Tokenizer ✅

**File**: `_tokenizer.py`

| Type | Description | Status |
|------|-------------|--------|
| `ProviderAwareTokenizer` | Protocol for tokenizers | ✅ |
| `TiktokenTokenizer` | Accurate tiktoken-based counting | ✅ |
| `SimpleTokenizer` | Fallback character-based estimation | ✅ |
| `ModelProvider` | Enum: OPENAI, ANTHROPIC, AZURE_OPENAI, UNKNOWN | ✅ |
| `TokenBudget` | Budget with overhead tracking and rehydration reserves | ✅ |
| `get_tokenizer()` | Factory function with fallback | ✅ |

**Key Implementation Details**:
- `MODEL_ENCODING_MAP` maps model names to tiktoken encodings
- `TokenBudget.for_model()` creates budget with appropriate context limits
- Model lookup uses ordered list (more specific patterns first: gpt-4o before gpt-4)
- `TiktokenTokenizer.count_message()` handles multipart content and tool calls
- `TokenBudget` tracks: system_prompt_tokens, tool_schema_tokens, formatting_overhead, safety_buffer, rehydration_reserve

**Dependencies**:
- `tiktoken` (optional, falls back to SimpleTokenizer)

### Phase 3: Prompt Renderer ✅

**File**: `_renderer.py`

| Type | Description | Status |
|------|-------------|--------|
| `ArtifactStore` | Protocol for storing externalized content | ✅ |
| `ArtifactMetadata` | Metadata for artifacts (thread_id, sensitivity, ttl, etc.) | ✅ |
| `SecurityContext` | Access control context (requester_id, tenant_id, permissions) | ✅ |
| `RenderedPrompt` | Result of rendering with statistics | ✅ |
| `PromptRenderer` | Main renderer class | ✅ |
| `render_summary_text()` | Standalone function for summary rendering | ✅ |
| `render_externalization_text()` | Standalone function for externalization rendering | ✅ |

**Key Implementation Details**:
- `PromptRenderer.render()` applies CompactionPlan to AgentThread
- Summaries and externalizations render as `role="assistant"` messages
- Synthetic messages are inserted at position of first message in span
- One message per span (never merge adjacent summaries)
- Rehydrated content is injected at end (ephemeral, not part of canonical log)
- Format is versioned: `COMPACTION_RENDER_FORMAT_VERSION = "v1.0"`

**Rendering Actions**:
- `DROP`: Skip message entirely
- `INCLUDE`: Include message as-is
- `CLEAR`: Replace with placeholder preserving key fields
- `SUMMARIZE`: Replace span with summary (once per span)
- `EXTERNALIZE`: Replace span with pointer + summary (once per span)

### Phase 4: Compaction Strategies ✅

**File**: `_strategies.py`

| Type | Description | Status |
|------|-------------|--------|
| `CompactionProposal` | Proposed compaction action | ✅ |
| `CompactionStrategy` | Abstract base class for strategies | ✅ |
| `Summarizer` | Protocol for LLM-based summarization | ✅ |
| `ClearStrategy` | Clear tool results respecting durability | ✅ |
| `SummarizeStrategy` | LLM-compress older spans | ✅ |
| `ExternalizeStrategy` | Store to artifact store | ✅ |
| `DropStrategy` | Remove content entirely (last resort) | ✅ |
| `CompactionResult` | Result of compaction run | ✅ |
| `CompactionCoordinator` | Orchestrates strategies | ✅ |

**Strategy Aggressiveness Order** (lower = less aggressive, applied first):
1. `ClearStrategy` (aggressiveness=1)
2. `SummarizeStrategy` (aggressiveness=2)
3. `ExternalizeStrategy` (aggressiveness=3)
4. `DropStrategy` (aggressiveness=4)

**Key Implementation Details**:

**ClearStrategy**:
- Respects durability: EPHEMERAL (always clear), ANCHORING (keep key fields), REPLAYABLE (check determinism), NON_REPLAYABLE (don't clear)
- Configurable: `min_tokens_to_clear`, `preserve_recent_turns`
- Uses `DEFAULT_DURABILITY_POLICIES` for tool-specific defaults

**SummarizeStrategy**:
- Requires `Summarizer` protocol implementation
- Configurable: `target_token_ratio`, `min_span_tokens`, `min_span_messages`, `preserve_recent_turns`
- Skips if `turn_context.should_skip_aggressive_compaction()` (rehydration happened)

**ExternalizeStrategy**:
- Requires `ArtifactStore` and `Summarizer`
- Prioritizes NON_REPLAYABLE content
- Configurable: `externalize_threshold_tokens`, `preserve_recent_turns`, `default_sensitivity`

**DropStrategy**:
- Only drops already-cleared EPHEMERAL content
- Most conservative - requires content to have been cleared first

**CompactionCoordinator**:
- Applies strategies in order of aggressiveness
- Stops when `tokens_to_free` target is met
- Configurable: `max_proposals_per_run`
- Returns `CompactionResult` with plan, tokens_freed, proposals stats

## Supporting Types (Completed)

### Durability Types (`_durability.py`) ✅

| Type | Description |
|------|-------------|
| `ToolDurability` | Enum: EPHEMERAL, ANCHORING, REPLAYABLE, NON_REPLAYABLE |
| `DeterminismMetadata` | Metadata to verify REPLAYABLE results (content_hash, etag, mtime, version) |
| `ToolDurabilityPolicy` | Policy for tool result handling |
| `ToolResultEnvelope` | Structured envelope for tool results |
| `DEFAULT_DURABILITY_POLICIES` | Default policies for common tools |

### Summary Types (`_summary.py`) ✅

| Type | Description |
|------|-------------|
| `Decision` | A decision made during conversation |
| `OpenItem` | Unresolved item/TODO with priority |
| `ArtifactReference` | Reference to externalized artifact |
| `ToolOutcome` | Summary of tool result |
| `StructuredSummary` | Drift-resistant structured summary |
| `SummaryCacheKey` | Key for cached summaries |

**Schema Versioning**: `STRUCTURED_SUMMARY_SCHEMA_VERSION = "v1.0"`, `SUMMARY_RENDER_VERSION = "v1.0"`

### Turn Context Types (`_turn_context.py`) ✅

| Type | Description |
|------|-------------|
| `TurnContext` | Turn-level state (turn_number, rehydration tracking) |
| `RehydrationResult` | Result of rehydrating an artifact |
| `RehydrationConfig` | Configuration for auto-rehydration |

**Rehydration Loop Breaking**:
- `TurnContext.should_skip_aggressive_compaction()` returns True if rehydration happened
- Prevents oscillation: rehydrate → pressure → externalize → rehydrate loop

### Phase 5: Rehydration ✅

**File**: `_rehydration.py`

| Type | Description | Status |
|------|-------------|--------|
| `ToolCall` | Pending tool call data | ✅ |
| `RehydrationEvent` | Event emitted when artifact rehydrated | ✅ |
| `RehydrationBlockedEvent` | Event emitted when rehydration blocked | ✅ |
| `RehydrationState` | State tracking for rehydration cooldowns | ✅ |
| `RehydrationInterceptor` | Intercepts agent requests and auto-rehydrates | ✅ |
| `create_rehydration_interceptor()` | Factory function | ✅ |

**Key Implementation Details**:
- Detects artifact ID references in agent messages (regex pattern)
- Detects artifact IDs in tool call arguments (nested dict traversal)
- Sensitivity gating (only auto-rehydrate public/internal by default)
- Cooldown period to prevent re-injection loops (configurable)
- Budget enforcement (max tokens per artifact, total budget)
- Truncation for large artifacts with `[TRUNCATED]` marker

### Phase 6: Concurrency and Storage ✅

**File**: `_store.py`

| Type | Description | Status |
|------|-------------|--------|
| `CompactionStore` | Protocol for storing compaction plans | ✅ |
| `InMemoryCompactionStore` | In-memory implementation with thread-safety | ✅ |
| `CompactionTransaction` | Transactional wrapper with version tracking | ✅ |
| `SummaryCache` | Protocol for caching LLM summaries | ✅ |
| `InMemorySummaryCache` | In-memory cache with LRU eviction and TTL | ✅ |
| `CacheEntry` | Cache entry with expiration tracking | ✅ |
| `InMemoryArtifactStore` | In-memory artifact storage | ✅ |
| `ArtifactStoreEntry` | Artifact entry with metadata | ✅ |
| `compute_content_hash()` | SHA256 hash for cache keys | ✅ |
| `create_summary_cache_key()` | Factory for summary cache keys | ✅ |

**Key Implementation Details**:
- `CompactionStore.get_current_plan()` returns `(plan, version)` tuple
- `CompactionStore.commit_plan()` with optimistic concurrency (version check)
- Summary caching with content hash + schema version key
- LRU eviction when cache at capacity
- TTL-based expiration with cleanup method
- Thread-safe implementations using `threading.Lock()`

### Phase 7: Observability ✅

**File**: `_events.py`

| Type | Description | Status |
|------|-------------|--------|
| `CompactionEventType` | Enum of event types | ✅ |
| `CompactionEvent` | Base class for all events | ✅ |
| `CompactionCheckStartedEvent` | Budget check begins | ✅ |
| `CompactionCompletedEvent` | Full compaction cycle done | ✅ |
| `ProposalGeneratedEvent` | Strategy proposes action | ✅ |
| `ProposalRejectedEvent` | Proposal was rejected | ✅ |
| `ContentClearedEvent` | Content cleared | ✅ |
| `ContentSummarizedEvent` | Content summarized | ✅ |
| `ContentExternalizedEvent` | Content externalized | ✅ |
| `ContentDroppedEvent` | Content dropped | ✅ |
| `ContentRehydratedEvent` | Content rehydrated | ✅ |
| `RehydrationBlockedEvent` | Rehydration was blocked | ✅ |
| `CompactionErrorEvent` | Error occurred | ✅ |
| `VersionConflictEvent` | Version conflict | ✅ |
| `CompactionMetrics` | Aggregated statistics | ✅ |
| `CompactionEventEmitter` | Protocol for event emission | ✅ |
| `LoggingEventEmitter` | Logs events for debugging | ✅ |
| `MetricsCollector` | Collects metrics from events | ✅ |
| `CompositeEventEmitter` | Delegates to multiple emitters | ✅ |

**Key Implementation Details**:
- All events include thread_id, turn_number, timestamp, and metadata
- Events serialize to dict for logging/metrics backends
- MetricsCollector tracks per-thread and global statistics
- CompositeEventEmitter allows multiple observers

### Phase 8: Testing ✅

**File**: `tests/harness/test_compaction.py`

| Test Category | Description | Status |
|---------------|-------------|--------|
| `TestSpanReference` | SpanReference creation, properties, validation | ✅ |
| `TestCompactionPlan` | Plan creation, action lookup, precedence | ✅ |
| `TestStructuredSummary` | Summary creation with decisions, items, outcomes | ✅ |
| `TestSimpleTokenizer` | Character-based token counting | ✅ |
| `TestTokenBudgetV2` | Budget defaults, threshold, pressure detection | ✅ |
| `TestInMemoryCompactionStore` | Store operations, version conflict | ✅ |
| `TestCompactionTransaction` | Transaction lifecycle | ✅ |
| `TestInMemorySummaryCache` | Cache get, put, invalidate | ✅ |
| `TestInMemoryArtifactStore` | Store, retrieve, delete artifacts | ✅ |
| `TestCompactionMetrics` | Metrics recording | ✅ |
| `TestMetricsCollector` | Global/per-thread metrics | ✅ |
| `TestRehydrationInterceptor` | Interceptor creation, config | ✅ |
| `TestRehydrationState` | State creation, serialization | ✅ |
| `TestToolCall` | ToolCall creation, serialization | ✅ |
| `TestUtilityFunctions` | Hash computation, cache key creation | ✅ |

**Test Results**: 41 tests passing

## Integration Points

### With AgentThread
- `ChatMessage.message_id` used for compaction tracking
- Messages without IDs are included as-is (no compaction)
- `AgentThread.message_store.list_messages()` provides message list

### With Harness
- Exports available from `agent_framework._harness`
- `TokenBudgetV2` aliased to avoid conflict with existing `TokenBudget`
- Integrates with harness turn executor via `TurnContext`

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SpanReference storage | Explicit message_ids list | Self-sufficient, no off-by-one bugs |
| Summary role | `assistant` | Safe (not system), compatible with all providers |
| Synthetic message position | First message in span | Preserves chronological order |
| Rehydrated content | Ephemeral (not in log) | Prevents compaction loops |
| Strategy order | Aggressiveness-based | Least invasive first |
| Drop requirements | Must be cleared EPHEMERAL first | Maximum safety |

## Running Checks

```bash
# From python/packages/core directory
cd python/packages/core

# Lint check
uv run ruff check agent_framework/_harness/_compaction/

# Type check
uv run pyright agent_framework/_harness/_compaction/

# Auto-fix lint issues
uv run ruff check --fix --unsafe-fixes agent_framework/_harness/_compaction/

# Run tests
uv run pytest tests/harness/test_compaction.py -v
```

## Completed

All 9 phases of the context compaction implementation are complete:

1. **Core Data Structures** - SpanReference, CompactionPlan, records
2. **ProviderAwareTokenizer** - Tiktoken-based and simple tokenizers
3. **PromptRenderer** - Applies compaction plan to produce rendered prompts
4. **Compaction Strategies** - Clear, Summarize, Externalize, Drop
5. **Rehydration Interceptor** - Automatic content restoration
6. **Concurrency and Storage** - Optimistic locking, caching
7. **Observability Events** - Lifecycle events, metrics collection
8. **Testing Suite** - 41 tests covering all components
9. **Harness Integration** - CompactionExecutor, AgentTurnExecutor enhancements

## Phase 9: Harness Integration ✅

### Architecture Decision: Plan Pipeline

After analyzing flexibility requirements (supporting filesystem for Claude Code, government-approved databases for cloud deployments), we chose the **Plan Pipeline** architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Executor Chain (Lightweight)                  │
│  SharedState carries: plans, metadata, decisions (small, typed) │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                         ┌─────────────┐       │
│  │ Compaction  │  ← CompactionPlan →     │ AgentTurn   │       │
│  │  Executor   │  (in SharedState)       │  Executor   │       │
│  └─────────────┘                         └─────────────┘       │
│        │                                       │                │
│        ▼                                       ▼                │
│  ┌─────────────────────────────────────────────────────────────┐
│  │              Storage Protocol Layer                         │
│  │  CompactionStore | ArtifactStore | SummaryCache             │
│  └─────────────────────────────────────────────────────────────┘
│        │                                       │                │
│        ▼                                       ▼                │
│  ┌─────────────────────────────────────────────────────────────┐
│  │  Implementation (deployment-specific)                       │
│  │  • InMemory (default/testing)                               │
│  │  • FileSystem (Claude Code)                                 │
│  │  • Database (Government Cloud)                              │
│  └─────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

**Key Benefits**:
1. Only small CompactionPlan flows through SharedState (not large message caches)
2. Storage protocols are injectable at configuration time
3. Each deployment provides their own protocol implementations
4. AgentTurnExecutor manages messages internally using whatever storage is configured

### New Files

| File | Description | Status |
|------|-------------|--------|
| `_compaction_executor.py` | CompactionExecutor that produces CompactionPlan | ✅ |

### Changes to Existing Files

| File | Change | Status |
|------|--------|--------|
| `_constants.py` | Add `HARNESS_COMPACTION_PLAN_KEY`, `HARNESS_COMPACTION_METRICS_KEY` | ✅ |
| `_agent_turn_executor.py` | Apply CompactionPlan, `enable_compaction` flag | ✅ |
| `__init__.py` | Export `CompactionExecutor`, `CompactionComplete` | ✅ |

### SharedState Flow

```
RepairComplete
     │
     ▼
┌─────────────────────┐
│ CompactionExecutor  │
│  - Check budget     │
│  - Run coordinator  │
│  - Store plan       │
└─────────────────────┘
     │
     │ SharedState["harness.compaction_plan"] = CompactionPlan
     │
     ▼
CompactionComplete (extends RepairComplete)
     │
     ▼
┌─────────────────────┐
│ AgentTurnExecutor   │
│  - Load plan        │
│  - Apply via        │
│    PromptRenderer   │
│  - Run agent        │
└─────────────────────┘
```

### Configuration

```python
# Default (in-memory storage)
harness = AgentHarness(
    agent,
    enable_compaction=True,
)

# Custom storage (government cloud example)
harness = AgentHarness(
    agent,
    enable_compaction=True,
    compaction_store=ApprovedDatabaseStore(connection_string),
    artifact_store=ApprovedBlobStore(container),
    summary_cache=ApprovedCacheStore(cluster),
)
```

### New Types

| Type | Description |
|------|-------------|
| `CompactionExecutor` | Executor that checks token budget and coordinates compaction |
| `CompactionComplete` | Message indicating compaction check is complete (extends RepairComplete) |

### Key Implementation Details

- **CompactionExecutor**:
  - Checks token budget from SharedState
  - Signals context pressure via `HarnessLifecycleEvent`
  - Stores CompactionPlan in SharedState
  - Provides `compact_thread()` method for when thread access is available

- **AgentTurnExecutor** enhancements:
  - New `enable_compaction` constructor parameter
  - Loads CompactionPlan from SharedState before running agent
  - Applies compaction to message cache via `_apply_compaction_plan()`
  - Supports CLEAR, SUMMARIZE, EXTERNALIZE, DROP actions

## Future Work

- Add `from_dict()` to `CompactionPlan` for full deserialization
- Real LLM-based summarizer implementation
- Persistent storage implementations (Redis, SQLite, Azure Blob)
- Performance benchmarking
- End-to-end integration tests

## Public API

All types are exported from `agent_framework._harness._compaction` and `agent_framework._harness`:

**Core Types**: `SpanReference`, `CompactionPlan`, `CompactionAction`, `ClearRecord`, `SummarizationRecord`, `ExternalizationRecord`, `DropRecord`

**Summary Types**: `StructuredSummary`, `Decision`, `OpenItem`, `ToolOutcome`, `ArtifactReference`, `SummaryCacheKey`

**Durability Types**: `ToolDurability`, `ToolResultEnvelope`, `DeterminismMetadata`, `ToolDurabilityPolicy`

**Tokenizer Types**: `ProviderAwareTokenizer`, `TiktokenTokenizer`, `SimpleTokenizer`, `TokenBudget`, `ModelProvider`

**Renderer Types**: `PromptRenderer`, `RenderedPrompt`, `ArtifactStore`, `ArtifactMetadata`, `SecurityContext`

**Strategy Types**: `CompactionStrategy`, `ClearStrategy`, `SummarizeStrategy`, `ExternalizeStrategy`, `DropStrategy`, `CompactionCoordinator`, `CompactionProposal`, `CompactionResult`, `Summarizer`

**Rehydration Types**: `RehydrationInterceptor`, `RehydrationState`, `ToolCall`, `RehydrationEvent`, `RehydrationBlockedEvent`

**Storage Types**: `CompactionStore`, `InMemoryCompactionStore`, `CompactionTransaction`, `SummaryCache`, `InMemorySummaryCache`, `CacheEntry`, `InMemoryArtifactStore`, `ArtifactStoreEntry`

**Event Types**: `CompactionEvent`, `CompactionEventType`, `CompactionCheckStartedEvent`, `CompactionCompletedEvent`, `ProposalGeneratedEvent`, `ProposalRejectedEvent`, `ContentClearedEvent`, `ContentSummarizedEvent`, `ContentExternalizedEvent`, `ContentDroppedEvent`, `ContentRehydratedEvent`, `VersionConflictEvent`, `CompactionErrorEvent`, `CompactionMetrics`, `CompactionEventEmitter`, `LoggingEventEmitter`, `MetricsCollector`, `CompositeEventEmitter`

**Turn Context Types**: `TurnContext`, `RehydrationConfig`, `RehydrationResult`

**Executor Types**: `CompactionExecutor`, `CompactionComplete`

**Functions**: `get_tokenizer()`, `render_summary_text()`, `render_externalization_text()`, `create_rehydration_interceptor()`, `compute_content_hash()`, `create_summary_cache_key()`

## Notes

- All code passes ruff and pyright checks
- Uses `from __future__ import annotations` for forward references
- TYPE_CHECKING imports for circular dependency prevention
- Dataclasses with `field(default_factory=lambda: [])` for mutable defaults
- Public methods added where needed to avoid protected member access
