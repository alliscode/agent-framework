# Harness Compaction Configuration Guide

This document describes how context compaction works in the agent harness and how to configure it in `harness_test_runner.py` (or any script using `AgentHarness`).

## TL;DR

The `harness_test_runner.py` script is **already correctly configured** for compaction. No changes are needed. The key settings are:

```python
harness = AgentHarness(
    agent,
    enable_compaction=True,
    max_input_tokens=100_000,
    soft_threshold_percent=0.85,
    # Optional stores (not needed for basic clearing):
    compaction_store=InMemoryCompactionStore(),
    artifact_store=InMemoryArtifactStore(),
    summary_cache=InMemorySummaryCache(max_entries=100),
)
```

## How Compaction Works End-to-End

### The Executor Loop

When `enable_compaction=True`, the harness workflow runs this loop:

```
repair → compaction → agent_turn → stop_decision → repair → ...
```

Each turn:

1. **CompactionExecutor.check_compaction()** — Reads the token budget from SharedState. If `budget.current_estimate > budget.soft_threshold`, sets `compaction_needed=True` on the `CompactionComplete` message.

2. **AgentTurnExecutor.run_turn()** — Receives the `CompactionComplete` trigger. If `compaction_needed=True`, calls `_apply_direct_clear()` which replaces large `FunctionResultContent` values in older cache entries with `"[Tool result cleared to save context]"`. Then runs the agent with the lightened cache.

3. **AgentTurnExecutor._update_token_budget()** — After the agent responds, reads `response.usage_details.input_token_count` (the authoritative count from the API, including system prompt, tool schemas, and all formatting overhead) and writes it to the SharedState budget. This count is what CompactionExecutor reads on the next iteration.

### Token Counting

The token budget uses **API-reported usage data** as the primary source:

| Source | What it measures | Accuracy |
|--------|-----------------|----------|
| `response.usage_details.input_token_count` (primary) | Everything the API saw: system prompt + tool schemas + all messages + formatting overhead | **Exact** |
| Tokenizer-based fallback | Only raw message content text | ~3x under-count |

The API count includes items invisible to the harness:
- System prompt tokens (~800–2,000 depending on instructions length)
- Tool schema tokens (~5,000–20,000 for a full toolset like CodingTools)
- Per-message formatting overhead (role tags, separators, JSON structure)
- Function call/result serialization overhead

For OpenAI/Azure OpenAI in streaming mode, usage data arrives in the final streaming chunk. The client requests this via `stream_options={"include_usage": True}` (already configured). The `AgentRunResponse.from_agent_run_response_updates()` method aggregates the `UsageContent` from that chunk into `response.usage_details`.

### What Gets Cleared

When compaction triggers, `_apply_direct_clear()`:

1. Preserves the most recent 2 cache entries (configurable via `preserve_recent_turns`)
2. For all older entries, scans for `FunctionResultContent` items
3. Replaces any result longer than 100 characters with `"[Tool result cleared to save context]"`
4. Modifies the cache **in-place** — cleared content is permanently removed from the in-memory cache

This is the "clear" strategy — the lightest form of compaction. It targets tool results because:
- Tool results (file contents, command outputs) are typically the largest items
- The agent has already processed them and formed its understanding
- The agent can re-read files if needed

## Configuration Parameters

### Required

| Parameter | Value | Effect |
|-----------|-------|--------|
| `enable_compaction` | `True` | Adds `CompactionExecutor` to the workflow loop |

### Token Budget

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_input_tokens` | `100_000` | Maximum input token budget. Should be ≤ model context window. |
| `soft_threshold_percent` | `0.85` | Fraction of `max_input_tokens` at which compaction triggers. |

**Computed threshold**: `max_input_tokens × soft_threshold_percent = soft_threshold`

With the defaults: `100,000 × 0.85 = 85,000 tokens`

**Choosing `max_input_tokens`**:
- For GPT-4o (128K context): `100_000` to `120_000` is reasonable. Leave headroom for the model's output.
- For GPT-4o-mini (128K context): Same range.
- For GPT-4-turbo (128K context): Same range.
- Setting it too close to the model limit risks `context_length_exceeded` errors if the response pushes over.

**Choosing `soft_threshold_percent`**:
- `0.85` (default) — Compaction triggers when 85% full. Good balance between early action and not being too aggressive.
- `0.75` — More aggressive. Clears earlier, more headroom for large responses.
- `0.90` — Less aggressive. Only clears when close to the limit.

### Optional Stores (Future Use)

These stores enable advanced compaction strategies that aren't currently wired through the executor chain but are fully implemented in the `_compaction` package:

| Parameter | Purpose | Needed Now? |
|-----------|---------|-------------|
| `compaction_store` | Persists compaction plans across checkpoints | No — basic clearing doesn't need it |
| `artifact_store` | Stores externalized content (for externalize strategy) | No — not wired yet |
| `summary_cache` | Caches LLM summaries (for summarize strategy) | No — not wired yet |

Providing them is harmless and forward-compatible. When the full `CompactionCoordinator` pipeline is wired in the future, these stores will enable the more advanced strategies (summarize, externalize, drop).

## Verifying Compaction is Working

### Log Messages to Look For

With `logging.getLogger("agent_framework._harness").setLevel(logging.DEBUG)`, you should see:

**1. Budget updates after each turn** (INFO level):
```
AgentTurnExecutor: Updated token budget — 45230 tokens (method: api_usage, threshold: 85000)
```

The `method: api_usage` confirms the API-reported count is being used. The count should grow as the conversation progresses.

**2. Pressure detection** (INFO level, when threshold exceeded):
```
CompactionExecutor: Context pressure detected (87500/85000 tokens, 2500 over threshold)
```

**3. Cache clearing** (INFO level, when compaction applied):
```
AgentTurnExecutor: Cleared 15 old tool results to reduce context pressure
```

**4. No pressure** (DEBUG level, on turns where budget is fine):
```
CompactionExecutor: Under budget (45230/85000 tokens)
```

### Expected Timeline

For a typical research task (read many files, produce a document):

| Turn Range | Expected Tokens | Compaction? |
|-----------|----------------|-------------|
| 1–3 | 10,000–30,000 | No |
| 4–6 | 40,000–70,000 | No |
| 7–9 | 75,000–100,000 | Likely triggers |
| 10+ | Should stay under threshold after clearing | May trigger again if agent reads many new files |

After clearing, the token count drops because large tool results are replaced with ~40-byte placeholders. On the next turn, the API will report a lower `input_token_count`.

## Troubleshooting

### Compaction never triggers

1. **Check the budget logs**: Look for `"Updated token budget"` messages. If `method: api_usage` and tokens are growing but never reaching `soft_threshold`, the threshold may be set too high.
2. **Check the model**: Not all providers return usage data in streaming mode. If you see `method: tokenizer`, the fallback is being used — it under-counts by ~3x and may never reach the threshold. The OpenAI and Azure OpenAI clients both support streaming usage.

### Compaction triggers but context still grows

1. **Check what's being cleared**: If most messages contain small tool results (<100 chars), they won't be cleared (the 100-char threshold skips tiny results).
2. **Agent is re-reading files**: After clearing, if the agent reads the same files again, the context grows back. This is expected — the agent may need to re-read important files.

### `context_length_exceeded` errors despite compaction

1. **Lower `soft_threshold_percent`** to trigger earlier (e.g., `0.75`).
2. **Lower `max_input_tokens`** to be more conservative.
3. **Check if the agent instructions or tool schemas are very large** — these count toward the input tokens but can't be compacted.

## Complete Working Example

```python
from agent_framework._harness import AgentHarness
from agent_framework._harness._compaction import (
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
)

harness = AgentHarness(
    agent,
    max_turns=50,
    # -- Compaction configuration --
    enable_compaction=True,
    max_input_tokens=100_000,       # Budget cap
    soft_threshold_percent=0.85,    # Trigger at 85% = 85,000 tokens
    # -- Optional stores (forward-compatible) --
    compaction_store=InMemoryCompactionStore(),
    artifact_store=InMemoryArtifactStore(),
    summary_cache=InMemorySummaryCache(max_entries=100),
)

# Run with streaming
async for event in harness.run_stream("Your prompt here"):
    # Process events...
    pass
```

No additional configuration beyond what `harness_test_runner.py` already has is needed.
