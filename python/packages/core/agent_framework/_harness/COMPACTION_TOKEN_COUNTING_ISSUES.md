# Compaction & Token Counting Issues — Detailed Breakdown

## Problem Summary

Context compaction never triggers in the harness, causing the agent to hit OpenAI's 128K context limit around turn 7-9 and fail. The root cause is that the token budget's `current_estimate` is never updated with accurate counts, so the compaction executor always sees 0 (or a wildly inaccurate number) and thinks it's "under budget."

---

## Architecture Overview

The harness workflow runs as a loop of executors in this order:

```
repair → compaction → agent_turn → stop_decision → repair (loop)
```

Each executor is an independent unit communicating via messages and SharedState.

### Key Executors

| Executor | File | Role |
|----------|------|------|
| `RepairExecutor` | `_repair_executor.py` | Entry point, kicks off each loop iteration |
| `CompactionExecutor` | `_compaction_executor.py` | Checks if context is under pressure, signals for compaction |
| `AgentTurnExecutor` | `_agent_turn_executor.py` | Runs the LLM agent, manages the message cache |
| `StopDecisionExecutor` | `_stop_decision_executor.py` | Decides whether to stop or loop |

### Key Data

| What | Where | Description |
|------|-------|-------------|
| **Message cache** | `AgentTurnExecutor._cache` (in-memory list) | The full conversation history: user messages, assistant responses, tool calls, tool results. This is what gets sent to the LLM. |
| **Token budget** | SharedState at `harness.token_budget` | A `TokenBudget` (from `_context_pressure.py`) with fields: `max_input_tokens`, `soft_threshold_percent`, `current_estimate` |
| **Tokenizer** | `CompactionExecutor._tokenizer` | A tiktoken-based `TiktokenTokenizer` (from `_compaction/_tokenizer.py`) that does accurate token counting |
| **Transcript** | SharedState at `harness.transcript` | A list of small metadata dicts (event_type, turn_number, etc.) — NOT the actual messages |

---

## The Bug: Nobody Updates `budget.current_estimate`

### How It Should Work

1. After each agent turn, someone counts the tokens in the message cache
2. That count is written to `budget.current_estimate` in SharedState
3. `CompactionExecutor.check_compaction()` reads the budget and checks `budget.is_under_pressure`
4. If under pressure, compaction strategies are applied to shrink the context

### What Actually Happens

1. `AgentTurnExecutor` runs the agent, accumulates messages in `self._cache`
2. **Nobody counts the cache tokens or updates the budget**
3. `CompactionExecutor.check_compaction()` reads the budget from SharedState
4. `budget.current_estimate` is still 0 (the default)
5. `budget.is_under_pressure` returns `False` (0 < 85000)
6. Compaction never triggers
7. Cache grows unbounded until OpenAI rejects the request at ~128K tokens

### Why This Happens

The architecture splits token counting knowledge across two executors:

- **`CompactionExecutor`** has the **tokenizer** (`self._tokenizer`, tiktoken-based, accurate) but does NOT have the **message cache** (that's private to `AgentTurnExecutor._cache`)
- **`AgentTurnExecutor`** has the **message cache** (`self._cache`) but did NOT have a **tokenizer** (until our recent change)

There is no mechanism for one executor to access another's private state. They communicate only through SharedState (dict-based key-value store) and messages.

### The Legacy System Worked

The older `ContextPressureExecutor` (from `_context_pressure_executor.py`) DID work:

```python
# _context_pressure_executor.py, line 96-98
transcript = await self._get_transcript(ctx)
budget.current_estimate = estimate_transcript_tokens(transcript)
```

But this only worked by accident — it read the transcript (small metadata events) and used a rough chars/4 estimate. The numbers were wrong but happened to be in a useful range because the old system was simpler.

---

## What We've Changed So Far (Partial Fix)

### 1. Gave `AgentTurnExecutor` a tokenizer

In `_agent_turn_executor.py`, when `enable_compaction=True`, we now create a tokenizer:

```python
# __init__
self._tokenizer = None
if enable_compaction:
    from ._compaction._tokenizer import get_tokenizer
    self._tokenizer = get_tokenizer()
```

### 2. Wired `enable_compaction` through the builder

In `_harness_builder.py`, the `AgentTurnExecutor` now receives `enable_compaction`:

```python
# Was missing before — enable_compaction was never passed
AgentTurnExecutor(
    ...,
    enable_compaction=enable_compact,
    ...
)
```

### 3. Added `_update_token_budget` method

After each turn, `AgentTurnExecutor` counts cache tokens and writes to SharedState:

```python
async def _update_token_budget(self, ctx):
    # Count tokens from cache contents
    current_tokens = 0
    for msg in self._cache:
        for content in msg.contents:
            if isinstance(content, TextContent):
                current_tokens += self._tokenizer.count_tokens(content.text)
            elif isinstance(content, FunctionResultContent):
                current_tokens += self._tokenizer.count_tokens(str(content.result))
            # ... etc

    budget.current_estimate = current_tokens
    await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
```

### 4. Reverted `CompactionExecutor` to read from budget

`check_compaction()` now simply reads `budget.current_estimate` (which `AgentTurnExecutor` writes):

```python
current_tokens = budget.current_estimate  # Set by AgentTurnExecutor
if not budget.is_under_pressure:
    # Under budget, no compaction needed
    ...
```

---

## Remaining Issue: Token Count Mismatch

Our token count is **~3x too low** compared to what OpenAI actually sees.

### Evidence

In experiment 7:
- Our count: 45,051 tokens at turn 7
- OpenAI's count: 135,189 tokens (returned in error: "context_length_exceeded")
- Ratio: ~3x

### Why the Mismatch

The message cache (`self._cache`) contains `ChatMessage` objects with typed content items:
- `TextContent` — the assistant's text responses
- `FunctionCallContent` — tool call metadata (name, arguments, call_id)
- `FunctionResultContent` — tool result strings (file contents, command output)

When counting tokens, we iterate over these content items and tokenize their text/result fields.

**However**, what OpenAI actually receives is the **full wire format** including:

1. **System prompt / agent instructions** — sent as a system message by the ChatClient, NOT in our cache
2. **Tool schemas** — all tool definitions serialized as JSON, sent alongside messages, NOT in our cache
3. **Injected guidance messages** — work item guidance (~600 tokens), tool strategy guidance (~300 tokens), planning prompt (~100 tokens) — these ARE in the cache but might not be counted correctly if they use `TextContent` vs plain text
4. **Message formatting overhead** — role tokens, separators, special tokens per message
5. **Content serialization** — the way `AzureOpenAIChatClient._prepare_message_for_openai()` serializes each message may produce more tokens than our raw content extraction

### The Serialization Path

When the agent framework sends messages to OpenAI:

```
self._cache (ChatMessage objects)
    → ChatAgent._prepare_run() builds messages
    → OpenAIChatClient._prepare_messages_for_openai() converts to OpenAI dicts
    → OpenAI SDK serializes to JSON
    → OpenAI API tokenizes with tiktoken
```

Our token counting runs on step 1 (raw `ChatMessage.contents`). The actual token count is at step 4. Every step in between adds overhead.

### Key File: `openai/_chat_client.py`

The method `_prepare_message_for_openai()` (line 389) converts each `ChatMessage` into one or more OpenAI-format dicts. The conversion is non-trivial:

- `FunctionCallContent` → `{"role": "assistant", "tool_calls": [{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}]}`
- `FunctionResultContent` → `{"role": "tool", "tool_call_id": ..., "content": result_string}`
- `TextContent` → `{"role": "...", "content": [{"type": "text", "text": ...}]}`

Each of these conversions adds structural tokens (keys, values, delimiters).

---

## Proposed Fix: Count Tokens on the Wire Format

Instead of counting tokens on raw `ChatMessage.contents`, we should count on the OpenAI-format messages (the same dicts that `_prepare_message_for_openai` produces).

### Approach A: Use the existing client serialization

The cleanest fix would be to:

1. In `_update_token_budget`, get a reference to the chat client's message preparation method
2. Convert each cache message to OpenAI format
3. Use the tokenizer's `count_messages()` method which expects OpenAI-format dicts

**Challenge**: `AgentTurnExecutor` doesn't have a reference to the chat client. It has `self._agent` (an `AgentProtocol`), but the client is buried inside.

### Approach B: Replicate the serialization logic

Create a lightweight utility that converts `ChatMessage` → OpenAI dict format, then use `count_messages()`.

**Challenge**: Duplicates logic from the client, may drift over time.

### Approach C: Have the client report token usage

After each agent invocation, the client receives `usage` information from the API response (prompt_tokens, completion_tokens). This is the most accurate count possible.

**Implementation**:
1. The `AgentRunResponse` (returned by `_run_agent()` / `_run_agent_streaming()`) contains `usage_details` 
2. `AgentTurnExecutor` can read `response.usage_details.prompt_tokens` after each turn
3. Write that to `budget.current_estimate`

This is the **most architecturally correct** approach because:
- Uses the authoritative token count from the API provider
- No estimation or serialization mismatch possible
- Already available in the response data (the `UsageContent` is already parsed)
- Works for any provider, not just OpenAI

### Approach D: Multiplier heuristic (last resort)

Apply a multiplier (e.g., 2.5-3x) to our content-level count to approximate the wire-level count.

**Not recommended** — fragile, provider-dependent, would need tuning per model.

---

## Recommended Fix: Approach C (Use API usage data)

### Where to look

1. **`AgentRunResponse`** (in `_types.py`): Check `response.usage_details` after `_run_agent()` and `_run_agent_streaming()`

2. **`UsageDetails`** class (in `_types.py`): Has `prompt_tokens` and `completion_tokens` fields

3. **`_agent_turn_executor.py`**: After each `_run_agent()` call (line ~287-290), read `response.usage_details.prompt_tokens` and write to budget

### Implementation steps

1. In `AgentTurnExecutor.run_turn()`, after calling `_run_agent()` or `_run_agent_streaming()`:
   ```python
   if response and response.usage_details:
       prompt_tokens = response.usage_details.prompt_tokens
       if prompt_tokens:
           budget.current_estimate = prompt_tokens
           # Save to SharedState
   ```

2. The `_update_token_budget` method can be simplified or removed — the API provides the ground truth.

3. Handle the case where `usage_details` is None (some providers/streams may not return it). Fall back to content-level counting in that case.

### Verification

After implementing, the debug logs should show realistic numbers:
```
AgentTurnExecutor: Updated token budget — 25000 tokens (threshold: 85000)  # turn 2
AgentTurnExecutor: Updated token budget — 55000 tokens (threshold: 85000)  # turn 5
CompactionExecutor: Context pressure detected (90000/85000 tokens)         # turn 7 — TRIGGERS!
```

---

## Other Issues Found Along the Way

### 1. CompactionExecutor.check_compaction() doesn't actually compact

Even when pressure IS detected, `check_compaction()` (lines 205-270) doesn't do any actual compaction work. It just:
- Creates/loads a `CompactionPlan`
- Sets `plan_updated = False` and `tokens_freed = 0`
- Saves the empty plan
- Signals `CompactionComplete`

The actual compaction (`compact_thread()`) is a separate method that's never called from the workflow. The comment at line 216-222 says "AgentTurnExecutor calls compact_thread() when it has access to messages" — but I don't see evidence of this happening.

**This means even if token counting is fixed, compaction may still not actually reduce the context.**

### 2. Two `TokenBudget` classes

There are TWO `TokenBudget` classes:
- **v1**: `_context_pressure.py` → Simple: `max_input_tokens`, `soft_threshold_percent`, `current_estimate`
- **v2**: `_compaction/_tokenizer.py` → Rich: includes `system_prompt_tokens`, `tool_schema_tokens`, `safety_buffer_tokens`, etc.

`CompactionExecutor.check_compaction()` uses **v1** (imported from `_context_pressure`).
`CompactionExecutor.compact_thread()` uses **v2** (imported from `_compaction`).

This is confusing and may cause budget threshold mismatches.

### 3. `AgentTurnExecutor._apply_compaction_plan()` deserializes empty plans

`_load_compaction_plan()` (line 377-401) always returns an empty plan:
```python
# Note: Full deserialization will be added when from_dict is added to CompactionPlan
return CompactionPlan.create_empty(thread_id=thread_id, thread_version=version)
```

So even if a compaction plan was created and saved to SharedState, it would be loaded back as empty — no messages would actually be compacted.

---

## File Reference

| File | Purpose | Key lines |
|------|---------|-----------|
| `_agent_turn_executor.py` | Runs agent turns, owns message cache | `self._cache` (line 82), `_update_token_budget` (line ~672), `_apply_compaction_plan` (line ~405) |
| `_compaction_executor.py` | Checks pressure, creates compaction plans | `check_compaction` (line 176), `compact_thread` (line ~280) |
| `_context_pressure.py` | v1 TokenBudget, legacy estimate functions | `TokenBudget` (line 190), `estimate_transcript_tokens` (line 473) |
| `_compaction/_tokenizer.py` | Tiktoken tokenizer, v2 TokenBudget | `TiktokenTokenizer` (line 163), `TokenBudget` (line 468) |
| `_harness_builder.py` | Builds the workflow, wires executors | `build()` (line 206), executor registration and edge wiring |
| `_constants.py` | SharedState key names | `HARNESS_TOKEN_BUDGET_KEY` (line 14) |
| `openai/_chat_client.py` | Converts ChatMessage → OpenAI format | `_prepare_message_for_openai` (line 389) |

All paths relative to: `python/packages/core/agent_framework/_harness/`
