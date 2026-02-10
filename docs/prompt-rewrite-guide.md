# Prompt Rewrite Guide

> **Goal**: Replace all prompt text that is identical or closely derivative of GitHub Copilot CLI
> prompts with original text that achieves the same (or better) behavioral outcomes.
>
> **Why**: The `docs/prompt-parity-comparison.md` audit found several prompts that were modeled
> too closely on Copilot CLI's wording. We need original prompts that stand on their own.

---

## Scope

There are **5 items** to address — 4 prompt rewrites and 1 terminology rename. Code-level
patterns (JIT once-only firing, agentStop hook, etc.) are architectural patterns, not
copyrightable text, and are **out of scope**.

| # | What | File | Rating | Issue |
|---|------|------|--------|-------|
| **P1** | `DEFAULT_CONTINUATION_PROMPT` | `_agent_turn_executor.py:83` | IDENTICAL | Near word-for-word match to Copilot's continuation nudge |
| **P2** | `TASK_COMPLETION_INSTRUCTIONS` | `_context_providers.py:106` | VERY SIMILAR | Same structure/wording as Copilot's task_complete instructions |
| **P3** | `task_complete` tool docstring | `_done_tool.py:17` | VERY SIMILAR | Tool description follows Copilot's pattern closely |
| **P4** | `SUB_AGENT_GUIDANCE` (explore/task one-liners) | `_context_providers.py:119-120` | VERY SIMILAR | "<300 word answers" and "brief success, verbose failure" mirror Copilot |
| **P5** | Rename `task_complete` → `work_complete` | Multiple files | N/A | Aligns with our work item vocabulary instead of Copilot's terminology |

**Not in scope** (already original):
- `WORK_ITEM_GUIDANCE` — our own artifact/revision protocol
- `TOOL_STRATEGY_GUIDANCE` — our own progressive reading strategy
- `PLANNING_PROMPT` — our own iterative decomposition approach
- `RESPONSE_STYLE_GUIDANCE` — our own inline delivery rules
- All JIT instruction texts — original conditions and wording

All file paths are relative to `python/packages/core/agent_framework/_harness/`.

---

## Ordering

Do **P5 first** (or simultaneously with P1–P4), since P1, P2, and P3 all contain the text
`task_complete`. Writing new prompts with the old name and then renaming later means double work.

Recommended order:
1. **P5** — mechanical rename, low risk
2. **P3** — smallest surface area, lowest prompt risk
3. **P4** — small change, easy to validate
4. **P2** — medium risk, moderate surface area
5. **P1** — highest risk, fires on every incomplete turn

---

## P1. Rewrite `DEFAULT_CONTINUATION_PROMPT`

### Location
`_agent_turn_executor.py` line 83, class attribute on `AgentTurnExecutor`

### Current text (derivative — must be rewritten)
```
You have not yet marked the task as complete using the task_complete tool.
If you were planning, stop planning and start executing.
You are not done until you have fully completed the task.

IMPORTANT: Do NOT call task_complete if:
- You have open questions — use your best judgment and continue
- You encountered an error — try to resolve it or find an alternative
- There are remaining steps — complete them first

Keep working autonomously until the task is truly finished,
then call task_complete with a summary.
```

### Copilot's version (for reference — do NOT copy or closely paraphrase)
```
You have not yet marked the task as complete using the task_complete tool.
If you were planning, stop planning and start implementing.
You aren't done until you have fully completed the task.

IMPORTANT: Do NOT call task_complete if:
- You have open questions or ambiguities - use your best judgment and continue
- You encountered an error - try to resolve it or find an alternative approach
- There are remaining steps - complete them first

Keep working autonomously until the task is truly finished,
then call task_complete with a summary.
```

### When this fires
Injected as a user message whenever the agent tries to end its turn without calling
`task_complete` (or `work_complete` after P5). It's a nudge to keep working, not a
hard block. Can fire up to `max_continuation_prompts` times (default 5).

### Behavioral requirements
1. Remind the agent it hasn't signaled completion yet
2. Push the agent from analysis/planning into action
3. Prevent premature completion when errors, open questions, or remaining work exist
4. Instruct the agent to keep working independently
5. Direct the agent to call the completion tool when genuinely finished
6. **NEW**: Reference incomplete work items if any exist (our harness tracks these)

### Rewrite constraints
- Use a completely **different structure** (e.g., numbered steps, short paragraph +
  constraint list, two-part DO/DO NOT format)
- Use **different vocabulary** — avoid: "autonomously", "truly finished", "best judgment",
  "mark the task as complete"
- Consider framing **positively** ("continue working through…") rather than Copilot's
  negative-list approach ("Do NOT call if…")
- Can be shorter or longer — optimize for clarity and effectiveness
- **Litmus test**: If you read ONLY this prompt with no knowledge of Copilot's, would it
  seem independently authored? If not, revise further.
- After P5, use `work_complete` instead of `task_complete` in the text

### Example direction (do NOT use verbatim — write your own)
```
This work is still in progress — you have not called work_complete yet.

Action required:
1. If you have been planning, switch to execution now.
2. If you hit an error, work around it or try a different approach.
3. If there are remaining steps, complete them before finishing.
4. When all work is done and verified, call work_complete with a summary.

Do not stop early. Resolve uncertainties with your own judgment and keep going.
```

---

## P2. Rewrite `TASK_COMPLETION_INSTRUCTIONS`

### Location
`_context_providers.py` line 106, class attribute `TASK_COMPLETION_INSTRUCTIONS` on
`HarnessGuidanceProvider`. Injected into the **system prompt** (protected from compaction).

After P5, rename to `WORK_COMPLETION_INSTRUCTIONS`.

### Current text (derivative — must be rewritten)
```xml
<task_completion>
* A task is not complete until the expected outcome is verified and persistent
* After making changes, validate they work correctly
* If an initial approach fails, try alternative tools or methods before concluding
* You MUST call task_complete when finished — do not just stop responding
* Only call task_complete after all work items are done and deliverables are produced
</task_completion>
```

### Copilot's version (for reference — do NOT copy)
```
Use the task_complete tool to explicitly mark when you have finished.

IMPORTANT: Only call this tool when you are confident the task is fully complete.

When calling this tool:
- First, provide a brief 1-sentence summary in your message text
- Then call the tool with a more detailed summary

When to call:
- After you have completed ALL requested changes
- After tests pass (if applicable)
- After you have verified the solution works correctly
- When you are confident no further work is needed

When NOT to call:
- If you encountered errors you haven't resolved
- If there are remaining steps to complete
- If you're waiting for user input or clarification
- If you haven't verified your changes work
- If you're unsure whether the task is fully done

CRITICAL: Do NOT mark the task complete prematurely.
```

### Behavioral requirements
1. Make it clear the agent MUST actively call the completion tool (not just stop generating)
2. Completion requires verification — changes actually work
3. Agent should retry/pivot on failure before giving up
4. All work items and deliverables must be finished first
5. Include WHEN NOT to call guidance (our current version is weak here)
6. Include guidance on what to put in the summary parameter

### Rewrite constraints
- Our current version is too sparse (5 bullets). Copilot's is 17 lines. The rewrite should
  be more thorough while remaining original.
- Use a **different format** than both the current XML-wrapped bullets and Copilot's
  "When to / When NOT to" list structure.
- Consider a narrative or **scenario-based format**:
  "Before calling work_complete, verify: [checklist]. If any are false, keep working."
- Avoid Copilot's vocabulary: "fully complete", "high confidence", "truly finished",
  "err on the side of caution"
- After P5, use `work_complete` and `<work_completion>` tags

---

## P3. Rewrite `task_complete` Tool Docstring

### Location
`_done_tool.py` line 17, the docstring on the `task_complete` function (renamed to
`work_complete` after P5).

### Current text (derivative — must be rewritten)
```python
"""Signal that the current task is complete.

Call this tool when you have finished the user's request and have no more
actions to take. Provide a brief summary of what was accomplished.

Args:
    summary: Brief summary of what was accomplished.

Returns:
    Confirmation message.
"""
```

### Copilot's version (for reference — do NOT copy)
```
Mark the current task as complete.
Use this tool ONLY when you are confident the task is fully done.

Guidelines:
- Call this tool only after verifying your changes work correctly
- Include a brief summary of what was accomplished
- If you're unsure whether the task is complete, continue working instead
- Err on the side of caution - it's better to do extra verification
  than to mark incomplete work as done
```

### Behavioral requirements
1. Clearly convey this is the "I'm done" signal
2. The summary parameter should describe what was accomplished
3. Keep it brief — the heavy guidance lives in `WORK_COMPLETION_INSTRUCTIONS` (P2), not here

### Rewrite constraints
- Keep it short — 2-3 sentences for the description
- Use different phrasing than both "Signal that the current task is complete" and
  "Mark the current task as complete"
- The parameter description can stay generic but should be rephrased
- Focus the docstring on WHAT the tool does; leave WHEN/HOW guidance to P2
- After P5, the function is named `work_complete` so phrase accordingly

---

## P4. Rewrite Sub-Agent One-Liners in `SUB_AGENT_GUIDANCE`

### Location
`_context_providers.py` lines 119-120, inside the `SUB_AGENT_GUIDANCE` class attribute.

**Only the one-liner descriptions** need rewriting. The rest of `SUB_AGENT_GUIDANCE`
(parallel research block, document agent guidance) is original and does NOT need changes.

### Current text (derivative phrases in bold)
```
- explore(prompt): Fast codebase Q&A (cheap model, **<300 word answers**, parallel-safe)
- run_task(prompt): Execute commands — builds, tests, linting (**brief success, verbose failure**)
```

### Copilot's version (for reference — do NOT copy)
```yaml
# explore agent
CRITICAL: Keep your answer under 300 words

# task agent
On SUCCESS: Return brief one-line summary ("All 247 tests passed")
On FAILURE: Return full error output for debugging
```

### Behavioral requirements
1. **explore**: convey it's lightweight, fast, returns concise answers, can be called in parallel
2. **run_task**: convey it executes commands and only surfaces details on failure (success is quiet)

### Rewrite constraints
- Rephrase the descriptions. Instead of "<300 word answers" use a different way to express
  conciseness (e.g., "returns focused summaries", "short-form answers")
- Instead of "brief success, verbose failure" express the asymmetric reporting differently
  (e.g., "quiet on success, detailed on errors", "minimal output unless something fails")
- Keep them as one-liners — these appear in a bullet list

---

## P5. Rename `task_complete` → `work_complete`

### Why
The harness uses "work item" terminology everywhere (`WorkItem`, `WorkItemLedger`,
`work_item_add`, `work_item_update`, etc.) but the completion tool uses Copilot's "task"
vocabulary. This rename aligns our terminology and removes the most visible borrowed name.

### Suggested new name: `work_complete`
Feels natural alongside `work_item_add`, `work_item_update`, `work_item_list`.

### Files to change

This is a **mechanical rename** — no logic changes. Every change is a string or identifier swap.

#### Core tool definition (`_done_tool.py`)
| Line | Current | New |
|------|---------|-----|
| 3 | `"""Built-in task completion tool…"""` | Update module docstring |
| 10 | `TASK_COMPLETE_TOOL_NAME = "task_complete"` | `WORK_COMPLETE_TOOL_NAME = "work_complete"` |
| 13 | `@ai_function(name=TASK_COMPLETE_TOOL_NAME, ...)` | `@ai_function(name=WORK_COMPLETE_TOOL_NAME, ...)` |
| 14 | `def task_complete(` | `def work_complete(` |
| 17–27 | Tool docstring | New docstring (see P3) |
| 28 | Return string | Update text |
| 31 | `def get_task_complete_tool()` | `def get_work_complete_tool()` |
| 37 | `return task_complete` | `return work_complete` |

#### Agent turn executor (`_agent_turn_executor.py`)
| What | Change |
|------|--------|
| Import (line ~27) | `TASK_COMPLETE_TOOL_NAME` → `WORK_COMPLETE_TOOL_NAME`, `task_complete` → `work_complete` |
| `DEFAULT_CONTINUATION_PROMPT` (line ~84) | All `task_complete` → `work_complete` (also being rewritten per P1) |
| Event data key (line ~317) | `"called_task_complete"` → `"called_work_complete"` |
| `TurnComplete` field (line ~321) | `called_task_complete=` → `called_work_complete=` |
| Tool injection (lines ~357, 362, 415, 420) | `task_complete` → `work_complete` |
| Method `_has_task_complete_call` (line ~824) | Rename to `_has_work_complete_call` |
| All log messages and comments | Update references |

#### Context providers (`_context_providers.py`)
| What | Change |
|------|--------|
| `TASK_COMPLETION_INSTRUCTIONS` | Rename to `WORK_COMPLETION_INSTRUCTIONS` (also being rewritten per P2) |
| XML tags | `<task_completion>` → `<work_completion>` |
| All prompt text | `task_complete` → `work_complete` |

#### State (`_state.py`)
| What | Change |
|------|--------|
| `TurnComplete.called_task_complete` (line ~272) | Rename to `called_work_complete` |
| Docstring | Update |

#### Stop decision executor (`_stop_decision_executor.py`)
| What | Change |
|------|--------|
| `require_task_complete` param (line ~68) | → `require_work_complete` |
| `self._require_task_complete` (line ~88) | → `self._require_work_complete` |
| Comments, log messages, event data | Update all references |

#### Hooks (`_hooks.py`)
| What | Change |
|------|--------|
| `AgentStopHookEvent.called_task_complete` (line ~54) | → `called_work_complete` |

#### Harness builder (`_harness_builder.py`)
| What | Change |
|------|--------|
| `require_task_complete=True` (line ~393) | → `require_work_complete=True` |

#### JIT instructions (`_jit_instructions.py`)
| What | Change |
|------|--------|
| Prompt text (line ~77) | `task_complete` → `work_complete` |

#### Work items (`_work_items.py`)
| What | Change |
|------|--------|
| Regex pattern (line ~86) | `\btask_complete\b` → `\bwork_complete\b` |

#### Public API (`__init__.py`)
| What | Change |
|------|--------|
| Imports (line ~182) | `TASK_COMPLETE_TOOL_NAME, get_task_complete_tool, task_complete` → new names |
| `__all__` entries | Update to new names |

#### Tests (5 files)
| File | Approx. changes |
|------|----------------|
| `test_context_providers.py` | 1 reference |
| `test_harness.py` | 1 reference |
| `test_hooks.py` | ~13 references |
| `test_task_contract.py` | 1 reference |
| `test_work_items.py` | ~12 references |

All test changes are mechanical renames of tool names, field names, and string assertions.

### Backward compatibility

Add **deprecated aliases** in `_done_tool.py` after the rename:
```python
# Deprecated aliases — use work_complete instead
task_complete = work_complete
TASK_COMPLETE_TOOL_NAME = WORK_COMPLETE_TOOL_NAME
get_task_complete_tool = get_work_complete_tool
```

Keep old names in `__init__.py` `__all__` with a deprecation comment. Remove after one
release cycle.

---

## Validation Strategy

Prompt rewrites are high-risk — subtle wording changes can degrade agent behavior.

### Step 1: Baseline capture (before any changes)
Run this test prompt 3× with current prompts and record metrics:

**Test prompt**: *"Investigate this repo and find the python based workflow engine. Research
the code and create a detailed architectural design."*

**Metrics to record**:
| Metric | What to capture |
|--------|----------------|
| Turns taken | Total turns before task_complete |
| Tool calls | Total count, breakdown by tool type |
| File reads | Number of read_file calls |
| Deliverable size | Approximate KB of output |
| Deliverable quality | Correct focus? Specific references? Sufficient depth? |
| Completion tool called? | Yes/No |
| Continuation nudges | How many fired? |

### Step 2: Rewrite one prompt at a time
Change only ONE prompt constant per experiment. This isolates which rewrite caused any
regression. Follow the order in the **Ordering** section above.

### Step 3: Compare each rewrite against baseline

| Metric | Regression Threshold | Action if Breached |
|--------|---------------------|--------------------|
| Turns | <50% of baseline avg | Revert, try different wording |
| File reads | <50% of baseline avg | Revert, check if nudge is too weak |
| Deliverable size | <70% of baseline avg | Revert, check completion guidance |
| Focus correct | Any miss vs baseline | Investigate root cause |
| Completion tool called | Must always be called | Revert immediately — critical |
| Continuation nudges | >2× baseline count | Nudge may be confusing the agent |

### Step 4: Combined test
After all 5 pass individually, run 3× with ALL changes active. Compare against original
baseline. If combined performance matches or exceeds baseline, the rewrites are validated.

### Step 5: Update parity comparison
After validated rewrites are merged, update `docs/prompt-parity-comparison.md`:
- Move P1–P4 from "Identical/Very Similar" to a new "✅ Original" category
- Document the new prompt text and tool name
- Note validation results

---

## Implementation Checklist

- [ ] Record baseline metrics (3 runs with current prompts)
- [ ] P5: Rename `task_complete` → `work_complete` across all files
- [ ] P5: Add deprecated aliases for backward compatibility
- [ ] P5: Update all tests for new names
- [ ] A/B test P5 alone → validate (no behavioral change expected)
- [ ] P3: Write new `work_complete` tool docstring
- [ ] A/B test P3 alone → validate
- [ ] P4: Write new explore/run_task one-liners in `SUB_AGENT_GUIDANCE`
- [ ] A/B test P4 alone → validate
- [ ] P2: Write new `WORK_COMPLETION_INSTRUCTIONS`
- [ ] A/B test P2 alone → validate
- [ ] P1: Write new `DEFAULT_CONTINUATION_PROMPT` (use `work_complete` in text)
- [ ] A/B test P1 alone → validate
- [ ] Combined test (all 5) → validate
- [ ] Update `docs/prompt-parity-comparison.md` with new ratings
- [ ] Update harness-parity-plan.md checklist items as done
