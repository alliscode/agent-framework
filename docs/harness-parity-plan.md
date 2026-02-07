# Closing the Gap: Harness → Copilot CLI Parity Plan

> A prioritized, multi-phase plan for closing functional gaps between the Agent Harness
> and GitHub Copilot CLI. Each phase builds on the previous one.
>
> Reference docs:
> - `docs/copilot-cli-architecture.md` — What Copilot does
> - `docs/agent-harness-architecture.md` — What we do today

---

## Guiding Principles

1. **Highest-impact changes first.** Prioritize by expected quality improvement per effort.
2. **Don't break what works.** Every phase must be backward-compatible — existing consumers (DevUI, harness_repl) keep working.
3. **Additive, not rewrite.** Extend the existing executor graph and configuration surface.
4. **Testable in isolation.** Each phase should be independently verifiable via `harness_test_runner.py`.

---

## Existing Framework Capabilities (What We Already Have)

Before diving into work, it's critical to understand the agent framework already provides
much of the infrastructure these phases need. This dramatically reduces effort for
several phases.

### Middleware System (`_middleware.py`)

Three middleware layers with full pre/post interception — **no new hook infrastructure needed**:

| Layer | Class | Intercepts |
|-------|-------|-----------|
| **Function** | `FunctionMiddleware` | Individual tool calls. Receives `FunctionInvocationContext` with `function`, `arguments`, `result`, `terminate`. Can override execution via `context.result` or call `await next(context)`. |
| **Agent** | `AgentMiddleware` | Agent invocations. Receives `AgentRunContext` with `agent`, `messages`, `thread`, `result`. Can intercept or override entire agent runs. |
| **Chat** | `ChatMiddleware` | Chat client requests. Receives `ChatContext` with `chat_client`, `messages`, `chat_options`, `result`. Can modify messages before LLM call. |

All three support:
- Pipeline chaining (multiple middleware in sequence)
- `context.terminate = True` to short-circuit
- `context.result = ...` to override without executing
- Decorator syntax: `@function_middleware`, `@agent_middleware`, `@chat_middleware`
- Class decorators: `@use_agent_middleware`, `@use_chat_middleware`

### Agent-as-Tool (`_agents.py` → `BaseAgent.as_tool()`)

Any agent can be converted to an `AIFunction` tool callable by other agents:

```python
explore_agent = ChatAgent(chat_client=mini_client, name="explore", instructions="...")
explore_tool = explore_agent.as_tool(
    name="explore",
    description="Explore codebase and answer questions",
    arg_name="prompt",
)
# Use in main agent
main_agent = ChatAgent(chat_client=client, tools=[explore_tool, ...])
```

This is **exactly** the pattern Copilot uses for sub-agents. The framework handles:
- Input model creation (dynamic Pydantic model)
- Streaming support via `stream_callback`
- Fresh context per invocation (each call creates a new thread)
- Result returned as string tool result to parent

### YAML Agent Definitions (`declarative` package)

`AgentFactory.create_agent_from_yaml_path()` loads agents from YAML:

```yaml
kind: Agent
name: explore
model:
  id: gpt-4o-mini
  provider: OpenAI
instructions: "You are an exploration agent..."
tools:
  - kind: Function
    name: read_file
  - kind: Function
    name: list_directory
```

Supports: model configuration, tool lists (Function, MCP, OpenAPI, Custom), instructions,
response format, and metadata.

### Multi-Agent Patterns (workflows)

| Pattern | Class | Location |
|---------|-------|----------|
| **Handoff** | `HandoffBuilder` | `_workflows/_handoff.py` — coordinator + specialists with auto-generated handoff tools |
| **Group Chat** | `GroupChatBuilder` | `_workflows/_group_chat.py` — manager-directed multi-agent conversation |
| **Concurrent** | Fan-out/Fan-in executors | `_workflows/_concurrent.py` — parallel agent execution |
| **Magentic** | `MagenticOrchestratorExecutor` | `_workflows/_magentic.py` — structured orchestration |

### System Prompt Construction

Instructions flow: `ChatAgent(instructions=...)` → `ChatOptions.instructions` →
`BaseChatClient` creates `ChatMessage(role="system")` → prepended to messages.

Context providers can merge additional instructions via `_prepare_thread_and_messages()`.

### Approval & Filtering

- `approval_mode` on tools: `"always_require"`, `"never_require"`, selective
- `FunctionApprovalRequestContent` / `FunctionApprovalResponseContent` for user gating
- MCP-level approval: `HostedMCPSpecificApproval` with allow/deny lists

---

## Phase 1 — Assertive Stop Control (Highest Impact)

**Gap**: Our harness passively accepts when the agent stops. Copilot forces the agent to explicitly declare completion and aggressively nudges if it doesn't.

**Why first**: This is the single largest quality gap. Our agent currently does 3 turns of shallow work, then stops. Copilot's agent keeps going because it is not allowed to stop until it calls `task_complete`.

### 1a. Auto-inject `task_complete` tool

**Current state**: The `task_complete` tool exists in `_done_tool.py` (37 lines) but must be manually included in the agent's tool set. The harness detects it via `_has_task_complete_call()` (line 642) but doesn't inject it.

**What already exists**: Work item tools are already injected at two identical points:

- `_agent_turn_executor.py` line 291-294 (`_run_agent`):
  ```python
  if self._task_list is not None:
      run_kwargs["tools"] = self._task_list.get_tools()
      if self._work_item_middleware is not None:
          run_kwargs["middleware"] = self._work_item_middleware
  ```
- `_agent_turn_executor.py` line 332-335 (`_run_agent_streaming`): identical block.

The `task_complete` function and `get_task_complete_tool()` helper are exported from `_done_tool.py`.

**Exact changes**:

1. **`_agent_turn_executor.py` line 16** — add import:
   ```python
   from ._done_tool import task_complete
   ```

2. **`_agent_turn_executor.py` lines 291-294** — in `_run_agent()`, change to:
   ```python
   if self._task_list is not None:
       run_kwargs["tools"] = self._task_list.get_tools()
       run_kwargs["tools"].append(task_complete)
       if self._work_item_middleware is not None:
           run_kwargs["middleware"] = self._work_item_middleware
   ```

3. **`_agent_turn_executor.py` lines 332-335** — in `_run_agent_streaming()`, same change.

4. **Also handle the case where work items are disabled but we still want task_complete**.
   Add after the existing `if self._task_list is not None` block in both methods:
   ```python
   else:
       # Even without work items, inject task_complete for stop control
       run_kwargs.setdefault("tools", [])
       if isinstance(run_kwargs["tools"], list):
           run_kwargs["tools"].append(task_complete)
   ```

### 1b. Assertive continuation nudge

**Current state**: `_agent_turn_executor.py` line 54:
```python
DEFAULT_CONTINUATION_PROMPT = "Continue if there's more to do, or just say 'done' if finished."
```
Default `max_continuation_prompts` in constructor (line 62): `2`.
Default in `AgentHarness.__init__()` (`_harness_builder.py` line 401): `max_continuation_prompts: int = 2`.

**Exact changes**:

1. **`_agent_turn_executor.py` line 54** — replace `DEFAULT_CONTINUATION_PROMPT`:
   ```python
   DEFAULT_CONTINUATION_PROMPT = (
       "You have not yet marked the task as complete using the task_complete tool.\n"
       "If you were planning, stop planning and start executing.\n"
       "You are not done until you have fully completed the task.\n\n"
       "IMPORTANT: Do NOT call task_complete if:\n"
       "- You have open questions — use your best judgment and continue\n"
       "- You encountered an error — try to resolve it or find an alternative\n"
       "- There are remaining steps — complete them first\n\n"
       "Keep working autonomously until the task is truly finished,\n"
       "then call task_complete with a summary."
   )
   ```

2. **`_agent_turn_executor.py` line 62** — change default:
   ```python
   max_continuation_prompts: int = 5,
   ```

3. **`_harness_builder.py` line 401** — change default in `AgentHarness.__init__()`:
   ```python
   max_continuation_prompts: int = 5,
   ```

### 1c. Require `task_complete` in stop decision

**Current state**: `TurnComplete` (`_state.py` lines 257-269) only has `agent_done: bool` and `error: str | None`. The stop decision (`_stop_decision_executor.py` line 138) checks `if turn_result.agent_done:` and proceeds directly to contract/work item verification.

`AgentTurnExecutor` already computes `called_task_complete` at line 214:
```python
called_task_complete = self._has_task_complete_call(response)
```
But it's only logged (line 250-255), never passed to the stop decision.

**Exact changes**:

1. **`_state.py` lines 257-269** — add field to `TurnComplete`:
   ```python
   @dataclass
   class TurnComplete:
       """Message indicating an agent turn is complete."""
       agent_done: bool = False
       called_task_complete: bool = False  # ← NEW
       error: str | None = None
   ```

2. **`_agent_turn_executor.py` line 257** — pass the flag:
   ```python
   await ctx.send_message(TurnComplete(agent_done=agent_done, called_task_complete=called_task_complete))
   ```

3. **`_stop_decision_executor.py`** — add constructor parameter and check. In `__init__()` (line 59-82):
   ```python
   def __init__(
       self,
       *,
       enable_contract_verification: bool = False,
       enable_stall_detection: bool = False,
       enable_work_item_verification: bool = False,
       require_task_complete: bool = True,  # ← NEW
       stall_threshold: int = DEFAULT_STALL_THRESHOLD,
       id: str = "harness_stop_decision",
   ):
       ...
       self._require_task_complete = require_task_complete
   ```

4. **`_stop_decision_executor.py` line 138** — add check before existing Layer 3:
   ```python
   # Layer 3: Agent signals done
   if turn_result.agent_done:
       # 3-pre: Require explicit task_complete call
       if self._require_task_complete and not turn_result.called_task_complete:
           logger.info(
               "StopDecisionExecutor: Agent signaled done but did not call task_complete"
           )
           await self._append_event(
               ctx,
               HarnessEvent(
                   event_type="stop_decision",
                   data={
                       "decision": "continue",
                       "reason": "task_complete_not_called",
                       "turn": turn_count,
                   },
               ),
           )
           await ctx.send_message(RepairTrigger())
           return

       # Existing Layer 3a: Contract verification...
       if self._enable_contract_verification:
           ...
   ```

5. **`_harness_builder.py`** — pass the new param when constructing `StopDecisionExecutor` (lines 289-297):
   ```python
   builder.register_executor(
       lambda: StopDecisionExecutor(
           enable_contract_verification=enable_contract_verification,
           enable_stall_detection=self._enable_stall_detection,
           enable_work_item_verification=enable_work_item_verification,
           require_task_complete=True,  # ← NEW (always on when using harness)
           stall_threshold=self._stall_threshold,
           id="harness_stop_decision",
       ),
       name="stop_decision",
   )
   ```

### Testing Phase 1

```bash
cd python
uv run python samples/getting_started/workflows/harness/harness_test_runner.py 2>&1
```

**Expected outcome**: Agent should be unable to stop after 3 turns. It will receive continuation nudges and be forced to keep working until it explicitly calls `task_complete`. Target: 8-15+ turns, 10+ file reads.

### Phase 1 Implementation Summary — COMPLETE

All three sub-tasks of Phase 1 have been implemented and verified:

**1a. Auto-inject `task_complete` tool**: The `task_complete` function is now imported and
automatically appended to the tool list in both `_run_agent()` and `_run_agent_streaming()`.
When work items are enabled, it's appended alongside work item tools. When work items are
disabled, it's still injected via the `else` branch using `run_kwargs.setdefault("tools", [])`.

**1b. Assertive continuation nudge**: `DEFAULT_CONTINUATION_PROMPT` replaced with a multi-line
assertive prompt that references `task_complete` by name, instructs the agent to stop planning
and start executing, and tells it to keep working autonomously. `max_continuation_prompts`
default raised from 2 to 5 in `AgentTurnExecutor`, `HarnessWorkflowBuilder`, and `AgentHarness`.

**1c. Require `task_complete` in stop decision**: `TurnComplete` dataclass now carries a
`called_task_complete: bool` field. `AgentTurnExecutor` passes this flag when sending
`TurnComplete`. `StopDecisionExecutor` gained a `require_task_complete` parameter (default `True`)
and checks it in a new "Layer 3-pre" gate before contract/work-item verification. If the agent
signals done without calling `task_complete`, a `RepairTrigger` is sent to continue execution.
`HarnessWorkflowBuilder` passes `require_task_complete=True` when constructing the executor.

**Files modified**:
- `_agent_turn_executor.py` — import, prompt, default, tool injection (×2 methods), TurnComplete field
- `_state.py` — `called_task_complete` field on `TurnComplete`
- `_stop_decision_executor.py` — `require_task_complete` param and Layer 3-pre gate
- `_harness_builder.py` — defaults (×2), docstrings (×2), `require_task_complete` wiring

**Tests**: All 308 existing harness tests pass. Tests updated to pass `called_task_complete=True`
in `TurnComplete` where agent completion is expected.

---

## Phase 2 — Rich System Prompt Construction

**Gap**: Copilot builds a rich, structured system prompt with environment context, tool usage instructions, task completion guidelines, and code change rules. Our harness delegates system prompt entirely to the agent constructor and injects supplementary messages as `role: user`.

**Why second**: Even with assertive stop control, the agent needs the right framing to know *how* to do thorough work. The system prompt is the foundational instruction set.

### 2a. Environment context auto-injection

**Current state**: No environment context is injected. The agent has to guess the OS, cwd, etc.

**What already exists**: The `ContextProvider` protocol (`_memory.py` line 76-186) has one required method:

```python
class ContextProvider(ABC):
    @abstractmethod
    async def invoking(self, messages, **kwargs) -> Context: ...
```

`Context` (`_memory.py` line 30-71) holds:
```python
class Context:
    def __init__(self, instructions=None, messages=None, tools=None):
        self.instructions = instructions    # Merged into system prompt
        self.messages = messages or []      # Prepended to thread
        self.tools = tools or []            # Added to tool set
```

The merging happens automatically in `_agents.py` line 1299-1304:
```python
if context.instructions:
    chat_options.instructions = (
        context.instructions
        if not chat_options.instructions
        else f"{chat_options.instructions}\n{context.instructions}"
    )
```

`ChatAgent` accepts `context_providers` in its constructor (`_agents.py` line 597):
```python
context_providers: ContextProvider | list[ContextProvider] | AggregateContextProvider | None = None
```

**New file**: `_harness/_context_providers.py`

```python
# Copyright (c) Microsoft. All rights reserved.
"""Context providers for harness-level instruction injection."""

import os
import platform
from typing import Any, MutableSequence, Sequence

from .._memory import Context, ContextProvider
from .._types import ChatMessage


class EnvironmentContextProvider(ContextProvider):
    """Injects environment context (cwd, OS, directory listing) into system prompt."""

    def __init__(self, sandbox_path: str | None = None, max_entries: int = 50):
        self._sandbox = sandbox_path
        self._max_entries = max_entries

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        cwd = self._sandbox or os.getcwd()
        os_name = platform.system()
        try:
            entries = sorted(os.listdir(cwd))[:self._max_entries]
            listing = "\n".join(f"  {e}" for e in entries)
        except OSError:
            listing = "  (unable to list directory)"

        instructions = (
            "<environment_context>\n"
            f"Working directory: {cwd}\n"
            f"Operating system: {os_name}\n"
            f"Directory contents (top-level):\n{listing}\n"
            "</environment_context>"
        )
        return Context(instructions=instructions)
```

**Wiring**: In `_harness_builder.py`, in the `AgentHarness.__init__()` method (line 378+),
add a `sandbox_path` parameter and wire the provider:

```python
# In AgentHarness.__init__() — add parameter:
sandbox_path: str | None = None,

# In the body, after agent is stored:
if sandbox_path:
    from ._context_providers import EnvironmentContextProvider
    from .._memory import AggregateContextProvider

    env_provider = EnvironmentContextProvider(sandbox_path=sandbox_path)
    # Append to existing context providers
    if hasattr(agent, 'context_provider') and agent.context_provider:
        existing = agent.context_provider
        if isinstance(existing, AggregateContextProvider):
            existing.providers.append(env_provider)
        else:
            agent.context_provider = AggregateContextProvider([existing, env_provider])
    else:
        agent.context_provider = env_provider
```

### 2b. Move guidance from user messages to system prompt

**Current state**: Three guidance blocks are injected as `role: user` messages on turn 1:
- `_inject_tool_strategy_guidance()` — line 925-931 (always)
- `_inject_work_item_guidance()` — line 891-897 (when work items enabled)
- `_inject_planning_prompt()` — line 946-952 (when work items enabled)

Called from `run_turn()` at lines 148-153:
```python
self._inject_tool_strategy_guidance()
if self._task_list is not None:
    self._inject_work_item_guidance()
    self._inject_planning_prompt()
```

These user messages are vulnerable to compaction and carry less weight than system prompt.

**New class in `_context_providers.py`**:

```python
class HarnessGuidanceProvider(ContextProvider):
    """Injects tool strategy, work item, and planning guidance into system prompt."""

    def __init__(
        self,
        enable_work_items: bool = False,
        enable_tool_guidance: bool = True,
    ):
        self._enable_work_items = enable_work_items
        self._enable_tool_guidance = enable_tool_guidance

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        from ._agent_turn_executor import AgentTurnExecutor

        sections: list[str] = []
        if self._enable_tool_guidance:
            sections.append(AgentTurnExecutor.TOOL_STRATEGY_GUIDANCE)
        if self._enable_work_items:
            sections.append(AgentTurnExecutor.WORK_ITEM_GUIDANCE)
            sections.append(AgentTurnExecutor.PLANNING_PROMPT)

        if not sections:
            return Context()
        return Context(instructions="\n\n".join(sections))
```

**Changes to `_agent_turn_executor.py`**: Remove the three injection calls from
`run_turn()` (lines 148-153). The guidance now comes via the system prompt automatically.

**Important**: Keep the turn-1 `_inject_work_item_state()` and `_maybe_inject_work_item_reminder()` calls — those are dynamic per-turn injections that belong as user messages.

**Wiring**: Same pattern as 2a — add `HarnessGuidanceProvider` to the agent in `AgentHarness.__init__()`.

### 2c. Task completion instructions in system prompt

**Add to `HarnessGuidanceProvider`** — always included:

```python
TASK_COMPLETION_INSTRUCTIONS = (
    "<task_completion>\n"
    "* A task is not complete until the expected outcome is verified and persistent\n"
    "* After making changes, validate they work correctly\n"
    "* If an initial approach fails, try alternative tools or methods before concluding\n"
    "* You MUST call task_complete when finished — do not just stop responding\n"
    "* Only call task_complete after all work items are done and deliverables are written\n"
    "</task_completion>"
)
```

Add to the `invoking()` method:
```python
sections.append(self.TASK_COMPLETION_INSTRUCTIONS)  # Always included
```

### Testing Phase 2

After implementing, the agent's effective system prompt (visible in debug logs from `BaseChatClient`)
should contain the environment context block, tool strategy guidance, work item guidance, planning
prompt, and task completion instructions — all as system-level content.

Verify by setting `logging.getLogger("agent_framework").setLevel(logging.DEBUG)` and inspecting
the system message in the first LLM request.

### Phase 2 — Implementation Summary ✅ COMPLETE

**What was implemented:**

1. **`_harness/_context_providers.py`** (new file) — Two context provider classes using the existing `ContextProvider` protocol:
   - `EnvironmentContextProvider`: Injects working directory, OS name, and sorted directory listing into the system prompt via `<environment_context>` tags. Accepts an optional `sandbox_path` override and `max_entries` limit. Handles invalid paths gracefully.
   - `HarnessGuidanceProvider`: Consolidates tool strategy guidance, work item/planning guidance, and task completion instructions into the system prompt. Configurable via `enable_tool_guidance` and `enable_work_items` flags. Task completion instructions (`<task_completion>` block) are always included.

2. **`_harness/_harness_builder.py`** — Added `sandbox_path` parameter to both `HarnessWorkflowBuilder` and `AgentHarness`. Added `_wire_context_providers()` method that appends `EnvironmentContextProvider` (when `sandbox_path` is set) and `HarnessGuidanceProvider` (always) to the agent's existing `context_provider`, preserving any user-configured providers via `AggregateContextProvider`.

3. **`_harness/_agent_turn_executor.py`** — Removed the three first-turn user message injection calls (`_inject_tool_strategy_guidance()`, `_inject_work_item_guidance()`, `_inject_planning_prompt()`) from `run_turn()`. These are now delivered as system prompt content through the context providers. Dynamic per-turn injections (`_inject_work_item_state`, `_maybe_inject_work_item_reminder`) are preserved as user messages.

4. **`_harness/__init__.py`** — Exported `EnvironmentContextProvider` and `HarnessGuidanceProvider`.

5. **`tests/harness/test_context_providers.py`** (new file) — 14 tests covering:
   - Environment context injection (cwd, OS, directory listing, sandbox override, max_entries, invalid path fallback, XML tags)
   - Guidance provider (task completion always included, tool guidance toggle, work items toggle, all sections combined)
   - Integration with `AggregateContextProvider`
   - Verification that providers return only instructions (no messages or tools)

**All 321 harness tests pass** (307 existing + 14 new).

---

## Phase 3 — Hooks System

**Gap**: Copilot has lifecycle hooks (`preToolUse`, `postToolUse`, `agentStop`, `sessionStart`, `sessionEnd`) that allow external code to intercept, modify, and control execution. We have none at the harness level.

**Why third**: Hooks enable all further quality enforcement without modifying the core harness. They're the extension mechanism that makes everything else possible.

**What already exists — this phase is 80% done**:

The framework's middleware system already provides **all three interception layers** Copilot uses:

| Copilot Hook | Framework Equivalent | How to Use |
|-------------|---------------------|------------|
| `preToolUse` + `postToolUse` | `FunctionMiddleware.process()` | Receives `FunctionInvocationContext` with `function.name`, `arguments`. Can set `context.result` to deny, or call `await next(context)` then inspect `context.result`. |
| `agentStop` | `AgentMiddleware.process()` | Wraps entire `agent.run()`. Can inspect `context.result` (the `AgentRunResponse`) after `await next(context)` and override it. |
| N/A | `ChatMiddleware.process()` | Wraps LLM requests. Can modify `messages` or `chat_options` before LLM call. |

The middleware is already wired into the agent's invocation pipeline. The harness already uses `FunctionMiddleware` for work item event tracking (`WorkItemEventMiddleware`).

### 3a. `HarnessStopGateMiddleware` (agentStop equivalent)

**Change**: Create an `AgentMiddleware` that inspects `AgentRunResponse` after each inner turn and can inject continuation messages:

```python
class HarnessStopGateMiddleware(AgentMiddleware):
    """Post-turn quality gate — the agentStop hook equivalent."""

    def __init__(self, min_turns: int = 3, min_reads: int = 5):
        self._min_turns = min_turns
        self._min_reads = min_reads
        self._turn_count = 0
        self._tool_usage: dict[str, int] = {}

    async def process(self, context: AgentRunContext, next):
        await next(context)  # Let the agent run
        self._turn_count += 1

        # Track tool usage from response
        for msg in context.result.messages:
            for content in getattr(msg, "contents", []):
                if hasattr(content, "name"):
                    self._tool_usage[content.name] = self._tool_usage.get(content.name, 0) + 1

        # Quality gate: if agent stops too early, we can signal to harness
        # (actual blocking happens in StopDecisionExecutor via the hooks below)
```

However, note that `AgentMiddleware` wraps `agent.run()` which is the **inner loop** — it fires per-turn. The harness's outer loop (`StopDecisionExecutor`) is where the real stop gating happens. So the more practical approach is:

### 3b. `HarnessHooks` dataclass for the stop decision layer

Since `StopDecisionExecutor` already implements the outer-loop stop logic, we just need to make it extensible:

```python
@dataclass
class HarnessHooks:
    """Harness-level lifecycle hooks for external extensibility."""
    agent_stop: list[Callable[[AgentStopEvent], Awaitable[AgentStopResult | None]]] = field(default_factory=list)
    session_start: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    session_end: list[Callable[[str], Awaitable[None]]] = field(default_factory=list)
```

Wire into `StopDecisionExecutor`:

```python
# After all layer checks pass, before accepting stop:
for hook in self._hooks.agent_stop:
    result = await hook(AgentStopEvent(turn=turn, tool_usage=usage, ...))
    if result and result.decision == "block":
        inject_message(result.reason)
        send_message(RepairTrigger())  # Continue
        return
```

### 3c. `HarnessToolMiddleware` for pre/post tool interception

**What already exists**: `FunctionMiddleware` is the exact hook needed. The harness already passes middleware via `run_kwargs["middleware"]`.

**Change**: Create a composable `HarnessToolMiddleware` that delegates to registered callbacks:

```python
class HarnessToolMiddleware(FunctionMiddleware):
    """Delegates to registered pre/post tool callbacks."""

    def __init__(self):
        self._pre_hooks: list[Callable] = []
        self._post_hooks: list[Callable] = []

    async def process(self, context: FunctionInvocationContext, next):
        # Pre-tool hooks
        for hook in self._pre_hooks:
            result = await hook(context.function.name, context.arguments)
            if result and result.decision == "deny":
                context.result = f"Denied: {result.reason}"
                context.terminate = True
                return

        await next(context)

        # Post-tool hooks
        for hook in self._post_hooks:
            await hook(context.function.name, context.arguments, context.result)
```

Compose with the existing work item middleware:

```python
# In AgentTurnExecutor._run_agent
middlewares = []
if self._work_item_middleware:
    middlewares.append(self._work_item_middleware)
if self._harness_tool_middleware:
    middlewares.append(self._harness_tool_middleware)
run_kwargs["middleware"] = middlewares
```

**Effort**: Small-Medium — the framework does the heavy lifting. We just need thin wrapper classes and wiring.

### Expected Impact

Hooks are the foundation for all extensible quality control. The `agent_stop` hook alone replicates Copilot's quality gate pattern. External consumers (DevUI, automation) can register custom hooks without modifying the harness.

### Phase 3 — Implementation Summary ✅ COMPLETE

All three sub-tasks of Phase 3 have been implemented and verified:

**3a/3b. `HarnessHooks` dataclass and agent-stop hooks**: New file `_hooks.py` defines the
`HarnessHooks` dataclass with three hook lists: `pre_tool`, `post_tool`, and `agent_stop`.
Supporting dataclasses `AgentStopEvent`, `AgentStopResult`, and `ToolHookResult` define the
hook contracts. Type aliases `PreToolHook`, `PostToolHook`, and `AgentStopHook` are exported
for consumer convenience.

The `agent_stop` hooks are wired into `StopDecisionExecutor` via a new `hooks` parameter.
After all verification layers (contract, work items) pass but before accepting the stop,
`_run_agent_stop_hooks()` iterates the registered callbacks. If any hook returns
`AgentStopResult(decision="block", reason=...)`, a `RepairTrigger` is sent to continue
execution and the block reason is recorded in the transcript. Hook exceptions are logged
and swallowed — they never prevent a stop.

**3c. `HarnessToolMiddleware`**: A `FunctionMiddleware` subclass in `_hooks.py` that delegates
to registered pre/post tool callbacks. Pre-tool hooks can return `ToolHookResult(decision="deny")`
to block execution (sets `context.result` and `context.terminate`). Post-tool hooks are purely
observational. Both layers catch and log exceptions without propagating.

The middleware is composed with the existing `WorkItemEventMiddleware` via a new
`_collect_middlewares()` helper in `AgentTurnExecutor`. Both `_run_agent()` and
`_run_agent_streaming()` now build the middleware list dynamically rather than passing
a single middleware.

**Wiring**: `HarnessHooks` flows through `HarnessWorkflowBuilder` and `AgentHarness` via a
new `hooks` parameter. The builder passes hooks to both `AgentTurnExecutor` (for tool hooks)
and `StopDecisionExecutor` (for agent-stop hooks).

**Files modified**:
- `_hooks.py` — NEW: `HarnessHooks`, `AgentStopEvent`, `AgentStopResult`, `ToolHookResult`,
  `HarnessToolMiddleware`, type aliases
- `_stop_decision_executor.py` — `hooks` param, `_run_agent_stop_hooks()` method
- `_agent_turn_executor.py` — `hooks` param, `_collect_middlewares()`, composed middleware wiring
- `_harness_builder.py` — `hooks` param on `HarnessWorkflowBuilder` and `AgentHarness`,
  forwarded to executors
- `__init__.py` — exports for all new types

**Files added**:
- `tests/harness/test_hooks.py` — 20 tests covering: pre/post tool hooks (deny, allow, None,
  error resilience, multiple hooks, composition), dataclass defaults, `HarnessHooks` instance
  independence, and full workflow integration tests for agent-stop hooks (block, allow, no hooks,
  exception resilience, event payload verification)

**All 359 harness tests pass** (339 existing + 20 new).

---

## Phase 4 — Wire Up Full Compaction Pipeline

**Gap**: Our compaction subsystem is fully built (9 phases, 41 tests, 11 files) but the
workflow doesn't call it. `CompactionExecutor` detects pressure and signals
`compaction_needed=True`, then `AgentTurnExecutor` only does `_apply_direct_clear()`
(dumb placeholder replacement). Nobody calls `compact_thread()` which runs the full
`CompactionCoordinator` strategy ladder.

**Why fourth**: With Phases 1-3 in place, the agent will run longer sessions (10-20+ turns).
Longer sessions need intelligent compaction, not just placeholder clearing.

### What's Already Built

All of this exists and is tested in `_harness/_compaction/`:

| Component | File | Status |
|-----------|------|--------|
| `CompactionCoordinator` | `_strategies.py:810+` | ✅ Orchestrates strategy ladder |
| `ClearStrategy` | `_strategies.py:200+` | ✅ Durability-aware tool result clearing |
| `SummarizeStrategy` | `_strategies.py:400+` | ✅ LLM compression (needs `Summarizer` impl) |
| `ExternalizeStrategy` | `_strategies.py:550+` | ✅ Artifact storage (needs stores) |
| `DropStrategy` | `_strategies.py:700+` | ✅ Last resort content removal |
| `CompactionPlan` | `_types.py` | ✅ Immutable plan with action map |
| `PromptRenderer` | `_renderer.py` | ✅ Applies plan to produce rendered messages |
| `TiktokenTokenizer` | `_tokenizer.py` | ✅ Accurate token counting |
| `TokenBudgetV2` | `_tokenizer.py` | ✅ Budget with overhead tracking |
| `InMemoryCompactionStore` | `_store.py` | ✅ Thread-safe plan storage |
| `InMemoryArtifactStore` | `_store.py` | ✅ Artifact storage |
| `InMemorySummaryCache` | `_store.py` | ✅ LRU+TTL summary caching |
| `CompactionExecutor.compact_thread()` | `_compaction_executor.py:272-327` | ✅ Full pipeline entry point |

### What's Missing (The Wiring Gap)

**Gap 1**: `CompactionExecutor.check_compaction()` (line 178-270) signals `compaction_needed=True`
but never calls `compact_thread()`. The executor has the coordinator but doesn't use it on the
hot path.

**Gap 2**: `AgentTurnExecutor._is_compaction_needed()` (line 549-560) checks the signal, then
`run_turn()` calls `_apply_direct_clear()` (line 562-615) — a naive O(n) placeholder replacement
that ignores durability, doesn't summarize, and doesn't use the CompactionPlan pipeline.

**Gap 3**: `compact_thread()` expects an `AgentThread` (line 272-278), but `AgentTurnExecutor`
works with a flat `self._cache: list[Any]` (line 83). The cache is a list of `ChatMessage` objects,
not an `AgentThread`.

**Gap 4**: No `Summarizer` implementation ships. `SummarizeStrategy` and `ExternalizeStrategy`
require a `Summarizer` protocol impl (defined at `_strategies.py`):
```python
class Summarizer(Protocol):
    async def summarize(self, messages: Sequence[ChatMessage], max_tokens: int = 200) -> StructuredSummary: ...
```

### Exact Changes

#### 4a. Bridge the cache-to-thread gap

The `CompactionCoordinator.compact()` method (`_strategies.py` line 854) takes an `AgentThread`.
Each strategy's `analyze()` method reads messages from `thread.message_store.list_messages()`.

**Option A (Recommended)**: Create a lightweight adapter that wraps the cache list:

```python
# New in _agent_turn_executor.py or _compaction/_adapters.py
class CacheThreadAdapter:
    """Wraps AgentTurnExecutor._cache as an AgentThread-like for compaction."""

    def __init__(self, cache: list[ChatMessage]):
        self.message_store = CacheMessageStore(cache)

class CacheMessageStore:
    """Read-only message store backed by the executor cache."""

    def __init__(self, cache: list[ChatMessage]):
        self._cache = cache

    async def list_messages(self) -> list[ChatMessage]:
        return list(self._cache)
```

**Option B**: Pass the cache directly to strategies. This requires modifying
`CompactionStrategy.analyze()` to accept `list[ChatMessage]` instead of `AgentThread` — more
invasive but cleaner long-term.

#### 4b. Call `compact_thread()` from `AgentTurnExecutor`

In `_agent_turn_executor.py`, in `run_turn()`, where compaction is currently handled:

**Current code** (around lines 170-180 in run_turn, after checking `_is_compaction_needed`):
```python
if self._is_compaction_needed(trigger):
    cleared = self._apply_direct_clear(turn_count)
```

**Replace with**:
```python
if self._is_compaction_needed(trigger):
    # Try full compaction pipeline first
    plan = await self._run_full_compaction(ctx, turn_count)
    if plan is not None and not plan.is_empty:
        # Apply the plan to the cache (existing _apply_compaction_plan method)
        pass  # Plan is loaded in _get_messages_for_agent via SharedState
    else:
        # Fallback to direct clearing
        cleared = self._apply_direct_clear(turn_count)
```

New method on `AgentTurnExecutor`:
```python
async def _run_full_compaction(self, ctx: WorkflowContext, turn_count: int) -> CompactionPlan | None:
    """Run the full compaction pipeline via CompactionExecutor."""
    from ._compaction import CompactionPlan, TokenBudget as TokenBudgetV2
    from ._compaction_executor import CompactionExecutor

    # Get compaction executor reference (stored during build)
    # OR: instantiate coordinator directly here
    if self._tokenizer is None:
        return None

    # Build a thread adapter from cache
    thread_adapter = CacheThreadAdapter(self._cache)

    # Load current plan
    plan = await self._load_compaction_plan(ctx)

    # Create v2 budget from current state
    budget_v2 = TokenBudgetV2.for_model("gpt-4o")

    # Calculate current tokens
    current_tokens = sum(
        self._tokenizer.count_tokens(str(getattr(msg, "text", "")))
        for msg in self._cache
    )

    tokens_over = budget_v2.tokens_over_threshold(current_tokens)
    if tokens_over <= 0:
        return plan

    tokens_to_free = tokens_over + int(budget_v2.max_input_tokens * 0.1)

    # Run coordinator
    from ._compaction import ClearStrategy, CompactionCoordinator, DropStrategy, TurnContext
    coordinator = CompactionCoordinator(strategies=[ClearStrategy(), DropStrategy()])
    result = await coordinator.compact(
        thread_adapter, plan, budget_v2, self._tokenizer,
        tokens_to_free=tokens_to_free,
        turn_context=TurnContext(turn_number=turn_count),
    )

    # Store updated plan
    if result.plan and not result.plan.is_empty:
        await ctx.set_shared_state(HARNESS_COMPACTION_PLAN_KEY, result.plan.to_dict())

    return result.plan
```

#### 4c. Dual thresholds (background + blocking)

**Current state**: Single threshold at 85% (`_constants.py` line 29:
`DEFAULT_SOFT_THRESHOLD_PERCENT = 0.85`).

**Change**: Add a second threshold to `_context_pressure.py` `TokenBudget`:

```python
@dataclass
class TokenBudget:
    max_input_tokens: int = 100000
    soft_threshold_percent: float = 0.80       # ← lowered from 0.85
    blocking_threshold_percent: float = 0.95   # ← NEW
    current_estimate: int = 0

    @property
    def is_under_pressure(self) -> bool:
        """True at 80% — trigger background compaction."""
        return self.current_estimate >= self.soft_threshold

    @property
    def is_blocking(self) -> bool:
        """True at 95% — must compact before sending to LLM."""
        return self.current_estimate >= int(self.max_input_tokens * self.blocking_threshold_percent)
```

In `CompactionExecutor.check_compaction()` (line 198), use `is_blocking` to determine urgency:
- At 80%: Signal `compaction_needed=True` (AgentTurnExecutor runs ClearStrategy)
- At 95%: Signal `compaction_needed=True` AND `blocking=True` (AgentTurnExecutor must fully compact before proceeding)

#### 4d. LLM Summarizer (Follow-up — Not Required for Initial Wire-up)

The `Summarizer` protocol is defined but no implementation ships. For the initial wire-up,
use only `ClearStrategy` + `DropStrategy` (no LLM dependency). This is already significantly
better than `_apply_direct_clear()` because `ClearStrategy` respects tool durability policies.

A real `Summarizer` implementation would call the LLM:

```python
class LLMSummarizer:
    """Calls the LLM to produce structured summaries."""

    def __init__(self, chat_client: ChatClientProtocol, model_id: str = "gpt-4o-mini"):
        self._client = chat_client
        self._model = model_id

    async def summarize(self, messages: Sequence[ChatMessage], max_tokens: int = 200) -> StructuredSummary:
        prompt = f"Summarize this conversation segment in {max_tokens} tokens..."
        response = await self._client.get_response(messages=[...], chat_options=...)
        return StructuredSummary.from_text(response.text)
```

This can be added later as a separate PR without changing any wiring.

### Testing Phase 4

1. Set `max_input_tokens=50000` (low threshold) to trigger compaction earlier.
2. Run a long session (15+ turns).
3. Verify via debug logs:
   - `CompactionExecutor: Context pressure detected` at 80%
   - `CompactionCoordinator: Applied ClearStrategy proposal` when compacting
   - Agent continues beyond where it previously hit context limits

### Phase 4 — Implementation Summary ✅ COMPLETE

**Implemented by**: Copilot CLI agent session  
**Status**: Complete and tested

**Changes made:**

1. **4a. CacheThreadAdapter** (`_compaction/_adapters.py` — new file):
   - `CacheMessageStore`: Read-only message store wrapping the executor's `list[Any]` cache, 
     implementing `list_messages()` for compaction strategies.
   - `CacheThreadAdapter`: Lightweight adapter exposing `message_store` attribute so 
     `CompactionCoordinator.compact()` can operate on the executor cache without modifying 
     strategy interfaces.

2. **4b. Full compaction wiring** (`_agent_turn_executor.py`):
   - Added `_run_full_compaction()` method that bridges cache → `CacheThreadAdapter` → 
     `CompactionCoordinator` with `ClearStrategy` + `DropStrategy`.
   - Updated `run_turn()` to try full compaction pipeline first, falling back to 
     `_apply_direct_clear()` if the coordinator fails or finds nothing to compact.
   - CompactionPlan stored in SharedState for `_get_messages_for_agent()` to apply via 
     existing `_apply_compaction_plan()` renderer.

3. **4c. Dual thresholds** (`_context_pressure.py`, `_compaction_executor.py`, `_constants.py`):
   - `TokenBudget` now has `blocking_threshold_percent` (default 0.95) and `is_blocking` property.
   - `soft_threshold_percent` default lowered from 0.85 → 0.80 to trigger compaction earlier.
   - `CompactionComplete` dataclass extended with `blocking: bool` field.
   - `CompactionExecutor.check_compaction()` signals `blocking=True` when above 95%.

4. **4d. LLM Summarizer**: Deferred as specified — only `ClearStrategy` + `DropStrategy` for 
   initial wire-up.

**Files changed:**
- `_compaction/_adapters.py` (new)
- `_compaction/__init__.py` (exports)
- `_agent_turn_executor.py` (`_run_full_compaction`, updated `run_turn`)
- `_context_pressure.py` (`TokenBudget` dual thresholds)
- `_compaction_executor.py` (`CompactionComplete.blocking`, soft threshold default)
- `_constants.py` (`DEFAULT_BLOCKING_THRESHOLD_PERCENT`, updated `DEFAULT_SOFT_THRESHOLD_PERCENT`)
- `_harness/__init__.py` (exports)

**Tests:** 17 new tests in `tests/harness/test_compaction_pipeline.py` covering:
- CacheMessageStore / CacheThreadAdapter behavior
- Dual threshold logic (soft, blocking, between, serialization roundtrip)
- CompactionComplete.blocking field
- CompactionExecutor signaling blocking vs non-blocking at different utilization levels

All 307+ harness tests pass (existing + new), zero regressions.

---

## Phase 5 — Sub-Agent Delegation

**Gap**: Copilot delegates codebase exploration, test execution, and code review to specialized sub-agents with their own context windows. Our harness runs everything in a single context.

**Why fifth**: Sub-agents keep the main agent's context clean and use cheaper/faster models for routine work.

**What already exists — this phase is 90% framework-supported**:

| Need | Framework Feature | Location |
|------|------------------|----------|
| Agent-as-tool | `BaseAgent.as_tool()` | `_agents.py:394-476` |
| Fresh context per call | `as_tool()` calls `self.run(input_text)` with no shared thread | Built-in |
| Return value | `as_tool()` returns `response.text` as the tool result string | `_agents.py:464` |
| Streaming | `as_tool(stream_callback=cb)` uses `run_stream()` | `_agents.py:468` |
| YAML definitions | `AgentFactory.create_agent_from_yaml_path()` | `declarative/_loader.py` |
| Tool filtering | `ChatAgent(tools=[...])` restricts sub-agent tools | Constructor param |

### 5a. New file: `_harness/_sub_agents.py`

```python
# Copyright (c) Microsoft. All rights reserved.
"""Sub-agent definitions and factory for harness-level agent delegation."""

from typing import TYPE_CHECKING, Any, Callable, Sequence

from .._agents import ChatAgent
from .._tools import AIFunction

if TYPE_CHECKING:
    from .._clients import ChatClientProtocol
    from .._tools import ToolProtocol


def create_explore_tool(
    chat_client: "ChatClientProtocol",
    tools: "Sequence[ToolProtocol | Callable[..., Any]]",
) -> "AIFunction[Any, str]":
    """Create an explore sub-agent tool for fast codebase Q&A.

    Args:
        chat_client: Chat client (ideally a fast/cheap model like gpt-4o-mini).
        tools: Read-only tools available to the explore agent (read_file, list_directory, run_command).

    Returns:
        An AIFunction tool that the main agent can call.
    """
    explore = ChatAgent(
        chat_client=chat_client,
        name="explore",
        description="Fast codebase exploration agent for answering questions about code",
        instructions=(
            "You are an exploration agent specialized in rapid codebase analysis.\n"
            "CRITICAL: Keep your answer under 300 words.\n"
            "CRITICAL: MAXIMIZE PARALLEL TOOL CALLING — make multiple independent "
            "tool calls in a single response.\n"
            "Aim to answer questions in 1-3 tool calls when possible.\n"
            "Return a focused, factual answer. No preamble, no hedging."
        ),
        tools=list(tools),
    )
    return explore.as_tool(
        name="explore",
        description=(
            "Launch a fast exploration agent to answer codebase questions. "
            "Returns focused answers under 300 words. Safe to call in parallel. "
            "Use for: finding files, understanding code structure, answering questions. "
            "Do NOT use for: making changes, running builds, complex multi-step work."
        ),
        arg_name="prompt",
        arg_description="The question or exploration task for the agent.",
    )


def create_task_tool(
    chat_client: "ChatClientProtocol",
    tools: "Sequence[ToolProtocol | Callable[..., Any]]",
) -> "AIFunction[Any, str]":
    """Create a task sub-agent tool for command execution.

    Args:
        chat_client: Chat client (ideally a fast/cheap model).
        tools: Tools available to the task agent (typically all tools).

    Returns:
        An AIFunction tool that the main agent can call.
    """
    task_agent = ChatAgent(
        chat_client=chat_client,
        name="task",
        description="Command execution agent for builds, tests, and linting",
        instructions=(
            "You are a command execution agent that runs development commands.\n"
            "On SUCCESS: Return a brief one-line summary (e.g., 'All 247 tests passed').\n"
            "On FAILURE: Return the full error output for debugging.\n"
            "Do NOT attempt to fix errors or make suggestions — just execute and report."
        ),
        tools=list(tools),
    )
    return task_agent.as_tool(
        name="run_task",
        description=(
            "Launch a task agent to execute commands (builds, tests, linting). "
            "Returns brief success summary or verbose error output. "
            "Use for: running tests, building code, checking linting. "
            "Do NOT use for: exploring code or making changes."
        ),
        arg_name="prompt",
        arg_description="The command or task to execute.",
    )
```

### 5b. Wire into `AgentHarness.__init__()` and `AgentTurnExecutor`

In `_harness_builder.py` `AgentHarness.__init__()` (line 378+), add parameters:

```python
# New parameters in AgentHarness.__init__():
sub_agent_client: "ChatClientProtocol | None" = None,
sub_agent_tools: "Sequence[ToolProtocol | Callable] | None" = None,
```

In the body:
```python
# Build sub-agent tools if client provided
self._sub_agent_tools: list[Any] = []
if sub_agent_client is not None:
    from ._sub_agents import create_explore_tool, create_task_tool
    agent_tools = list(sub_agent_tools or [])
    self._sub_agent_tools = [
        create_explore_tool(sub_agent_client, agent_tools),
        create_task_tool(sub_agent_client, agent_tools),
    ]
```

Pass to `AgentTurnExecutor` constructor (line 270-282):
```python
lambda: AgentTurnExecutor(
    self._agent,
    agent_thread=self._agent_thread,
    ...
    sub_agent_tools=self._sub_agent_tools,  # ← NEW
),
```

In `AgentTurnExecutor.__init__()` (line 56-103), add:
```python
sub_agent_tools: list[Any] | None = None,
```
Store as `self._sub_agent_tools = sub_agent_tools or []`.

In both `_run_agent()` (line 291) and `_run_agent_streaming()` (line 332), add after existing tool injection:
```python
# Inject sub-agent tools
if self._sub_agent_tools:
    existing_tools = run_kwargs.get("tools", [])
    if not isinstance(existing_tools, list):
        existing_tools = list(existing_tools)
    existing_tools.extend(self._sub_agent_tools)
    run_kwargs["tools"] = existing_tools
```

### 5c. Sub-agent guidance in system prompt

Add to `HarnessGuidanceProvider` (Phase 2b) when sub-agents are enabled:

```python
SUB_AGENT_GUIDANCE = (
    "<sub_agents>\n"
    "You have access to specialized sub-agents:\n"
    "- explore(prompt): Fast codebase Q&A (cheap model, <300 word answers, parallel-safe)\n"
    "- run_task(prompt): Execute commands — builds, tests, linting (brief success, verbose failure)\n\n"
    "Use explore proactively for codebase questions before making changes.\n"
    "Use run_task for builds/tests where you only need success/failure status.\n"
    "Do your own deep reading and writing — don't delegate core work to sub-agents.\n"
    "</sub_agents>"
)
```

### 5d. Usage example

```python
from openai import AsyncOpenAI
from agent_framework import ChatAgent, OpenAIChatClient

# Main agent (powerful model)
main_client = OpenAIChatClient(AsyncOpenAI(), model="gpt-4o")
# Sub-agent client (fast/cheap model)
mini_client = OpenAIChatClient(AsyncOpenAI(), model="gpt-4o-mini")

harness = AgentHarness(
    agent=ChatAgent(main_client, instructions="You are a helpful assistant."),
    enable_work_items=True,
    sub_agent_client=mini_client,
    sub_agent_tools=[read_file, list_directory, run_command],  # Tools for sub-agents
)
```

### Testing Phase 5

Run the test scenario and verify in output:
- Agent calls `explore(prompt="What files are in python/packages/core/...")`
- Explore agent responds with focused answer
- Main agent's context stays clean (explore result is a single tool result message)

### Phase 5 — Implementation Complete

**Status: ✅ COMPLETE**

**What was implemented:**

1. **5a. `_harness/_sub_agents.py`** (new file): Two factory functions:
   - `create_explore_tool(chat_client, tools)` — creates an `explore` sub-agent tool for fast 
     codebase Q&A with <300 word answers and parallel tool calling emphasis.
   - `create_task_tool(chat_client, tools)` — creates a `run_task` sub-agent tool for command 
     execution with brief-success/verbose-failure reporting.
   Both use `ChatAgent.as_tool()` so each invocation gets a fresh context window.

2. **5b. Wiring into builders and executor:**
   - `HarnessWorkflowBuilder.__init__()` accepts `sub_agent_client` and `sub_agent_tools` params.
     When `sub_agent_client` is provided, explore + run_task tools are created and stored.
   - `AgentHarness.__init__()` forwards both params to the builder.
   - `AgentTurnExecutor.__init__()` accepts `sub_agent_tools` list.
   - Both `_run_agent()` and `_run_agent_streaming()` inject sub-agent tools into `run_kwargs["tools"]`
     after existing tool injection (task_complete, work items).

3. **5c. Sub-agent guidance in system prompt:**
   - `HarnessGuidanceProvider` gains `enable_sub_agents` parameter.
   - When enabled, injects `SUB_AGENT_GUIDANCE` XML block describing explore/run_task usage patterns.
   - `HarnessWorkflowBuilder._wire_context_providers()` passes `enable_sub_agents=True` when 
     sub-agent tools exist.

4. **5d. Exports:** `create_explore_tool` and `create_task_tool` exported from `_harness/__init__.py`.

**Files changed:**
- `_harness/_sub_agents.py` (new)
- `_harness/_harness_builder.py` (sub_agent_client/sub_agent_tools params, wire_context_providers)
- `_harness/_agent_turn_executor.py` (sub_agent_tools param, injection in both run methods)
- `_harness/_context_providers.py` (SUB_AGENT_GUIDANCE, enable_sub_agents flag)
- `_harness/__init__.py` (exports)

**Tests:** 20 new tests in `tests/harness/test_sub_agents.py` covering:
- Factory function output (name, description, tools handling)
- AgentTurnExecutor sub_agent_tools storage and defaults
- HarnessGuidanceProvider sub-agent guidance inclusion/exclusion
- HarnessWorkflowBuilder/AgentHarness wiring and parameter forwarding
- Module-level exports

All 379 harness tests pass (359 existing + 20 new), zero regressions.

---

### Phase 5b — Documentation Sub-Agent (NOT YET IMPLEMENTED)

**Status: ✅ COMPLETE**

**What was implemented:**

1. **5b-1. `create_document_tool()` in `_harness/_sub_agents.py`**: New factory function that
   creates a `document` sub-agent tool. The agent receives specialized instructions emphasizing
   depth over breadth: read every relevant source file, reference specific class names and method
   signatures, include code examples and ASCII diagrams, and use `write_file` to save the
   deliverable. Follows the same `ChatAgent.as_tool()` pattern as explore and run_task.

2. **5b-2. Wired into builder**: `HarnessWorkflowBuilder.__init__()` now creates three sub-agent
   tools (explore, run_task, document) when `sub_agent_client` is provided. No new parameters
   needed — the document tool is created alongside the others.

3. **5b-3. Updated sub-agent guidance**: `SUB_AGENT_GUIDANCE` in `HarnessGuidanceProvider` now
   includes the `document` agent with usage guidance: provide the topic, target files/directories,
   and output file path. Updated guidance advises the main agent to delegate heavy writing to
   the document agent while retaining exploration and planning responsibilities.

4. **5b-4. Updated exports**: `create_document_tool` exported from `_harness/__init__.py`.

**Files changed:**
- `_harness/_sub_agents.py` — added `create_document_tool()` factory function
- `_harness/_harness_builder.py` — added `create_document_tool` to sub-agent creation block
- `_harness/_context_providers.py` — updated `SUB_AGENT_GUIDANCE` with document agent
- `_harness/__init__.py` — added `create_document_tool` to imports and `__all__`
- `tests/harness/test_sub_agents.py` — added 10 new tests, updated 3 existing tests for new counts

**Tests:** 30 tests in `tests/harness/test_sub_agents.py` (20 existing updated + 10 new):
- `TestCreateDocumentTool`: factory output (name, description, tools handling, depth emphasis)
- `TestSubAgentGuidanceIncludesDocument`: guidance mentions document agent and usage hints
- `TestBuilderCreatesThreeSubAgents`: builder and harness create 3 tools (explore, run_task, document)
- Updated existing builder/harness tests from 2→3 tool count expectations

All 389 harness tests pass (379 existing + 10 new), zero regressions.

**Problem**: Experiments with Phases 1-5 show the main agent is effective at *finding*
code (10 file reads) but consistently produces shallow deliverables (~3KB). The target
is comprehensive, deep documentation that references specific classes, methods, module
paths, and includes code examples. The main agent's context is cluttered with work item
history, tool call logs, and planning overhead — leaving little room for synthesis.

**Solution**: Add a `document` sub-agent that receives a focused brief from the main
agent, reads the relevant source files itself in a fresh context window, and produces
a thorough deliverable. This follows the same `ChatAgent.as_tool()` pattern as the
existing `explore` and `run_task` sub-agents.

**Why this helps**:
- Fresh context window — the doc agent's entire context is dedicated to reading code
  and writing the document, not polluted with harness overhead
- Specialized instructions — tuned for depth, thoroughness, and technical precision
- The main agent becomes an orchestrator: explore → read/understand → delegate writing

#### 5b-1. Add `create_document_tool()` to `_harness/_sub_agents.py`

Add a third factory function after `create_task_tool()`:

```python
def create_document_tool(
    chat_client: "ChatClientProtocol",
    tools: "Sequence[ToolProtocol | Callable[..., Any]]",
) -> "AIFunction[Any, str]":
    """Create a documentation sub-agent tool for producing thorough technical documents.

    Args:
        chat_client: Chat client for the documentation agent.
        tools: Tools available to the doc agent (read_file, list_directory, write_file, run_command).

    Returns:
        An AIFunction tool that the main agent can call.
    """
    doc_agent = ChatAgent(
        chat_client=chat_client,
        name="document",
        description="Documentation agent specialized in producing comprehensive technical documents",
        instructions=(
            "You are a technical documentation specialist. Your job is to produce\n"
            "thorough, high-quality technical documents about software systems.\n\n"
            "APPROACH:\n"
            "1. Read EVERY source file relevant to the topic — do not skip files or guess.\n"
            "2. Go deep on the most important components to answer the request thoroughly.\n"
            "3. Reference specific class names, method signatures, and module paths.\n"
            "4. Include source code examples where they help communicate concepts and principles.\n"
            "5. Include ASCII diagrams for architecture, data flow, or component relationships\n"
            "   where they help the reader understand the system.\n"
            "6. Organize with clear sections, tables, and hierarchical structure.\n\n"
            "QUALITY STANDARDS:\n"
            "- Every claim must be backed by something you read in a source file.\n"
            "- Do not summarize from directory listings — read the actual code.\n"
            "- Prioritize depth over breadth: deeply explain the most important parts\n"
            "  rather than shallowly listing everything.\n"
            "- Your document should give a reader genuine understanding of how the\n"
            "  system works, not just a surface inventory of its parts.\n"
            "- Include constructor signatures, key method signatures, and important\n"
            "  type definitions when documenting classes.\n\n"
            "OUTPUT:\n"
            "- Use write_file to save your document to the path specified in the prompt.\n"
            "- Return a brief summary of what you produced and its location."
        ),
        tools=list(tools),
    )
    return doc_agent.as_tool(
        name="document",
        description=(
            "Launch a documentation agent to produce a comprehensive technical document. "
            "The agent reads source files itself in a fresh context window and writes "
            "a thorough deliverable. Provide: the topic/request, which files or directories "
            "to focus on, and the output file path. "
            "Use for: creating architectural designs, API docs, code analysis reports. "
            "Do NOT use for: quick questions (use explore) or running commands (use run_task)."
        ),
        arg_name="prompt",
        arg_description="The documentation request including topic, scope, target files/directories, and output file path.",
    )
```

**Key design decisions**:
- Instructions stress depth on what matters most, not a specific size target — the
  agent should produce whatever length is needed to genuinely explain the system.
- Explicit instruction to include code examples and diagrams "where they help" — not
  mandatory for every section, but encouraged when they add clarity.
- The agent reads files itself rather than receiving pre-read content — this keeps the
  main agent's context lighter and lets the doc agent decide what's important.
- The prompt parameter should include the output file path so the doc agent can
  `write_file` directly.

#### 5b-2. Wire into builder

In `_harness/_harness_builder.py`, in the block where `create_explore_tool` and
`create_task_tool` are called (inside `__init__` when `sub_agent_client` is provided),
add:

```python
from ._sub_agents import create_document_tool

# After existing explore + run_task creation:
sub_tools.append(create_document_tool(sub_agent_client, sub_agent_tools))
```

No new parameters needed — the document tool is created whenever sub-agents are enabled.

#### 5b-3. Update sub-agent guidance

In `_harness/_context_providers.py`, update `SUB_AGENT_GUIDANCE` to include the
document agent:

```python
SUB_AGENT_GUIDANCE = (
    "<sub_agents>\n"
    "You have access to specialized sub-agents:\n"
    "- explore(prompt): Fast codebase Q&A (cheap model, <300 word answers, parallel-safe)\n"
    "- run_task(prompt): Execute commands — builds, tests, linting (brief success, verbose failure)\n"
    "- document(prompt): Produce comprehensive technical documents (reads files in fresh context)\n\n"
    "Use explore proactively for codebase questions before making changes.\n"
    "Use run_task for builds/tests where you only need success/failure status.\n"
    "Use document when you need to produce a detailed deliverable — provide the topic,\n"
    "which files/directories to focus on, and the output file path. The document agent\n"
    "reads source files itself, so you don't need to pre-read everything.\n"
    "Do your own exploration and planning — delegate the heavy writing to document.\n"
    "</sub_agents>"
)
```

#### 5b-4. Update exports

In `_harness/__init__.py`, add `create_document_tool` to the exports alongside
`create_explore_tool` and `create_task_tool`.

#### Testing Phase 5b

1. **Unit tests** — add to `tests/harness/test_sub_agents.py`:
   - `test_create_document_tool_returns_ai_function`: verify name="document",
     description contains "comprehensive", has tools
   - `test_document_tool_instructions_stress_depth`: verify instructions contain
     "specific class names", "source code examples", "do not skip files"
   - `test_sub_agent_guidance_includes_document`: verify `SUB_AGENT_GUIDANCE`
     mentions "document" when `enable_sub_agents=True`
   - `test_builder_creates_three_sub_agents`: verify builder creates 3 sub-agent
     tools (explore, run_task, document) when `sub_agent_client` is provided

2. **Integration test** — run the test harness and verify:
   - The `document` tool appears in tool call logs
   - The deliverable file is written by the document sub-agent
   - The deliverable contains specific class names, method signatures, and code examples
   - The main agent uses explore for discovery and document for the final deliverable

**Files to change**:
- `_harness/_sub_agents.py` — add `create_document_tool()` factory function
- `_harness/_harness_builder.py` — add `create_document_tool` call in sub-agent creation block
- `_harness/_context_providers.py` — update `SUB_AGENT_GUIDANCE` constant
- `_harness/__init__.py` — add export
- `tests/harness/test_sub_agents.py` — add tests

**Estimated effort**: S (small) — follows exact same pattern as existing sub-agents.

---

## Phase 6 — Custom Instructions Discovery

**Gap**: Copilot auto-loads AGENTS.md, CLAUDE.md, .github/copilot-instructions.md, etc. We require all instructions in the agent constructor.

**What already exists**: `ContextProvider` protocol (`_memory.py` line 76). Same pattern as Phase 2.

### 6a. Add `InstructionFileProvider` to `_harness/_context_providers.py`

```python
class InstructionFileProvider(ContextProvider):
    """Auto-discovers and loads instruction files from the workspace.

    Searches for well-known instruction files (AGENTS.md, CLAUDE.md, etc.)
    and wraps their content in <custom_instruction> tags for the system prompt.
    Deduplicates by content hash (same behavior as Copilot CLI).
    """

    SEARCH_FILES: list[str] = [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".github/copilot-instructions.md",
    ]
    SEARCH_DIRS: list[str] = [
        ".github/instructions",
    ]

    def __init__(self, workspace_root: str):
        self._root = workspace_root
        self._cache: str | None = None

    async def invoking(
        self,
        messages: ChatMessage | MutableSequence[ChatMessage],
        **kwargs: Any,
    ) -> Context:
        if self._cache is not None:
            return Context(instructions=self._cache) if self._cache else Context()

        sections: list[str] = []
        seen_content: set[int] = set()

        for rel_path in self.SEARCH_FILES:
            full_path = os.path.join(self._root, rel_path)
            if os.path.isfile(full_path):
                try:
                    content = Path(full_path).read_text(encoding="utf-8")
                    h = hash(content)
                    if h not in seen_content:
                        seen_content.add(h)
                        sections.append(
                            f'<custom_instruction source="{rel_path}">\n{content}\n</custom_instruction>'
                        )
                except (OSError, UnicodeDecodeError):
                    pass

        for rel_dir in self.SEARCH_DIRS:
            full_dir = os.path.join(self._root, rel_dir)
            if os.path.isdir(full_dir):
                for md_file in sorted(Path(full_dir).glob("*.md")):
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        h = hash(content)
                        if h not in seen_content:
                            seen_content.add(h)
                            rel = md_file.relative_to(self._root)
                            sections.append(
                                f'<custom_instruction source="{rel}">\n{content}\n</custom_instruction>'
                            )
                    except (OSError, UnicodeDecodeError):
                        pass

        self._cache = "\n\n".join(sections) if sections else ""
        return Context(instructions=self._cache) if self._cache else Context()
```

### 6b. Wire into `AgentHarness.__init__()`

In `_harness_builder.py` (line 378+), add parameter:
```python
auto_discover_instructions: bool = False,
```

In the body (after sandbox_path handling from Phase 2a):
```python
if auto_discover_instructions and sandbox_path:
    from ._context_providers import InstructionFileProvider
    instruction_provider = InstructionFileProvider(workspace_root=sandbox_path)
    # Append to agent's context providers (same wiring pattern as Phase 2a)
```

### Testing Phase 6

1. Create `AGENTS.md` in test repo root with custom instructions.
2. Run harness, verify instructions appear in system prompt (debug logs).
3. Create both `AGENTS.md` and `CLAUDE.md` with identical content — verify deduplication.

---

## Phase 7 — JIT Instructions System

**Gap**: Copilot injects context-sensitive guidance during execution based on time remaining, files changed, etc. Our injections are limited to work item state.

**Why seventh**: JIT instructions handle edge cases that static prompts miss — catching agents stuck in patterns (all planning, no execution; all reading, no writing).

### 7a. New file: `_harness/_jit_instructions.py`

```python
# Copyright (c) Microsoft. All rights reserved.
"""Just-In-Time instruction injection based on execution state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class JitContext:
    """Execution state available to JIT conditions."""
    turn: int
    max_turns: int
    tool_usage: dict[str, int]        # Tool name → call count
    work_items_complete: int
    work_items_total: int


@dataclass
class JitInstruction:
    """A conditional instruction that fires when its condition is met."""
    id: str
    instruction: str | Callable[[JitContext], str]
    condition: Callable[[JitContext], bool]
    once: bool = True  # Only inject once per session


# Default instructions that handle common stuck patterns
DEFAULT_JIT_INSTRUCTIONS: list[JitInstruction] = [
    JitInstruction(
        id="no_reads_after_5_turns",
        instruction=(
            "You have been working for several turns but haven't read any files yet. "
            "Use read_file to examine source code before drawing conclusions."
        ),
        condition=lambda ctx: ctx.turn >= 5 and ctx.tool_usage.get("read_file", 0) == 0,
    ),
    JitInstruction(
        id="no_writes_after_reads",
        instruction=(
            "You have read many files but haven't produced any deliverables yet. "
            "Start writing your output using write_file."
        ),
        condition=lambda ctx: (
            ctx.turn >= 10
            and ctx.tool_usage.get("read_file", 0) >= 5
            and ctx.tool_usage.get("write_file", 0) == 0
        ),
    ),
    JitInstruction(
        id="approaching_turn_limit",
        instruction=(
            "You are approaching the turn limit. Prioritize completing your "
            "most important remaining work items and call task_complete."
        ),
        condition=lambda ctx: ctx.turn >= int(ctx.max_turns * 0.8),
    ),
    JitInstruction(
        id="all_planning_no_execution",
        instruction=(
            "You have created work items but haven't started executing any of them. "
            "Stop planning and begin working on your first item."
        ),
        condition=lambda ctx: (
            ctx.turn >= 3
            and ctx.work_items_total > 0
            and ctx.work_items_complete == 0
            and ctx.tool_usage.get("work_item_add", 0) > 0
            and ctx.tool_usage.get("read_file", 0) == 0
        ),
    ),
]


@dataclass
class JitInstructionProcessor:
    """Evaluates JIT instructions and returns those that should fire."""

    instructions: list[JitInstruction] = field(default_factory=lambda: list(DEFAULT_JIT_INSTRUCTIONS))
    _fired: set[str] = field(default_factory=set)

    def evaluate(self, context: JitContext) -> list[str]:
        """Return instruction texts for all conditions that are met."""
        results: list[str] = []
        for jit in self.instructions:
            if jit.once and jit.id in self._fired:
                continue
            if jit.condition(context):
                text = jit.instruction if isinstance(jit.instruction, str) else jit.instruction(context)
                results.append(text)
                if jit.once:
                    self._fired.add(jit.id)
        return results
```

### 7b. Integration in `AgentTurnExecutor`

**Where**: In `run_turn()`, after work item state injection (around line 155), before
running the agent.

Add to `AgentTurnExecutor.__init__()` (line 56):
```python
# Add parameter:
jit_instructions: list[JitInstruction] | None = None,

# In body:
from ._jit_instructions import JitInstructionProcessor, DEFAULT_JIT_INSTRUCTIONS
self._jit_processor = JitInstructionProcessor(
    instructions=jit_instructions or DEFAULT_JIT_INSTRUCTIONS
)
```

Add new method:
```python
async def _inject_jit_instructions(self, ctx: WorkflowContext[Any], turn_count: int) -> None:
    """Evaluate and inject JIT instructions based on current execution state."""
    from .._types import ChatMessage
    from ._jit_instructions import JitContext
    from ._work_items import WorkItemLedger

    # Build context from current state
    tool_usage = self._get_tool_usage()  # Already computed in _inject_work_item_state
    work_complete, work_total = 0, 0
    try:
        ledger_data = await ctx.get_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY)
        if ledger_data and isinstance(ledger_data, dict):
            ledger = WorkItemLedger.from_dict(ledger_data)
            work_total = len(ledger.items)
            work_complete = work_total - len(ledger.get_incomplete_items())
    except KeyError:
        pass

    jit_ctx = JitContext(
        turn=turn_count,
        max_turns=50,  # Should come from config
        tool_usage=tool_usage,
        work_items_complete=work_complete,
        work_items_total=work_total,
    )

    instructions = self._jit_processor.evaluate(jit_ctx)
    for text in instructions:
        self._cache.append(ChatMessage(role="user", text=text))
        logger.info("AgentTurnExecutor: Injected JIT instruction: %s", text[:80])
```

Call from `run_turn()`:
```python
# After work item state injection, before running agent:
await self._inject_jit_instructions(ctx, turn_count)
```

### 7c. Wire through builder

In `_harness_builder.py`, add optional parameter to `AgentHarness.__init__()`:
```python
jit_instructions: "list[JitInstruction] | None" = None,
```

Pass to `AgentTurnExecutor` constructor.

### Testing Phase 7

1. Set `max_turns=20` and run with a prompt that triggers planning but no execution.
2. Verify `"all_planning_no_execution"` fires after turn 3.
3. Verify `"approaching_turn_limit"` fires at turn 16 (80% of 20).
4. Verify `once=True` instructions don't fire twice.

---

## Phase 8 — Per-Tool-Call Interception

**Gap**: Copilot can intercept every tool call before and after execution. Our harness only sees the final `AgentRunResponse` after all tools in a turn have already executed.

**What already exists — this phase is essentially done**:

The framework's `FunctionMiddleware` (`_middleware.py` line 337-400) already provides full
per-tool-call interception. The harness already uses it for `WorkItemEventMiddleware`.

`FunctionMiddleware.process()` receives `FunctionInvocationContext` (`_middleware.py` line 134-197):
- `context.function` — the `AIFunction` being called (has `.name`, `.description`)
- `context.arguments` — validated Pydantic `BaseModel` of arguments
- `context.result` — set before `await next(context)` to override execution, or read after to inspect result
- `context.terminate` — set `True` to prevent downstream middleware and execution
- `context.metadata` — arbitrary `dict[str, Any]`

Middleware is already passed via `run_kwargs["middleware"]` at lines 293-294 and 334-335 in
`_agent_turn_executor.py`. Currently only one middleware is passed; need to support a list.

### 8a. New file: `_harness/_tool_middleware.py`

```python
# Copyright (c) Microsoft. All rights reserved.
"""Middleware implementations for tool-level interception in the harness."""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from .._middleware import FunctionInvocationContext, FunctionMiddleware

logger = logging.getLogger(__name__)


class ToolUsageTrackingMiddleware(FunctionMiddleware):
    """Tracks tool call counts for JIT instructions and quality hooks.

    Usage:
        tracker = ToolUsageTrackingMiddleware()
        # ... run agent with tracker as middleware ...
        print(tracker.usage)  # {"read_file": 5, "write_file": 2, ...}
    """

    def __init__(self) -> None:
        self.usage: dict[str, int] = {}

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        name = context.function.name
        self.usage[name] = self.usage.get(name, 0) + 1
        await next(context)


class ShellSafetyMiddleware(FunctionMiddleware):
    """Validates shell commands before execution.

    Denies commands matching dangerous patterns (rm -rf /, format drives, etc.).
    """

    DANGEROUS_PATTERNS: list[str] = [
        r"rm\s+-rf\s+/(?!\S)",       # rm -rf / (root)
        r"rm\s+-rf\s+~",              # rm -rf ~ (home)
        r"format\s+[A-Z]:",           # format C: (Windows)
        r"mkfs\.",                     # mkfs.ext4 etc.
        r"dd\s+.*of=/dev/",           # dd to device
        r">\s*/dev/sd[a-z]",          # redirect to block device
        r"chmod\s+-R\s+777\s+/",      # chmod 777 / (root)
    ]

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self._patterns = self.DANGEROUS_PATTERNS + (extra_patterns or [])
        self._compiled = [re.compile(p) for p in self._patterns]

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        if context.function.name == "run_command":
            # Extract command string from arguments
            command = ""
            if hasattr(context.arguments, "command"):
                command = str(context.arguments.command)
            elif hasattr(context.arguments, "dict"):
                command = str(context.arguments.dict().get("command", ""))

            for pattern in self._compiled:
                if pattern.search(command):
                    logger.warning("ShellSafetyMiddleware: Denied command: %s", command[:100])
                    context.result = f"Command denied for safety reasons: {command[:100]}"
                    context.terminate = True
                    return

        await next(context)


class CompositeMiddleware(FunctionMiddleware):
    """Chains multiple FunctionMiddleware into a single middleware.

    Use when the framework expects a single middleware but you need multiple.
    """

    def __init__(self, middlewares: list[FunctionMiddleware]) -> None:
        self._middlewares = [m for m in middlewares if m is not None]

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        if not self._middlewares:
            await next(context)
            return

        # Build chain: each middleware calls the next, last calls `next`
        async def build_chain(index: int) -> None:
            if index >= len(self._middlewares):
                await next(context)
            else:
                await self._middlewares[index].process(
                    context,
                    lambda ctx: build_chain(index + 1),
                )

        await build_chain(0)
```

### 8b. Update middleware injection in `AgentTurnExecutor`

In `_agent_turn_executor.py`, update both `_run_agent()` (line 291-294) and
`_run_agent_streaming()` (line 332-335) to compose all middleware:

```python
# Replace the single middleware injection with composite:
middlewares: list[FunctionMiddleware] = []
if self._work_item_middleware is not None:
    middlewares.append(self._work_item_middleware)
if self._tracking_middleware is not None:
    middlewares.append(self._tracking_middleware)
if self._safety_middleware is not None:
    middlewares.append(self._safety_middleware)

if middlewares:
    from ._tool_middleware import CompositeMiddleware
    run_kwargs["middleware"] = CompositeMiddleware(middlewares)
```

Add to `AgentTurnExecutor.__init__()` (line 56):
```python
# New parameters:
enable_shell_safety: bool = True,

# In body:
from ._tool_middleware import ShellSafetyMiddleware, ToolUsageTrackingMiddleware
self._tracking_middleware = ToolUsageTrackingMiddleware()
self._safety_middleware = ShellSafetyMiddleware() if enable_shell_safety else None
```

The `_tracking_middleware.usage` dict can then be used by JIT instructions (Phase 7) and
work item state injection (existing code at lines 982-1019) instead of scanning the cache.

### Testing Phase 8

1. Verify `ToolUsageTrackingMiddleware` counts match manual cache scanning.
2. Test `ShellSafetyMiddleware` with a dangerous command — verify it returns denial message.
3. Verify `CompositeMiddleware` chains correctly (work items + tracking + safety).

---

## Summary: Revised Prioritized Roadmap

The framework investigation reveals that most infrastructure already exists. The effort
estimates drop dramatically — several "Large" phases are now "Small" because we're
writing thin wrappers over existing framework capabilities.

| Phase | Name | Key Deliverable | Old Effort | **Revised** | Impact | Framework Leverage |
|-------|------|----------------|------------|-------------|--------|--------------------|
| **1** | Assertive Stop Control | `task_complete` required + assertive nudge | S-M | **XS-S** | ★★★★★ | Tool injection already works |
| **2** | Rich System Prompt | ContextProviders for env + guidance | M | **S** | ★★★★☆ | `ContextProvider` protocol exists |
| **3** | Hooks System | `HarnessHooks` + middleware wrappers | L | **S-M** | ★★★★☆ | `FunctionMiddleware` + `AgentMiddleware` exist |
| **4** | Wire Full Compaction | Connect CompactionCoordinator + dual thresholds | L | **S-M** | ★★★☆☆ | All 9 phases built, just needs wiring |
| **5** | Sub-Agent Delegation | `ChatAgent.as_tool()` wrappers | XL | **S-M** | ★★★☆☆ | `as_tool()` + YAML loader exist |
| **6** | Custom Instructions | `InstructionFileProvider` | M | **XS** | ★★☆☆☆ | `ContextProvider` protocol exists |
| **7** | JIT Instructions | Condition-based dynamic injection | M | **S** | ★★☆☆☆ | Turn executor injection points exist |
| **8** | Per-Tool Interception | `FunctionMiddleware` implementations | L | **XS** | ★★☆☆☆ | `FunctionMiddleware` is the exact abstraction |
| **9** | Model Selection | Configurable model + provider support | — | **XS** | ★★★★★ | `AzureOpenAIChatClient` already parameterized |
| **10** | Response Demeanor | Acknowledgment + progress narration | — | **XS** | ★★★☆☆ | System prompt guidance in `HarnessGuidanceProvider` |
| **11** | Smart File Tools | Line-range reads, grep, glob, deep listing | — | **S-M** | ★★★★★ | `coding_tools.py` already has the sandbox/resolve infrastructure |

---

## Phase 9 — Model Selection

**Impact: ★★★★★ (Highest)**

**Gap**: The harness had no guidance on which LLM to use. Model choice turned out to be
the single biggest quality lever discovered during experimentation — larger than any
infrastructure phase.

**Why this matters**: All infrastructure improvements (Phases 1-8) are force multipliers,
but they multiply the base capability of the underlying model. A weaker model with
perfect infrastructure still produces shallow results. A stronger model with basic
infrastructure produces dramatically better output.

### Experimental Evidence

We ran the same prompt ("Investigate this repo and find the python based workflow engine.
Research the code and create a detailed architectural design.") with identical harness
configuration (Phases 1+2+3+4) across two models:

#### gpt-4o Results

| Metric | Value | Assessment |
|--------|-------|-----------|
| Turns | 6-8 | Adequate |
| File reads | 6-8 | Below target (15-20) |
| Directory listings | 7-14 | Often excessive — broad browsing instead of targeted search |
| Work items | 4-6 | Adequate |
| Tool calls | 32-36 | Good |
| Deliverable size | 3.0-3.8 KB | **Poor** — shallow, generic summaries |
| Deliverable quality | Low | Surface-level class inventories, fabricated code examples, sometimes documented the wrong packages entirely |
| Exploration strategy | Weak | Got lost in sibling packages, rarely used grep to find targets, took 10+ list_directory calls to reach `_workflows/` (or never found it) |
| task_complete behavior | Worked | Phase 1 hooks effective |
| Compaction | Sometimes hit 128K limit | Phase 4 helped but agent generated too much noise |

**Key failure modes with gpt-4o:**
- Premature completion — declared "done" with shallow deliverable
- Wrong focus — documented `a2a`, `ag_ui`, `anthropic` packages instead of `_workflows/`
- Fabricated content — code examples that don't match actual source code
- Inefficient exploration — 14 directory listings but only 6 file reads

#### gpt-4.1 Results

| Metric | Value | Assessment |
|--------|-------|-----------|
| Turns | 2 | Very efficient — batched heavily |
| File reads | 7 | Good — targeted reads |
| Directory listings | 18 | Systematic but more than needed |
| Work items | 12 | **Excellent** — granular, all completed |
| Tool calls | 56 | **Excellent** — high parallelism per turn |
| Deliverable size | 8.5 KB | **Good** — 2.2x best gpt-4o result |
| Deliverable quality | High | Module-by-module table, specific class names (`Workflow`, `WorkflowBuilder`, `Runner`, `AgentExecutor`, `GroupChatBuilder`), execution flow, extensibility points, data flow sections |
| Exploration strategy | Strong | Found `_workflows/` quickly, wrote ARCHITECTURE.md directly inside the target directory |
| task_complete behavior | Worked | Clean single call |
| Compaction | No issues | Efficient context usage |

**Key strengths of gpt-4.1:**
- Correct focus immediately — identified `_workflows/` and stayed on target
- High parallelism — 56 tool calls in just 2 turns (batched many parallel calls)
- Granular planning — 12 specific work items vs 4-6 vague ones from gpt-4o
- Accurate content — referenced real classes, methods, and module paths
- Wrote deliverable in the correct location (`_workflows/ARCHITECTURE.md`)

#### Target Benchmark (Claude Sonnet 4.6)

The reference "good output" was produced by Claude Sonnet 4.6 (via GitHub Copilot CLI):

| Metric | Value |
|--------|-------|
| File reads | 15-20+ |
| Deliverable size | ~29 KB |
| Deliverable quality | Exceptional — includes ASCII diagrams, code examples from actual source, constructor signatures, type definitions, orchestration pattern details, and best practices |
| Exploration strategy | Systematic multi-pass: broad directory scan → targeted file reads → supporting files → deliverable creation |

### Model Selection Guidance

Based on experimental data, model capabilities that matter most for agentic harness
workloads are:

1. **Instruction following** — the model must reliably follow complex system prompt
   guidance (tool strategy, work item protocols, quality standards) across many turns.
   Weaker models ignore or "forget" guidance mid-conversation.

2. **Tool call parallelism** — models that batch multiple independent tool calls in a
   single response are dramatically more efficient. gpt-4.1 made 56 calls in 2 turns;
   gpt-4o made 34 in 8 turns.

3. **Planning granularity** — stronger models create more specific, actionable work items.
   gpt-4.1 created 12 granular items; gpt-4o created 4-6 vague ones.

4. **Exploration precision** — the model must navigate deep directory structures
   efficiently. Weaker models do broad listing; stronger models use grep/find to jump
   to targets.

5. **Synthesis depth** — the model must produce deliverables that reference specific
   classes, methods, and code patterns from files it actually read. Weaker models produce
   generic summaries regardless of how many files they read.

### Implementation

Model selection is already fully configurable — no code changes needed:

```python
# Via environment variable
# AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4.1

# Or via constructor parameter
chat_client = AzureOpenAIChatClient(
    credential=AzureCliCredential(),
    deployment_name="gpt-4.1",  # or "claude-sonnet-4.6", "gpt-4o", etc.
)
```

**Recommended defaults:**
- **Primary agent**: gpt-4.1 or Claude Sonnet 4.6 — both strong at multi-turn agentic work
- **Sub-agents (explore/task)**: gpt-4o-mini or equivalent — fast/cheap models are fine
  for short-lived sub-agent tasks. The harness already supports separate `sub_agent_client`
  for this purpose.

**What to document for harness users:**
- The `devui_harness.py` and `harness_test_runner.py` samples should note that model
  choice dramatically affects output quality
- Recommended minimum: gpt-4.1 tier or equivalent for the primary agent
- Budget option: gpt-4o works but produces shallower output; compensate with more
  aggressive JIT instructions (Phase 7) and quality hooks (Phase 3)

### Remaining Gap to Target

Even with gpt-4.1, there's still a gap to the Claude 4.6 reference output:
- 8.5KB vs 29KB deliverable
- Fewer file reads (7 vs 15-20)
- No ASCII diagrams or inline code examples from actual source

Closing this remaining gap likely requires:
1. **Phase 7 (JIT Instructions)** — detect shallow deliverables and inject "read more
   files, add code examples" nudges
2. **Phase 5b (Document sub-agent)** — revisit once the main agent's exploration is
   reliable; the doc agent could expand a solid brief into a richer document
3. **Model upgrade** — Claude Sonnet 4.6 or equivalent as the primary model would likely
   close the gap entirely based on the reference output quality

---

## Phase 10 — Response Demeanor & Acknowledgment

**Impact: ★★★☆☆**

**Gap**: When the user sends a request like "research this thing for me", the agent
jumps straight into tool calls (creating work items, listing directories, reading files)
with **zero text output** until the task is complete. The user sees nothing streaming
for potentially minutes — no acknowledgment, no indication work has started, no
progress narrative. This creates a poor interactive experience even though the agent
is doing good work behind the scenes.

By contrast, GitHub Copilot CLI and other polished agents emit a brief acknowledgment
immediately ("I'll investigate the repository to find the workflow engine…") and then
provide brief progress narration between tool call batches ("Found the target directory,
now reading the core modules…").

**What we want**:
1. **Immediate acknowledgment** — a 1-2 sentence blurb before any tool calls confirming
   the request and indicating work is starting
2. **Progress narration** — brief text between tool call batches describing what was
   found and what's next (not every turn, but enough to keep the user informed)
3. **Not verbose** — this should be a few sentences, not paragraphs. The agent's
   narrative should not dominate over the actual work.

**Root cause**: The LLM is choosing to emit only tool calls with no interleaved text.
This is a prompting/instruction issue — the model can produce text alongside tool calls
but isn't being told to.

### Implementation Options

#### Option A: System prompt guidance (XS effort, recommended first)

Add a `<response_style>` section to `HarnessGuidanceProvider` that instructs the model
on demeanor:

```python
RESPONSE_STYLE_GUIDANCE = (
    "<response_style>\n"
    "When you receive a request:\n"
    "1. Start with a brief acknowledgment (1-2 sentences) before your first tool calls.\n"
    "   Example: 'I'll investigate the workflow engine in this repository and create\n"
    "   a detailed architectural design. Let me start by exploring the directory structure.'\n"
    "2. Between batches of tool calls, briefly narrate what you found and what you're\n"
    "   doing next. Keep it to 1-2 sentences.\n"
    "3. Do NOT narrate every single tool call — just the transitions between investigation\n"
    "   phases (e.g., 'Found the core modules, now reading each one in detail.').\n"
    "4. Your final message before task_complete should summarize what was produced.\n"
    "</response_style>"
)
```

Add this to the `invoking()` method in `HarnessGuidanceProvider`, always included
(like `TASK_COMPLETION_INSTRUCTIONS`).

**File to change**: `_harness/_context_providers.py`
- Add `RESPONSE_STYLE_GUIDANCE` constant to `HarnessGuidanceProvider`
- Append to `sections` list in `invoking()` (always included)

#### Option B: ChatMiddleware interception (S effort, if Option A insufficient)

If the model ignores system prompt guidance about text output, a `ChatMiddleware` could
post-process the LLM response to prepend an acknowledgment on the first turn. This is
heavier and should only be pursued if Option A doesn't work.

```python
class AcknowledgmentMiddleware(ChatMiddleware):
    """Ensures the first LLM response includes text, not just tool calls."""

    def __init__(self):
        self._first_turn = True

    async def process(self, context: ChatContext, next: NextChatHandler) -> None:
        await next(context)
        if self._first_turn and context.result:
            # Check if response has tool calls but no text
            # If so, inject a synthetic acknowledgment
            self._first_turn = False
```

This is more complex and fragile — Option A should be tried first.

### Testing

1. Run `harness_repl.py` with a request like "research the workflow engine"
2. Verify the agent emits text BEFORE its first tool call
3. Verify brief progress text appears between investigation phases
4. Verify the final message summarizes the deliverable
5. Verify the agent does NOT become verbose (no multi-paragraph narration per turn)

**Estimated effort**: XS (Option A) — single constant addition + 1 line in `invoking()`

### Implementation Summary — ✅ COMPLETE

Implemented Option A (system prompt guidance). Changes:

1. **`_harness/_context_providers.py`**: Added `RESPONSE_STYLE_GUIDANCE` constant to
   `HarnessGuidanceProvider` with `<response_style>` XML-tagged instructions covering
   immediate acknowledgment, progress narration between tool batches, conciseness, and
   final summary before `task_complete`. Appended to `sections` in `invoking()` so it
   is always included (alongside `TASK_COMPLETION_INSTRUCTIONS`).

2. **`tests/harness/test_context_providers.py`**: Added `test_always_includes_response_style`
   to verify the guidance is always present. Updated integration test to assert
   `<response_style>` tag is included in aggregated output. All 15 tests pass.

---

## Phase 11 — Smart File Tools (Progressive Reading, Search, Deep Listing)

**Impact: ★★★★★ (Highest — addresses root cause of shallow output AND slowness)**

**Gap**: Our `CodingTools` provide only whole-file reads, single-level directory listing,
and no built-in search. By contrast, GitHub Copilot CLI has `view` with line ranges,
`grep` for content search, and `glob` for file finding. This forces our agent to dump
entire files into context, rapidly filling the token budget and triggering expensive
compaction — while Copilot reads just the 40 lines it needs.

**Current CodingTools** (`python/samples/getting_started/workflows/harness/coding_tools.py`):

| Tool | Behavior | Problem |
|------|----------|---------|
| `read_file(path)` | Reads **entire file** — no line range support | A 1000-line file = ~30K tokens. 7 reads = 200K+ tokens → compaction storm |
| `list_directory(path)` | Single level only, no recursion | Agent needs 10+ calls to navigate 4 levels deep |
| No `grep` | Must use `run_command("grep ...")` | Agent often doesn't think to use shell grep |
| No `glob` | Must use `run_command("find ...")` | Same problem — agents prefer native tools over shell |

**Copilot CLI equivalents:**

| Tool | Behavior |
|------|----------|
| `view(path, view_range=[11, 50])` | Read specific line ranges — keeps only what's needed in context |
| `grep(pattern, path, glob, output_mode)` | Native content search with glob filtering, line numbers, context lines |
| `glob(pattern)` | Fast file pattern matching (`**/*.py`, `src/**/*.ts`) |
| `list_directory` | 2 levels deep by default |

**Impact on our experiments:**
- Agent reads 7 files → fills 200K+ tokens → compaction fires → slowness
- After compaction, summaries lose detail → shallow deliverable
- Agent can't efficiently find target code → wastes turns on broad listing
- With smart tools: agent could read 20+ file excerpts in the same token budget

### 11a. Add line range support to `read_file`

**File**: `coding_tools.py`, `read_file` method (line 116-136)

```python
@ai_function(approval_mode="never_require")
def read_file(
    self,
    path: Annotated[str, "The path to the file to read, relative to the working directory"],
    start_line: Annotated[int | None, "First line to read (1-indexed). Omit to read from start."] = None,
    end_line: Annotated[int | None, "Last line to read (1-indexed, inclusive). Use -1 for end of file. Omit to read to end."] = None,
) -> str:
    """Read the contents of a file, optionally a specific line range.

    Use this to examine code, configuration files, or any text file.
    For large files, use start_line/end_line to read specific sections
    rather than loading the entire file.
    """
    try:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"Error: File '{path}' does not exist"
        if not resolved.is_file():
            return f"Error: '{path}' is not a file"

        content = resolved.read_text(encoding="utf-8")

        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            total = len(lines)
            start = max(1, start_line or 1) - 1  # Convert to 0-indexed
            end = total if (end_line is None or end_line == -1) else min(end_line, total)

            selected = lines[start:end]
            # Prefix with line numbers for context
            numbered = [f"{start + i + 1}. {line}" for i, line in enumerate(selected)]
            header = f"[Lines {start + 1}-{end} of {total} in {path}]\n"
            return header + "".join(numbered)

        return content
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"
```

**Key design decisions:**
- Line numbers are 1-indexed (matches what developers expect and what editors show)
- Output includes line numbers as prefixes (`42. def foo():`) so the agent can
  reference specific lines in its analysis
- Header shows `[Lines X-Y of Z in path]` so agent knows what it's looking at
- `end_line=-1` means "to end of file" (common convention)
- Fully backward compatible — omitting both params reads the entire file as before

### 11b. Add `grep_files` search tool

**File**: `coding_tools.py`, new method + add to `get_tools()` list

```python
@ai_function(approval_mode="never_require")
def grep_files(
    self,
    pattern: Annotated[str, "The text or regex pattern to search for in file contents"],
    path: Annotated[str, "Directory to search in, relative to working directory. Use '.' for current."] = ".",
    file_glob: Annotated[str | None, "Glob pattern to filter files (e.g., '*.py', '*.{ts,tsx}'). Omit to search all files."] = None,
    include_lines: Annotated[bool, "If true, show matching lines with line numbers. If false, show only file paths."] = False,
    max_results: Annotated[int, "Maximum number of matching files to return."] = 20,
) -> str:
    """Search for a pattern in file contents across the workspace.

    Use this to find files containing specific classes, functions, imports,
    or any text pattern. Much faster than reading files one by one.
    """
    import re
    try:
        resolved = self._resolve_path(path)
        if not resolved.exists() or not resolved.is_dir():
            return f"Error: '{path}' is not a valid directory"

        try:
            regex = re.compile(pattern)
        except re.error:
            # Fall back to literal match if not valid regex
            regex = re.compile(re.escape(pattern))

        # Collect matching files
        if file_glob:
            files = sorted(resolved.rglob(file_glob))
        else:
            files = sorted(f for f in resolved.rglob("*") if f.is_file())

        matches = []
        for f in files:
            if not f.is_file():
                continue
            # Skip binary files, hidden dirs, common noise
            rel = str(f.relative_to(self.working_directory))
            if any(part.startswith(".") for part in f.parts):
                continue
            if "__pycache__" in rel or "node_modules" in rel or ".git" in rel:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if include_lines:
                matching_lines = []
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        matching_lines.append(f"  {i}: {line.rstrip()}")
                if matching_lines:
                    matches.append(f"{rel}\n" + "\n".join(matching_lines[:10]))
                    if len(matches) >= max_results:
                        break
            else:
                if regex.search(text):
                    matches.append(rel)
                    if len(matches) >= max_results:
                        break

        if not matches:
            return f"No files matching pattern '{pattern}' found in '{path}'"

        header = f"Found {len(matches)} file(s) matching '{pattern}':\n"
        return header + "\n".join(matches)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error searching files: {e}"
```

### 11c. Add `find_files` glob tool

**File**: `coding_tools.py`, new method + add to `get_tools()` list

```python
@ai_function(approval_mode="never_require")
def find_files(
    self,
    pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', 'src/**/*.ts', '*.md')"],
    path: Annotated[str, "Directory to search in, relative to working directory. Use '.' for current."] = ".",
    max_results: Annotated[int, "Maximum number of results to return."] = 50,
) -> str:
    """Find files by name pattern using glob matching.

    Use this to locate files by extension or naming convention across the workspace.
    Supports ** for recursive matching across directories.
    """
    try:
        resolved = self._resolve_path(path)
        if not resolved.exists() or not resolved.is_dir():
            return f"Error: '{path}' is not a valid directory"

        files = []
        for f in sorted(resolved.rglob(pattern) if "**" in pattern else resolved.glob(pattern)):
            if not f.is_file():
                continue
            rel = str(f.relative_to(self.working_directory))
            if any(part.startswith(".") for part in f.parts):
                continue
            if "__pycache__" in rel or "node_modules" in rel:
                continue
            files.append(rel)
            if len(files) >= max_results:
                break

        if not files:
            return f"No files matching '{pattern}' found in '{path}'"

        return f"Found {len(files)} file(s):\n" + "\n".join(files)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error finding files: {e}"
```

### 11d. Add depth support to `list_directory`

**File**: `coding_tools.py`, modify `list_directory` method (line 162-195)

```python
@ai_function(approval_mode="never_require")
def list_directory(
    self,
    path: Annotated[str, "The directory path relative to working directory. Use '.' for current."] = ".",
    depth: Annotated[int, "How many levels deep to list. 1 = immediate children only, 2 = children and grandchildren. Default is 2."] = 2,
) -> str:
    """List the contents of a directory up to the specified depth.

    Use this to explore the file structure. Default depth of 2 shows
    files and one level of subdirectory contents, giving a good overview
    without being overwhelming.
    """
    try:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"Error: Directory '{path}' does not exist"
        if not resolved.is_dir():
            return f"Error: '{path}' is not a directory"

        items = []

        def _list(dir_path: Path, current_depth: int, prefix: str = "") -> None:
            if current_depth > depth:
                return
            try:
                children = sorted(dir_path.iterdir())
            except PermissionError:
                return
            for item in children:
                # Skip hidden files/dirs
                if item.name.startswith("."):
                    continue
                if item.name in ("__pycache__", "node_modules", ".git"):
                    continue
                rel_path = item.relative_to(self.working_directory)
                if item.is_dir():
                    items.append(f"{prefix}[DIR]  {rel_path}/")
                    _list(item, current_depth + 1, prefix + "  ")
                else:
                    size = item.stat().st_size
                    items.append(f"{prefix}[FILE] {rel_path} ({size} bytes)")

        _list(resolved, 1)

        if not items:
            return f"Directory '{path}' is empty"

        return "\n".join(items)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error listing directory: {e}"
```

### 11e. Update `get_tools()` and tool strategy guidance

**`coding_tools.py`** — add new tools to the list:
```python
def get_tools(self) -> list:
    return [
        self.read_file,
        self.write_file,
        self.list_directory,
        self.grep_files,       # NEW
        self.find_files,       # NEW
        self.run_command,
        self.create_directory,
        self.get_background_output,
        self.stop_background_process,
        self.list_background_processes,
    ]
```

**`_agent_turn_executor.py`** — update `TOOL_STRATEGY_GUIDANCE` to reference new tools:

Replace the DISCOVERY section with:
```
"DISCOVERY — find what's relevant before reading:\n"
"- grep_files(pattern, path, file_glob) to find files containing a term\n"
"- find_files(pattern) to locate files by name/extension (e.g., '**/*.py')\n"
"- list_directory('.', depth=2) to see structure two levels deep\n\n"
```

Replace the THOROUGH READING section with:
```
"THOROUGH READING — depth matters more than speed:\n"
"- Use read_file with start_line/end_line for large files — read the section you need,\n"
"  not the entire file. A 500-line file should be read in 2-3 targeted chunks.\n"
"- After reading, note specific class names, methods, and patterns before moving on.\n"
"- Use grep_files to find specific classes or functions across multiple files.\n\n"
```

### Expected Impact

With smart file tools, the agent should:
- **Read 20+ file excerpts** instead of 7 whole files in the same token budget
- **Find target code in 1-2 calls** (`grep_files("workflow", file_glob="*.py")`)
  instead of 10+ `list_directory` calls
- **Trigger compaction less often** — targeted reads keep context lean
- **Produce deeper deliverables** — more files read = more detail available

### Testing

1. **Unit tests** for each new tool:
   - `test_read_file_line_range`: verify start/end lines, line numbering, header
   - `test_read_file_backward_compat`: verify no params = full file (unchanged)
   - `test_grep_files_basic`: verify pattern matching, file filtering, max_results
   - `test_grep_files_with_lines`: verify `include_lines=True` shows line numbers
   - `test_find_files_glob`: verify glob patterns, recursive matching
   - `test_list_directory_depth`: verify depth=1 vs depth=2 behavior

2. **Integration test** — run `harness_test_runner.py` and verify:
   - Agent uses `grep_files` to find the workflow engine (instead of listing 10+ dirs)
   - Agent uses `read_file` with line ranges for large files
   - Total tokens consumed per turn decreases
   - Compaction triggers less frequently
   - Deliverable references more files with more detail

**Files to change:**
- `coding_tools.py` — modify `read_file`, `list_directory`; add `grep_files`, `find_files`; update `get_tools()`
- `_agent_turn_executor.py` — update `TOOL_STRATEGY_GUIDANCE` to reference new tools
- `_context_providers.py` — if sub-agent guidance mentions tools, update there too

**Estimated effort**: S-M (3-6 hours) — straightforward Python but needs testing across
edge cases (binary files, permissions, large directories, regex errors)

---

### Effort Key

- **XS**: < 1 hour. String changes, config tweaks, or single-class additions.
- **S**: 1-3 hours. New class + wiring into existing code.
- **S-M**: 3-6 hours. Multiple new classes + integration testing.

### Quick Wins (Same Day)

These can all be done in a single session:

1. **Phase 1a+1b**: Auto-inject `task_complete` + assertive nudge (~30 min)
2. **Phase 1c**: Require `task_complete` in stop decision (~1 hour)
3. **Phase 2a**: `EnvironmentContextProvider` (~30 min)
4. **Phase 2c**: Task completion instructions (~15 min)
5. **Phase 6**: `InstructionFileProvider` (~30 min)
6. **Phase 8**: `ShellSafetyMiddleware` + `ToolUsageTrackingMiddleware` (~30 min)

**Total quick wins: ~3.5 hours for Phases 1 + 2a + 2c + 6 + 8**

### Recommended First Sprint

**Phase 1 (complete) + Phase 2a + 2b + 2c** — assertive stop + system prompt improvements.

This combination addresses the root cause of shallow agent output with minimal code:
- Agent cannot stop without calling `task_complete` (~10 lines changed)
- Continuation nudge is assertive, not passive (~1 string replaced)
- Environment context is in the system prompt (~20 line class)
- Tool guidance is in the system prompt, not lossy user messages (~20 line class)
- Task completion rules tell the agent what "done" means (~5 lines added)

### Recommended Second Sprint

**Phase 3 + Phase 5** — hooks + sub-agents.

With `FunctionMiddleware` and `as_tool()` already in the framework, these are primarily
configuration and wiring work. Together they give us:
- Extensible quality gates (stop hooks)
- Per-tool-call interception (function middleware)
- Context-efficient exploration (explore sub-agent)
- Clean build/test execution (task sub-agent)

### Validation Checkpoint

After Phase 1 implementation, re-run the test scenario:
- Prompt: "Investigate this repo and find the python based workflow engine. Research the code and create a detailed architectural design."
- Target: 10+ turns, 10+ file reads, 15KB+ document with specific class/method references
- Previous baseline: 3 turns, 3 reads, shallow output
