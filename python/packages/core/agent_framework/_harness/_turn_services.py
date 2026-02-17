# Copyright (c) Microsoft. All rights reserved.

"""Internal turn services used by AgentTurnExecutor.

These services are intentionally harness-internal for now, but they use
executor-agnostic contracts to support later promotion as reusable units.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

from opentelemetry import trace

from .._agents import AgentProtocol, ChatAgent
from .._types import AgentRunResponse, AgentRunResponseUpdate
from .._workflows._events import AgentRunEvent, AgentRunUpdateEvent
from ._compaction_view import apply_compaction_plan_to_messages
from ._compaction_owner_mode import normalize_compaction_owner_mode
from ._turn_prompt_catalog import PLANNING_PROMPT, TOOL_STRATEGY_GUIDANCE, WORK_ITEM_GUIDANCE
from ._constants import HARNESS_CONTINUATION_COUNT_KEY
from ._done_tool import WORK_COMPLETE_TOOL_NAME, work_complete
from ._state_store import HarnessStateStore
from ._state import HarnessEvent, HarnessLifecycleEvent, RepairComplete

if TYPE_CHECKING:
    from ._compaction import CompactionPlan
    from .._workflows._workflow_context import WorkflowContext
    from ._compaction import ProviderAwareTokenizer
    from ._turn_buffer import SharedStateTurnBuffer, TurnBuffer
    from ._work_items import WorkItemTaskListProtocol

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("agent_framework.harness")


class TurnToolingService:
    """Composes tools and middleware for agent invocation."""

    def __init__(
        self,
        *,
        task_list: "WorkItemTaskListProtocol | None",
        sub_agent_tools: list[Any],
        work_item_middleware: Any | None,
        harness_tool_middleware: Any | None,
    ) -> None:
        self._task_list = task_list
        self._sub_agent_tools = sub_agent_tools
        self._work_item_middleware = work_item_middleware
        self._harness_tool_middleware = harness_tool_middleware
        self._state_store = HarnessStateStore()

    async def build_run_kwargs(self, ctx: "WorkflowContext[Any]") -> dict[str, Any]:
        """Build invocation kwargs from shared state plus harness runtime policy."""
        run_kwargs = await self._state_store.get_run_kwargs(ctx)

        # Filter out 'thread' because harness owns turn history.
        run_kwargs = {k: v for k, v in run_kwargs.items() if k != "thread"}

        if self._task_list is not None:
            run_kwargs["tools"] = self._task_list.get_tools()
            run_kwargs["tools"].append(work_complete)
        else:
            run_kwargs.setdefault("tools", [])
            if isinstance(run_kwargs["tools"], list):
                run_kwargs["tools"].append(work_complete)

        if self._sub_agent_tools:
            existing_tools = run_kwargs.get("tools", [])
            if not isinstance(existing_tools, list):
                existing_tools = list(existing_tools)
            existing_tools.extend(self._sub_agent_tools)
            run_kwargs["tools"] = existing_tools

        middlewares = self._collect_middlewares()
        if middlewares:
            run_kwargs["middleware"] = middlewares if len(middlewares) > 1 else middlewares[0]

        return run_kwargs

    def _collect_middlewares(self) -> list[Any]:
        middlewares: list[Any] = []
        if self._work_item_middleware is not None:
            middlewares.append(self._work_item_middleware)
        if self._harness_tool_middleware is not None:
            middlewares.append(self._harness_tool_middleware)
        return middlewares


class TurnOwnerModeService:
    """Resolves compaction owner mode from shared state with normalization."""

    def __init__(self) -> None:
        self._state_store = HarnessStateStore()

    async def get_owner_mode(self, ctx: "WorkflowContext[Any]") -> str:
        return normalize_compaction_owner_mode(await self._state_store.get_compaction_owner_mode(ctx))


class TurnInitialMessageService:
    """Loads and normalizes the initial user message from shared state."""

    def __init__(self) -> None:
        self._state_store = HarnessStateStore()

    async def get_initial_message(self, ctx: "WorkflowContext[Any]") -> Any:
        message = await self._state_store.get_initial_message(ctx)
        if not message:
            return None
        if not isinstance(message, str):
            return message
        from .._types import ChatMessage

        return ChatMessage(role="user", text=message)


class TurnCompactionViewService:
    """Builds the per-turn message view by applying compaction plan state when needed."""

    def __init__(self, *, enable_compaction: bool, turn_buffer: "TurnBuffer") -> None:
        self._enable_compaction = enable_compaction
        self._turn_buffer = turn_buffer
        self._state_store = HarnessStateStore()

    async def get_messages_for_agent(self, ctx: "WorkflowContext[Any]", owner_mode: str) -> list[Any]:
        cache = self._turn_buffer.load_messages()
        if not self._enable_compaction:
            return cache
        if owner_mode == "compaction_executor":
            return cache

        plan = await self.load_compaction_plan(ctx)
        if plan is None or plan.is_empty:
            return cache

        with _tracer.start_as_current_span(
            "harness.apply_compaction_plan",
            attributes={"harness.cache_size": len(cache)},
        ) as span:
            result = self.apply_compaction_plan(plan)
            span.set_attribute("harness.messages_after_plan", len(result))
            return result

    async def load_compaction_plan(self, ctx: "WorkflowContext[Any]") -> "CompactionPlan | None":
        plan_data = await self._state_store.get_compaction_plan_data(ctx)
        if plan_data and isinstance(plan_data, dict):
            from ._compaction import CompactionPlan

            return CompactionPlan.from_dict(plan_data)
        return None

    def apply_compaction_plan(self, plan: "CompactionPlan") -> list[Any]:
        return apply_compaction_plan_to_messages(self._turn_buffer.load_messages(), plan)


@dataclass
class TurnPreamble:
    turn_count: int
    max_turns: int


@dataclass(frozen=True)
class PreparedTurnContext:
    """Prepared turn context passed from prep/compaction into invocation."""

    turn_count: int
    max_turns: int
    owner_mode: str
    streaming: bool
    cache_size: int


class TurnPreambleService:
    """Initializes per-turn counters and emits turn-start events."""

    def __init__(self, *, event_writer: "TurnEventWriter") -> None:
        self._event_writer = event_writer
        self._state_store = HarnessStateStore()

    async def begin_turn(self, ctx: "WorkflowContext[Any]", *, repairs_made: int) -> TurnPreamble:
        turn_count = await self._state_store.get_turn_count(ctx)
        turn_count += 1
        await self._state_store.set_turn_count(ctx, turn_count)
        max_turns = await self._state_store.get_max_turns(ctx)

        await self._event_writer.emit_lifecycle_event(
            ctx,
            event_type="turn_started",
            turn_number=turn_count,
            max_turns=max_turns,
        )
        await self._event_writer.append_transcript_event(
            ctx,
            HarnessEvent(
                event_type="turn_start",
                data={"turn_number": turn_count, "repairs_made": repairs_made},
            ),
        )
        return TurnPreamble(turn_count=turn_count, max_turns=max_turns)


class TurnInvocationService:
    """Executes agent invocation and appends response messages to the turn buffer."""

    def __init__(
        self,
        *,
        agent: AgentProtocol,
        executor_id: str,
        tooling_service: TurnToolingService,
        work_item_state_service: "TurnWorkItemStateService",
        turn_buffer: "TurnBuffer",
        get_messages_for_agent: Callable[["WorkflowContext[Any]", str], Awaitable[list[Any]]],
    ) -> None:
        self._agent = agent
        self._executor_id = executor_id
        self._tooling_service = tooling_service
        self._work_item_state_service = work_item_state_service
        self._turn_buffer = turn_buffer
        self._get_messages_for_agent = get_messages_for_agent

    async def invoke(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        prepared_context: PreparedTurnContext,
    ) -> "InvocationResult":
        run_kwargs = await self._tooling_service.build_run_kwargs(ctx)
        if prepared_context.streaming:
            return await self._invoke_streaming(ctx, run_kwargs=run_kwargs, owner_mode=prepared_context.owner_mode)
        return await self._invoke_non_streaming(ctx, run_kwargs=run_kwargs, owner_mode=prepared_context.owner_mode)

    async def _invoke_non_streaming(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        run_kwargs: dict[str, Any],
        owner_mode: str,
    ) -> "InvocationResult":
        messages_to_send = await self._get_messages_for_agent(ctx, owner_mode)
        response = await self._agent.run(messages_to_send, **run_kwargs)
        await ctx.add_event(AgentRunEvent(self._executor_id, response))
        await self._work_item_state_service.emit_work_item_change_events(ctx)
        self._turn_buffer.append_messages(response.messages)
        input_tokens = response.usage_details.input_token_count if response.usage_details else 0
        output_tokens = response.usage_details.output_token_count if response.usage_details else 0
        return InvocationResult(response=response, input_tokens=input_tokens or 0, output_tokens=output_tokens or 0)

    async def _invoke_streaming(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        run_kwargs: dict[str, Any],
        owner_mode: str,
    ) -> "InvocationResult":
        msg_build_start = time.monotonic()
        messages_to_send = await self._get_messages_for_agent(ctx, owner_mode)
        msg_build_elapsed = (time.monotonic() - msg_build_start) * 1000
        logger.info(
            "AgentTurnExecutor: Prepared %d messages for agent in %.1fms",
            len(messages_to_send),
            msg_build_elapsed,
        )

        updates: list[AgentRunResponseUpdate] = []
        stream_start = time.monotonic()
        first_token_logged = False
        async for update in self._agent.run_stream(messages_to_send, **run_kwargs):
            if not first_token_logged:
                ttft = (time.monotonic() - stream_start) * 1000
                logger.info("AgentTurnExecutor: Time-to-first-token: %.1fms", ttft)
                first_token_logged = True
            updates.append(update)
            update_text = getattr(update, "text", None)
            update_contents = getattr(update, "contents", [])
            content_types = [type(c).__name__ for c in (update_contents or [])]
            text_preview = update_text[:50] if update_text else None
            logger.debug(
                "AgentTurnExecutor: Streaming update - text=%s..., contents=%s",
                text_preview,
                content_types,
            )
            await ctx.add_event(AgentRunUpdateEvent(self._executor_id, update))

        await self._work_item_state_service.emit_work_item_change_events(ctx)
        if isinstance(self._agent, ChatAgent):
            response_format = self._agent.chat_options.response_format
            response = AgentRunResponse.from_agent_run_response_updates(
                updates,
                output_format_type=response_format,
            )
        else:
            response = AgentRunResponse.from_agent_run_response_updates(updates)

        self._turn_buffer.append_messages(response.messages)
        input_tokens = response.usage_details.input_token_count if response.usage_details else 0
        output_tokens = response.usage_details.output_token_count if response.usage_details else 0
        return InvocationResult(response=response, input_tokens=input_tokens or 0, output_tokens=output_tokens or 0)


class TurnTokenBudgetService:
    """Counts cache tokens and persists unified token budget to shared state."""

    def __init__(self, *, tokenizer: "ProviderAwareTokenizer | None") -> None:
        self._tokenizer = tokenizer
        self._state_store = HarnessStateStore()

    async def update_token_budget(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        cache: list[Any],
        response: "AgentRunResponse | None" = None,
    ) -> None:
        """Update token budget using current cache contents plus API overhead hints."""
        from .._types import FunctionCallContent, FunctionResultContent, TextContent
        from ._compaction import TokenBudget

        if self._tokenizer is None:
            return

        cache_tokens = 0
        for msg in cache:
            contents = getattr(msg, "contents", None)
            if not contents:
                continue
            for content in contents:
                if isinstance(content, TextContent):
                    cache_tokens += self._tokenizer.count_tokens(content.text or "")
                elif isinstance(content, FunctionCallContent):
                    cache_tokens += self._tokenizer.count_tokens(content.name or "")
                    args = content.arguments
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    cache_tokens += self._tokenizer.count_tokens(str(args or ""))
                elif isinstance(content, FunctionResultContent):
                    cache_tokens += self._tokenizer.count_tokens(str(content.result or ""))
                else:
                    cache_tokens += self._tokenizer.count_tokens(str(content))
            # Per-message framing overhead.
            cache_tokens += 4

        budget_data = await self._state_store.get_token_budget_data(ctx)
        if budget_data and isinstance(budget_data, dict):
            budget = TokenBudget.from_dict(cast("dict[str, Any]", budget_data))
        else:
            budget = TokenBudget()

        overhead = budget.system_prompt_tokens + budget.tool_schema_tokens
        if response and response.usage_details and response.usage_details.input_token_count:
            api_input = response.usage_details.input_token_count
            output_tokens = response.usage_details.output_token_count or 0
            cache_at_send_approx = max(0, cache_tokens - output_tokens)
            computed_overhead = max(0, api_input - cache_at_send_approx)
            if computed_overhead > 0:
                overhead = computed_overhead
                budget.system_prompt_tokens = overhead

        budget.current_estimate = cache_tokens + overhead
        await self._state_store.set_token_budget_data(ctx, budget.to_dict())

    async def get_budget_estimate(self, ctx: "WorkflowContext[Any]") -> int:
        """Read current token estimate from shared state budget."""
        from ._compaction import TokenBudget

        budget_data = await self._state_store.get_token_budget_data(ctx)
        if budget_data and isinstance(budget_data, dict):
            budget = TokenBudget.from_dict(cast("dict[str, Any]", budget_data))
            return budget.current_estimate
        return 0


class TurnBudgetSyncService:
    """Updates shared token budget and logs the resulting estimate."""

    def __init__(self, *, token_budget_service: TurnTokenBudgetService, turn_buffer: "TurnBuffer") -> None:
        self._token_budget_service = token_budget_service
        self._turn_buffer = turn_buffer

    async def sync(self, ctx: "WorkflowContext[Any]", response: "AgentRunResponse | None" = None) -> None:
        await self._token_budget_service.update_token_budget(
            ctx,
            cache=self._turn_buffer.load_messages(),
            response=response,
        )
        current_tokens = await self._token_budget_service.get_budget_estimate(ctx)
        logger.info("AgentTurnExecutor: Updated token budget — %d tokens", current_tokens)


class TurnPromptAssembler:
    """Orchestrates prompt preparation for a turn."""

    def __init__(
        self,
        *,
        task_list_enabled: bool,
        get_initial_message: Callable[["WorkflowContext[Any]"], Awaitable[Any | None]],
        append_to_cache: Callable[[Any], None],
        inject_work_item_guidance: Callable[[], None],
        inject_tool_strategy_guidance: Callable[[], None],
        inject_planning_prompt: Callable[[], None],
        inject_work_item_state: Callable[["WorkflowContext[Any]"], Awaitable[None]],
        maybe_inject_work_item_reminder: Callable[["WorkflowContext[Any]"], Awaitable[None]],
        inject_jit_instructions: Callable[["WorkflowContext[Any]", int], Awaitable[None]],
    ) -> None:
        self._task_list_enabled = task_list_enabled
        self._get_initial_message = get_initial_message
        self._append_to_cache = append_to_cache
        self._inject_work_item_guidance = inject_work_item_guidance
        self._inject_tool_strategy_guidance = inject_tool_strategy_guidance
        self._inject_planning_prompt = inject_planning_prompt
        self._inject_work_item_state = inject_work_item_state
        self._maybe_inject_work_item_reminder = maybe_inject_work_item_reminder
        self._inject_jit_instructions = inject_jit_instructions

    async def prepare_turn(self, ctx: "WorkflowContext[Any]", *, turn_count: int) -> None:
        if turn_count == 1:
            initial_message = await self._get_initial_message(ctx)
            if initial_message:
                self._append_to_cache(initial_message)

            if self._task_list_enabled:
                self._inject_work_item_guidance()
                self._inject_tool_strategy_guidance()
                self._inject_planning_prompt()

        if turn_count > 1 and self._task_list_enabled:
            await self._inject_work_item_state(ctx)

        if self._task_list_enabled:
            await self._maybe_inject_work_item_reminder(ctx)

        await self._inject_jit_instructions(ctx, turn_count)


class TurnGuidanceService:
    """Injects static harness guidance prompts into the turn buffer."""

    def __init__(self, *, turn_buffer: "TurnBuffer") -> None:
        self._turn_buffer = turn_buffer

    def inject_work_item_guidance(self) -> None:
        from .._types import ChatMessage

        self._turn_buffer.append_message(ChatMessage(role="system", text=WORK_ITEM_GUIDANCE))
        logger.info("AgentTurnExecutor: Injected work item guidance message")

    def inject_tool_strategy_guidance(self) -> None:
        from .._types import ChatMessage

        self._turn_buffer.append_message(ChatMessage(role="system", text=TOOL_STRATEGY_GUIDANCE))
        logger.info("AgentTurnExecutor: Injected tool strategy guidance")

    def inject_planning_prompt(self) -> None:
        from .._types import ChatMessage

        self._turn_buffer.append_message(ChatMessage(role="system", text=PLANNING_PROMPT))
        logger.info("AgentTurnExecutor: Injected planning prompt")


class TurnSystemMessageService:
    """Appends system-role messages into the turn buffer."""

    def __init__(self, *, turn_buffer: "TurnBuffer") -> None:
        self._turn_buffer = turn_buffer

    def append_system_message(self, text: str) -> None:
        from .._types import ChatMessage

        self._turn_buffer.append_message(ChatMessage(role="system", text=text))


class TurnCheckpointService:
    """Serializes and restores turn-buffer state for workflow checkpoints."""

    def __init__(self, *, turn_buffer: "TurnBuffer") -> None:
        self._turn_buffer = turn_buffer

    async def save(self) -> dict[str, Any]:
        from .._workflows._conversation_state import encode_chat_messages

        return {
            "cache": encode_chat_messages(self._turn_buffer.load_messages()),
        }

    async def restore(self, state: dict[str, Any]) -> None:
        from .._workflows._conversation_state import decode_chat_messages

        cache_payload = state.get("cache")
        if cache_payload:
            try:
                self._turn_buffer.replace_messages(decode_chat_messages(cache_payload))
                return
            except Exception as exc:
                logger.warning("Failed to restore cache: %s", exc)
        self._turn_buffer.replace_messages([])

    def set_initial_messages(self, messages: list[Any]) -> None:
        self._turn_buffer.replace_messages(list(messages))


class TurnBufferSyncService:
    """Synchronizes local/shared turn-buffer snapshots by owner mode."""

    def __init__(self, *, shared_turn_buffer: "SharedStateTurnBuffer") -> None:
        self._shared_turn_buffer = shared_turn_buffer

    async def adopt_shared_snapshot_if_owner_mode(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        owner_mode: str,
        local_turn_buffer: "TurnBuffer",
    ) -> bool:
        if owner_mode != "compaction_executor":
            return False

        shared_messages, shared_version = await self._shared_turn_buffer.read_snapshot(ctx)
        if not shared_messages:
            return False
        if shared_version <= local_turn_buffer.snapshot_version():
            return False

        local_turn_buffer.replace_messages(shared_messages)
        return True

    async def publish_shadow_snapshot(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        owner_mode: str,
        local_turn_buffer: "TurnBuffer",
    ) -> None:
        if owner_mode != "shadow":
            return
        await self._shared_turn_buffer.write_snapshot(
            ctx,
            messages=local_turn_buffer.load_messages(),
            version=local_turn_buffer.snapshot_version(),
        )

    async def publish_owner_snapshot(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        owner_mode: str,
        local_turn_buffer: "TurnBuffer",
    ) -> None:
        if owner_mode != "compaction_executor":
            return
        await self._shared_turn_buffer.write_snapshot(
            ctx,
            messages=local_turn_buffer.load_messages(),
            version=local_turn_buffer.snapshot_version(),
        )


class TurnCompactionTelemetryService:
    """Emits compaction lifecycle and shadow parity telemetry payloads."""

    def __init__(self) -> None:
        self._state_store = HarnessStateStore()

    async def emit_compaction_completed(
        self,
        *,
        ctx: "WorkflowContext[Any]",
        turn_count: int,
        owner_mode: str,
        strategies_applied: list[str],
        tokens_before: int,
        tokens_after: int,
        tokens_freed: int,
        duration_ms: float,
        compaction_level: str,
    ) -> None:
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="compaction_completed",
                turn_number=turn_count,
                data={
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "tokens_freed": tokens_freed,
                    "duration_ms": round(duration_ms, 1),
                    "strategies_applied": strategies_applied,
                    "compaction_level": compaction_level,
                    "compaction_owner_mode": owner_mode,
                },
            ),
        )

    async def emit_shadow_compare(
        self,
        *,
        ctx: "WorkflowContext[Any]",
        owner_mode: str,
        turn_count: int,
        actual_attempted: bool,
        actual_effective: bool,
        strategies_applied: list[str],
        tokens_before: int,
        tokens_after: int,
        tokens_freed: int,
        local_message_count: int,
        local_buffer_version: int,
    ) -> None:
        if owner_mode != "shadow":
            return

        candidate_data = await self._state_store.get_compaction_shadow_candidate(ctx)

        candidate_would_compact = bool((candidate_data or {}).get("would_compact", False))
        candidate_blocking = bool((candidate_data or {}).get("blocking", False))
        candidate_effective_present = "candidate_effective_compaction" in (candidate_data or {})
        candidate_effective_compaction = bool((candidate_data or {}).get("candidate_effective_compaction", False))
        diverged = (
            candidate_effective_compaction != actual_effective
            if candidate_effective_present
            else candidate_would_compact != actual_effective
        )
        shared_buffer = await self._get_shared_buffer_info(ctx)
        buffer_parity_match = (
            bool(shared_buffer.get("present"))
            and int(shared_buffer.get("message_count", -1)) == local_message_count
            and int(shared_buffer.get("version", -1)) == local_buffer_version
        )

        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="context_pressure",
                turn_number=turn_count,
                data={
                    "shadow_compare": {
                        "owner_mode": owner_mode,
                        "candidate_present": candidate_data is not None,
                        "candidate_would_compact": candidate_would_compact,
                        "candidate_blocking": candidate_blocking,
                        "candidate_effective_compaction": candidate_effective_compaction,
                        "candidate_effective_present": candidate_effective_present,
                        "candidate_strategies_applied": (candidate_data or {}).get("candidate_strategies_applied", []),
                        "actual_attempted": actual_attempted,
                        "actual_effective": actual_effective,
                        "actual_strategies_applied": strategies_applied,
                        "tokens_before": tokens_before,
                        "tokens_after": tokens_after,
                        "tokens_freed": tokens_freed,
                        "diverged": diverged,
                        "buffer_parity": {
                            "local_message_count": local_message_count,
                            "local_version": local_buffer_version,
                            "shared_present": bool(shared_buffer.get("present", False)),
                            "shared_message_count": int(shared_buffer.get("message_count", 0)),
                            "shared_version": int(shared_buffer.get("version", 0)),
                            "match": buffer_parity_match,
                        },
                    }
                },
            ),
        )

    async def _get_shared_buffer_info(self, ctx: "WorkflowContext[Any]") -> dict[str, Any]:
        payload = await self._state_store.get_shared_turn_buffer_payload(ctx)
        if not isinstance(payload, dict):
            return {"present": False, "version": 0, "message_count": 0}
        return {
            "present": True,
            "version": int(payload.get("version", 0)),
            "message_count": int(payload.get("message_count", 0)),
        }

    async def publish_owner_snapshot(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        owner_mode: str,
        local_turn_buffer: "TurnBuffer",
    ) -> None:
        if owner_mode != "compaction_executor":
            return
        await self._shared_turn_buffer.write_snapshot(
            ctx,
            messages=local_turn_buffer.load_messages(),
            version=local_turn_buffer.snapshot_version(),
        )


@dataclass
class CompactionExecutionResult:
    strategies_applied: list[str]
    tokens_before: int
    tokens_after: int
    tokens_freed: int
    duration_ms: float


@dataclass(frozen=True)
class CompactionDecision:
    """Typed handoff for compaction outcome between stages."""

    attempted: bool
    effective: bool
    result: CompactionExecutionResult | None


@dataclass(frozen=True)
class InvocationResult:
    """Typed handoff for invocation outcome and usage metadata."""

    response: AgentRunResponse | None
    input_tokens: int
    output_tokens: int


class TurnCompactionService:
    """Orchestrates compaction execution for a turn."""

    def __init__(
        self,
        *,
        enable_compaction: bool,
        is_compaction_needed: Callable[[RepairComplete], bool],
        ensure_message_ids: Callable[[list[Any]], None],
        get_budget_estimate: Callable[["WorkflowContext[Any]"], Awaitable[int]],
        run_full_compaction: Callable[["WorkflowContext[Any]", int], Awaitable[list[str]]],
        apply_direct_clear: Callable[[int, int, int], int],
        update_token_budget: Callable[["WorkflowContext[Any]"], Awaitable[None]],
        classify_compaction_level: Callable[[list[str], int, int], str],
        increment_compaction_count: Callable[[], None],
        owner_mode_service: TurnOwnerModeService | None = None,
        telemetry_service: "TurnCompactionTelemetryService | None" = None,
    ) -> None:
        self._enable_compaction = enable_compaction
        self._owner_mode_service = owner_mode_service or TurnOwnerModeService()
        self._is_compaction_needed = is_compaction_needed
        self._ensure_message_ids = ensure_message_ids
        self._get_budget_estimate = get_budget_estimate
        self._run_full_compaction = run_full_compaction
        self._apply_direct_clear = apply_direct_clear
        self._update_token_budget = update_token_budget
        self._classify_compaction_level = classify_compaction_level
        self._increment_compaction_count = increment_compaction_count
        self._telemetry_service = telemetry_service or TurnCompactionTelemetryService()

    async def maybe_compact(
        self,
        *,
        trigger: RepairComplete,
        ctx: "WorkflowContext[Any]",
        turn_count: int,
        cache: list[Any],
        local_buffer_version: int = 0,
    ) -> CompactionExecutionResult | None:
        owner_mode = await self._get_owner_mode(ctx)
        compaction_signaled = self._enable_compaction and self._is_compaction_needed(trigger)
        if not compaction_signaled:
            await self._telemetry_service.emit_shadow_compare(
                ctx=ctx,
                owner_mode=owner_mode,
                turn_count=turn_count,
                actual_attempted=False,
                actual_effective=False,
                strategies_applied=[],
                tokens_before=0,
                tokens_after=0,
                tokens_freed=0,
                local_message_count=len(cache),
                local_buffer_version=local_buffer_version,
            )
            return None

        self._ensure_message_ids(cache)
        tokens_before = await self._get_budget_estimate(ctx)
        start_time = time.monotonic()

        strategies_applied = await self._run_full_compaction(ctx, turn_count)
        if strategies_applied:
            self._increment_compaction_count()
        else:
            target_after = int(tokens_before * 0.50) if tokens_before > 0 else 0
            target_to_free = max(0, tokens_before - target_after) if tokens_before > 0 else 0
            cleared_count = self._apply_direct_clear(turn_count, 2, target_to_free)
            if cleared_count > 0:
                strategies_applied = ["clear"]

        duration_ms = (time.monotonic() - start_time) * 1000
        await self._update_token_budget(ctx)
        tokens_after = await self._get_budget_estimate(ctx)
        tokens_freed = max(0, tokens_before - tokens_after)

        compaction_level = self._classify_compaction_level(strategies_applied, tokens_before, tokens_after)
        await self._telemetry_service.emit_compaction_completed(
            ctx=ctx,
            turn_count=turn_count,
            owner_mode=owner_mode,
            strategies_applied=strategies_applied,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_freed=tokens_freed,
            duration_ms=duration_ms,
            compaction_level=compaction_level,
        )
        await self._telemetry_service.emit_shadow_compare(
            ctx=ctx,
            owner_mode=owner_mode,
            turn_count=turn_count,
            actual_attempted=True,
            actual_effective=(bool(strategies_applied) or tokens_freed > 0),
            strategies_applied=strategies_applied,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_freed=tokens_freed,
            local_message_count=len(cache),
            local_buffer_version=local_buffer_version,
        )

        return CompactionExecutionResult(
            strategies_applied=strategies_applied,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_freed=tokens_freed,
            duration_ms=duration_ms,
        )

    async def _get_owner_mode(self, ctx: "WorkflowContext[Any]") -> str:
        return await self._owner_mode_service.get_owner_mode(ctx)


class TurnCompactionEngine:
    """Core compaction mechanics operating on the live turn cache."""

    def __init__(
        self,
        *,
        turn_buffer: "TurnBuffer",
        tokenizer: "ProviderAwareTokenizer | None",
        summarizer: Any | None,
        artifact_store: Any | None,
        load_compaction_plan: Callable[["WorkflowContext[Any]"], Awaitable["CompactionPlan | None"]],
        apply_compaction_plan: Callable[["CompactionPlan"], list[Any]],
    ) -> None:
        self._turn_buffer = turn_buffer
        self._tokenizer = tokenizer
        self._summarizer = summarizer
        self._artifact_store = artifact_store
        self._load_compaction_plan = load_compaction_plan
        self._apply_compaction_plan = apply_compaction_plan
        self._state_store = HarnessStateStore()

    def ensure_message_ids(self, messages: list[Any]) -> None:
        for msg in messages:
            if getattr(msg, "message_id", None) is None:
                msg.message_id = f"msg-{uuid.uuid4().hex[:12]}"

    def apply_direct_clear(
        self,
        current_turn: int,
        preserve_recent_turns: int = 2,
        target_tokens_to_free: int = 0,
    ) -> int:
        from .._types import ChatMessage, FunctionResultContent

        cache = self._turn_buffer.load_messages()
        if len(cache) <= preserve_recent_turns:
            return 0

        cleared_count = 0
        tokens_freed_estimate = 0
        cutoff = len(cache) - preserve_recent_turns

        for i in range(cutoff):
            if target_tokens_to_free > 0 and tokens_freed_estimate >= target_tokens_to_free:
                break

            msg = cache[i]
            contents = getattr(msg, "contents", None)
            if not contents:
                continue

            new_contents: list[Any] = []
            modified = False
            for content in contents:
                if isinstance(content, FunctionResultContent):
                    result_str = str(content.result or "")
                    if len(result_str) > 100:
                        if target_tokens_to_free > 0 and tokens_freed_estimate >= target_tokens_to_free:
                            new_contents.append(content)
                            continue
                        placeholder = "[Tool result cleared to save context]"
                        tokens_freed_estimate += max(0, len(result_str) - len(placeholder)) // 4
                        content = FunctionResultContent(
                            call_id=content.call_id,
                            result=placeholder,
                        )
                        modified = True
                        cleared_count += 1
                new_contents.append(content)

            if modified:
                role = getattr(msg, "role", "tool")
                role_value: str = str(getattr(role, "value", role))
                cache[i] = ChatMessage(
                    role=role_value,  # type: ignore[arg-type]
                    contents=new_contents,
                    message_id=getattr(msg, "message_id", None),
                )

        return cleared_count

    async def run_full_compaction(
        self,
        ctx: "WorkflowContext[Any]",
        turn_count: int,
        get_budget_estimate: Callable[["WorkflowContext[Any]"], Awaitable[int]],
    ) -> list[str]:
        from ._compaction import (
            CacheThreadAdapter,
            ClearStrategy,
            CompactionCoordinator,
            DropStrategy,
            ExternalizeStrategy,
            SummarizeStrategy,
            TokenBudget,
            TurnContext,
        )

        if self._tokenizer is None:
            return []

        cache = self._turn_buffer.load_messages()
        existing_plan = await self._load_compaction_plan(ctx)
        if existing_plan and not existing_plan.is_empty:
            self._turn_buffer.replace_messages(self._apply_compaction_plan(existing_plan))
            cache = self._turn_buffer.load_messages()
            await self._state_store.set_compaction_plan_data(ctx, None)
            logger.debug(
                "TurnCompactionEngine: Flattened existing plan into cache (%d messages)",
                len(cache),
            )

        thread_adapter = CacheThreadAdapter(cache)

        budget_data = await self._state_store.get_token_budget_data(ctx)
        if budget_data and isinstance(budget_data, dict):
            budget = TokenBudget.from_dict(cast("dict[str, Any]", budget_data))
        else:
            budget = TokenBudget()

        current_tokens = await get_budget_estimate(ctx)
        if current_tokens <= 0:
            for msg in cache:
                text = getattr(msg, "text", None) or ""
                current_tokens += self._tokenizer.count_tokens(str(text))

        tokens_over = budget.tokens_over_threshold(current_tokens)
        logger.info(
            "TurnCompactionEngine: Full compaction check — current_tokens=%d, "
            "budget_max=%d, soft_threshold=%d, available_for_messages=%d, "
            "tokens_over=%d, cache_size=%d",
            current_tokens,
            budget.max_input_tokens,
            budget.soft_threshold,
            budget.available_for_messages,
            tokens_over,
            len(cache),
        )
        if tokens_over <= 0:
            logger.debug("TurnCompactionEngine: Full compaction skipped — under threshold (%d tokens)", current_tokens)
            return []

        tokens_to_free = tokens_over + int(budget.max_input_tokens * 0.1)
        strategies: list[ClearStrategy | SummarizeStrategy | DropStrategy] = [ClearStrategy()]
        if self._summarizer is not None:
            if self._artifact_store is not None:
                strategies.append(ExternalizeStrategy(self._artifact_store, self._summarizer))
            strategies.append(SummarizeStrategy(self._summarizer))
        strategies.append(DropStrategy())

        turn_context = TurnContext(turn_number=turn_count)
        coordinator = CompactionCoordinator(strategies=strategies)  # type: ignore[arg-type]

        try:
            result = await coordinator.compact(
                thread_adapter,  # type: ignore[arg-type]
                None,
                budget,
                self._tokenizer,
                tokens_to_free=tokens_to_free,
                turn_context=turn_context,
            )
        except Exception:
            logger.warning(
                "TurnCompactionEngine: Full compaction failed (cache_size=%d, tokens=%d, tokens_to_free=%d)",
                len(cache),
                current_tokens,
                tokens_to_free,
                exc_info=True,
            )
            return []

        logger.info(
            "TurnCompactionEngine: Compaction result — plan=%s, is_empty=%s, "
            "proposals_generated=%d, proposals_applied=%d, tokens_freed=%d",
            "present" if result.plan else "None",
            result.plan.is_empty if result.plan else "N/A",
            result.proposals_generated,
            result.proposals_applied,
            result.tokens_freed,
        )
        if result.plan and not result.plan.is_empty:
            self._turn_buffer.replace_messages(self._apply_compaction_plan(result.plan))
            cache = self._turn_buffer.load_messages()
            await self._state_store.set_compaction_plan_data(ctx, None)
            logger.info(
                "TurnCompactionEngine: Compaction complete — freed ~%d tokens, applied %d/%d proposals, "
                "cache flattened to %d messages",
                result.tokens_freed,
                result.proposals_applied,
                result.proposals_generated,
                len(cache),
            )
            strategies_applied: list[str] = []
            if result.plan.clearings:
                strategies_applied.append("clear")
            if result.plan.summarizations:
                strategies_applied.append("summarize")
            if result.plan.externalizations:
                strategies_applied.append("externalize")
            if result.plan.drops:
                strategies_applied.append("drop")
            return strategies_applied

        return []


class TurnEventWriter:
    """Writes transcript and lifecycle events for a harness turn."""

    def __init__(self) -> None:
        self._state_store = HarnessStateStore()

    async def append_transcript_event(self, ctx: "WorkflowContext[Any]", event: HarnessEvent) -> None:
        await self._state_store.append_transcript_event(ctx, event)

    async def emit_lifecycle_event(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        event_type: str,
        turn_number: int = 0,
        max_turns: int = 0,
        data: dict[str, Any] | None = None,
    ) -> None:
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type=cast(Any, event_type),
                turn_number=turn_number,
                max_turns=max_turns,
                data=data,
            )
        )


@dataclass
class TurnOutcome:
    has_tool_calls: bool
    called_work_complete: bool
    agent_done: bool
    continuation_prompt_sent: bool = False


@dataclass
class TurnPostprocessResult:
    agent_done: bool
    has_tool_calls: bool
    called_work_complete: bool
    continuation_prompt_sent: bool
    token_estimate: int
    cache_size: int


class TurnOutcomeEvaluator:
    """Evaluates whether the turn is done and if continuation prompting is needed."""

    def __init__(
        self,
        *,
        enable_continuation_prompts: bool,
        max_continuation_prompts: int,
        get_continuation_count: Callable[["WorkflowContext[Any]"], Awaitable[int]],
        inject_continuation_prompt: Callable[["WorkflowContext[Any]", int], Awaitable[None]],
    ) -> None:
        self._enable_continuation_prompts = enable_continuation_prompts
        self._max_continuation_prompts = max_continuation_prompts
        self._get_continuation_count = get_continuation_count
        self._inject_continuation_prompt = inject_continuation_prompt

    def has_tool_calls(self, response: "AgentRunResponse") -> bool:
        if response.user_input_requests:
            return True

        for message in response.messages:
            if hasattr(message, "contents") and message.contents:
                for content in message.contents:
                    if hasattr(content, "__class__") and "FunctionCall" in content.__class__.__name__:
                        tool_name = getattr(content, "name", None)
                        if tool_name and tool_name != WORK_COMPLETE_TOOL_NAME:
                            return True
        return False

    def has_work_complete_call(self, response: "AgentRunResponse") -> bool:
        for message in response.messages:
            if hasattr(message, "contents") and message.contents:
                for content in message.contents:
                    if hasattr(content, "__class__") and "FunctionCall" in content.__class__.__name__:
                        tool_name = getattr(content, "name", None)
                        if tool_name == WORK_COMPLETE_TOOL_NAME:
                            return True
        return False

    async def evaluate(self, response: "AgentRunResponse", ctx: "WorkflowContext[Any]") -> TurnOutcome:
        has_tool_calls = self.has_tool_calls(response)
        called_work_complete = self.has_work_complete_call(response)

        if called_work_complete:
            return TurnOutcome(
                has_tool_calls=has_tool_calls,
                called_work_complete=True,
                agent_done=True,
                continuation_prompt_sent=False,
            )
        if has_tool_calls:
            return TurnOutcome(
                has_tool_calls=True,
                called_work_complete=False,
                agent_done=False,
                continuation_prompt_sent=False,
            )

        continuation_count = await self._get_continuation_count(ctx)
        if self._enable_continuation_prompts and continuation_count < self._max_continuation_prompts:
            await self._inject_continuation_prompt(ctx, continuation_count)
            return TurnOutcome(
                has_tool_calls=False,
                called_work_complete=False,
                agent_done=False,
                continuation_prompt_sent=True,
            )

        return TurnOutcome(
            has_tool_calls=False,
            called_work_complete=False,
            agent_done=True,
            continuation_prompt_sent=False,
        )


class TurnContinuationService:
    """Manages continuation prompt state and transcript/lifecycle emissions."""

    def __init__(
        self,
        *,
        continuation_prompt: str,
        max_continuation_prompts: int,
        task_list: "WorkItemTaskListProtocol | None",
        event_writer: TurnEventWriter,
        append_system_message: Callable[[str], None],
    ) -> None:
        self._continuation_prompt = continuation_prompt
        self._max_continuation_prompts = max_continuation_prompts
        self._task_list = task_list
        self._event_writer = event_writer
        self._append_system_message = append_system_message
        self._state_store = HarnessStateStore()

    async def get_continuation_count(self, ctx: "WorkflowContext[Any]") -> int:
        return await self._state_store.get_continuation_count(ctx, key=HARNESS_CONTINUATION_COUNT_KEY)

    async def inject_continuation_prompt(self, ctx: "WorkflowContext[Any]", current_count: int) -> None:
        prompt_parts = [self._continuation_prompt]
        if self._task_list is not None and self._task_list.ledger:
            incomplete = self._task_list.ledger.get_incomplete_items()
            if incomplete:
                items_summary = ", ".join(f'"{item.title}" ({item.status.value})' for item in incomplete)
                prompt_parts.append(f"\nOpen work items ({len(incomplete)}): {items_summary}")
            else:
                prompt_parts.append("\nNo open work items remain.")

        self._append_system_message("\n".join(prompt_parts))
        await self._state_store.set_continuation_count(
            ctx,
            key=HARNESS_CONTINUATION_COUNT_KEY,
            value=current_count + 1,
        )
        await self._event_writer.append_transcript_event(
            ctx,
            HarnessEvent(
                event_type="continuation_prompt",
                data={
                    "count": current_count + 1,
                    "max": self._max_continuation_prompts,
                    "prompt": self._continuation_prompt,
                },
            ),
        )
        await self._event_writer.emit_lifecycle_event(
            ctx,
            event_type="continuation_prompt",
            data={
                "count": current_count + 1,
                "max": self._max_continuation_prompts,
            },
        )


class TurnJitInstructionService:
    """Evaluates and injects JIT instructions into the turn buffer."""

    def __init__(self, *, jit_processor: Any, turn_buffer: "TurnBuffer") -> None:
        self._jit_processor = jit_processor
        self._turn_buffer = turn_buffer
        self._state_store = HarnessStateStore()

    async def inject_instructions(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        turn_count: int,
        compaction_count: int,
        tool_usage: dict[str, int],
    ) -> None:
        from .._types import ChatMessage
        from ._jit_instructions import JitContext
        from ._work_items import WorkItemLedger

        work_complete, work_total = 0, 0
        ledger_data = await self._state_store.get_work_item_ledger_data(ctx)
        if ledger_data and isinstance(ledger_data, dict):
            ledger = WorkItemLedger.from_dict(ledger_data)
            work_total = len(ledger.items)
            work_complete = work_total - len(ledger.get_incomplete_items())

        max_turns = await self._state_store.get_max_turns(ctx)

        jit_ctx = JitContext(
            turn=turn_count,
            max_turns=max_turns,
            tool_usage=tool_usage,
            work_items_complete=work_complete,
            work_items_total=work_total,
            compaction_count=compaction_count,
        )

        instructions = self._jit_processor.evaluate(jit_ctx)
        for text in instructions:
            self._turn_buffer.append_message(ChatMessage(role="system", text=text))
            logger.info("AgentTurnExecutor: Injected JIT instruction: %s", text[:80])


class TurnWorkItemStateService:
    """Manages work-item event emission, ledger syncing, and control invariants."""

    def __init__(
        self,
        *,
        task_list: "WorkItemTaskListProtocol | None",
        work_item_middleware: Any | None,
        event_writer: TurnEventWriter,
        append_system_message: Callable[[str], None],
    ) -> None:
        self._task_list = task_list
        self._work_item_middleware = work_item_middleware
        self._event_writer = event_writer
        self._append_system_message = append_system_message
        self._state_store = HarnessStateStore()

    async def emit_work_item_change_events(self, ctx: "WorkflowContext[Any]") -> None:
        if self._work_item_middleware is None:
            return

        events = self._work_item_middleware.drain_events()
        for event_data in events:
            await self._event_writer.emit_lifecycle_event(
                ctx,
                event_type="work_item_changed",
                data=event_data,
            )
            logger.debug(
                "TurnWorkItemStateService: Emitted work_item_changed event for tool %s",
                event_data.get("tool", "unknown"),
            )

    async def sync_ledger(self, ctx: "WorkflowContext[Any]") -> None:
        if self._task_list is None:
            return

        ledger = self._task_list.ledger
        await self._state_store.set_work_item_ledger_data(ctx, ledger.to_dict())

        if ledger.items:
            deliverables = ledger.get_deliverables()
            total_items = len(ledger.items)
            done_items = sum(1 for item in ledger.items.values() if item.status.value in ("done", "skipped"))
            await self._event_writer.emit_lifecycle_event(
                ctx,
                event_type="deliverables_updated",
                data={
                    "count": len(deliverables),
                    "total_items": total_items,
                    "done_items": done_items,
                    "items": [
                        {
                            "item_id": i.id,
                            "title": i.title,
                            "content": i.artifact,
                        }
                        for i in deliverables
                    ],
                    "all_items": [
                        {
                            "id": item.id,
                            "title": item.title,
                            "status": item.status.value,
                            "priority": item.priority.value,
                            "artifact_role": item.artifact_role.value,
                            "notes": item.notes,
                            "requires_revision": item.requires_revision,
                            "created_at": item.created_at,
                            "updated_at": item.updated_at,
                        }
                        for item in ledger.items.values()
                    ],
                },
            )

        invariant_prompt = self.check_control_invariants(ledger)
        if invariant_prompt:
            self._append_system_message(invariant_prompt)
            await self._event_writer.append_transcript_event(
                ctx,
                HarnessEvent(
                    event_type="control_invariant_violation",
                    data={"prompt": invariant_prompt},
                ),
            )
            logger.info("TurnWorkItemStateService: Control invariant violated, injected continuation prompt")

    @staticmethod
    def check_control_invariants(ledger: Any) -> str | None:
        import json

        from ._work_items import ArtifactRole, WorkItemStatus

        failed_control_items: list[tuple[str, list[dict[str, Any]]]] = []

        for item in ledger.items.values():
            if (
                item.artifact_role == ArtifactRole.CONTROL
                and item.artifact
                and item.status in (WorkItemStatus.DONE, WorkItemStatus.IN_PROGRESS)
            ):
                try:
                    data = json.loads(item.artifact)
                    if data.get("verdict") == "fail":
                        failed_checks = [c for c in data.get("checks", []) if c.get("result") == "fail"]
                        if failed_checks:
                            failed_control_items.append((item.id, failed_checks))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    continue

        if not failed_control_items:
            return None

        has_revisions = any(item.revision_of for item in ledger.items.values())
        if has_revisions:
            return None

        check_names: list[str] = []
        for _, checks in failed_control_items:
            check_names.extend(c.get("name", "unnamed") for c in checks)

        return (
            "Your control audit reported a 'fail' verdict with failed checks: "
            f"{', '.join(check_names)}. "
            "You must call flag_revision on the work items that need correction, "
            "then complete the revision items with corrected artifacts before finishing."
        )


class TurnWorkItemPromptService:
    """Builds and injects work-item progress/reminder prompts for turn preparation."""

    def __init__(
        self,
        *,
        turn_buffer: "TurnBuffer",
        event_writer: TurnEventWriter,
    ) -> None:
        self._turn_buffer = turn_buffer
        self._event_writer = event_writer
        self._state_store = HarnessStateStore()

    def get_tool_usage(self) -> dict[str, int]:
        tool_usage: dict[str, int] = {}
        for msg in self._turn_buffer.load_messages():
            contents = getattr(msg, "contents", None)
            if not contents:
                continue
            for content in contents:
                content_type = getattr(content, "type", None)
                if content_type == "function_call":
                    tool_name = getattr(content, "name", None)
                    if tool_name and not tool_name.startswith("work_item_"):
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage

    async def inject_work_item_state(self, ctx: "WorkflowContext[Any]") -> None:
        from .._types import ChatMessage
        from ._work_items import WorkItemLedger, WorkItemStatus

        ledger_data = await self._state_store.get_work_item_ledger_data(ctx)
        if not ledger_data or not isinstance(ledger_data, dict):
            return
        ledger = WorkItemLedger.from_dict(ledger_data)

        if not ledger.items:
            return

        incomplete = ledger.get_incomplete_items()
        if not incomplete:
            return

        tool_usage = self.get_tool_usage()
        status_icons = {
            WorkItemStatus.PENDING: "[ ]",
            WorkItemStatus.IN_PROGRESS: "[~]",
            WorkItemStatus.DONE: "[x]",
            WorkItemStatus.SKIPPED: "[-]",
        }

        lines = [f"Work items ({len(incomplete)} remaining):"]
        for item in incomplete:
            icon = status_icons.get(item.status, "[ ]")
            revision_tag = " [NEEDS REVISION]" if item.requires_revision else ""
            lines.append(f"  {icon} [{item.id}] {item.title}{revision_tag}")

        if tool_usage:
            lines.append("")
            lines.append(
                "Your tool usage so far: " + ", ".join(f"{name}: {count}" for name, count in sorted(tool_usage.items()))
            )
            read_count = tool_usage.get("read_file", 0)
            if read_count < 10:
                lines.append(
                    f"NOTE: You have only read {read_count} file(s). For thorough "
                    "code investigation, aim to read 10-20+ relevant source files."
                )

        lines.append("")
        lines.append("Continue working on the next incomplete item.")
        self._turn_buffer.append_message(ChatMessage(role="system", text="\n".join(lines)))
        logger.info("AgentTurnExecutor: Injected work item state (%d remaining)", len(incomplete))

    async def maybe_inject_work_item_reminder(self, ctx: "WorkflowContext[Any]") -> None:
        from .._types import ChatMessage
        from ._work_items import WorkItemLedger, format_work_item_reminder

        transcript = await self._state_store.get_transcript_data(ctx)
        if not transcript:
            return

        last_stop = None
        for event_data in reversed(transcript):
            if event_data.get("event_type") == "stop_decision":
                last_stop = event_data
                break
        if last_stop is None:
            return

        data = last_stop.get("data", {})
        if data.get("reason") != "work_items_incomplete":
            return

        ledger_data = await self._state_store.get_work_item_ledger_data(ctx)
        if not ledger_data or not isinstance(ledger_data, dict):
            return
        ledger = WorkItemLedger.from_dict(ledger_data)

        reminder_text = format_work_item_reminder(ledger)
        if not reminder_text:
            return

        self._turn_buffer.append_message(ChatMessage(role="system", text=reminder_text))
        await self._event_writer.append_transcript_event(
            ctx,
            HarnessEvent(
                event_type="work_item_reminder",
                data={"incomplete_count": len(ledger.get_incomplete_items())},
            ),
        )
        logger.info(
            "AgentTurnExecutor: Injected work item reminder (%d items remaining)",
            len(ledger.get_incomplete_items()),
        )


class TurnPostprocessService:
    """Finalizes turn outcomes, token budgets, and completion lifecycle emission."""

    def __init__(
        self,
        *,
        event_writer: TurnEventWriter,
        outcome_evaluator: TurnOutcomeEvaluator,
        work_item_state_service: TurnWorkItemStateService,
        task_list_enabled: bool,
        update_token_budget: Callable[["WorkflowContext[Any]", AgentRunResponse | None], Awaitable[None]],
        get_budget_estimate: Callable[["WorkflowContext[Any]"], Awaitable[int]],
        buffer_sync_service: TurnBufferSyncService,
        turn_buffer: "TurnBuffer",
    ) -> None:
        self._event_writer = event_writer
        self._outcome_evaluator = outcome_evaluator
        self._work_item_state_service = work_item_state_service
        self._task_list_enabled = task_list_enabled
        self._update_token_budget = update_token_budget
        self._get_budget_estimate = get_budget_estimate
        self._buffer_sync_service = buffer_sync_service
        self._turn_buffer = turn_buffer

    async def finalize_turn(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        response: AgentRunResponse,
        turn_count: int,
        max_turns: int,
        owner_mode: str,
    ) -> TurnPostprocessResult:
        await self._event_writer.append_transcript_event(
            ctx,
            HarnessEvent(
                event_type="agent_response",
                data={
                    "turn_number": turn_count,
                    "message_count": len(response.messages),
                    "has_tool_calls": self._outcome_evaluator.has_tool_calls(response),
                },
            ),
        )

        if self._task_list_enabled:
            await self._work_item_state_service.sync_ledger(ctx)

        outcome = await self._outcome_evaluator.evaluate(response, ctx)
        await self._update_token_budget(ctx, response)
        await self._buffer_sync_service.publish_owner_snapshot(
            ctx,
            owner_mode=owner_mode,
            local_turn_buffer=self._turn_buffer,
        )
        token_estimate = await self._get_budget_estimate(ctx)
        cache_size = len(self._turn_buffer.load_messages())

        await self._event_writer.emit_lifecycle_event(
            ctx,
            event_type="turn_completed",
            turn_number=turn_count,
            max_turns=max_turns,
            data={
                "agent_done": outcome.agent_done,
                "has_tool_calls": outcome.has_tool_calls,
                "called_work_complete": outcome.called_work_complete,
                "token_estimate": token_estimate,
                "cache_size": cache_size,
            },
        )

        return TurnPostprocessResult(
            agent_done=outcome.agent_done,
            has_tool_calls=outcome.has_tool_calls,
            called_work_complete=outcome.called_work_complete,
            continuation_prompt_sent=outcome.continuation_prompt_sent,
            token_estimate=token_estimate,
            cache_size=cache_size,
        )


class TurnCompletionSignalService:
    """Maps turn outcome into workflow completion message signaling."""

    async def send_completion(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        agent_done: bool,
        called_work_complete: bool,
        error: str | None = None,
    ) -> None:
        from ._state import TurnComplete

        await ctx.send_message(
            TurnComplete(
                agent_done=agent_done,
                called_work_complete=called_work_complete,
                error=error,
            )
        )


class TurnErrorHandlingService:
    """Handles turn exceptions with consistent transcript + completion signaling."""

    def __init__(
        self,
        *,
        event_writer: TurnEventWriter,
        completion_signal_service: TurnCompletionSignalService,
    ) -> None:
        self._event_writer = event_writer
        self._completion_signal_service = completion_signal_service

    async def handle_turn_error(
        self,
        ctx: "WorkflowContext[Any]",
        *,
        turn_count: int,
        error: Exception,
    ) -> None:
        error_text = str(error)
        await self._event_writer.append_transcript_event(
            ctx,
            HarnessEvent(
                event_type="agent_response",
                data={
                    "turn_number": turn_count,
                    "error": error_text,
                },
            ),
        )
        await self._completion_signal_service.send_completion(
            ctx,
            agent_done=False,
            called_work_complete=False,
            error=error_text,
        )


class TurnExecutionCoordinator:
    """Coordinates the end-to-end sequence for one turn execution."""

    def __init__(
        self,
        *,
        preamble_service: TurnPreambleService,
        owner_mode_service: TurnOwnerModeService,
        buffer_sync_service: TurnBufferSyncService,
        prompt_assembler: TurnPromptAssembler,
        compaction_service: TurnCompactionService,
        invocation_service: TurnInvocationService,
        postprocess_service: TurnPostprocessService,
        completion_signal_service: TurnCompletionSignalService,
        error_handling_service: TurnErrorHandlingService,
        logging_service: "TurnLoggingService",
        turn_buffer: "TurnBuffer",
        get_compaction_count: Callable[[], int],
    ) -> None:
        self._preamble_service = preamble_service
        self._owner_mode_service = owner_mode_service
        self._buffer_sync_service = buffer_sync_service
        self._prompt_assembler = prompt_assembler
        self._compaction_service = compaction_service
        self._invocation_service = invocation_service
        self._postprocess_service = postprocess_service
        self._completion_signal_service = completion_signal_service
        self._error_handling_service = error_handling_service
        self._logging_service = logging_service
        self._turn_buffer = turn_buffer
        self._get_compaction_count = get_compaction_count

    async def execute_turn(self, trigger: RepairComplete, ctx: "WorkflowContext[Any]") -> None:
        preamble = await self._preamble_service.begin_turn(ctx, repairs_made=trigger.repairs_made)
        turn_count = preamble.turn_count
        max_turns = preamble.max_turns

        self._logging_service.log_turn_start(turn_count)

        owner_mode = await self._owner_mode_service.get_owner_mode(ctx)
        prepared_context = PreparedTurnContext(
            turn_count=turn_count,
            max_turns=max_turns,
            owner_mode=owner_mode,
            streaming=ctx.is_streaming(),
            cache_size=len(self._turn_buffer.load_messages()),
        )
        adopted = await self._buffer_sync_service.adopt_shared_snapshot_if_owner_mode(
            ctx,
            owner_mode=prepared_context.owner_mode,
            local_turn_buffer=self._turn_buffer,
        )
        if adopted:
            self._logging_service.log_shared_buffer_adopted(
                version=self._turn_buffer.snapshot_version(),
                message_count=len(self._turn_buffer.load_messages()),
            )
        await self._prompt_assembler.prepare_turn(ctx, turn_count=turn_count)
        await self._buffer_sync_service.publish_shadow_snapshot(
            ctx,
            owner_mode=prepared_context.owner_mode,
            local_turn_buffer=self._turn_buffer,
        )

        with _tracer.start_as_current_span("harness.compaction", attributes={"harness.turn": turn_count}) as span:
            compaction_result = await self._compaction_service.maybe_compact(
                trigger=trigger,
                ctx=ctx,
                turn_count=turn_count,
                cache=self._turn_buffer.load_messages(),
                local_buffer_version=self._turn_buffer.snapshot_version(),
            )
            compaction_decision = CompactionDecision(
                attempted=compaction_result is not None,
                effective=(
                    compaction_result is not None
                    and (bool(compaction_result.strategies_applied) or compaction_result.tokens_freed > 0)
                ),
                result=compaction_result,
            )
            if compaction_decision.result is not None:
                span.set_attribute("harness.compaction.tokens_before", compaction_result.tokens_before)
                span.set_attribute("harness.compaction.tokens_after", compaction_result.tokens_after)
                span.set_attribute("harness.compaction.tokens_freed", compaction_result.tokens_freed)
                span.set_attribute("harness.compaction.duration_ms", round(compaction_result.duration_ms, 1))
                span.set_attribute("harness.compaction.strategies", compaction_result.strategies_applied)
                self._logging_service.log_compaction_executed(
                    compaction_count=self._get_compaction_count(),
                    strategies_applied=compaction_result.strategies_applied,
                )

        agent_call_start = time.monotonic()
        self._logging_service.log_agent_call_start(
            turn_count=turn_count,
            cache_size=len(self._turn_buffer.load_messages()),
            streaming=prepared_context.streaming,
        )
        try:
            with _tracer.start_as_current_span(
                "harness.agent_call",
                attributes={
                    "harness.turn": turn_count,
                    "harness.cache_size": len(self._turn_buffer.load_messages()),
                    "harness.streaming": prepared_context.streaming,
                },
            ) as agent_span:
                invocation = await self._invocation_service.invoke(
                    ctx,
                    prepared_context=prepared_context,
                )
                if invocation.response and invocation.response.usage_details:
                    agent_span.set_attribute("harness.input_tokens", invocation.input_tokens)
                    agent_span.set_attribute("harness.output_tokens", invocation.output_tokens)

            agent_call_elapsed = (time.monotonic() - agent_call_start) * 1000
            self._logging_service.log_agent_call_completed(
                turn_count=turn_count,
                elapsed_ms=agent_call_elapsed,
                input_tokens=invocation.input_tokens,
                output_tokens=invocation.output_tokens,
            )

            if invocation.response is None:
                self._logging_service.log_agent_awaiting_user_input(turn_count)
                await self._completion_signal_service.send_completion(
                    ctx,
                    agent_done=False,
                    called_work_complete=False,
                )
                return

            postprocess = await self._postprocess_service.finalize_turn(
                ctx,
                response=invocation.response,
                turn_count=turn_count,
                max_turns=max_turns,
                owner_mode=prepared_context.owner_mode,
            )
            self._logging_service.log_turn_outcome(
                turn_count=turn_count,
                called_work_complete=postprocess.called_work_complete,
                has_tool_calls=postprocess.has_tool_calls,
                continuation_prompt_sent=postprocess.continuation_prompt_sent,
            )
            await self._completion_signal_service.send_completion(
                ctx,
                agent_done=postprocess.agent_done,
                called_work_complete=postprocess.called_work_complete,
            )
        except Exception as e:
            self._logging_service.log_turn_failed(turn_count=turn_count, error=e)
            await self._error_handling_service.handle_turn_error(ctx, turn_count=turn_count, error=e)


class TurnLoggingService:
    """Centralizes turn-level logging semantics for orchestration flow."""

    def log_turn_start(self, turn_count: int) -> None:
        logger.info("AgentTurnExecutor: Starting turn %d", turn_count)

    def log_shared_buffer_adopted(self, *, version: int, message_count: int) -> None:
        logger.debug(
            "AgentTurnExecutor: Adopted shared turn buffer snapshot (version=%d, messages=%d)",
            version,
            message_count,
        )

    def log_compaction_executed(self, *, compaction_count: int, strategies_applied: list[str]) -> None:
        logger.info(
            "AgentTurnExecutor: Compaction executed (count=%d, strategies=%s)",
            compaction_count,
            strategies_applied,
        )

    def log_agent_call_start(self, *, turn_count: int, cache_size: int, streaming: bool) -> None:
        logger.info(
            "AgentTurnExecutor: Turn %d - Starting agent call (cache_size=%d, streaming=%s)",
            turn_count,
            cache_size,
            streaming,
        )

    def log_agent_call_completed(self, *, turn_count: int, elapsed_ms: float, input_tokens: int, output_tokens: int) -> None:
        logger.info(
            "AgentTurnExecutor: Turn %d - Agent call completed in %.1fms (input=%d, output=%d tokens)",
            turn_count,
            elapsed_ms,
            input_tokens,
            output_tokens,
        )

    def log_agent_awaiting_user_input(self, turn_count: int) -> None:
        logger.info("AgentTurnExecutor: Turn %d - Agent awaiting user input", turn_count)

    def log_turn_outcome(
        self,
        *,
        turn_count: int,
        called_work_complete: bool,
        has_tool_calls: bool,
        continuation_prompt_sent: bool,
    ) -> None:
        if called_work_complete:
            logger.info("AgentTurnExecutor: Turn %d complete, agent called work_complete", turn_count)
        elif has_tool_calls:
            logger.info("AgentTurnExecutor: Turn %d complete, agent making tool calls", turn_count)
        elif continuation_prompt_sent:
            logger.info("AgentTurnExecutor: Turn %d - agent stopped, sending continuation prompt", turn_count)
        else:
            logger.info("AgentTurnExecutor: Turn %d complete, agent_done=True", turn_count)

    def log_turn_failed(self, *, turn_count: int, error: Exception) -> None:
        logger.error("AgentTurnExecutor: Turn %d failed with error: %s", turn_count, error)
