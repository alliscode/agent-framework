# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any

import pytest

from agent_framework import AgentRunResponse, AgentRunResponseUpdate, ChatMessage, FunctionCallContent, FunctionResultContent, UsageDetails
from agent_framework._harness._constants import (
    HARNESS_COMPACTION_PLAN_KEY,
    HARNESS_COMPACTION_OWNER_MODE_KEY,
    HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY,
    HARNESS_CONTINUATION_COUNT_KEY,
    HARNESS_INITIAL_MESSAGE_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_SHARED_TURN_BUFFER_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
    HARNESS_WORK_ITEM_LEDGER_KEY,
)
from agent_framework._harness._state import HarnessEvent, HarnessLifecycleEvent, RepairComplete
from agent_framework._harness._state import TurnComplete
from agent_framework._harness._agent_turn_executor import AgentTurnExecutor
from agent_framework._harness._done_tool import work_complete
from agent_framework._harness._turn_services import (
    TurnBudgetSyncService,
    TurnBufferSyncService,
    TurnCompactionService,
    TurnCompactionTelemetryService,
    TurnCompactionViewService,
    TurnCompletionSignalService,
    TurnContinuationService,
    TurnErrorHandlingService,
    TurnEventWriter,
    TurnGuidanceService,
    TurnInitialMessageService,
    TurnInvocationService,
    TurnJitInstructionService,
    TurnOutcomeEvaluator,
    TurnOwnerModeService,
    PreparedTurnContext,
    TurnPostprocessService,
    TurnPreambleService,
    TurnPromptAssembler,
    TurnCheckpointService,
    TurnSystemMessageService,
    TurnTokenBudgetService,
    TurnToolingService,
    TurnWorkItemPromptService,
    TurnWorkItemStateService,
)
from agent_framework._workflows._events import AgentRunEvent, AgentRunUpdateEvent
from agent_framework._harness._turn_buffer import ExecutorLocalTurnBuffer, SharedStateTurnBuffer
from agent_framework._harness._compaction_view import (
    apply_compaction_plan_to_messages,
    apply_direct_clear_to_messages,
    ensure_message_ids,
)
from agent_framework._harness._compaction._tokenizer import get_tokenizer
from agent_framework._harness._compaction import CompactionPlan, DropRecord, SpanReference, TokenBudget
from agent_framework._harness._compaction_owner_mode import (
    is_valid_compaction_owner_mode,
    normalize_compaction_owner_mode,
)
from agent_framework._harness._compaction_telemetry import (
    OwnerCompactionOutcome,
    compaction_started_payload,
    context_pressure_payload,
    pressure_metrics_payload,
)
from agent_framework._harness._work_items import ArtifactRole, WorkItem, WorkItemLedger, WorkItemStatus
from agent_framework._harness._work_items import WorkItemTaskList
from agent_framework._workflows._const import WORKFLOW_RUN_KWARGS_KEY


class _FakeWorkflowContext:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self._state = dict(state or {})
        self.events: list[Any] = []
        self.messages: list[Any] = []

    async def get_shared_state(self, key: str) -> Any:
        if key not in self._state:
            raise KeyError(key)
        return self._state[key]

    async def set_shared_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    async def add_event(self, event: Any) -> None:
        self.events.append(event)

    async def send_message(self, message: Any) -> None:
        self.messages.append(message)


class _DummyTaskList:
    def __init__(self, tools: list[Any]) -> None:
        self._tools = tools

    def get_tools(self) -> list[Any]:
        return list(self._tools)


class _DummyWorkItemStateService:
    def __init__(self) -> None:
        self.emit_calls = 0
        self.sync_calls = 0

    async def emit_work_item_change_events(self, ctx: Any) -> None:
        self.emit_calls += 1

    async def sync_ledger(self, ctx: Any) -> None:
        self.sync_calls += 1


class _DummyAgent:
    async def run(self, messages: list[Any], **_: Any) -> AgentRunResponse:
        assert len(messages) == 1
        return AgentRunResponse(messages=[ChatMessage(role="assistant", text="ok")])

    async def run_stream(self, messages: list[Any], **_: Any):  # type: ignore[no-untyped-def]
        assert len(messages) == 1
        yield AgentRunResponseUpdate(text="hello", role="assistant")


class _DummyJitProcessor:
    def __init__(self, instructions: list[str]) -> None:
        self._instructions = instructions

    def evaluate(self, _ctx: Any) -> list[str]:
        return list(self._instructions)


class _InjectedCheckpointService:
    def __init__(self) -> None:
        self.saved = False
        self.restored: dict[str, Any] | None = None
        self.set_messages: list[Any] | None = None

    async def save(self) -> dict[str, Any]:
        self.saved = True
        return {"cache": [{"type": "chat_message", "role": "user", "contents": []}]}

    async def restore(self, state: dict[str, Any]) -> None:
        self.restored = state

    def set_initial_messages(self, messages: list[Any]) -> None:
        self.set_messages = list(messages)


class _InjectedInitialMessageService:
    def __init__(self) -> None:
        self.called = False

    async def get_initial_message(self, ctx: Any) -> Any:
        self.called = True
        return ChatMessage(role="user", text="injected-initial")


class _InjectedGuidanceService:
    def __init__(self) -> None:
        self.work_item_calls = 0
        self.tool_strategy_calls = 0
        self.planning_calls = 0

    def inject_work_item_guidance(self) -> None:
        self.work_item_calls += 1

    def inject_tool_strategy_guidance(self) -> None:
        self.tool_strategy_calls += 1

    def inject_planning_prompt(self) -> None:
        self.planning_calls += 1


class _InjectedJitInstructionService:
    def __init__(self) -> None:
        self.calls = 0

    async def inject_instructions(
        self,
        ctx: Any,
        *,
        turn_count: int,
        compaction_count: int,
        tool_usage: dict[str, int],
    ) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_turn_tooling_service_builds_kwargs_and_filters_thread() -> None:
    ctx = _FakeWorkflowContext(
        {
            WORKFLOW_RUN_KWARGS_KEY: {
                "thread": object(),
                "temperature": 0.1,
                "tools": ["base_tool"],
            }
        }
    )
    tooling = TurnToolingService(
        task_list=None,
        sub_agent_tools=["sub_tool"],
        work_item_middleware="mw_work_items",
        harness_tool_middleware="mw_hooks",
    )

    run_kwargs = await tooling.build_run_kwargs(ctx)  # type: ignore[arg-type]

    assert "thread" not in run_kwargs
    assert run_kwargs["temperature"] == 0.1
    assert work_complete in run_kwargs["tools"]
    assert "sub_tool" in run_kwargs["tools"]
    assert "base_tool" in run_kwargs["tools"]
    assert run_kwargs["middleware"] == ["mw_work_items", "mw_hooks"]


@pytest.mark.asyncio
async def test_turn_tooling_service_prefers_task_list_tools() -> None:
    ctx = _FakeWorkflowContext({WORKFLOW_RUN_KWARGS_KEY: {"tools": ["ignored_base"]}})
    tooling = TurnToolingService(
        task_list=_DummyTaskList(["task_tool"]),
        sub_agent_tools=[],
        work_item_middleware=None,
        harness_tool_middleware=None,
    )

    run_kwargs = await tooling.build_run_kwargs(ctx)  # type: ignore[arg-type]

    assert "task_tool" in run_kwargs["tools"]
    assert work_complete in run_kwargs["tools"]
    assert "ignored_base" not in run_kwargs["tools"]


@pytest.mark.asyncio
async def test_turn_initial_message_service_normalizes_string_and_passthrough() -> None:
    service = TurnInitialMessageService()
    ctx_str = _FakeWorkflowContext({HARNESS_INITIAL_MESSAGE_KEY: "hello"})
    msg = await service.get_initial_message(ctx_str)  # type: ignore[arg-type]
    assert msg is not None
    assert msg.role.value == "user"
    assert msg.text == "hello"

    existing = ChatMessage(role="user", text="existing")
    ctx_obj = _FakeWorkflowContext({HARNESS_INITIAL_MESSAGE_KEY: existing})
    same = await service.get_initial_message(ctx_obj)  # type: ignore[arg-type]
    assert same is existing


@pytest.mark.asyncio
async def test_turn_budget_sync_service_updates_and_reads_estimate() -> None:
    ctx = _FakeWorkflowContext()
    token_service = TurnTokenBudgetService(tokenizer=get_tokenizer())
    buffer = ExecutorLocalTurnBuffer([ChatMessage(role="user", text="count me")])
    sync = TurnBudgetSyncService(token_budget_service=token_service, turn_buffer=buffer)

    await sync.sync(ctx)  # type: ignore[arg-type]
    estimate = await token_service.get_budget_estimate(ctx)  # type: ignore[arg-type]
    assert estimate > 0


@pytest.mark.asyncio
async def test_turn_jit_instruction_service_injects_instructions() -> None:
    ctx = _FakeWorkflowContext()
    buffer = ExecutorLocalTurnBuffer()
    jit = TurnJitInstructionService(
        jit_processor=_DummyJitProcessor(["first", "second"]),
        turn_buffer=buffer,
    )

    await jit.inject_instructions(
        ctx,  # type: ignore[arg-type]
        turn_count=2,
        compaction_count=1,
        tool_usage={"read_file": 3},
    )

    assert len(buffer.load_messages()) == 2
    assert buffer.load_messages()[0].text == "first"
    assert buffer.load_messages()[1].text == "second"


@pytest.mark.asyncio
async def test_agent_turn_executor_uses_injected_checkpoint_service() -> None:
    injected = _InjectedCheckpointService()
    executor = AgentTurnExecutor(
        _DummyAgent(),  # type: ignore[arg-type]
        checkpoint_service=injected,  # type: ignore[arg-type]
    )

    saved = await executor.on_checkpoint_save()
    await executor.on_checkpoint_restore({"cache": []})
    executor.set_initial_messages([ChatMessage(role="user", text="seed")])

    assert injected.saved is True
    assert saved.get("cache") is not None
    assert injected.restored == {"cache": []}
    assert injected.set_messages is not None
    assert len(injected.set_messages) == 1


@pytest.mark.asyncio
async def test_agent_turn_executor_uses_injected_initial_guidance_and_jit_services() -> None:
    initial = _InjectedInitialMessageService()
    guidance = _InjectedGuidanceService()
    jit = _InjectedJitInstructionService()
    executor = AgentTurnExecutor(
        _DummyAgent(),  # type: ignore[arg-type]
        task_list=WorkItemTaskList(),
        initial_message_service=initial,  # type: ignore[arg-type]
        guidance_service=guidance,  # type: ignore[arg-type]
        jit_instruction_service=jit,  # type: ignore[arg-type]
    )
    ctx = _FakeWorkflowContext()

    await executor._prompt_assembler.prepare_turn(ctx, turn_count=1)  # type: ignore[arg-type]

    assert initial.called is True
    assert guidance.work_item_calls == 1
    assert guidance.tool_strategy_calls == 1
    assert guidance.planning_calls == 1
    assert jit.calls == 1


def test_turn_guidance_service_injects_three_guidance_messages() -> None:
    buffer = ExecutorLocalTurnBuffer()
    service = TurnGuidanceService(turn_buffer=buffer)

    service.inject_work_item_guidance()
    service.inject_tool_strategy_guidance()
    service.inject_planning_prompt()

    messages = buffer.load_messages()
    assert len(messages) == 3
    assert "work item tracking tools" in messages[0].text
    assert "TOOL STRATEGY GUIDE" in messages[1].text
    assert "Assess the user's request before taking action" in messages[2].text


def test_turn_system_message_service_appends_system_role_message() -> None:
    buffer = ExecutorLocalTurnBuffer()
    service = TurnSystemMessageService(turn_buffer=buffer)
    service.append_system_message("hello system")

    messages = buffer.load_messages()
    assert len(messages) == 1
    assert messages[0].role.value == "system"
    assert messages[0].text == "hello system"


@pytest.mark.asyncio
async def test_turn_checkpoint_service_round_trip_and_fallback() -> None:
    buffer = ExecutorLocalTurnBuffer([ChatMessage(role="user", text="seed")])
    service = TurnCheckpointService(turn_buffer=buffer)

    saved = await service.save()
    assert "cache" in saved

    buffer.replace_messages([])
    await service.restore(saved)
    assert len(buffer.load_messages()) == 1
    assert buffer.load_messages()[0].text == "seed"

    await service.restore({"cache": "bad-payload"})  # type: ignore[arg-type]
    assert buffer.load_messages() == []


@pytest.mark.asyncio
async def test_turn_preamble_service_increments_turn_and_emits_start_events() -> None:
    ctx = _FakeWorkflowContext(
        {
            HARNESS_TURN_COUNT_KEY: 2,
            HARNESS_MAX_TURNS_KEY: 15,
        }
    )
    writer = TurnEventWriter()
    service = TurnPreambleService(event_writer=writer)

    preamble = await service.begin_turn(
        ctx,  # type: ignore[arg-type]
        repairs_made=1,
    )

    assert preamble.turn_count == 3
    assert preamble.max_turns == 15
    assert await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) == 3
    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert any(e.get("event_type") == "turn_start" for e in transcript)
    assert any(isinstance(e, HarnessLifecycleEvent) and e.event_type == "turn_started" for e in ctx.events)


@pytest.mark.asyncio
async def test_turn_preamble_service_uses_default_max_turns_when_missing() -> None:
    ctx = _FakeWorkflowContext({HARNESS_TURN_COUNT_KEY: 0})
    writer = TurnEventWriter()
    service = TurnPreambleService(event_writer=writer)

    preamble = await service.begin_turn(
        ctx,  # type: ignore[arg-type]
        repairs_made=0,
    )

    assert preamble.turn_count == 1
    assert preamble.max_turns == 50


@pytest.mark.asyncio
async def test_turn_continuation_service_reads_count_and_injects_prompt() -> None:
    ledger = WorkItemLedger()
    ledger.add_item(
        WorkItem(
            id="w1",
            title="Open task",
            status=WorkItemStatus.IN_PROGRESS,
        )
    )
    task_list = type("TaskList", (), {"ledger": ledger})()
    injected: list[str] = []
    ctx = _FakeWorkflowContext({HARNESS_CONTINUATION_COUNT_KEY: 1})
    service = TurnContinuationService(
        continuation_prompt="continue now",
        max_continuation_prompts=5,
        task_list=task_list,  # type: ignore[arg-type]
        event_writer=TurnEventWriter(),
        append_system_message=injected.append,
    )

    count = await service.get_continuation_count(ctx)  # type: ignore[arg-type]
    assert count == 1

    await service.inject_continuation_prompt(
        ctx,  # type: ignore[arg-type]
        current_count=count,
    )
    assert await ctx.get_shared_state(HARNESS_CONTINUATION_COUNT_KEY) == 2
    assert len(injected) == 1
    assert "continue now" in injected[0]
    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert any(e.get("event_type") == "continuation_prompt" for e in transcript)
    assert any(isinstance(e, HarnessLifecycleEvent) and e.event_type == "continuation_prompt" for e in ctx.events)


@pytest.mark.asyncio
async def test_turn_completion_signal_service_sends_turn_complete_message() -> None:
    ctx = _FakeWorkflowContext()
    service = TurnCompletionSignalService()

    await service.send_completion(
        ctx,  # type: ignore[arg-type]
        agent_done=True,
        called_work_complete=True,
        error=None,
    )

    assert len(ctx.messages) == 1
    assert isinstance(ctx.messages[0], TurnComplete)
    assert ctx.messages[0].agent_done is True
    assert ctx.messages[0].called_work_complete is True


@pytest.mark.asyncio
async def test_turn_error_handling_service_appends_error_and_sends_completion() -> None:
    ctx = _FakeWorkflowContext()
    event_writer = TurnEventWriter()
    completion = TurnCompletionSignalService()
    service = TurnErrorHandlingService(
        event_writer=event_writer,
        completion_signal_service=completion,
    )

    await service.handle_turn_error(
        ctx,  # type: ignore[arg-type]
        turn_count=7,
        error=RuntimeError("boom"),
    )

    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert any(
        e.get("event_type") == "agent_response" and e.get("data", {}).get("error") == "boom"
        for e in transcript
    )
    assert len(ctx.messages) == 1
    assert isinstance(ctx.messages[0], TurnComplete)
    assert ctx.messages[0].agent_done is False
    assert ctx.messages[0].error == "boom"


@pytest.mark.asyncio
async def test_turn_invocation_service_non_streaming_appends_response_and_event() -> None:
    ctx = _FakeWorkflowContext()
    tooling = TurnToolingService(
        task_list=None,
        sub_agent_tools=[],
        work_item_middleware=None,
        harness_tool_middleware=None,
    )
    work_item_state = _DummyWorkItemStateService()
    buffer = ExecutorLocalTurnBuffer()
    requested_modes: list[str] = []

    async def _messages_for_agent(_ctx: Any, mode: str) -> list[Any]:
        requested_modes.append(mode)
        return [ChatMessage(role="user", text="go")]

    invocation = TurnInvocationService(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        executor_id="turn_exec",
        tooling_service=tooling,
        work_item_state_service=work_item_state,  # type: ignore[arg-type]
        turn_buffer=buffer,
        get_messages_for_agent=_messages_for_agent,
    )

    result = await invocation.invoke(
        ctx,  # type: ignore[arg-type]
        prepared_context=PreparedTurnContext(
            turn_count=1,
            max_turns=10,
            owner_mode="agent_turn",
            streaming=False,
            cache_size=0,
        ),
    )

    assert result.response is not None
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert requested_modes == ["agent_turn"]
    assert work_item_state.emit_calls == 1
    assert len(buffer.load_messages()) == 1
    assert buffer.load_messages()[0].text == "ok"
    assert any(isinstance(e, AgentRunEvent) for e in ctx.events)


@pytest.mark.asyncio
async def test_turn_invocation_service_streaming_emits_updates_and_appends_response() -> None:
    ctx = _FakeWorkflowContext()
    tooling = TurnToolingService(
        task_list=None,
        sub_agent_tools=[],
        work_item_middleware=None,
        harness_tool_middleware=None,
    )
    work_item_state = _DummyWorkItemStateService()
    buffer = ExecutorLocalTurnBuffer()
    invocation = TurnInvocationService(
        agent=_DummyAgent(),  # type: ignore[arg-type]
        executor_id="turn_exec",
        tooling_service=tooling,
        work_item_state_service=work_item_state,  # type: ignore[arg-type]
        turn_buffer=buffer,
        get_messages_for_agent=lambda _ctx, _mode: _async_value([ChatMessage(role="user", text="go")]),
    )

    result = await invocation.invoke(
        ctx,  # type: ignore[arg-type]
        prepared_context=PreparedTurnContext(
            turn_count=1,
            max_turns=10,
            owner_mode="shadow",
            streaming=True,
            cache_size=0,
        ),
    )

    assert result.response is not None
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert work_item_state.emit_calls == 1
    assert len(buffer.load_messages()) == 1
    assert buffer.load_messages()[0].text == "hello"
    assert any(isinstance(e, AgentRunUpdateEvent) for e in ctx.events)


@pytest.mark.asyncio
async def test_turn_compaction_view_service_applies_plan_for_agent_turn_mode() -> None:
    dropped = ChatMessage(role="user", text="drop me", message_id="u1")
    kept = ChatMessage(role="assistant", text="keep me", message_id="a1")
    plan = CompactionPlan.create_empty(thread_id="h")
    plan.drops.append(
        DropRecord(span=SpanReference(message_ids=["u1"], first_turn=1, last_turn=1), reason="test-drop")
    )
    ctx = _FakeWorkflowContext({HARNESS_COMPACTION_PLAN_KEY: plan.to_dict()})
    buffer = ExecutorLocalTurnBuffer([dropped, kept])
    service = TurnCompactionViewService(enable_compaction=True, turn_buffer=buffer)

    result = await service.get_messages_for_agent(
        ctx,  # type: ignore[arg-type]
        owner_mode="agent_turn",
    )

    assert len(result) == 1
    assert result[0].message_id == "a1"


@pytest.mark.asyncio
async def test_turn_compaction_view_service_skips_plan_in_owner_mode() -> None:
    dropped = ChatMessage(role="user", text="drop me", message_id="u1")
    plan = CompactionPlan.create_empty(thread_id="h")
    plan.drops.append(
        DropRecord(span=SpanReference(message_ids=["u1"], first_turn=1, last_turn=1), reason="test-drop")
    )
    ctx = _FakeWorkflowContext({HARNESS_COMPACTION_PLAN_KEY: plan.to_dict()})
    buffer = ExecutorLocalTurnBuffer([dropped])
    service = TurnCompactionViewService(enable_compaction=True, turn_buffer=buffer)

    result = await service.get_messages_for_agent(
        ctx,  # type: ignore[arg-type]
        owner_mode="compaction_executor",
    )

    assert len(result) == 1
    assert result[0].message_id == "u1"


@pytest.mark.asyncio
async def test_turn_compaction_telemetry_service_emits_completed_and_shadow_compare() -> None:
    ctx = _FakeWorkflowContext(
        {
            HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY: {
                "would_compact": True,
                "blocking": False,
                "candidate_effective_compaction": False,
            },
            HARNESS_SHARED_TURN_BUFFER_KEY: {
                "version": 3,
                "message_count": 2,
                "messages": [],
            },
        }
    )
    telemetry = TurnCompactionTelemetryService()

    await telemetry.emit_compaction_completed(
        ctx=ctx,  # type: ignore[arg-type]
        turn_count=4,
        owner_mode="agent_turn",
        strategies_applied=["clear"],
        tokens_before=1000,
        tokens_after=600,
        tokens_freed=400,
        duration_ms=10.0,
        compaction_level="optimized",
    )
    await telemetry.emit_shadow_compare(
        ctx=ctx,  # type: ignore[arg-type]
        owner_mode="shadow",
        turn_count=4,
        actual_attempted=True,
        actual_effective=True,
        strategies_applied=["clear"],
        tokens_before=1000,
        tokens_after=600,
        tokens_freed=400,
        local_message_count=2,
        local_buffer_version=3,
    )

    completed = [e for e in ctx.events if isinstance(e, HarnessLifecycleEvent) and e.event_type == "compaction_completed"]
    pressure = [e for e in ctx.events if isinstance(e, HarnessLifecycleEvent) and e.event_type == "context_pressure"]
    assert len(completed) == 1
    assert completed[0].data["tokens_freed"] == 400
    assert len(pressure) == 1
    assert pressure[0].data["shadow_compare"]["buffer_parity"]["match"] is True


@pytest.mark.asyncio
async def test_turn_postprocess_service_finalizes_turn_and_emits_lifecycle() -> None:
    ctx = _FakeWorkflowContext()
    event_writer = TurnEventWriter()
    work_item_state = _DummyWorkItemStateService()
    outcome = TurnOutcomeEvaluator(
        enable_continuation_prompts=False,
        max_continuation_prompts=1,
        get_continuation_count=lambda _: _async_value(0),
        inject_continuation_prompt=lambda *_: _async_none(),
    )
    local_buffer = ExecutorLocalTurnBuffer([ChatMessage(role="assistant", text="cached")])
    shared_buffer = SharedStateTurnBuffer(key=HARNESS_SHARED_TURN_BUFFER_KEY)
    sync_service = TurnBufferSyncService(shared_turn_buffer=shared_buffer)

    async def _update_budget(_ctx: Any, _response: AgentRunResponse | None) -> None:
        return None

    async def _budget(_ctx: Any) -> int:
        return 321

    service = TurnPostprocessService(
        event_writer=event_writer,
        outcome_evaluator=outcome,
        work_item_state_service=work_item_state,  # type: ignore[arg-type]
        task_list_enabled=True,
        update_token_budget=_update_budget,
        get_budget_estimate=_budget,
        buffer_sync_service=sync_service,
        turn_buffer=local_buffer,
    )

    response = AgentRunResponse(messages=[ChatMessage(role="assistant", text="done")])
    result = await service.finalize_turn(
        ctx,  # type: ignore[arg-type]
        response=response,
        turn_count=2,
        max_turns=10,
        owner_mode="compaction_executor",
    )

    assert result.agent_done is True
    assert result.has_tool_calls is False
    assert result.called_work_complete is False
    assert result.token_estimate == 321
    assert result.cache_size == 1
    assert work_item_state.sync_calls == 1
    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert any(e.get("event_type") == "agent_response" for e in transcript)
    assert any(isinstance(e, HarnessLifecycleEvent) and e.event_type == "turn_completed" for e in ctx.events)
    shared_info = await shared_buffer.read_snapshot_info(ctx)  # type: ignore[arg-type]
    assert shared_info["present"] is True


@pytest.mark.asyncio
async def test_turn_token_budget_service_updates_budget_estimate() -> None:
    ctx = _FakeWorkflowContext()
    service = TurnTokenBudgetService(tokenizer=get_tokenizer())
    cache = [
        ChatMessage(role="user", text="Analyze this request in detail."),
        ChatMessage(role="assistant", text="I will inspect files and report back."),
    ]
    response = AgentRunResponse(
        messages=[],
        usage_details=UsageDetails(input_token_count=1200, output_token_count=200, total_token_count=1400),
    )

    await service.update_token_budget(ctx, cache=cache, response=response)  # type: ignore[arg-type]
    saved_budget = await ctx.get_shared_state(HARNESS_TOKEN_BUDGET_KEY)

    assert isinstance(saved_budget, dict)
    assert saved_budget.get("current_estimate", 0) > 0
    assert saved_budget.get("system_prompt_tokens", 0) >= 0


@pytest.mark.asyncio
async def test_turn_prompt_assembler_sequence_first_turn() -> None:
    call_order: list[str] = []
    cache: list[Any] = []
    ctx = _FakeWorkflowContext()

    async def _get_initial_message(_: Any) -> Any:
        call_order.append("get_initial_message")
        return ChatMessage(role="user", text="hello")

    def _append_to_cache(msg: Any) -> None:
        call_order.append("append_to_cache")
        cache.append(msg)

    def _inject_work_item_guidance() -> None:
        call_order.append("inject_work_item_guidance")

    def _inject_tool_strategy_guidance() -> None:
        call_order.append("inject_tool_strategy_guidance")

    def _inject_planning_prompt() -> None:
        call_order.append("inject_planning_prompt")

    async def _inject_work_item_state(_: Any) -> None:
        call_order.append("inject_work_item_state")

    async def _maybe_inject_work_item_reminder(_: Any) -> None:
        call_order.append("maybe_inject_work_item_reminder")

    async def _inject_jit_instructions(_: Any, __: int) -> None:
        call_order.append("inject_jit_instructions")

    assembler = TurnPromptAssembler(
        task_list_enabled=True,
        get_initial_message=_get_initial_message,
        append_to_cache=_append_to_cache,
        inject_work_item_guidance=_inject_work_item_guidance,
        inject_tool_strategy_guidance=_inject_tool_strategy_guidance,
        inject_planning_prompt=_inject_planning_prompt,
        inject_work_item_state=_inject_work_item_state,
        maybe_inject_work_item_reminder=_maybe_inject_work_item_reminder,
        inject_jit_instructions=_inject_jit_instructions,
    )

    await assembler.prepare_turn(ctx, turn_count=1)  # type: ignore[arg-type]

    assert len(cache) == 1
    assert call_order == [
        "get_initial_message",
        "append_to_cache",
        "inject_work_item_guidance",
        "inject_tool_strategy_guidance",
        "inject_planning_prompt",
        "maybe_inject_work_item_reminder",
        "inject_jit_instructions",
    ]


@pytest.mark.asyncio
async def test_turn_compaction_service_emits_completion_event() -> None:
    ctx = _FakeWorkflowContext()
    cache = [ChatMessage(role="assistant", text="x")]
    counts = {"inc": 0}

    async def _get_budget_estimate(_: Any) -> int:
        # First call (before compaction), then second call (after budget update)
        return 1000 if not getattr(_get_budget_estimate, "_called", False) else 600

    setattr(_get_budget_estimate, "_called", False)

    async def _run_full_compaction(_: Any, __: int) -> list[str]:
        setattr(_get_budget_estimate, "_called", True)
        return ["clear"]

    def _apply_direct_clear(_: int, __: int, ___: int) -> int:
        return 0

    async def _update_token_budget(_: Any) -> None:
        return None

    service = TurnCompactionService(
        enable_compaction=True,
        is_compaction_needed=lambda _: True,
        ensure_message_ids=lambda __: None,
        get_budget_estimate=_get_budget_estimate,
        run_full_compaction=_run_full_compaction,
        apply_direct_clear=_apply_direct_clear,
        update_token_budget=_update_token_budget,
        classify_compaction_level=lambda *_: "optimized",
        increment_compaction_count=lambda: counts.__setitem__("inc", counts["inc"] + 1),
    )

    result = await service.maybe_compact(
        trigger=RepairComplete(repairs_made=0),
        ctx=ctx,  # type: ignore[arg-type]
        turn_count=2,
        cache=cache,
    )

    assert result is not None
    assert result.tokens_before == 1000
    assert result.tokens_after == 600
    assert result.tokens_freed == 400
    assert counts["inc"] == 1
    assert len(ctx.events) == 1
    assert isinstance(ctx.events[0], HarnessLifecycleEvent)
    assert ctx.events[0].event_type == "compaction_completed"


@pytest.mark.asyncio
async def test_turn_event_writer_writes_transcript_and_lifecycle() -> None:
    ctx = _FakeWorkflowContext()
    writer = TurnEventWriter()

    await writer.append_transcript_event(
        ctx,  # type: ignore[arg-type]
        event=HarnessEvent(event_type="turn_start", data={}),
    )
    await writer.emit_lifecycle_event(
        ctx,  # type: ignore[arg-type]
        event_type="turn_started",
        turn_number=1,
        max_turns=10,
    )

    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert isinstance(transcript, list)
    assert len(transcript) == 1
    assert transcript[0]["event_type"] == "turn_start"
    assert len(ctx.events) == 1
    assert isinstance(ctx.events[0], HarnessLifecycleEvent)
    assert ctx.events[0].event_type == "turn_started"


@pytest.mark.asyncio
async def test_turn_outcome_evaluator_prefers_work_complete() -> None:
    calls = {"inject": 0}

    async def _get_count(_: Any) -> int:
        return 0

    async def _inject(_: Any, __: int) -> None:
        calls["inject"] += 1

    evaluator = TurnOutcomeEvaluator(
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        get_continuation_count=_get_count,
        inject_continuation_prompt=_inject,
    )

    response = AgentRunResponse(
        messages=[
            ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(call_id="c1", name="work_complete", arguments={"summary": "done"})
                ],
            )
        ]
    )
    outcome = await evaluator.evaluate(response, _FakeWorkflowContext())  # type: ignore[arg-type]

    assert outcome.called_work_complete is True
    assert outcome.agent_done is True
    assert calls["inject"] == 0


@pytest.mark.asyncio
async def test_turn_outcome_evaluator_requests_continuation_when_idle() -> None:
    calls = {"inject": 0}

    async def _get_count(_: Any) -> int:
        return 0

    async def _inject(_: Any, __: int) -> None:
        calls["inject"] += 1

    evaluator = TurnOutcomeEvaluator(
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        get_continuation_count=_get_count,
        inject_continuation_prompt=_inject,
    )

    response = AgentRunResponse(messages=[ChatMessage(role="assistant", text="done?")])
    outcome = await evaluator.evaluate(response, _FakeWorkflowContext())  # type: ignore[arg-type]

    assert outcome.agent_done is False
    assert outcome.continuation_prompt_sent is True
    assert calls["inject"] == 1


@pytest.mark.asyncio
async def test_turn_work_item_state_service_syncs_and_emits_deliverables() -> None:
    ctx = _FakeWorkflowContext()
    writer = TurnEventWriter()
    injected_messages: list[str] = []
    ledger = WorkItemLedger()
    ledger.add_item(
        WorkItem(
            id="item1",
            title="Deliverable item",
            status=WorkItemStatus.DONE,
            artifact_role=ArtifactRole.DELIVERABLE,
            artifact="final output",
        )
    )
    task_list = type("TaskList", (), {"ledger": ledger})()

    service = TurnWorkItemStateService(
        task_list=task_list,
        work_item_middleware=None,
        event_writer=writer,
        append_system_message=injected_messages.append,
    )
    await service.sync_ledger(ctx)  # type: ignore[arg-type]

    saved = await ctx.get_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY)
    assert isinstance(saved, dict)
    assert len(ctx.events) == 1
    assert isinstance(ctx.events[0], HarnessLifecycleEvent)
    assert ctx.events[0].event_type == "deliverables_updated"
    assert injected_messages == []


@pytest.mark.asyncio
async def test_turn_work_item_state_service_invariant_violation_injects_message() -> None:
    ctx = _FakeWorkflowContext()
    writer = TurnEventWriter()
    injected_messages: list[str] = []
    control_artifact = '{"verdict":"fail","checks":[{"name":"schema","result":"fail","detail":"bad"}],"summary":"x"}'
    ledger = WorkItemLedger()
    ledger.add_item(
        WorkItem(
            id="ctrl1",
            title="Control audit",
            status=WorkItemStatus.DONE,
            artifact_role=ArtifactRole.CONTROL,
            artifact=control_artifact,
        )
    )
    task_list = type("TaskList", (), {"ledger": ledger})()

    service = TurnWorkItemStateService(
        task_list=task_list,
        work_item_middleware=None,
        event_writer=writer,
        append_system_message=injected_messages.append,
    )
    await service.sync_ledger(ctx)  # type: ignore[arg-type]

    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert isinstance(transcript, list)
    assert any(e.get("event_type") == "control_invariant_violation" for e in transcript)
    assert len(injected_messages) == 1
    assert "fail" in injected_messages[0]


@pytest.mark.asyncio
async def test_turn_work_item_prompt_service_injects_state_with_tool_usage() -> None:
    ctx = _FakeWorkflowContext()
    ledger = WorkItemLedger()
    ledger.add_item(
        WorkItem(
            id="w1",
            title="Pending step",
            status=WorkItemStatus.IN_PROGRESS,
        )
    )
    await ctx.set_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY, ledger.to_dict())

    buffer = ExecutorLocalTurnBuffer(
        [
            ChatMessage(
                role="assistant",
                contents=[FunctionCallContent(call_id="c1", name="read_file", arguments={"path": "x.py"})],
            )
        ]
    )
    service = TurnWorkItemPromptService(turn_buffer=buffer, event_writer=TurnEventWriter())
    await service.inject_work_item_state(ctx)  # type: ignore[arg-type]

    messages = buffer.load_messages()
    assert len(messages) == 2
    assert "Work items (1 remaining)" in messages[-1].text
    assert "read_file: 1" in messages[-1].text


@pytest.mark.asyncio
async def test_turn_work_item_prompt_service_injects_reminder_from_stop_reason() -> None:
    ledger = WorkItemLedger()
    ledger.add_item(
        WorkItem(
            id="w1",
            title="Pending step",
            status=WorkItemStatus.IN_PROGRESS,
        )
    )
    ctx = _FakeWorkflowContext(
        {
            HARNESS_TRANSCRIPT_KEY: [
                {
                    "event_type": "stop_decision",
                    "data": {"reason": "work_items_incomplete"},
                }
            ],
            HARNESS_WORK_ITEM_LEDGER_KEY: ledger.to_dict(),
        }
    )

    buffer = ExecutorLocalTurnBuffer()
    service = TurnWorkItemPromptService(turn_buffer=buffer, event_writer=TurnEventWriter())
    await service.maybe_inject_work_item_reminder(ctx)  # type: ignore[arg-type]

    assert len(buffer.load_messages()) == 1
    assert "work item" in buffer.load_messages()[0].text.lower()
    transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
    assert any(e.get("event_type") == "work_item_reminder" for e in transcript)


@pytest.mark.asyncio
async def test_turn_compaction_service_shadow_compare_without_compaction_signal() -> None:
    ctx = _FakeWorkflowContext(
        {
            HARNESS_COMPACTION_OWNER_MODE_KEY: "shadow",
            HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY: {
                "would_compact": False,
                "blocking": False,
            },
            HARNESS_SHARED_TURN_BUFFER_KEY: {
                "version": 0,
                "message_count": 0,
                "messages": [],
            },
        }
    )

    service = TurnCompactionService(
        enable_compaction=True,
        is_compaction_needed=lambda _: False,
        ensure_message_ids=lambda __: None,
        get_budget_estimate=lambda _: _async_value(0),
        run_full_compaction=lambda *_: _async_value([]),
        apply_direct_clear=lambda *_: 0,
        update_token_budget=lambda _: _async_none(),
        classify_compaction_level=lambda *_: "optimized",
        increment_compaction_count=lambda: None,
    )

    result = await service.maybe_compact(
        trigger=RepairComplete(repairs_made=0),
        ctx=ctx,  # type: ignore[arg-type]
        turn_count=3,
        cache=[],
    )

    assert result is None
    shadow_events = [e for e in ctx.events if isinstance(e, HarnessLifecycleEvent) and e.event_type == "context_pressure"]
    assert len(shadow_events) == 1
    compare = shadow_events[0].data["shadow_compare"]
    assert compare["actual_attempted"] is False
    assert compare["actual_effective"] is False
    assert compare["diverged"] is False
    assert compare["buffer_parity"]["match"] is True


@pytest.mark.asyncio
async def test_turn_compaction_service_shadow_compare_detects_divergence() -> None:
    ctx = _FakeWorkflowContext(
        {
            HARNESS_COMPACTION_OWNER_MODE_KEY: "shadow",
            HARNESS_COMPACTION_SHADOW_CANDIDATE_KEY: {
                "would_compact": True,
                "blocking": True,
            },
            HARNESS_SHARED_TURN_BUFFER_KEY: {
                "version": 0,
                "message_count": 0,
                "messages": [],
            },
        }
    )

    async def _budget_estimate(_: Any) -> int:
        return 1000

    service = TurnCompactionService(
        enable_compaction=True,
        is_compaction_needed=lambda _: True,
        ensure_message_ids=lambda __: None,
        get_budget_estimate=_budget_estimate,
        run_full_compaction=lambda *_: _async_value([]),
        apply_direct_clear=lambda *_: 0,
        update_token_budget=lambda _: _async_none(),
        classify_compaction_level=lambda *_: "optimized",
        increment_compaction_count=lambda: None,
    )

    result = await service.maybe_compact(
        trigger=RepairComplete(repairs_made=0),
        ctx=ctx,  # type: ignore[arg-type]
        turn_count=4,
        cache=[],
    )

    assert result is not None
    shadow_events = [e for e in ctx.events if isinstance(e, HarnessLifecycleEvent) and e.event_type == "context_pressure"]
    assert len(shadow_events) == 1
    compare = shadow_events[0].data["shadow_compare"]
    assert compare["actual_attempted"] is True
    assert compare["actual_effective"] is False
    assert compare["candidate_would_compact"] is True
    assert compare["diverged"] is True
    assert compare["buffer_parity"]["match"] is True


async def _async_none() -> None:
    return None


async def _async_value(value: Any) -> Any:
    return value


def test_compaction_owner_mode_helpers_validate_and_normalize() -> None:
    assert is_valid_compaction_owner_mode("agent_turn") is True
    assert is_valid_compaction_owner_mode("compaction_executor") is True
    assert is_valid_compaction_owner_mode("shadow") is True
    assert is_valid_compaction_owner_mode("invalid") is False

    assert normalize_compaction_owner_mode("shadow") == "shadow"
    assert normalize_compaction_owner_mode("invalid") == "compaction_executor"
    assert normalize_compaction_owner_mode(None) == "compaction_executor"


@pytest.mark.asyncio
async def test_turn_owner_mode_service_reads_and_normalizes_shared_state() -> None:
    service = TurnOwnerModeService()

    valid_ctx = _FakeWorkflowContext({HARNESS_COMPACTION_OWNER_MODE_KEY: "shadow"})
    assert await service.get_owner_mode(valid_ctx) == "shadow"  # type: ignore[arg-type]

    invalid_ctx = _FakeWorkflowContext({HARNESS_COMPACTION_OWNER_MODE_KEY: "bad-mode"})
    assert await service.get_owner_mode(invalid_ctx) == "compaction_executor"  # type: ignore[arg-type]

    missing_ctx = _FakeWorkflowContext()
    assert await service.get_owner_mode(missing_ctx) == "compaction_executor"  # type: ignore[arg-type]


def test_executor_local_turn_buffer_mutations_and_version() -> None:
    buffer = ExecutorLocalTurnBuffer()
    assert buffer.snapshot_version() == 0
    assert buffer.load_messages() == []

    buffer.append_message(ChatMessage(role="user", text="hello"))
    assert len(buffer.load_messages()) == 1
    assert buffer.snapshot_version() == 1

    buffer.append_messages([ChatMessage(role="assistant", text="world")])
    assert len(buffer.load_messages()) == 2
    assert buffer.snapshot_version() == 2

    buffer.replace_messages([ChatMessage(role="system", text="reset")])
    assert len(buffer.load_messages()) == 1
    assert buffer.snapshot_version() == 3


@pytest.mark.asyncio
async def test_shared_state_turn_buffer_round_trip() -> None:
    ctx = _FakeWorkflowContext()
    shared = SharedStateTurnBuffer(key=HARNESS_SHARED_TURN_BUFFER_KEY)
    messages = [
        ChatMessage(role="user", text="hello"),
        ChatMessage(role="assistant", text="world"),
    ]

    await shared.write_snapshot(ctx, messages=messages, version=7)  # type: ignore[arg-type]
    restored, version = await shared.read_snapshot(ctx)  # type: ignore[arg-type]
    info = await shared.read_snapshot_info(ctx)  # type: ignore[arg-type]

    assert version == 7
    assert len(restored) == 2
    assert restored[0].text == "hello"
    assert restored[1].text == "world"
    assert info["present"] is True
    assert info["version"] == 7
    assert info["message_count"] == 2


@pytest.mark.asyncio
async def test_turn_buffer_sync_service_adopts_newer_owner_snapshot() -> None:
    ctx = _FakeWorkflowContext()
    shared = SharedStateTurnBuffer(key=HARNESS_SHARED_TURN_BUFFER_KEY)
    local = ExecutorLocalTurnBuffer([ChatMessage(role="user", text="stale")])
    sync = TurnBufferSyncService(shared_turn_buffer=shared)

    await shared.write_snapshot(
        ctx,
        messages=[ChatMessage(role="user", text="fresh"), ChatMessage(role="assistant", text="state")],
        version=4,
    )  # type: ignore[arg-type]

    adopted = await sync.adopt_shared_snapshot_if_owner_mode(
        ctx,  # type: ignore[arg-type]
        owner_mode="compaction_executor",
        local_turn_buffer=local,
    )

    assert adopted is True
    assert local.snapshot_version() == 1
    assert len(local.load_messages()) == 2
    assert local.load_messages()[0].text == "fresh"


@pytest.mark.asyncio
async def test_turn_buffer_sync_service_publishes_shadow_and_owner_modes() -> None:
    ctx = _FakeWorkflowContext()
    shared = SharedStateTurnBuffer(key=HARNESS_SHARED_TURN_BUFFER_KEY)
    local = ExecutorLocalTurnBuffer([ChatMessage(role="user", text="one")])
    sync = TurnBufferSyncService(shared_turn_buffer=shared)

    await sync.publish_shadow_snapshot(
        ctx,  # type: ignore[arg-type]
        owner_mode="shadow",
        local_turn_buffer=local,
    )
    shadow_info = await shared.read_snapshot_info(ctx)  # type: ignore[arg-type]
    assert shadow_info["present"] is True
    assert shadow_info["version"] == local.snapshot_version()
    assert shadow_info["message_count"] == 1

    local.append_message(ChatMessage(role="assistant", text="two"))
    await sync.publish_owner_snapshot(
        ctx,  # type: ignore[arg-type]
        owner_mode="agent_turn",
        local_turn_buffer=local,
    )
    unchanged_info = await shared.read_snapshot_info(ctx)  # type: ignore[arg-type]
    assert unchanged_info["version"] == shadow_info["version"]
    assert unchanged_info["message_count"] == shadow_info["message_count"]

    await sync.publish_owner_snapshot(
        ctx,  # type: ignore[arg-type]
        owner_mode="compaction_executor",
        local_turn_buffer=local,
    )
    owner_info = await shared.read_snapshot_info(ctx)  # type: ignore[arg-type]
    assert owner_info["version"] == local.snapshot_version()
    assert owner_info["message_count"] == 2


def test_compaction_view_ensure_message_ids_assigns_missing() -> None:
    messages = [
        ChatMessage(role="user", text="a", message_id=None),
        ChatMessage(role="assistant", text="b", message_id="m2"),
    ]
    ensure_message_ids(messages)
    assert messages[0].message_id is not None
    assert messages[1].message_id == "m2"


def test_compaction_view_apply_plan_keeps_tool_pairing() -> None:
    assistant = ChatMessage(
        role="assistant",
        contents=[FunctionCallContent(call_id="c1", name="read_file", arguments={"path": "x"})],
        message_id="a1",
    )
    tool = ChatMessage(
        role="tool",
        text="tool output",
        message_id="t1",
    )

    plan = CompactionPlan.create_empty(thread_id="h")
    plan.drops.append(DropRecord(span=SpanReference(message_ids=["t1"], first_turn=1, last_turn=1), reason="test-drop"))
    plan = CompactionPlan(
        thread_id=plan.thread_id,
        thread_version=plan.thread_version,
        created_at=plan.created_at,
        drops=plan.drops,
    )

    compacted = apply_compaction_plan_to_messages([assistant, tool], plan)
    assert len(compacted) == 2
    assert compacted[0].message_id == "a1"
    assert compacted[1].role.value == "tool"


def test_compaction_view_apply_direct_clear_replaces_large_results() -> None:
    assistant = ChatMessage(
        role="assistant",
        contents=[FunctionCallContent(call_id="x", name="read_file", arguments={"path": "a.txt"})],
        message_id="a1",
    )
    tool_msg = ChatMessage(
        role="tool",
        contents=[FunctionResultContent(call_id="x", result="R" * 4000)],
        message_id="t1",
    )
    recent_user = ChatMessage(role="user", text="recent", message_id="u2")
    compacted, cleared_count, tokens_freed = apply_direct_clear_to_messages(
        [assistant, tool_msg, recent_user],
        preserve_recent_messages=1,
        target_tokens_to_free=100,
    )

    assert cleared_count >= 1
    assert tokens_freed > 0
    assert any(m.role.value == "tool" for m in compacted)


def test_compaction_telemetry_owner_outcome_payloads() -> None:
    outcome = OwnerCompactionOutcome(
        turn_number=3,
        tokens_before=2000,
        tokens_freed=500,
        proposals_applied=2,
        strategies_applied=["clear"],
        owner_mode="compaction_executor",
        under_pressure=True,
    )
    metrics = outcome.metrics_payload()
    completed = outcome.lifecycle_completed_payload()

    assert metrics["turn_number"] == 3
    assert metrics["tokens_before"] == 2000
    assert metrics["tokens_freed"] == 500
    assert metrics["owner_path_applied"] is True
    assert completed["tokens_after"] == 1500
    assert completed["compaction_owner_mode"] == "compaction_executor"


def test_compaction_telemetry_pressure_and_lifecycle_payloads() -> None:
    budget = TokenBudget(max_input_tokens=10000, soft_threshold_percent=0.80, current_estimate=3000)
    metrics = pressure_metrics_payload(
        turn_number=4,
        tokens_before=3000,
        tokens_freed=0,
        proposals_applied=0,
        under_pressure=True,
        owner_mode="compaction_executor",
    )
    started = compaction_started_payload(
        current_tokens=3000,
        budget=budget,
        strategies_available=["clear", "drop"],
        owner_mode="compaction_executor",
        owner_fallback_reason="empty_plan",
    )
    pressure = context_pressure_payload(
        plan_updated=False,
        tokens_freed=0,
        proposals_applied=0,
        owner_mode="compaction_executor",
        owner_fallback_reason="empty_plan",
    )

    assert metrics["compaction_owner_mode"] == "compaction_executor"
    assert started["owner_fallback_reason"] == "empty_plan"
    assert started["strategies_available"] == ["clear", "drop"]
    assert pressure["plan_updated"] is False
