# Copyright (c) Microsoft. All rights reserved.

"""Tests for the hooks system (Phase 3)."""

import pytest

from agent_framework._harness._constants import (
    HARNESS_TURN_COUNT_KEY,
)
from agent_framework._harness._hooks import (
    AgentStopEvent,
    AgentStopResult,
    HarnessHooks,
    HarnessToolMiddleware,
    ToolHookResult,
)
from agent_framework._harness._repair_executor import RepairExecutor
from agent_framework._harness._state import (
    HarnessResult,
    RepairComplete,
    RepairTrigger,
    TurnComplete,
)
from agent_framework._harness._stop_decision_executor import StopDecisionExecutor
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_builder import WorkflowBuilder
from agent_framework._workflows._workflow_context import WorkflowContext

# ---------------------------------------------------------------------------
# HarnessToolMiddleware tests
# ---------------------------------------------------------------------------


class _FakeFunctionContext:
    """Minimal stand-in for FunctionInvocationContext."""

    def __init__(self, name: str, arguments: dict | None = None):
        self.function = type("F", (), {"name": name})()
        self.arguments = arguments or {}
        self.result = None
        self.terminate = False


class TestHarnessToolMiddleware:
    """Tests for HarnessToolMiddleware (pre/post tool hooks)."""

    @pytest.mark.asyncio
    async def test_no_hooks_passes_through(self) -> None:
        """With empty hooks, middleware just calls next."""
        hooks = HarnessHooks()
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("read_file")
        called = False

        async def next_handler(c):
            nonlocal called
            called = True
            c.result = "file content"

        await mw.process(ctx, next_handler)
        assert called
        assert ctx.result == "file content"

    @pytest.mark.asyncio
    async def test_pre_hook_deny_blocks_execution(self) -> None:
        """A pre-tool hook returning deny prevents tool execution."""

        async def deny_all(name, args):
            return ToolHookResult(decision="deny", reason="Not allowed")

        hooks = HarnessHooks(pre_tool=[deny_all])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("dangerous_tool")
        next_called = False

        async def next_handler(c):
            nonlocal next_called
            next_called = True

        await mw.process(ctx, next_handler)
        assert not next_called
        assert ctx.terminate is True
        assert "Denied" in ctx.result
        assert "Not allowed" in ctx.result

    @pytest.mark.asyncio
    async def test_pre_hook_allow_continues(self) -> None:
        """A pre-tool hook returning allow (or None) lets tool execute."""

        async def allow_all(name, args):
            return ToolHookResult(decision="allow")

        hooks = HarnessHooks(pre_tool=[allow_all])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("safe_tool")

        async def next_handler(c):
            c.result = "ok"

        await mw.process(ctx, next_handler)
        assert ctx.result == "ok"
        assert ctx.terminate is False

    @pytest.mark.asyncio
    async def test_pre_hook_none_continues(self) -> None:
        """A pre-tool hook returning None lets tool execute."""

        async def no_opinion(name, args):
            return None

        hooks = HarnessHooks(pre_tool=[no_opinion])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("some_tool")

        async def next_handler(c):
            c.result = "ok"

        await mw.process(ctx, next_handler)
        assert ctx.result == "ok"

    @pytest.mark.asyncio
    async def test_post_hook_receives_result(self) -> None:
        """Post-tool hooks receive the tool name, args, and result."""
        captured = {}

        async def capture_result(name, args, result):
            captured["name"] = name
            captured["args"] = args
            captured["result"] = result

        hooks = HarnessHooks(post_tool=[capture_result])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("read_file", {"path": "/etc/hosts"})

        async def next_handler(c):
            c.result = "file content"

        await mw.process(ctx, next_handler)
        assert captured["name"] == "read_file"
        assert captured["result"] == "file content"

    @pytest.mark.asyncio
    async def test_pre_hook_exception_does_not_block(self) -> None:
        """A pre-tool hook that raises still allows tool execution."""

        async def buggy_hook(name, args):
            raise RuntimeError("hook crashed")

        hooks = HarnessHooks(pre_tool=[buggy_hook])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("tool")

        async def next_handler(c):
            c.result = "ok"

        await mw.process(ctx, next_handler)
        assert ctx.result == "ok"

    @pytest.mark.asyncio
    async def test_post_hook_exception_does_not_raise(self) -> None:
        """A post-tool hook that raises does not propagate."""

        async def buggy_hook(name, args, result):
            raise RuntimeError("post hook crashed")

        hooks = HarnessHooks(post_tool=[buggy_hook])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("tool")

        async def next_handler(c):
            c.result = "ok"

        await mw.process(ctx, next_handler)
        assert ctx.result == "ok"

    @pytest.mark.asyncio
    async def test_multiple_pre_hooks_first_deny_wins(self) -> None:
        """When multiple pre hooks exist, the first deny wins."""
        call_order = []

        async def hook_a(name, args):
            call_order.append("a")
            return ToolHookResult(decision="deny", reason="hook_a says no")

        async def hook_b(name, args):
            call_order.append("b")
            return ToolHookResult(decision="allow")

        hooks = HarnessHooks(pre_tool=[hook_a, hook_b])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("tool")

        async def next_handler(c):
            pass

        await mw.process(ctx, next_handler)
        assert call_order == ["a"]  # hook_b never called
        assert ctx.terminate is True

    @pytest.mark.asyncio
    async def test_multiple_post_hooks_all_called(self) -> None:
        """All post hooks are called even if earlier ones raise."""
        call_order = []

        async def hook_a(name, args, result):
            call_order.append("a")
            raise RuntimeError("boom")

        async def hook_b(name, args, result):
            call_order.append("b")

        hooks = HarnessHooks(post_tool=[hook_a, hook_b])
        mw = HarnessToolMiddleware(hooks)
        ctx = _FakeFunctionContext("tool")

        async def next_handler(c):
            c.result = "ok"

        await mw.process(ctx, next_handler)
        assert call_order == ["a", "b"]


# ---------------------------------------------------------------------------
# AgentStopEvent / AgentStopResult dataclass tests
# ---------------------------------------------------------------------------


class TestAgentStopDataclasses:
    """Tests for hook dataclasses."""

    def test_stop_event_defaults(self) -> None:
        event = AgentStopEvent()
        assert event.turn_count == 0
        assert event.tool_usage == {}
        assert event.called_task_complete is False

    def test_stop_result_allow_by_default(self) -> None:
        result = AgentStopResult()
        assert result.decision == "allow"

    def test_stop_result_block(self) -> None:
        result = AgentStopResult(decision="block", reason="Not enough turns")
        assert result.decision == "block"
        assert result.reason == "Not enough turns"

    def test_tool_hook_result_allow_by_default(self) -> None:
        result = ToolHookResult()
        assert result.decision == "allow"


# ---------------------------------------------------------------------------
# HarnessHooks dataclass tests
# ---------------------------------------------------------------------------


class TestHarnessHooks:
    """Tests for HarnessHooks dataclass."""

    def test_defaults_are_empty_lists(self) -> None:
        hooks = HarnessHooks()
        assert hooks.pre_tool == []
        assert hooks.post_tool == []
        assert hooks.agent_stop == []

    def test_independent_instances(self) -> None:
        """Each HarnessHooks instance has its own lists."""
        h1 = HarnessHooks()
        h2 = HarnessHooks()

        async def dummy(event):
            return None

        h1.agent_stop.append(dummy)
        assert len(h2.agent_stop) == 0


# ---------------------------------------------------------------------------
# StopDecisionExecutor agent_stop hook integration
# ---------------------------------------------------------------------------

HARNESS_CONFIG_KEY = "harness.config"


class TestStopDecisionAgentStopHooks:
    """Tests for agent_stop hooks wired into StopDecisionExecutor."""

    @pytest.mark.asyncio
    async def test_blocking_hook_prevents_stop(self) -> None:
        """An agent_stop hook returning block sends RepairTrigger instead of stopping."""

        async def block_always(event: AgentStopEvent) -> AgentStopResult:
            return AgentStopResult(decision="block", reason="Too few turns")

        hooks = HarnessHooks(agent_stop=[block_always])

        # Build a minimal workflow with the stop decision executor
        class MockAgent(Executor):
            @handler
            async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
                turn_count = (await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0) + 1
                await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)
                # Signal done with task_complete
                await ctx.send_message(TurnComplete(agent_done=True, called_task_complete=True))

        workflow = (
            WorkflowBuilder()
            .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
            .register_executor(lambda: MockAgent(id="agent"), name="agent")
            .register_executor(
                lambda: StopDecisionExecutor(
                    require_task_complete=True,
                    hooks=hooks,
                    id="stop",
                ),
                name="stop",
            )
            .add_edge("repair", "agent")
            .add_edge("agent", "stop")
            .add_edge("stop", "repair")
            .set_start_executor("repair")
            .set_max_iterations(100)
            .build()
        )
        result = await workflow.run(
            RepairTrigger(),
            **{HARNESS_CONFIG_KEY: {"max_turns": 3}},
        )

        # The hook blocks every stop, so we hit max_turns (3)
        outputs = result.get_outputs()
        assert len(outputs) == 1
        assert isinstance(outputs[0], HarnessResult)
        assert outputs[0].reason.kind == "max_turns"

    @pytest.mark.asyncio
    async def test_allow_hook_lets_stop(self) -> None:
        """An agent_stop hook returning allow (or None) lets stop proceed."""

        async def allow_always(event: AgentStopEvent) -> AgentStopResult | None:
            return AgentStopResult(decision="allow")

        hooks = HarnessHooks(agent_stop=[allow_always])

        class MockAgent(Executor):
            @handler
            async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
                turn_count = (await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0) + 1
                await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)
                await ctx.send_message(TurnComplete(agent_done=True, called_task_complete=True))

        workflow = (
            WorkflowBuilder()
            .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
            .register_executor(lambda: MockAgent(id="agent"), name="agent")
            .register_executor(
                lambda: StopDecisionExecutor(
                    require_task_complete=True,
                    hooks=hooks,
                    id="stop",
                ),
                name="stop",
            )
            .add_edge("repair", "agent")
            .add_edge("agent", "stop")
            .add_edge("stop", "repair")
            .set_start_executor("repair")
            .set_max_iterations(100)
            .build()
        )
        result = await workflow.run(
            RepairTrigger(),
            **{HARNESS_CONFIG_KEY: {"max_turns": 10}},
        )

        outputs = result.get_outputs()
        assert len(outputs) == 1
        assert isinstance(outputs[0], HarnessResult)
        assert outputs[0].reason.kind == "agent_done"

    @pytest.mark.asyncio
    async def test_no_hooks_stops_normally(self) -> None:
        """Without hooks, stop decision proceeds normally."""

        class MockAgent(Executor):
            @handler
            async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
                turn_count = (await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0) + 1
                await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)
                await ctx.send_message(TurnComplete(agent_done=True, called_task_complete=True))

        workflow = (
            WorkflowBuilder()
            .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
            .register_executor(lambda: MockAgent(id="agent"), name="agent")
            .register_executor(
                lambda: StopDecisionExecutor(
                    require_task_complete=True,
                    hooks=None,
                    id="stop",
                ),
                name="stop",
            )
            .add_edge("repair", "agent")
            .add_edge("agent", "stop")
            .add_edge("stop", "repair")
            .set_start_executor("repair")
            .set_max_iterations(100)
            .build()
        )
        result = await workflow.run(
            RepairTrigger(),
            **{HARNESS_CONFIG_KEY: {"max_turns": 10}},
        )

        outputs = result.get_outputs()
        assert len(outputs) == 1
        assert isinstance(outputs[0], HarnessResult)
        assert outputs[0].reason.kind == "agent_done"

    @pytest.mark.asyncio
    async def test_hook_exception_does_not_block_stop(self) -> None:
        """If an agent_stop hook raises, the stop proceeds normally."""

        async def crashy_hook(event: AgentStopEvent):
            raise RuntimeError("hook crashed")

        hooks = HarnessHooks(agent_stop=[crashy_hook])

        class MockAgent(Executor):
            @handler
            async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
                turn_count = (await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0) + 1
                await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)
                await ctx.send_message(TurnComplete(agent_done=True, called_task_complete=True))

        workflow = (
            WorkflowBuilder()
            .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
            .register_executor(lambda: MockAgent(id="agent"), name="agent")
            .register_executor(
                lambda: StopDecisionExecutor(
                    require_task_complete=True,
                    hooks=hooks,
                    id="stop",
                ),
                name="stop",
            )
            .add_edge("repair", "agent")
            .add_edge("agent", "stop")
            .add_edge("stop", "repair")
            .set_start_executor("repair")
            .set_max_iterations(100)
            .build()
        )
        result = await workflow.run(
            RepairTrigger(),
            **{HARNESS_CONFIG_KEY: {"max_turns": 10}},
        )

        outputs = result.get_outputs()
        assert len(outputs) == 1
        assert isinstance(outputs[0], HarnessResult)
        assert outputs[0].reason.kind == "agent_done"

    @pytest.mark.asyncio
    async def test_hook_receives_event_data(self) -> None:
        """The hook receives the correct AgentStopEvent payload."""
        captured_events = []

        async def capture_hook(event: AgentStopEvent) -> AgentStopResult | None:
            captured_events.append(event)
            return None  # allow

        hooks = HarnessHooks(agent_stop=[capture_hook])

        class MockAgent(Executor):
            @handler
            async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
                turn_count = (await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0) + 1
                await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)
                await ctx.send_message(TurnComplete(agent_done=True, called_task_complete=True))

        workflow = (
            WorkflowBuilder()
            .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
            .register_executor(lambda: MockAgent(id="agent"), name="agent")
            .register_executor(
                lambda: StopDecisionExecutor(
                    require_task_complete=True,
                    hooks=hooks,
                    id="stop",
                ),
                name="stop",
            )
            .add_edge("repair", "agent")
            .add_edge("agent", "stop")
            .add_edge("stop", "repair")
            .set_start_executor("repair")
            .set_max_iterations(100)
            .build()
        )
        await workflow.run(
            RepairTrigger(),
            **{HARNESS_CONFIG_KEY: {"max_turns": 10}},
        )

        assert len(captured_events) == 1
        assert captured_events[0].turn_count == 1
        assert captured_events[0].called_task_complete is True
