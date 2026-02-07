# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness module."""

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    AgentThread,
    BaseAgent,
    ChatMessage,
    Executor,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework._harness import (
    HarnessResult,
    HarnessStatus,
    HarnessWorkflowBuilder,
    RepairComplete,
    RepairExecutor,
    RepairTrigger,
    StopDecisionExecutor,
    TurnComplete,
)
from agent_framework._harness._constants import (
    HARNESS_MAX_TURNS_KEY,
    HARNESS_PENDING_TOOL_CALLS_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from agent_framework._harness._repair_executor import HARNESS_CONFIG_KEY
from agent_framework._harness._state import PendingToolCall


class MockAgent(BaseAgent):
    """A mock agent for testing that completes after a specified number of turns."""

    def __init__(
        self,
        *,
        name: str = "mock_agent",
        turns_to_complete: int = 1,
        should_fail: bool = False,
    ):
        super().__init__(name=name)
        self._turns_to_complete = turns_to_complete
        self._should_fail = should_fail
        self._current_turn = 0

    def get_new_thread(self) -> AgentThread:
        return AgentThread()

    async def run(
        self,
        messages: list[Any],
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        if self._should_fail:
            raise RuntimeError("Mock agent failure")

        self._current_turn += 1

        # Create a response message
        response_content = f"Turn {self._current_turn} response"
        response_message = ChatMessage(
            role=Role.ASSISTANT,
            contents=[TextContent(text=response_content)],
        )

        return AgentRunResponse(
            messages=[response_message],
            user_input_requests=[],
        )

    async def run_stream(
        self,
        messages: list[Any],
        thread: AgentThread | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        response = await self.run(messages, thread, **kwargs)
        for message in response.messages:
            yield AgentRunResponseUpdate(message=message)


# Test RepairExecutor


async def test_repair_executor_initializes_harness_state() -> None:
    """Test that RepairExecutor initializes harness state on first run."""

    @dataclass
    class TestOutput:
        status: str
        max_turns: int

    class TestExecutor(Executor):
        @handler
        async def handle(self, msg: RepairComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            status = await ctx.get_shared_state(HARNESS_STATUS_KEY)
            max_turns = await ctx.get_shared_state(HARNESS_MAX_TURNS_KEY)
            await ctx.yield_output(TestOutput(status=status, max_turns=max_turns))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: TestExecutor(id="test"), name="test")
        .add_edge("repair", "test")
        .set_start_executor("repair")
        .build()
    )

    # Pass max_turns via harness config
    result = await workflow.run(
        RepairTrigger(),
        **{HARNESS_CONFIG_KEY: {"max_turns": 25}},
    )
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.status == HarnessStatus.RUNNING.value
    assert output.max_turns == 25


async def test_repair_executor_repairs_dangling_tool_calls() -> None:
    """Test that RepairExecutor repairs dangling tool calls."""

    @dataclass
    class TestOutput:
        transcript: list[Any]
        repairs_made: int

    class SetupExecutor(Executor):
        """Executor that sets up initial state with dangling tool call."""

        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairTrigger]) -> None:
            # Set up state with dangling tool call AFTER workflow starts
            pending_call = PendingToolCall(
                call_id="test-call-1",
                tool_name="test_tool",
                args={"arg": "value"},
                turn_number=1,
            )
            await ctx.set_shared_state(HARNESS_PENDING_TOOL_CALLS_KEY, [pending_call.to_dict()])
            await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, [])
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 0)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            # Trigger repair
            await ctx.send_message(RepairTrigger())

    class TestRepairExecutor(Executor):
        """A test executor that captures repair output."""

        @handler
        async def handle_repair(self, complete: RepairComplete, ctx: WorkflowContext[None, TestOutput]) -> None:
            transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            await ctx.yield_output(TestOutput(transcript=transcript, repairs_made=complete.repairs_made))

    # Build workflow: setup -> repair -> test
    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: TestRepairExecutor(id="test"), name="test")
        .add_edge("setup", "repair")
        .add_edge("repair", "test")
        .set_start_executor("setup")
        .build()
    )

    # Run workflow
    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.repairs_made == 1
    assert len(output.transcript) == 1
    assert output.transcript[0]["event_type"] == "repair"
    assert output.transcript[0]["data"]["kind"] == "dangling_tool_call"


# Test StopDecisionExecutor


async def test_stop_decision_stops_on_max_turns() -> None:
    """Test that StopDecisionExecutor stops when max turns is reached."""

    class TestAgentTurnExecutor(Executor):
        """Mock executor that simulates agent turns."""

        @handler
        async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
            # Increment turn count
            turn = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
            turn = (turn or 0) + 1
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn)
            # Signal not done (to test max turns)
            await ctx.send_message(TurnComplete(agent_done=False))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: TestAgentTurnExecutor(id="agent"), name="agent")
        .register_executor(lambda: StopDecisionExecutor(id="stop"), name="stop")
        .add_edge("repair", "agent")
        .add_edge("agent", "stop")
        .add_edge("stop", "repair")
        .set_start_executor("repair")
        .set_max_iterations(100)  # Allow enough iterations
        .build()
    )

    # Pass max_turns=3 via harness config
    result = await workflow.run(
        RepairTrigger(),
        **{HARNESS_CONFIG_KEY: {"max_turns": 3}},
    )
    outputs = result.get_outputs()

    assert len(outputs) == 1
    harness_result = outputs[0]
    assert isinstance(harness_result, HarnessResult)
    assert harness_result.status == HarnessStatus.DONE
    assert harness_result.reason is not None
    assert harness_result.reason.kind == "max_turns"
    assert harness_result.turn_count == 3


async def test_stop_decision_stops_on_agent_done() -> None:
    """Test that StopDecisionExecutor stops when agent signals done."""

    class TestAgentTurnExecutor(Executor):
        """Mock executor that signals done on first turn."""

        @handler
        async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
            turn = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
            turn = (turn or 0) + 1
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn)
            # Signal done on first turn
            await ctx.send_message(TurnComplete(agent_done=True, called_task_complete=True))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: TestAgentTurnExecutor(id="agent"), name="agent")
        .register_executor(lambda: StopDecisionExecutor(id="stop"), name="stop")
        .add_edge("repair", "agent")
        .add_edge("agent", "stop")
        .add_edge("stop", "repair")
        .set_start_executor("repair")
        .build()
    )

    # Pass config via kwargs
    result = await workflow.run(
        RepairTrigger(),
        **{HARNESS_CONFIG_KEY: {"max_turns": 50}},
    )
    outputs = result.get_outputs()

    assert len(outputs) == 1
    harness_result = outputs[0]
    assert isinstance(harness_result, HarnessResult)
    assert harness_result.status == HarnessStatus.DONE
    assert harness_result.reason is not None
    assert harness_result.reason.kind == "agent_done"
    assert harness_result.turn_count == 1


async def test_stop_decision_stops_on_error() -> None:
    """Test that StopDecisionExecutor stops when agent fails."""

    class TestAgentTurnExecutor(Executor):
        """Mock executor that signals error."""

        @handler
        async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
            turn = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
            turn = (turn or 0) + 1
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn)
            await ctx.send_message(TurnComplete(agent_done=False, error="Test error"))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: TestAgentTurnExecutor(id="agent"), name="agent")
        .register_executor(lambda: StopDecisionExecutor(id="stop"), name="stop")
        .add_edge("repair", "agent")
        .add_edge("agent", "stop")
        .add_edge("stop", "repair")
        .set_start_executor("repair")
        .build()
    )

    # Pass config via kwargs
    result = await workflow.run(
        RepairTrigger(),
        **{HARNESS_CONFIG_KEY: {"max_turns": 50}},
    )
    outputs = result.get_outputs()

    assert len(outputs) == 1
    harness_result = outputs[0]
    assert isinstance(harness_result, HarnessResult)
    assert harness_result.status == HarnessStatus.FAILED
    assert harness_result.reason is not None
    assert harness_result.reason.kind == "failed"


# Test HarnessWorkflowBuilder


async def test_harness_workflow_builder_creates_valid_workflow() -> None:
    """Test that HarnessWorkflowBuilder creates a valid workflow."""
    agent = MockAgent(turns_to_complete=1)
    builder = HarnessWorkflowBuilder(agent, max_turns=10)

    workflow = builder.build()

    # Verify workflow was created
    assert workflow is not None
    assert workflow.name == "AgentHarness"


async def test_harness_workflow_builder_get_harness_kwargs() -> None:
    """Test that HarnessWorkflowBuilder provides correct kwargs."""
    agent = MockAgent()
    builder = HarnessWorkflowBuilder(agent, max_turns=25)

    kwargs = builder.get_harness_kwargs()

    assert HARNESS_CONFIG_KEY in kwargs
    assert kwargs[HARNESS_CONFIG_KEY]["max_turns"] == 25
