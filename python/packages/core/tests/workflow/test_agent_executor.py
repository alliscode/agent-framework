# Copyright (c) Microsoft. All rights reserved.

import logging
from collections.abc import AsyncIterable, Awaitable
from typing import TYPE_CHECKING, Any, Literal, overload

import pytest

from agent_framework import (
    AgentExecutor,
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    AgentSession,
    BaseAgent,
    Content,
    Message,
    ResponseStream,
    WorkflowBuilder,
    WorkflowEvent,
    WorkflowRunState,
)
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


class _CountingAgent(BaseAgent):
    """Agent that echoes messages with a counter to verify session state persistence."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.call_count = 0

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        self.call_count += 1
        if stream:

            async def _stream() -> AsyncIterable[AgentResponseUpdate]:
                yield AgentResponseUpdate(
                    contents=[Content.from_text(text=f"Response #{self.call_count}: {self.name}")]
                )

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates)

        async def _run() -> AgentResponse:
            return AgentResponse(messages=[Message("assistant", [f"Response #{self.call_count}: {self.name}"])])

        return _run()


class _StreamingHookAgent(BaseAgent):
    """Agent that exposes whether its streaming result hook was executed."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.result_hook_called = False

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        if stream:

            async def _stream() -> AsyncIterable[AgentResponseUpdate]:
                yield AgentResponseUpdate(
                    contents=[Content.from_text(text="hook test")],
                    role="assistant",
                )

            async def _mark_result_hook_called(
                response: AgentResponse,
            ) -> AgentResponse:
                self.result_hook_called = True
                return response

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates).with_result_hook(
                _mark_result_hook_called
            )

        async def _run() -> AgentResponse:
            return AgentResponse(messages=[Message("assistant", ["hook test"])])

        return _run()


async def test_agent_executor_streaming_finalizes_stream_and_runs_result_hooks() -> None:
    """AgentExecutor should call get_final_response() so stream result hooks execute."""
    agent = _StreamingHookAgent(id="hook_agent", name="HookAgent")
    executor = AgentExecutor(agent, id="hook_exec")
    workflow = WorkflowBuilder(start_executor=executor).build()

    output_events: list[Any] = []
    async for event in workflow.run("run hook test", stream=True):
        if event.type == "output":
            output_events.append(event)

    assert output_events
    assert agent.result_hook_called


async def test_agent_executor_checkpoint_stores_and_restores_state() -> None:
    """Test that workflow checkpoint stores AgentExecutor's cache and session states and restores them correctly."""
    storage = InMemoryCheckpointStorage()

    # Create two agents to form a two-step workflow
    initial_agent_a = _CountingAgent(id="agent_a", name="AgentA")
    initial_agent_b = _CountingAgent(id="agent_b", name="AgentB")
    initial_session = AgentSession()

    # Add some initial messages to the session state to verify session state persistence
    initial_messages = [
        Message(role="user", text="Initial message 1"),
        Message(role="assistant", text="Initial response 1"),
    ]
    initial_session.state["history"] = {"messages": initial_messages}

    # Create AgentExecutors — first executor gets the custom session
    exec_a = AgentExecutor(initial_agent_a, id="exec_a", session=initial_session)
    exec_b = AgentExecutor(initial_agent_b, id="exec_b")

    # Build two-executor workflow with checkpointing enabled
    wf = WorkflowBuilder(start_executor=exec_a, checkpoint_storage=storage).add_edge(exec_a, exec_b).build()

    # Run the workflow with a user message
    first_run_output: AgentExecutorResponse | None = None
    async for ev in wf.run("First workflow run", stream=True):
        if ev.type == "output":
            first_run_output = ev.data  # type: ignore[assignment]
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    assert first_run_output is not None
    assert initial_agent_a.call_count == 1

    # Verify checkpoint was created
    checkpoints = await storage.list_checkpoints(workflow_name=wf.name)
    assert len(checkpoints) >= 2, "Expected at least 2 checkpoints: one after exec_a and one after exec_b."

    # Get the first checkpoint that contains exec_a's state (taken after exec_a completes,
    # before exec_b runs)
    checkpoints.sort(key=lambda cp: cp.timestamp)
    restore_checkpoint = next(
        cp for cp in checkpoints if "_executor_state" in cp.state and "exec_a" in cp.state["_executor_state"]
    )

    # Verify checkpoint contains executor state with both cache and session
    executor_states = restore_checkpoint.state["_executor_state"]
    assert isinstance(executor_states, dict)
    assert exec_a.id in executor_states

    executor_state = executor_states[exec_a.id]  # type: ignore[index]
    assert "cache" in executor_state, "Checkpoint should store executor cache state"
    assert "agent_session" in executor_state, "Checkpoint should store executor session state"

    # Verify session state structure
    session_state = executor_state["agent_session"]  # type: ignore[index]
    assert "session_id" in session_state, "Session state should include session_id"
    assert "state" in session_state, "Session state should include state dict"

    # Verify checkpoint contains pending requests from agents and responses to be sent
    assert "pending_agent_requests" in executor_state
    assert "pending_responses_to_agent" in executor_state

    # Create new agents and executors for restoration
    # This simulates starting from a fresh state and restoring from checkpoint
    restored_agent_a = _CountingAgent(id="agent_a", name="AgentA")
    restored_agent_b = _CountingAgent(id="agent_b", name="AgentB")
    restored_session = AgentSession()
    restored_exec_a = AgentExecutor(restored_agent_a, id="exec_a", session=restored_session)
    restored_exec_b = AgentExecutor(restored_agent_b, id="exec_b")

    # Verify the restored agents start with a fresh state
    assert restored_agent_a.call_count == 0
    assert restored_agent_b.call_count == 0

    # Build new workflow with the restored executors
    wf_resume = (
        WorkflowBuilder(start_executor=restored_exec_a, checkpoint_storage=storage)
        .add_edge(restored_exec_a, restored_exec_b)
        .build()
    )

    # Resume from checkpoint — exec_a already ran, so exec_b should run and produce output
    resumed_output: AgentExecutorResponse | None = None
    async for ev in wf_resume.run(checkpoint_id=restore_checkpoint.checkpoint_id, stream=True):
        if ev.type == "output":
            resumed_output = ev.data  # type: ignore[assignment]
        if ev.type == "status" and ev.state in (
            WorkflowRunState.IDLE,
            WorkflowRunState.IDLE_WITH_PENDING_REQUESTS,
        ):
            break

    assert resumed_output is not None

    # Verify the restored executor's session state was restored
    restored_session_obj = restored_exec_a._session  # type: ignore[reportPrivateUsage]
    assert restored_session_obj is not None
    assert restored_session_obj.session_id == initial_session.session_id


async def test_agent_executor_save_and_restore_state_directly() -> None:
    """Test AgentExecutor's on_checkpoint_save and on_checkpoint_restore methods directly."""
    # Create agent with session containing state
    agent = _CountingAgent(id="direct_test_agent", name="DirectTestAgent")
    session = AgentSession()

    # Add messages to session state
    session_messages = [
        Message(role="user", text="Message in session 1"),
        Message(role="assistant", text="Session response 1"),
        Message(role="user", text="Message in session 2"),
    ]
    session.state["history"] = {"messages": session_messages}

    executor = AgentExecutor(agent, session=session)

    # Add messages to executor cache
    cache_messages = [
        Message(role="user", text="Cached user message"),
        Message(role="assistant", text="Cached assistant response"),
    ]
    executor._cache = list(cache_messages)  # type: ignore[reportPrivateUsage]

    # Snapshot the state
    state = await executor.on_checkpoint_save()

    # Verify snapshot contains both cache and session
    assert "cache" in state
    assert "agent_session" in state

    # Verify session state structure
    session_state = state["agent_session"]  # type: ignore[index]
    assert "session_id" in session_state
    assert "state" in session_state

    # Create new executor to restore into
    new_agent = _CountingAgent(id="direct_test_agent", name="DirectTestAgent")
    new_session = AgentSession()
    new_executor = AgentExecutor(new_agent, session=new_session)

    # Verify new executor starts empty
    assert len(new_executor._cache) == 0  # type: ignore[reportPrivateUsage]
    assert len(new_session.state) == 0

    # Restore state
    await new_executor.on_checkpoint_restore(state)

    # Verify cache is restored
    restored_cache = new_executor._cache  # type: ignore[reportPrivateUsage]
    assert len(restored_cache) == len(cache_messages)
    assert restored_cache[0].text == "Cached user message"
    assert restored_cache[1].text == "Cached assistant response"

    # Verify session was restored with correct session_id
    restored_session = new_executor._session  # type: ignore[reportPrivateUsage]
    assert restored_session.session_id == session.session_id


async def test_agent_executor_run_with_session_kwarg_does_not_raise() -> None:
    """Passing session= via workflow.run() should not cause a duplicate-keyword TypeError (#4295)."""
    agent = _CountingAgent(id="session_kwarg_agent", name="SessionKwargAgent")
    executor = AgentExecutor(agent, id="session_kwarg_exec")
    workflow = WorkflowBuilder(start_executor=executor).build()

    # This previously raised: TypeError: run() got multiple values for keyword argument 'session'
    result = await workflow.run("hello", session="user-supplied-value")
    assert result is not None
    assert agent.call_count == 1


async def test_agent_executor_run_streaming_with_stream_kwarg_does_not_raise() -> None:
    """Passing stream= via workflow.run() kwargs should not cause a duplicate-keyword TypeError."""
    agent = _CountingAgent(id="stream_kwarg_agent", name="StreamKwargAgent")
    executor = AgentExecutor(agent, id="stream_kwarg_exec")
    workflow = WorkflowBuilder(start_executor=executor).build()

    # stream=True at workflow level triggers streaming mode (returns async iterable)
    events: list[WorkflowEvent] = []
    async for event in workflow.run("hello", stream=True):
        events.append(event)
    assert len(events) > 0
    assert agent.call_count == 1


@pytest.mark.parametrize("reserved_kwarg", ["session", "stream", "messages"])
async def test_prepare_agent_run_args_strips_reserved_kwargs(reserved_kwarg: str, caplog: "LogCaptureFixture") -> None:
    """_prepare_agent_run_args must remove reserved kwargs and log a warning."""
    raw: dict[str, Any] = {
        reserved_kwarg: "should-be-stripped",
        "custom_key": "keep-me",
    }

    with caplog.at_level(logging.WARNING):
        run_kwargs, options = AgentExecutor._prepare_agent_run_args(raw)  # pyright: ignore[reportPrivateUsage]

    assert reserved_kwarg not in run_kwargs
    assert "custom_key" in run_kwargs
    assert options is not None
    assert options["additional_function_arguments"]["custom_key"] == "keep-me"
    assert any(reserved_kwarg in record.message for record in caplog.records)


async def test_prepare_agent_run_args_preserves_non_reserved_kwargs() -> None:
    """Non-reserved workflow kwargs should pass through unchanged."""
    raw: dict[str, Any] = {"custom_param": "value", "another": 42}
    run_kwargs, _options = AgentExecutor._prepare_agent_run_args(raw)  # pyright: ignore[reportPrivateUsage]
    assert run_kwargs["custom_param"] == "value"
    assert run_kwargs["another"] == 42


async def test_prepare_agent_run_args_strips_all_reserved_kwargs_at_once(
    caplog: "LogCaptureFixture",
) -> None:
    """All reserved kwargs should be stripped when supplied together, each emitting a warning."""
    raw: dict[str, Any] = {"session": "x", "stream": True, "messages": [], "custom": 1}

    with caplog.at_level(logging.WARNING):
        run_kwargs, options = AgentExecutor._prepare_agent_run_args(raw)  # pyright: ignore[reportPrivateUsage]

    assert "session" not in run_kwargs
    assert "stream" not in run_kwargs
    assert "messages" not in run_kwargs
    assert run_kwargs["custom"] == 1
    assert options is not None
    assert options["additional_function_arguments"]["custom"] == 1

    warned_keys = {r.message.split("'")[1] for r in caplog.records if "reserved" in r.message.lower()}
    assert warned_keys == {"session", "stream", "messages"}


async def test_agent_executor_run_with_messages_kwarg_does_not_raise() -> None:
    """Passing messages= via workflow.run() kwargs should not cause a duplicate-keyword TypeError."""
    agent = _CountingAgent(id="messages_kwarg_agent", name="MessagesKwargAgent")
    executor = AgentExecutor(agent, id="messages_kwarg_exec")
    workflow = WorkflowBuilder(start_executor=executor).build()

    result = await workflow.run("hello", messages=["stale"])
    assert result is not None
    assert agent.call_count == 1


class _NonCopyableRaw:
    """Simulates an LLM SDK response object that cannot be deep-copied (e.g., proto/gRPC)."""

    def __deepcopy__(self, memo: dict) -> Any:
        raise TypeError("Cannot deepcopy this object")


class _AgentWithRawRepr(BaseAgent):
    """Agent that returns responses with a non-copyable raw_representation."""

    def __init__(self, raw: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self._raw = raw

    def run(
        self,
        messages: str | Message | list[str] | list[Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse] | ResponseStream[AgentResponseUpdate, AgentResponse]:
        async def _run() -> AgentResponse:
            return AgentResponse(
                messages=[Message("assistant", [f"reply from {self.name}"])],
                raw_representation=self._raw,
            )

        return _run()


async def test_agent_executor_workflow_with_non_copyable_raw_representation() -> None:
    """Workflow should complete when AgentResponse contains a raw_representation that cannot be deep-copied."""
    raw = _NonCopyableRaw()

    agent_a = _AgentWithRawRepr(raw=raw, id="a", name="AgentA")
    agent_b = _CountingAgent(id="b", name="AgentB")

    exec_a = AgentExecutor(agent_a, id="exec_a")
    exec_b = AgentExecutor(agent_b, id="exec_b")

    workflow = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()
    events = await workflow.run("hello")

    completed = [e for e in events if isinstance(e, WorkflowEvent) and e.type == "executor_completed"]
    completed_a = [e for e in completed if e.executor_id == "exec_a"]

    assert len(completed_a) == 1
    assert completed_a[0].data is not None

    # The yielded AgentResponse should preserve its raw_representation reference
    agent_responses = [d for d in completed_a[0].data if isinstance(d, AgentResponse)]
    assert len(agent_responses) > 0
    assert agent_responses[0].text == "reply from AgentA"
    assert agent_responses[0].raw_representation is raw


# ---------------------------------------------------------------------------
# Context mode tests
# ---------------------------------------------------------------------------


class _MessageCapturingAgent(BaseAgent):
    """Agent that records the messages it received and returns a configurable reply."""

    def __init__(self, *, reply_text: str = "reply", **kwargs: Any):
        super().__init__(**kwargs)
        self.reply_text = reply_text
        self.last_messages: list[Message] = []

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        captured: list[Message] = []
        if messages:
            for m in messages:  # type: ignore[union-attr]
                if isinstance(m, Message):
                    captured.append(m)
                elif isinstance(m, str):
                    captured.append(Message("user", [m]))
        self.last_messages = captured

        if stream:

            async def _stream() -> AsyncIterable[AgentResponseUpdate]:
                yield AgentResponseUpdate(contents=[Content.from_text(text=self.reply_text)])

            return ResponseStream(_stream(), finalizer=AgentResponse.from_updates)

        async def _run() -> AgentResponse:
            return AgentResponse(messages=[Message("assistant", [self.reply_text])])

        return _run()


def test_context_mode_custom_requires_context_filter() -> None:
    """context_mode='custom' without context_filter must raise ValueError."""
    agent = _CountingAgent(id="a", name="A")
    with pytest.raises(ValueError, match="context_filter must be provided"):
        AgentExecutor(agent, context_mode="custom")


def test_context_mode_custom_with_filter_succeeds() -> None:
    """context_mode='custom' with a context_filter should not raise."""
    agent = _CountingAgent(id="a", name="A")
    executor = AgentExecutor(agent, context_mode="custom", context_filter=lambda msgs: msgs[-1:])
    assert executor._context_mode == "custom"  # pyright: ignore[reportPrivateUsage]
    assert executor._context_filter is not None  # pyright: ignore[reportPrivateUsage]


def test_context_mode_defaults_to_full() -> None:
    """Default context_mode should be 'full'."""
    agent = _CountingAgent(id="a", name="A")
    executor = AgentExecutor(agent)
    assert executor._context_mode == "full"  # pyright: ignore[reportPrivateUsage]


def test_context_mode_invalid_value_raises() -> None:
    """Invalid context_mode value should raise ValueError."""
    agent = _CountingAgent(id="a", name="A")
    with pytest.raises(ValueError, match="context_mode must be one of"):
        AgentExecutor(agent, context_mode="invalid_mode")  # type: ignore


async def test_from_response_context_mode_full_passes_full_conversation() -> None:
    """context_mode='full' (default) should pass full_conversation to the second agent."""
    first = _MessageCapturingAgent(id="first", name="First", reply_text="first reply")
    second = _MessageCapturingAgent(id="second", name="Second", reply_text="second reply")

    exec_a = AgentExecutor(first, id="exec_a")
    exec_b = AgentExecutor(second, id="exec_b", context_mode="full")

    wf = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()

    async for ev in wf.run("hello", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    # Second agent should see full conversation: [user("hello"), assistant("first reply")]
    seen = second.last_messages
    assert len(seen) == 2
    assert seen[0].role == "user" and "hello" in (seen[0].text or "")
    assert seen[1].role == "assistant" and "first reply" in (seen[1].text or "")


async def test_from_response_context_mode_last_agent_passes_only_agent_messages() -> None:
    """context_mode='last_agent' should pass only the previous agent's response messages."""
    first = _MessageCapturingAgent(id="first", name="First", reply_text="first reply")
    second = _MessageCapturingAgent(id="second", name="Second", reply_text="second reply")

    exec_a = AgentExecutor(first, id="exec_a")
    exec_b = AgentExecutor(second, id="exec_b", context_mode="last_agent")

    wf = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()

    async for ev in wf.run("hello", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    # Second agent should see only the assistant message from first: [assistant("first reply")]
    seen = second.last_messages
    assert len(seen) == 1
    assert seen[0].role == "assistant" and "first reply" in (seen[0].text or "")


async def test_from_response_context_mode_custom_uses_filter() -> None:
    """context_mode='custom' should invoke context_filter on full_conversation."""
    first = _MessageCapturingAgent(id="first", name="First", reply_text="first reply")
    second = _MessageCapturingAgent(id="second", name="Second", reply_text="second reply")

    # Custom filter: keep only user messages
    def only_user_messages(msgs: list[Message]) -> list[Message]:
        return [m for m in msgs if m.role == "user"]

    exec_a = AgentExecutor(first, id="exec_a")
    exec_b = AgentExecutor(second, id="exec_b", context_mode="custom", context_filter=only_user_messages)

    wf = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()

    async for ev in wf.run("hello", stream=True):
        if ev.type == "status" and ev.state == WorkflowRunState.IDLE:
            break

    # Second agent should see only user messages: [user("hello")]
    seen = second.last_messages
    assert len(seen) == 1
    assert seen[0].role == "user" and "hello" in (seen[0].text or "")


async def test_checkpoint_save_does_not_include_context_mode() -> None:
    """on_checkpoint_save should not include context_mode in the saved state."""
    agent = _CountingAgent(id="a", name="A")
    executor = AgentExecutor(agent, context_mode="last_agent")

    state = await executor.on_checkpoint_save()

    assert "context_mode" not in state
    assert "cache" in state
    assert "agent_session" in state


async def test_checkpoint_restore_works_without_context_mode_in_state() -> None:
    """on_checkpoint_restore should succeed when state does not contain context_mode."""
    agent = _CountingAgent(id="a", name="A")
    executor = AgentExecutor(agent, context_mode="last_agent")

    # Simulate a checkpoint state without context_mode (as saved by the new code)
    state: dict[str, Any] = {
        "cache": [Message(role="user", text="cached msg")],
        "full_conversation": [],
        "agent_session": AgentSession().to_dict(),
        "pending_agent_requests": {},
        "pending_responses_to_agent": [],
    }

    await executor.on_checkpoint_restore(state)

    cache = executor._cache  # pyright: ignore[reportPrivateUsage]
    assert len(cache) == 1
    assert cache[0].text == "cached msg"
    # context_mode should remain as configured in the constructor, not changed by restore
    assert executor._context_mode == "last_agent"  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Regression: stale service_session_id between workflow runs
# ---------------------------------------------------------------------------


class _SessionTrackingAgent(BaseAgent):
    """Agent that simulates the Responses API behaviour of setting service_session_id.

    Records the session ID that was present at the START of each run(), then
    sets a new one — mimicking what the Responses API client does when it stores
    the response ID back into the session.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.observed_session_ids: list[str | None] = []

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        # Record the session ID that was visible when we were called.
        # In a real Responses API scenario this becomes previous_response_id.
        self.observed_session_ids.append(session.service_session_id if session else None)

        # Simulate the Responses API client storing its response ID back
        if session is not None:
            call_num = len(self.observed_session_ids)
            session.service_session_id = f"resp_{self.name}_{call_num}"

        async def _run() -> AgentResponse:
            return AgentResponse(messages=[Message("assistant", [f"Reply from {self.name}"])])

        return _run()


@pytest.mark.asyncio
async def test_workflow_rerun_preserves_session_for_continuation() -> None:
    """Multiple workflow.run() calls on the same instance are continuations.

    The session's service_session_id should carry over between runs so
    that the Responses API treats them as a continued conversation (via
    previous_response_id). The workflow engine resets its own plumbing
    (superstep counter, message queues, state) but executor session
    state is preserved.
    """
    agent_a = _SessionTrackingAgent(id="A", name="A")
    agent_b = _SessionTrackingAgent(id="B", name="B")

    exec_a = AgentExecutor(agent_a)
    exec_b = AgentExecutor(agent_b)

    workflow = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()

    # --- First run ---
    result1 = await workflow.run("Hello")
    assert result1 is not None

    # First run: both agents start with a fresh session (no service_session_id)
    assert agent_a.observed_session_ids[0] is None, "Run 1: agent A should start with no session ID"
    assert agent_b.observed_session_ids[0] is None, "Run 1: agent B should start with no session ID"

    # After the first run, sessions hold the response IDs set during the run
    assert exec_a._session.service_session_id == "resp_A_1"  # pyright: ignore[reportPrivateUsage]
    assert exec_b._session.service_session_id == "resp_B_1"  # pyright: ignore[reportPrivateUsage]

    # --- Second run (continuation) ---
    result2 = await workflow.run("Hello again")
    assert result2 is not None

    # Second run: agents see the session IDs from run 1 (continuation model)
    assert agent_a.observed_session_ids[1] == "resp_A_1", (
        f"Run 2: agent A should see run 1's session ID for continuation, "
        f"got '{agent_a.observed_session_ids[1]}'"
    )
    assert agent_b.observed_session_ids[1] == "resp_B_1", (
        f"Run 2: agent B should see run 1's session ID for continuation, "
        f"got '{agent_b.observed_session_ids[1]}'"
    )

    # After run 2, sessions hold updated response IDs
    assert exec_a._session.service_session_id == "resp_A_2"  # pyright: ignore[reportPrivateUsage]
    assert exec_b._session.service_session_id == "resp_B_2"  # pyright: ignore[reportPrivateUsage]


class _ToolUsingSessionTrackingAgent(BaseAgent):
    """Agent that simulates tool calls and Responses API session tracking.

    Like _SessionTrackingAgent but responses include function_call content,
    mimicking an agent that invokes tools as part of its workflow.
    """

    def __init__(self, tool_name: str = "get_data", **kwargs: Any):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.observed_session_ids: list[str | None] = []

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        self.observed_session_ids.append(session.service_session_id if session else None)

        if session is not None:
            call_num = len(self.observed_session_ids)
            session.service_session_id = f"resp_{self.name}_{call_num}"

        call_num = len(self.observed_session_ids)
        call_id = f"call_{self.name}_{call_num}"

        async def _run() -> AgentResponse:
            return AgentResponse(
                messages=[
                    # Tool call message
                    Message(
                        "assistant",
                        [Content.from_function_call(call_id, self.tool_name, arguments={"query": "test"})],
                    ),
                    # Tool result message
                    Message(
                        "tool",
                        [Content.from_function_result(call_id, result=f"Result from {self.name} run {call_num}")],
                    ),
                    # Final text response
                    Message("assistant", [f"Based on {self.tool_name}: answer from {self.name}"]),
                ]
            )

        return _run()


@pytest.mark.asyncio
async def test_workflow_rerun_preserves_session_with_tool_calls() -> None:
    """Continuation works when agents make tool calls.

    Agents that invoke tools produce function_call and function_result
    content in their responses. The session should still carry over
    between runs, and each run's tool call IDs are distinct.
    """
    agent_a = _ToolUsingSessionTrackingAgent(id="planner", name="planner", tool_name="search")
    agent_b = _ToolUsingSessionTrackingAgent(id="executor", name="executor", tool_name="run_action")

    exec_a = AgentExecutor(agent_a)
    exec_b = AgentExecutor(agent_b)

    workflow = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()

    # --- First run ---
    result1 = await workflow.run("Plan a trip to Paris")
    assert result1 is not None

    assert agent_a.observed_session_ids[0] is None, "Run 1: planner starts fresh"
    assert agent_b.observed_session_ids[0] is None, "Run 1: executor starts fresh"

    assert exec_a._session.service_session_id == "resp_planner_1"  # pyright: ignore[reportPrivateUsage]
    assert exec_b._session.service_session_id == "resp_executor_1"  # pyright: ignore[reportPrivateUsage]

    # --- Second run (continuation with tool-using agents) ---
    result2 = await workflow.run("Now book the flights")
    assert result2 is not None

    # Both agents see the session from run 1 (continuation)
    assert agent_a.observed_session_ids[1] == "resp_planner_1", (
        "Run 2: planner should continue from run 1's session"
    )
    assert agent_b.observed_session_ids[1] == "resp_executor_1", (
        "Run 2: executor should continue from run 1's session"
    )

    # Sessions updated after run 2
    assert exec_a._session.service_session_id == "resp_planner_2"  # pyright: ignore[reportPrivateUsage]
    assert exec_b._session.service_session_id == "resp_executor_2"  # pyright: ignore[reportPrivateUsage]


class _InspectingToolAgent(BaseAgent):
    """Agent that records received messages and simulates tool use.

    Designed for integration tests: records the full message list received
    on each invocation so tests can verify the message chain flowing through
    the workflow engine across multiple runs.
    """

    def __init__(self, tool_name: str = "lookup", **kwargs: Any):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.received_messages: list[list[Message]] = []
        self.observed_session_ids: list[str | None] = []
        self._call_count = 0

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]]: ...
    @overload
    def run(
        self,
        messages: AgentRunInputs | None = ...,
        *,
        stream: Literal[True],
        session: AgentSession | None = ...,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        self._call_count += 1
        call = self._call_count

        # Record what we received
        msg_list = list(messages) if messages else []
        self.received_messages.append(msg_list)
        self.observed_session_ids.append(session.service_session_id if session else None)

        # Simulate Responses API updating service_session_id
        if session is not None:
            session.service_session_id = f"resp_{self.name}_{call}"

        call_id = f"call_{self.name}_{call}"

        async def _run() -> AgentResponse:
            return AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [Content.from_function_call(call_id, self.tool_name, arguments={"q": f"run{call}"})],
                    ),
                    Message(
                        "tool",
                        [Content.from_function_result(call_id, result=f"{self.tool_name}_result_{call}")],
                    ),
                    Message("assistant", [f"{self.name} final answer (run {call})"]),
                ]
            )

        return _run()


@pytest.mark.asyncio
async def test_two_agent_workflow_with_tools_runs_twice_integration() -> None:
    """Full integration test: 2-agent sequential workflow with tools, run twice.

    Exercises the real workflow engine end-to-end:
    - Message delivery through edges (str → from_str, AgentExecutorResponse → from_response)
    - Cache accumulation and clearing within _run_agent_and_emit
    - Session preservation across runs (continuation model)
    - Full conversation chain flowing from agent A to agent B
    - Workflow outputs accessible via get_outputs()
    """
    agent_a = _InspectingToolAgent(id="research", name="research", tool_name="web_search")
    agent_b = _InspectingToolAgent(id="writer", name="writer", tool_name="draft_doc")

    exec_a = AgentExecutor(agent_a)
    exec_b = AgentExecutor(agent_b)

    workflow = WorkflowBuilder(start_executor=exec_a).add_edge(exec_a, exec_b).build()

    # ===================== First run =====================
    result1 = await workflow.run("Research AI agents")
    assert result1 is not None
    assert result1.get_final_state() == WorkflowRunState.IDLE

    # Agent A received the user message
    assert len(agent_a.received_messages[0]) == 1
    assert agent_a.received_messages[0][0].role == "user"
    assert "Research AI agents" in (agent_a.received_messages[0][0].text or "")

    # Agent B received the full conversation (user msg + A's tool call + tool result + A's reply)
    b_msgs_run1 = agent_b.received_messages[0]
    assert len(b_msgs_run1) == 4, f"Expected 4 messages (user + 3 from A), got {len(b_msgs_run1)}"
    assert b_msgs_run1[0].role == "user"
    assert b_msgs_run1[1].role == "assistant"  # A's tool call
    assert b_msgs_run1[2].role == "tool"  # tool result
    assert b_msgs_run1[3].role == "assistant"  # A's final answer

    # Sessions: both started fresh
    assert agent_a.observed_session_ids[0] is None
    assert agent_b.observed_session_ids[0] is None

    # Outputs: both agents produced AgentResponse outputs
    outputs1 = result1.get_outputs()
    assert len(outputs1) == 2  # one per agent
    assert isinstance(outputs1[0], AgentResponse)
    assert isinstance(outputs1[1], AgentResponse)

    # ===================== Second run (continuation) =====================
    result2 = await workflow.run("Now summarize the findings")
    assert result2 is not None
    assert result2.get_final_state() == WorkflowRunState.IDLE

    # Agent A received the new user message (cache was cleared after run 1)
    assert len(agent_a.received_messages[1]) == 1
    assert "summarize the findings" in (agent_a.received_messages[1][0].text or "")

    # Agent B received the full conversation from this run
    b_msgs_run2 = agent_b.received_messages[1]
    assert len(b_msgs_run2) == 4, f"Expected 4 messages for run 2, got {len(b_msgs_run2)}"
    assert b_msgs_run2[0].role == "user"
    assert "summarize" in (b_msgs_run2[0].text or "")

    # Sessions: both agents see the session from run 1 (continuation)
    assert agent_a.observed_session_ids[1] == "resp_research_1", (
        f"Agent A should see run 1's session, got '{agent_a.observed_session_ids[1]}'"
    )
    assert agent_b.observed_session_ids[1] == "resp_writer_1", (
        f"Agent B should see run 1's session, got '{agent_b.observed_session_ids[1]}'"
    )

    # Sessions updated after run 2
    assert exec_a._session.service_session_id == "resp_research_2"  # pyright: ignore[reportPrivateUsage]
    assert exec_b._session.service_session_id == "resp_writer_2"  # pyright: ignore[reportPrivateUsage]

    # Outputs: second run also produced 2 outputs
    outputs2 = result2.get_outputs()
    assert len(outputs2) == 2
    # Verify output content is from run 2 (not stale run 1 data)
    assert "run 2" in outputs2[0].messages[-1].text  # type: ignore[operator]
    assert "run 2" in outputs2[1].messages[-1].text  # type: ignore[operator]

    # Both agents were called exactly twice total
    assert agent_a._call_count == 2
    assert agent_b._call_count == 2
