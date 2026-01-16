# Copyright (c) Microsoft. All rights reserved.

"""AgentTurnExecutor for running agent turns within the harness."""

import contextlib
import logging
from typing import Any

from .._agents import AgentProtocol, ChatAgent
from .._threads import AgentThread
from .._types import AgentRunResponse, AgentRunResponseUpdate
from .._workflows._const import WORKFLOW_RUN_KWARGS_KEY
from .._workflows._events import AgentRunEvent, AgentRunUpdateEvent
from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    HARNESS_CONTINUATION_COUNT_KEY,
    HARNESS_INITIAL_MESSAGE_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from ._done_tool import TASK_COMPLETE_TOOL_NAME
from ._state import HarnessEvent, HarnessLifecycleEvent, RepairComplete, TurnComplete

logger = logging.getLogger(__name__)


class AgentTurnExecutor(Executor):
    """Executor that runs a single agent turn with harness event tracking.

    This executor wraps an agent and executes it as a single turn within the
    harness loop. It tracks turn counts, records events to the harness transcript,
    and signals completion to the stop decision executor.
    """

    # Default continuation prompt - gentle nudge, not accusatory
    DEFAULT_CONTINUATION_PROMPT = (
        "Continue if there's more to do, or just say 'done' if finished."
    )

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        agent_thread: AgentThread | None = None,
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 2,
        continuation_prompt: str | None = None,
        id: str = "harness_agent_turn",
    ):
        """Initialize the AgentTurnExecutor.

        Args:
            agent: The agent to run for each turn.
            agent_thread: Optional thread to use for the agent. If None, creates a new thread.
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done signal.
            continuation_prompt: Custom continuation prompt text.
            id: Unique identifier for this executor.
        """
        super().__init__(id)
        self._agent = agent
        self._agent_thread = agent_thread or self._agent.get_new_thread()
        self._cache: list[Any] = []
        self._enable_continuation_prompts = enable_continuation_prompts
        self._max_continuation_prompts = max_continuation_prompts
        self._continuation_prompt = continuation_prompt or self.DEFAULT_CONTINUATION_PROMPT

    @handler
    async def run_turn(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
        """Execute a single agent turn.

        Args:
            trigger: The repair complete message indicating we can proceed.
            ctx: Workflow context for state access and message sending.
        """
        # 1. Increment turn count
        try:
            turn_count = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
        except KeyError:
            turn_count = 0
        turn_count += 1
        await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)

        # Get max_turns for lifecycle event
        try:
            max_turns = await ctx.get_shared_state(HARNESS_MAX_TURNS_KEY)
        except KeyError:
            max_turns = 50

        logger.info(f"AgentTurnExecutor: Starting turn {turn_count}")

        # Emit lifecycle event for DevUI
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="turn_started",
                turn_number=turn_count,
                max_turns=max_turns,
            )
        )

        # 1.5. On first turn of each run, inject the initial message into the cache
        # This accumulates history across multiple harness.run() calls
        if turn_count == 1:
            initial_message = await self._get_initial_message(ctx)
            if initial_message:
                # Log a preview of the message text
                preview = getattr(initial_message, "text", str(initial_message))[:100]
                logger.info(f"AgentTurnExecutor: Injecting initial message: {preview}...")
                self._cache.append(initial_message)

        # 2. Record turn start event
        await self._append_event(
            ctx,
            HarnessEvent(
                event_type="turn_start",
                data={"turn_number": turn_count, "repairs_made": trigger.repairs_made},
            ),
        )

        # 3. Run the agent
        try:
            if ctx.is_streaming():
                response = await self._run_agent_streaming(ctx)
            else:
                response = await self._run_agent(ctx)

            if response is None:
                # Agent did not complete (e.g., waiting for user input)
                logger.info(f"AgentTurnExecutor: Turn {turn_count} - Agent awaiting user input")
                await ctx.send_message(TurnComplete(agent_done=False))
                return

            # 4. Record agent response event
            await self._append_event(
                ctx,
                HarnessEvent(
                    event_type="agent_response",
                    data={
                        "turn_number": turn_count,
                        "message_count": len(response.messages),
                        "has_tool_calls": self._has_tool_calls(response),
                    },
                ),
            )

            # 5. Determine if agent is done
            # Priority: task_complete tool > no tool calls > continuation prompts
            has_tool_calls = self._has_tool_calls(response)
            called_task_complete = self._has_task_complete_call(response)

            if called_task_complete:
                # Agent explicitly signaled completion via task_complete tool
                agent_done = True
                logger.info(f"AgentTurnExecutor: Turn {turn_count} complete, agent called task_complete")
            elif has_tool_calls:
                # Agent is still working
                agent_done = False
                logger.info(f"AgentTurnExecutor: Turn {turn_count} complete, agent making tool calls")
            else:
                # Agent stopped making tool calls - check if we should prompt to continue
                continuation_count = await self._get_continuation_count(ctx)

                if self._enable_continuation_prompts and continuation_count < self._max_continuation_prompts:
                    # Inject continuation prompt to verify agent is truly done
                    logger.info(
                        f"AgentTurnExecutor: Turn {turn_count} - agent stopped, "
                        f"sending continuation prompt ({continuation_count + 1}/{self._max_continuation_prompts})"
                    )
                    await self._inject_continuation_prompt(ctx, continuation_count)
                    agent_done = False
                else:
                    # Accept the done signal
                    agent_done = True
                    logger.info(f"AgentTurnExecutor: Turn {turn_count} complete, agent_done=True")

            # 6. Emit lifecycle event and signal completion
            await ctx.add_event(
                HarnessLifecycleEvent(
                    event_type="turn_completed",
                    turn_number=turn_count,
                    max_turns=max_turns,
                    data={
                        "agent_done": agent_done,
                        "has_tool_calls": has_tool_calls,
                        "called_task_complete": called_task_complete,
                    },
                )
            )
            await ctx.send_message(TurnComplete(agent_done=agent_done))

        except Exception as e:
            logger.error(f"AgentTurnExecutor: Turn {turn_count} failed with error: {e}")
            await self._append_event(
                ctx,
                HarnessEvent(
                    event_type="agent_response",
                    data={
                        "turn_number": turn_count,
                        "error": str(e),
                    },
                ),
            )
            await ctx.send_message(TurnComplete(agent_done=False, error=str(e)))

    async def _run_agent(self, ctx: WorkflowContext[Any]) -> AgentRunResponse | None:
        """Execute the agent in non-streaming mode.

        Args:
            ctx: The workflow context.

        Returns:
            The agent response, or None if waiting for user input.
        """
        run_kwargs: dict[str, Any] = {}
        with contextlib.suppress(KeyError):
            run_kwargs = await ctx.get_shared_state(WORKFLOW_RUN_KWARGS_KEY)

        # Filter out 'thread' from run_kwargs since we manage our own thread
        # This prevents conflicts when DevUI passes its own thread
        run_kwargs = {k: v for k, v in run_kwargs.items() if k != "thread"}

        response = await self._agent.run(
            self._cache,
            thread=self._agent_thread,
            **run_kwargs,
        )
        await ctx.add_event(AgentRunEvent(self.id, response))

        # Update cache with response messages
        self._cache.extend(response.messages)

        return response

    async def _run_agent_streaming(self, ctx: WorkflowContext[Any]) -> AgentRunResponse | None:
        """Execute the agent in streaming mode.

        Args:
            ctx: The workflow context.

        Returns:
            The agent response, or None if waiting for user input.
        """
        run_kwargs: dict[str, Any] = {}
        with contextlib.suppress(KeyError):
            run_kwargs = await ctx.get_shared_state(WORKFLOW_RUN_KWARGS_KEY)

        # Filter out 'thread' from run_kwargs since we manage our own thread
        # This prevents conflicts when DevUI passes its own thread
        run_kwargs = {k: v for k, v in run_kwargs.items() if k != "thread"}

        updates: list[AgentRunResponseUpdate] = []
        async for update in self._agent.run_stream(
            self._cache,
            thread=self._agent_thread,
            **run_kwargs,
        ):
            updates.append(update)
            # Debug: log update contents
            update_text = getattr(update, 'text', None)
            update_contents = getattr(update, 'contents', [])
            logger.debug(f"AgentTurnExecutor: Streaming update - text={update_text[:50] if update_text else None}..., contents={[type(c).__name__ for c in (update_contents or [])]}")
            await ctx.add_event(AgentRunUpdateEvent(self.id, update))

        # Build the final response
        if isinstance(self._agent, ChatAgent):
            response_format = self._agent.chat_options.response_format
            response = AgentRunResponse.from_agent_run_response_updates(
                updates,
                output_format_type=response_format,
            )
        else:
            response = AgentRunResponse.from_agent_run_response_updates(updates)

        # Update cache with response messages
        self._cache.extend(response.messages)

        return response

    def _has_tool_calls(self, response: AgentRunResponse) -> bool:
        """Check if the response contains tool calls (excluding task_complete).

        Args:
            response: The agent response to check.

        Returns:
            True if the response contains pending tool calls (other than task_complete).
        """
        # Check for function approval requests (tool calls awaiting approval)
        if response.user_input_requests:
            return True

        # Check message contents for function calls (excluding task_complete)
        for message in response.messages:
            if hasattr(message, "contents") and message.contents:
                for content in message.contents:
                    if hasattr(content, "__class__") and "FunctionCall" in content.__class__.__name__:
                        tool_name = getattr(content, "name", None)
                        # Don't count task_complete as a tool call that keeps the agent running
                        if tool_name and tool_name != TASK_COMPLETE_TOOL_NAME:
                            return True

        return False

    def _has_task_complete_call(self, response: AgentRunResponse) -> bool:
        """Check if the response contains a task_complete tool call.

        Args:
            response: The agent response to check.

        Returns:
            True if the agent called the task_complete tool.
        """
        for message in response.messages:
            if hasattr(message, "contents") and message.contents:
                for content in message.contents:
                    if hasattr(content, "__class__") and "FunctionCall" in content.__class__.__name__:
                        tool_name = getattr(content, "name", None)
                        if tool_name == TASK_COMPLETE_TOOL_NAME:
                            return True
        return False

    async def _get_continuation_count(self, ctx: WorkflowContext[Any]) -> int:
        """Get the current continuation prompt count.

        Args:
            ctx: Workflow context for state access.

        Returns:
            The number of continuation prompts sent so far.
        """
        try:
            count = await ctx.get_shared_state(HARNESS_CONTINUATION_COUNT_KEY)
            return int(count) if count else 0
        except KeyError:
            return 0

    async def _inject_continuation_prompt(self, ctx: WorkflowContext[Any], current_count: int) -> None:
        """Inject a continuation prompt into the conversation.

        Args:
            ctx: Workflow context for state access.
            current_count: Current continuation count.
        """
        from .._types import ChatMessage

        # Add continuation prompt to cache
        continuation_msg = ChatMessage(role="user", text=self._continuation_prompt)
        self._cache.append(continuation_msg)

        # Update continuation count
        await ctx.set_shared_state(HARNESS_CONTINUATION_COUNT_KEY, current_count + 1)

        # Record event in transcript
        await self._append_event(
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

        # Emit lifecycle event for DevUI
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="continuation_prompt",
                data={
                    "count": current_count + 1,
                    "max": self._max_continuation_prompts,
                },
            )
        )

    async def _get_initial_message(self, ctx: WorkflowContext[Any]) -> Any:
        """Get the initial message from shared state and convert to chat message.

        Args:
            ctx: Workflow context for state access.

        Returns:
            A chat message object, or the raw message if conversion fails.
        """
        try:
            message = await ctx.get_shared_state(HARNESS_INITIAL_MESSAGE_KEY)
            if not message:
                return None

            # If it's already a message object, return as-is
            if not isinstance(message, str):
                return message

            # Convert string to ChatMessage with user role
            from .._types import ChatMessage

            return ChatMessage(role="user", text=message)
        except KeyError:
            return None

    async def _append_event(self, ctx: WorkflowContext[Any], event: HarnessEvent) -> None:
        """Append an event to the transcript.

        Args:
            ctx: Workflow context for state access.
            event: The event to append.
        """
        transcript: list[dict[str, Any]] = []
        try:
            stored = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            if stored is not None:
                transcript = list(stored)
        except KeyError:
            pass

        transcript.append(event.to_dict())
        await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, transcript)

    async def on_checkpoint_save(self) -> dict[str, Any]:
        """Save executor state for checkpointing.

        Returns:
            State dictionary for restoration.
        """
        from .._workflows._conversation_state import encode_chat_messages

        serialized_thread = await self._agent_thread.serialize()
        return {
            "cache": encode_chat_messages(self._cache),
            "agent_thread": serialized_thread,
        }

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        """Restore executor state from checkpoint.

        Args:
            state: The saved state dictionary.
        """
        from .._workflows._conversation_state import decode_chat_messages

        cache_payload = state.get("cache")
        if cache_payload:
            try:
                self._cache = decode_chat_messages(cache_payload)
            except Exception as exc:
                logger.warning(f"Failed to restore cache: {exc}")
                self._cache = []
        else:
            self._cache = []

        thread_payload = state.get("agent_thread")
        if thread_payload:
            try:
                self._agent_thread = await AgentThread.deserialize(thread_payload)
            except Exception as exc:
                logger.warning(f"Failed to restore agent thread: {exc}")
                self._agent_thread = self._agent.get_new_thread()

    def set_initial_messages(self, messages: list[Any]) -> None:
        """Set initial messages for the agent conversation.

        Args:
            messages: The initial messages to seed the conversation with.
        """
        self._cache = list(messages)
