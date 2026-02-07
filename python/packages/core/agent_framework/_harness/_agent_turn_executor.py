# Copyright (c) Microsoft. All rights reserved.

"""AgentTurnExecutor for running agent turns within the harness."""

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

from .._agents import AgentProtocol, ChatAgent
from .._threads import AgentThread
from .._types import AgentRunResponse, AgentRunResponseUpdate
from .._workflows._const import WORKFLOW_RUN_KWARGS_KEY
from .._workflows._events import AgentRunEvent, AgentRunUpdateEvent
from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    HARNESS_COMPACTION_PLAN_KEY,
    HARNESS_CONTINUATION_COUNT_KEY,
    HARNESS_INITIAL_MESSAGE_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
    HARNESS_WORK_ITEM_LEDGER_KEY,
)
from ._done_tool import TASK_COMPLETE_TOOL_NAME, task_complete
from ._state import HarnessEvent, HarnessLifecycleEvent, RepairComplete, TurnComplete

if TYPE_CHECKING:
    from ._compaction import CompactionPlan
    from ._hooks import HarnessHooks
    from ._work_items import WorkItemEventMiddleware, WorkItemLedger, WorkItemTaskListProtocol

logger = logging.getLogger(__name__)


class AgentTurnExecutor(Executor):
    """Executor that runs a single agent turn with harness event tracking.

    This executor wraps an agent and executes it as a single turn within the
    harness loop. It tracks turn counts, records events to the harness transcript,
    and signals completion to the stop decision executor.

    Context Compaction Integration:
    When a CompactionPlan is present in SharedState, this executor applies it
    to the message cache before passing to the agent. The full cache is preserved
    for durability, while the agent sees a compacted view. This supports the
    Plan Pipeline architecture where:
    - CompactionPlan flows through SharedState (small, serializable)
    - Message cache is local to this executor
    - Storage protocols are injectable for different environments
    """

    # Default continuation prompt - assertive nudge requiring task_complete
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

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        agent_thread: AgentThread | None = None,
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 5,
        continuation_prompt: str | None = None,
        enable_compaction: bool = False,
        task_list: "WorkItemTaskListProtocol | None" = None,
        hooks: "HarnessHooks | None" = None,
        sub_agent_tools: list[Any] | None = None,
        id: str = "harness_agent_turn",
    ):
        """Initialize the AgentTurnExecutor.

        Args:
            agent: The agent to run for each turn.
            agent_thread: Optional thread to use for the agent. If None, creates a new thread.
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done signal.
            continuation_prompt: Custom continuation prompt text.
            enable_compaction: Whether to apply CompactionPlan from SharedState.
            task_list: Optional work item task list for self-critique tracking.
            hooks: Optional harness hooks for pre/post tool interception.
            sub_agent_tools: Optional list of sub-agent tools (explore, run_task) to inject.
            id: Unique identifier for this executor.
        """
        super().__init__(id)
        self._agent = agent
        self._agent_thread = agent_thread or self._agent.get_new_thread()
        self._cache: list[Any] = []
        self._enable_continuation_prompts = enable_continuation_prompts
        self._max_continuation_prompts = max_continuation_prompts
        self._continuation_prompt = continuation_prompt or self.DEFAULT_CONTINUATION_PROMPT
        self._enable_compaction = enable_compaction
        self._task_list = task_list

        # Initialize tokenizer for cache token counting when compaction is enabled
        self._tokenizer: "ProviderAwareTokenizer | None" = None
        if enable_compaction:
            from ._compaction._tokenizer import ProviderAwareTokenizer, get_tokenizer

            self._tokenizer = get_tokenizer()

        # Create work item event middleware if task_list is provided
        self._work_item_middleware: "WorkItemEventMiddleware | None" = None
        if task_list is not None:
            from ._work_items import WorkItemEventMiddleware

            self._work_item_middleware = WorkItemEventMiddleware(task_list.ledger)

        # Create harness tool middleware if hooks have pre/post tool callbacks
        from ._hooks import HarnessToolMiddleware

        self._harness_tool_middleware: HarnessToolMiddleware | None = None
        if hooks and (hooks.pre_tool or hooks.post_tool):
            self._harness_tool_middleware = HarnessToolMiddleware(hooks)

        # Phase 5: Sub-agent tools
        self._sub_agent_tools = sub_agent_tools or []

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

            # Note: Tool strategy, work item, and planning guidance are also injected
            # via HarnessGuidanceProvider as system prompt content (Phase 2).
            # First-turn user messages provide the immediate planning kick.
            if self._task_list is not None:
                self._inject_work_item_guidance()
                self._inject_tool_strategy_guidance()
                self._inject_planning_prompt()

        # 1.6. On turns after the first, inject current work item state
        if turn_count > 1 and self._task_list is not None:
            await self._inject_work_item_state(ctx)

        # 2. Inject work item reminder if last stop was work_items_incomplete
        if self._task_list is not None:
            await self._maybe_inject_work_item_reminder(ctx)

        # 3. Apply compaction if needed — prefer full pipeline, fall back to direct clear
        if self._enable_compaction and self._is_compaction_needed(trigger):
            compacted = await self._run_full_compaction(ctx, turn_count)
            if compacted:
                logger.info("AgentTurnExecutor: Full compaction pipeline applied successfully")
            else:
                # Fall back to direct clearing
                cleared_count = self._apply_direct_clear(turn_count)
                if cleared_count > 0:
                    logger.info(
                        "AgentTurnExecutor: Cleared %d old tool results to reduce context pressure",
                        cleared_count,
                    )

        # 4. Record turn start event
        await self._append_event(
            ctx,
            HarnessEvent(
                event_type="turn_start",
                data={"turn_number": turn_count, "repairs_made": trigger.repairs_made},
            ),
        )

        # 5. Run the agent
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

            # 5. Sync work item ledger to SharedState
            if self._task_list is not None:
                await self._sync_work_item_ledger(ctx)

            # 6. Determine if agent is done
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

            # 6. Update token budget in SharedState so CompactionExecutor has accurate counts
            await self._update_token_budget(ctx, response)

            # 7. Emit lifecycle event and signal completion
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
            await ctx.send_message(TurnComplete(agent_done=agent_done, called_task_complete=called_task_complete))

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

        # Inject work item tools and middleware if task_list is set
        if self._task_list is not None:
            run_kwargs["tools"] = self._task_list.get_tools()
            run_kwargs["tools"].append(task_complete)
        else:
            # Even without work items, inject task_complete for stop control
            run_kwargs.setdefault("tools", [])
            if isinstance(run_kwargs["tools"], list):
                run_kwargs["tools"].append(task_complete)

        # Inject sub-agent tools
        if self._sub_agent_tools:
            existing_tools = run_kwargs.get("tools", [])
            if not isinstance(existing_tools, list):
                existing_tools = list(existing_tools)
            existing_tools.extend(self._sub_agent_tools)
            run_kwargs["tools"] = existing_tools

        # Compose function middlewares (work item + harness tool hooks)
        middlewares = self._collect_middlewares()
        if middlewares:
            run_kwargs["middleware"] = middlewares if len(middlewares) > 1 else middlewares[0]

        # Apply compaction if enabled and plan exists
        messages_to_send = await self._get_messages_for_agent(ctx)

        response = await self._agent.run(
            messages_to_send,
            thread=self._agent_thread,
            **run_kwargs,
        )
        await ctx.add_event(AgentRunEvent(self.id, response))

        # Emit work item change events (real-time updates)
        await self._emit_work_item_events(ctx)

        # Update cache with response messages (full cache, not compacted)
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

        # Inject work item tools and middleware if task_list is set
        if self._task_list is not None:
            run_kwargs["tools"] = self._task_list.get_tools()
            run_kwargs["tools"].append(task_complete)
        else:
            # Even without work items, inject task_complete for stop control
            run_kwargs.setdefault("tools", [])
            if isinstance(run_kwargs["tools"], list):
                run_kwargs["tools"].append(task_complete)

        # Inject sub-agent tools
        if self._sub_agent_tools:
            existing_tools = run_kwargs.get("tools", [])
            if not isinstance(existing_tools, list):
                existing_tools = list(existing_tools)
            existing_tools.extend(self._sub_agent_tools)
            run_kwargs["tools"] = existing_tools

        # Compose function middlewares (work item + harness tool hooks)
        middlewares = self._collect_middlewares()
        if middlewares:
            run_kwargs["middleware"] = middlewares if len(middlewares) > 1 else middlewares[0]

        # Apply compaction if enabled and plan exists
        messages_to_send = await self._get_messages_for_agent(ctx)

        updates: list[AgentRunResponseUpdate] = []
        async for update in self._agent.run_stream(
            messages_to_send,
            thread=self._agent_thread,
            **run_kwargs,
        ):
            updates.append(update)
            # Debug: log update contents
            update_text = getattr(update, "text", None)
            update_contents = getattr(update, "contents", [])
            content_types = [type(c).__name__ for c in (update_contents or [])]
            text_preview = update_text[:50] if update_text else None
            logger.debug(
                "AgentTurnExecutor: Streaming update - text=%s..., contents=%s",
                text_preview,
                content_types,
            )
            await ctx.add_event(AgentRunUpdateEvent(self.id, update))

        # Emit work item change events (real-time updates)
        await self._emit_work_item_events(ctx)

        # Build the final response
        if isinstance(self._agent, ChatAgent):
            response_format = self._agent.chat_options.response_format
            response = AgentRunResponse.from_agent_run_response_updates(
                updates,
                output_format_type=response_format,
            )
        else:
            response = AgentRunResponse.from_agent_run_response_updates(updates)

        # Update cache with response messages (full cache, not compacted)
        self._cache.extend(response.messages)

        return response

    def _collect_middlewares(self) -> list[Any]:
        """Collect all function middlewares to inject into agent.run().

        Returns:
            List of FunctionMiddleware instances (may be empty).
        """
        middlewares: list[Any] = []
        if self._work_item_middleware is not None:
            middlewares.append(self._work_item_middleware)
        if self._harness_tool_middleware is not None:
            middlewares.append(self._harness_tool_middleware)
        return middlewares

    async def _get_messages_for_agent(self, ctx: WorkflowContext[Any]) -> list[Any]:
        """Get messages to send to agent, applying compaction if enabled.

        Args:
            ctx: The workflow context.

        Returns:
            Messages to send - either the full cache or a compacted view.
        """
        if not self._enable_compaction:
            return self._cache

        # Load compaction plan from SharedState
        plan = await self._load_compaction_plan(ctx)
        if plan is None or plan.is_empty:
            return self._cache

        # Apply compaction to cache
        return self._apply_compaction_plan(plan)

    async def _load_compaction_plan(self, ctx: WorkflowContext[Any]) -> "CompactionPlan | None":
        """Load the compaction plan from SharedState.

        Args:
            ctx: The workflow context.

        Returns:
            The CompactionPlan if found, None otherwise.
        """
        try:
            plan_data = await ctx.get_shared_state(HARNESS_COMPACTION_PLAN_KEY)
            if plan_data and isinstance(plan_data, dict):
                from ._compaction import CompactionPlan

                return CompactionPlan.from_dict(plan_data)
        except KeyError:
            pass
        return None

    def _apply_compaction_plan(self, plan: "CompactionPlan") -> list[Any]:
        """Apply compaction plan to the message cache.

        This produces a compacted view of the cache for the agent while
        preserving the original cache for durability.

        Args:
            plan: The compaction plan to apply.

        Returns:
            Compacted message list.
        """
        from ._compaction import CompactionAction

        compacted: list[Any] = []
        processed_spans: set[str] = set()

        for msg in self._cache:
            # Get message ID - include as-is if no ID
            msg_id = getattr(msg, "message_id", None)
            if msg_id is None:
                compacted.append(msg)
                continue

            # Look up action in plan
            action, record = plan.get_action(msg_id)

            # Apply action
            if action == CompactionAction.DROP:
                continue  # Skip this message

            if action == CompactionAction.INCLUDE:
                compacted.append(msg)
                continue

            if action == CompactionAction.CLEAR:
                compacted.append(self._render_cleared_message(msg, record))
                continue

            # For SUMMARIZE and EXTERNALIZE, only render once per span
            if record is None:
                continue

            span_key = record.span.start_message_id
            if span_key in processed_spans:
                continue
            processed_spans.add(span_key)

            if action == CompactionAction.SUMMARIZE:
                compacted.append(self._render_summary_message(record))

            elif action == CompactionAction.EXTERNALIZE:
                compacted.append(self._render_externalization_message(record))

        return compacted

    def _render_cleared_message(self, original_msg: Any, record: Any) -> Any:
        """Render a cleared message as a placeholder.

        Args:
            original_msg: The original message.
            record: The clear record.

        Returns:
            A placeholder message.
        """
        from .._types import ChatMessage

        role_attr = getattr(original_msg, "role", "assistant")
        # Handle both enum-style roles and string roles
        role_value: str = str(getattr(role_attr, "value", role_attr))

        # Build placeholder
        parts = [f"[Cleared: {role_value} message]"]
        if record is not None and hasattr(record, "preserved_fields") and record.preserved_fields:
            fields = ", ".join(f"{k}={v}" for k, v in sorted(record.preserved_fields.items()))
            parts.append(f"Key data: {fields}")

        return ChatMessage(  # type: ignore[call-overload]
            role=role_value,  # type: ignore[arg-type]
            text="\n".join(parts),
            message_id=getattr(original_msg, "message_id", None),
        )

    def _render_summary_message(self, record: Any) -> Any:
        """Render a summary as an assistant message.

        Args:
            record: The summarization record.

        Returns:
            A summary message.
        """
        from .._types import ChatMessage

        span = record.span
        summary_text = record.summary.render_as_message()

        content = f"[Context Summary - Turns {span.first_turn}-{span.last_turn}]\n{summary_text}"

        return ChatMessage(
            role="assistant",
            text=content,
            message_id=f"summary-{span.start_message_id}",
        )

    def _render_externalization_message(self, record: Any) -> Any:
        """Render an externalization pointer as an assistant message.

        Args:
            record: The externalization record.

        Returns:
            An externalization pointer message.
        """
        from .._types import ChatMessage

        span = record.span
        summary_text = record.summary.render_as_message()

        content = (
            f"[Externalized Content - artifact:{record.artifact_id}]\n"
            f"Summary: {summary_text}\n"
            f'To retrieve full content, call: read_artifact("{record.artifact_id}")'
        )

        return ChatMessage(
            role="assistant",
            text=content,
            message_id=f"external-{span.start_message_id}",
        )

    @staticmethod
    def _is_compaction_needed(trigger: RepairComplete) -> bool:
        """Check if the trigger signals that compaction is needed.

        Args:
            trigger: The trigger message (may be CompactionComplete with compaction_needed flag).

        Returns:
            True if compaction should be applied to the cache.
        """
        from ._compaction_executor import CompactionComplete

        return isinstance(trigger, CompactionComplete) and trigger.compaction_needed

    def _apply_direct_clear(self, current_turn: int, preserve_recent_turns: int = 2) -> int:
        """Clear old tool results in the cache to reduce context pressure.

        Replaces FunctionResultContent in older messages with short placeholders.
        This is a lightweight version of ClearStrategy that operates directly on
        the in-memory cache without needing an AgentThread.

        Args:
            current_turn: The current turn number.
            preserve_recent_turns: Number of recent cache entries to keep intact.

        Returns:
            Number of tool results that were cleared.
        """
        from .._types import ChatMessage, FunctionResultContent

        if len(self._cache) <= preserve_recent_turns:
            return 0

        cleared_count = 0
        cutoff = len(self._cache) - preserve_recent_turns

        for i in range(cutoff):
            msg = self._cache[i]
            contents = getattr(msg, "contents", None)
            if not contents:
                continue

            new_contents: list[Any] = []
            modified = False
            for content in contents:
                if isinstance(content, FunctionResultContent):
                    result_str = str(content.result or "")
                    # Only clear results that are large enough to matter (>100 chars)
                    if len(result_str) > 100:
                        content = FunctionResultContent(
                            call_id=content.call_id,
                            result="[Tool result cleared to save context]",
                        )
                        modified = True
                        cleared_count += 1
                new_contents.append(content)

            if modified:
                # Replace the message in cache with updated contents
                role = getattr(msg, "role", "tool")
                role_value: str = str(getattr(role, "value", role))
                self._cache[i] = ChatMessage(
                    role=role_value,  # type: ignore[arg-type]
                    contents=new_contents,
                    message_id=getattr(msg, "message_id", None),
                )

        return cleared_count

    async def _run_full_compaction(self, ctx: WorkflowContext[Any], turn_count: int) -> bool:
        """Run the full compaction pipeline via CompactionCoordinator.

        Bridges the cache to the compaction strategies using CacheThreadAdapter,
        then runs ClearStrategy + DropStrategy to produce a CompactionPlan that
        is stored in SharedState for _get_messages_for_agent to apply.

        Args:
            ctx: Workflow context for state access.
            turn_count: Current turn number.

        Returns:
            True if compaction was applied, False if no compaction was needed or possible.
        """
        from ._compaction import (
            CacheThreadAdapter,
            ClearStrategy,
            CompactionCoordinator,
            DropStrategy,
            TurnContext,
        )
        from ._compaction import TokenBudget as TokenBudgetV2

        if self._tokenizer is None:
            return False

        # Build a thread adapter from cache
        thread_adapter = CacheThreadAdapter(self._cache)

        # Load current plan from SharedState
        plan = await self._load_compaction_plan(ctx)

        # Create v2 budget for compaction internals
        budget_v2 = TokenBudgetV2()

        # Estimate current tokens from cache
        current_tokens = 0
        for msg in self._cache:
            text = getattr(msg, "text", None) or ""
            current_tokens += self._tokenizer.count_tokens(str(text))

        tokens_over = budget_v2.tokens_over_threshold(current_tokens)
        if tokens_over <= 0:
            logger.debug(
                "AgentTurnExecutor: Full compaction skipped — under threshold (%d tokens)",
                current_tokens,
            )
            return False

        tokens_to_free = tokens_over + int(budget_v2.max_input_tokens * 0.1)

        # Run coordinator with ClearStrategy + DropStrategy
        turn_context = TurnContext(turn_number=turn_count)
        coordinator = CompactionCoordinator(strategies=[ClearStrategy(), DropStrategy()])

        try:
            result = await coordinator.compact(
                thread_adapter,  # type: ignore[arg-type]
                plan,
                budget_v2,
                self._tokenizer,
                tokens_to_free=tokens_to_free,
                turn_context=turn_context,
            )
        except Exception:
            logger.warning("AgentTurnExecutor: Full compaction failed", exc_info=True)
            return False

        # Store updated plan if it has content
        if result.plan and not result.plan.is_empty:
            await ctx.set_shared_state(HARNESS_COMPACTION_PLAN_KEY, result.plan.to_dict())
            logger.info(
                "AgentTurnExecutor: Compaction complete — freed ~%d tokens, applied %d/%d proposals",
                result.tokens_freed,
                result.proposals_applied,
                result.proposals_generated,
            )
            return True

        return False

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

    async def _update_token_budget(self, ctx: WorkflowContext[Any], response: "AgentRunResponse | None" = None) -> None:
        """Update the token budget in SharedState so CompactionExecutor has accurate counts.

        Uses the API-reported input token count (from response.usage_details) as the
        primary source — this is the authoritative count including system prompt, tool
        schemas, and all formatting overhead. Falls back to content-level counting with
        the tokenizer when usage_details is not available.

        Args:
            ctx: Workflow context for state access.
            response: The agent response (may contain usage_details with input_token_count).
        """
        from .._types import FunctionCallContent, FunctionResultContent, TextContent
        from ._context_pressure import TokenBudget

        current_tokens = 0
        counting_method = "none"

        # Primary: use API-reported input token count (includes system prompt, tool schemas, formatting)
        if response and response.usage_details and response.usage_details.input_token_count:
            current_tokens = response.usage_details.input_token_count
            counting_method = "api_usage"
        elif self._tokenizer is not None:
            # Fallback: count tokens from cache contents
            for msg in self._cache:
                contents = getattr(msg, "contents", None)
                if not contents:
                    continue
                for content in contents:
                    if isinstance(content, TextContent):
                        current_tokens += self._tokenizer.count_tokens(content.text or "")
                    elif isinstance(content, FunctionCallContent):
                        current_tokens += self._tokenizer.count_tokens(content.name or "")
                        args = content.arguments
                        if isinstance(args, dict):
                            import json

                            args = json.dumps(args)
                        current_tokens += self._tokenizer.count_tokens(str(args or ""))
                    elif isinstance(content, FunctionResultContent):
                        current_tokens += self._tokenizer.count_tokens(str(content.result or ""))
                    else:
                        current_tokens += self._tokenizer.count_tokens(str(content))
                # Per-message overhead (role, formatting)
                current_tokens += 4
            counting_method = "tokenizer"
        else:
            return  # No way to count tokens

        # Load or create the budget and update its estimate
        try:
            budget_data = await ctx.get_shared_state(HARNESS_TOKEN_BUDGET_KEY)
            if budget_data and isinstance(budget_data, dict):
                budget = TokenBudget.from_dict(cast("dict[str, Any]", budget_data))
            else:
                budget = TokenBudget()
        except KeyError:
            budget = TokenBudget()

        budget.current_estimate = current_tokens
        await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())
        logger.info(
            "AgentTurnExecutor: Updated token budget — %d tokens (method: %s, threshold: %d)",
            current_tokens,
            counting_method,
            budget.soft_threshold,
        )

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

    # Default guidance message for work item tools
    WORK_ITEM_GUIDANCE = (
        "You have access to work item tracking tools for planning, self-review, "
        "and quality control:\n\n"
        "PLANNING & PROGRESS:\n"
        "- work_item_add(title, priority?, notes?): Plan subtasks for this request\n"
        "- work_item_update(item_id, status, notes?): Track progress (done/skipped)\n"
        "- work_item_list(filter_status?): Review your progress before finishing\n\n"
        "ARTIFACT & REVISION PROTOCOL:\n"
        "- work_item_set_artifact(item_id, artifact): Store the output of a completed step\n"
        "- work_item_flag_revision(item_id, reason): Flag a prior step's output for correction\n\n"
        "WORKFLOW:\n"
        "1. Plan: Add work items for each distinct step\n"
        "2. Execute: Work through items one at a time. After storing an artifact for a step, "
        "immediately mark it done with work_item_update. Do NOT batch status updates.\n"
        "3. Audit: When reviewing prior work, if you find issues, call flag_revision to "
        "create a mandatory correction item\n"
        "4. Revise: Complete any revision items with corrected artifacts\n"
        "5. Verify: Call work_item_list to confirm all items are done before signaling completion\n\n"
        "IMPORTANT: Mark each item done AS SOON AS you complete it and store its artifact. "
        "All items (including revisions) must be done or skipped before the task "
        "can finish. Audit steps that find issues MUST call flag_revision - "
        "noting problems in prose is not sufficient.\n\n"
        "ARTIFACT CONTENT RULES:\n"
        "Artifacts are pure deliverables. Your response text is the narrative channel.\n"
        "- GOOD: The structured content itself (text, JSON, markdown, code)\n"
        "- BAD: 'I will now write...\\n## Timeline\\n...\\nI have stored this.'\n"
        "- Never start with 'I will...', 'Let me...', 'Here is...'\n"
        "- Never end with 'I have stored...', 'Moving on...', 'Next I will...'\n"
        "- Never reference work items or task management within artifact content\n"
        "- Process commentary belongs in response text BEFORE/AFTER the tool call\n\n"
        "ARTIFACT ROLES:\n"
        "When adding work items, you MUST classify their role correctly:\n"
        "- deliverable: Any output the user asked for or will receive. Reports, documents,\n"
        "  analyses, plans, summaries, code — if it answers the user's request, it is a deliverable.\n"
        "  ALWAYS store deliverable content via work_item_set_artifact so it persists in state.\n"
        "  Only ALSO write it as a file (write_file) if the user explicitly asked for a file,\n"
        "  or the content is very large (5KB+). Otherwise, present the deliverable inline in\n"
        "  your response text — users prefer seeing results directly.\n"
        "- working: Your own scratch notes, intermediate reasoning, or data you won't show the user.\n"
        "- control: Validation checks and audit results (must use structured JSON format below).\n"
        "IMPORTANT: Most work items in a typical task should be 'deliverable'. Only use 'working'\n"
        "for items that are purely internal to your process and not part of the user's answer.\n\n"
        "CONTROL ARTIFACT FORMAT (REQUIRED):\n"
        "When storing a control-role artifact, it MUST be valid JSON with this structure:\n"
        '{"verdict": "pass" or "fail",\n'
        ' "checks": [{"name": "what was checked", "result": "pass" or "fail", "detail": "evidence"}],\n'
        ' "summary": "overall reasoning for the verdict"}\n\n'
        "If your verdict is 'fail', you MUST call flag_revision on items whose checks failed.\n"
        "A pass verdict must still include the checks showing what was validated.\n\n"
        "ANTI-DOUBLE-EMISSION (CRITICAL):\n"
        "Your response text is a brief narrative of progress. Artifact content goes ONLY in tool calls.\n"
        "- NEVER reproduce artifact content in your response text\n"
        "- NEVER mention tool names, artifacts, work items, or internal mechanics in response text\n"
        "- NEVER say 'I stored/saved this as an artifact' or 'I will call work_item_set_artifact'\n"
        "- DO describe what you are working on at a high level\n"
        "- DO say what you produced and what you will do next\n\n"
        "CORRECT response style:\n"
        "  'I have completed the incident timeline covering 10:00-10:47 AM across 3 systems. "
        "Now I will work on the stakeholder impact analysis.'\n\n"
        "WRONG response style:\n"
        "  '### Timeline\\n10:00 AM - Deploy...\\n10:47 AM - Recovery...\\n"
        "I will store these findings as an artifact for this step.'"
    )

    def _inject_work_item_guidance(self) -> None:
        """Inject guidance about work item tools on the first turn."""
        from .._types import ChatMessage

        guidance_msg = ChatMessage(role="user", text=self.WORK_ITEM_GUIDANCE)
        self._cache.append(guidance_msg)
        logger.info("AgentTurnExecutor: Injected work item guidance message")

    TOOL_STRATEGY_GUIDANCE = (
        "TOOL STRATEGY GUIDE:\n"
        "Use these patterns for effective investigation and task execution.\n\n"
        "DISCOVERY — map the codebase before reading anything:\n"
        "- find_files('**/*.py') to get a complete map of all source files regardless of depth\n"
        "- grep_files(pattern, file_glob='*.py') to find files containing specific terms\n"
        "- list_directory('.', depth=2) for a quick top-level overview\n"
        "- IMPORTANT: When grep_files returns matches in many different packages or modules,\n"
        "  look at the file PATHS to identify which directory is the dedicated implementation\n"
        "  vs which files merely reference or import it. A directory named after what you're\n"
        "  looking for (e.g., '_workflows/') is more likely the core implementation than\n"
        "  scattered references in other packages.\n\n"
        "PROGRESSIVE EXPLORATION — go broad then deep:\n"
        "1. Start with find_files to get the full file map — this reveals deeply nested\n"
        "   directories that list_directory might miss (it only shows 2 levels)\n"
        "2. Use grep_files to narrow down which files contain the core logic you need\n"
        "3. When grep returns results in multiple packages, focus on the package/directory\n"
        "   that has the MOST matches and the most specific file names\n"
        "4. Drill into the target directory with list_directory to see its full contents\n"
        "5. read_file on each relevant file — use line ranges for large files\n\n"
        "THOROUGH READING — be context-efficient:\n"
        "- Your context window is a limited resource. Every line you read stays in memory\n"
        "  and competes for space. Read what you NEED, not everything that EXISTS.\n"
        "- For large files (200+ lines): first scan the structure with\n"
        "  read_file(path, start_line=1, end_line=50) to see imports and class definitions,\n"
        "  then read specific sections of interest. Do NOT read the entire file.\n"
        "- For small files (<100 lines): reading the whole file is fine.\n"
        "- Use grep_files to find specific classes or functions across multiple files —\n"
        "  this tells you WHERE to look without loading everything into context.\n"
        "- When investigating a module with many files, scan each file's first ~50 lines\n"
        "  to understand its purpose, then deep-read only the files that matter for\n"
        "  your task.\n"
        "- SKIP: test files, boilerplate, generated code, and files unrelated to your task.\n"
        "  Not every file in a directory deserves to be read.\n\n"
        "CONTEXT EFFICIENCY — keep your working memory lean:\n"
        "- Prefer grep_files over reading entire files to find what you need.\n"
        "- Prefer read_file with line ranges over reading whole files.\n"
        "- After reading a section, decide if you need more of that file or can move on.\n"
        "- If you've already found the information you need, stop reading more files.\n"
        "- A good investigation reads many files partially rather than few files fully.\n\n"
        "DELIVERABLE CREATION — only after thorough investigation:\n"
        "- Do NOT write deliverables until you have read all relevant source files.\n"
        "- ALWAYS store deliverable content via work_item_set_artifact for persistence.\n"
        "- Write to a file (write_file) ONLY when the user asked for a file, or the content\n"
        "  is large (5KB+). Otherwise, present it inline in your response.\n"
        "- Reference specific class names, method signatures, and module paths you found by reading.\n"
        "- If you haven't read_file'd a file, do not write about its contents."
    )

    def _inject_tool_strategy_guidance(self) -> None:
        """Inject guidance about effective tool usage patterns on the first turn."""
        from .._types import ChatMessage

        guidance_msg = ChatMessage(role="user", text=self.TOOL_STRATEGY_GUIDANCE)
        self._cache.append(guidance_msg)
        logger.info("AgentTurnExecutor: Injected tool strategy guidance")

    PLANNING_PROMPT = (
        "Assess the user's request before taking action:\n"
        "- If the request is a KNOWLEDGE QUESTION (e.g., 'explain X', 'tell me about Y',\n"
        "  'what is Z') that you can answer from your training knowledge WITHOUT needing\n"
        "  to read files, just answer it directly. No planning or tool use needed.\n"
        "- If the request requires INVESTIGATING the workspace, READING files, or MAKING\n"
        "  changes, create a detailed plan using work_item_add for each step:\n"
        "  - BAD work item: 'Analyze the core modules' (vague, unverifiable)\n"
        "  - GOOD work item: 'Read and analyze each file in the target directory' "
        "(specific, measurable)\n"
        "  - Each work item should involve concrete tool usage (file reads, directory "
        "listings, command execution).\n"
        "  - Do not mark a work item as done until you have actually used tools to "
        "complete it and stored meaningful artifacts from the work."
    )

    def _inject_planning_prompt(self) -> None:
        """Inject a prompt asking the agent to plan before executing."""
        from .._types import ChatMessage

        planning_msg = ChatMessage(role="user", text=self.PLANNING_PROMPT)
        self._cache.append(planning_msg)
        logger.info("AgentTurnExecutor: Injected planning prompt")

    async def _inject_work_item_state(self, ctx: WorkflowContext[Any]) -> None:
        """Inject current work item state so the agent sees its progress.

        Called on turns after the first to give the agent visibility into
        which items are done and which remain. Only injects if there are
        incomplete items to avoid redundant messages.

        Args:
            ctx: Workflow context for state access.
        """
        from .._types import ChatMessage
        from ._work_items import WorkItemLedger, WorkItemStatus

        try:
            ledger_data = await ctx.get_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY)
            if not ledger_data or not isinstance(ledger_data, dict):
                return
            ledger = WorkItemLedger.from_dict(ledger_data)
        except KeyError:
            return

        if not ledger.items:
            return

        incomplete = ledger.get_incomplete_items()
        if not incomplete:
            return  # all done, no need to inject state

        # Count tool usage from the message cache for agent self-awareness
        tool_usage: dict[str, int] = {}
        for msg in self._cache:
            contents = getattr(msg, "contents", None)
            if not contents:
                continue
            for content in contents:
                content_type = getattr(content, "type", None)
                if content_type == "function_call":
                    tool_name = getattr(content, "name", None)
                    if tool_name and not tool_name.startswith("work_item_"):
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

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

        # Include tool usage summary for self-awareness
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

        state_msg = ChatMessage(role="user", text="\n".join(lines))
        self._cache.append(state_msg)
        logger.info(f"AgentTurnExecutor: Injected work item state ({len(incomplete)} remaining)")

    async def _maybe_inject_work_item_reminder(self, ctx: WorkflowContext[Any]) -> None:
        """Inject a work item reminder if the last stop decision was work_items_incomplete.

        Args:
            ctx: Workflow context for state access.
        """
        from .._types import ChatMessage
        from ._work_items import WorkItemLedger, format_work_item_reminder

        # Check the transcript for the most recent stop_decision event
        try:
            transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            if not transcript:
                return
        except KeyError:
            return

        # Find the last stop_decision event
        last_stop = None
        for event_data in reversed(transcript):
            if event_data.get("event_type") == "stop_decision":
                last_stop = event_data
                break

        if last_stop is None:
            return

        # Check if it was a work_items_incomplete rejection
        data = last_stop.get("data", {})
        if data.get("reason") != "work_items_incomplete":
            return

        # Get the ledger and format reminder
        try:
            ledger_data = await ctx.get_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY)
            if not ledger_data or not isinstance(ledger_data, dict):
                return
            ledger = WorkItemLedger.from_dict(ledger_data)
        except KeyError:
            return

        reminder_text = format_work_item_reminder(ledger)
        if not reminder_text:
            return

        # Inject reminder as user message
        reminder_msg = ChatMessage(role="user", text=reminder_text)
        self._cache.append(reminder_msg)

        # Record event
        await self._append_event(
            ctx,
            HarnessEvent(
                event_type="work_item_reminder",
                data={"incomplete_count": len(ledger.get_incomplete_items())},
            ),
        )

        logger.info(
            f"AgentTurnExecutor: Injected work item reminder ({len(ledger.get_incomplete_items())} items remaining)"
        )

    async def _emit_work_item_events(self, ctx: WorkflowContext[Any]) -> None:
        """Emit events for work item changes captured by the middleware.

        This drains the middleware's event queue and emits each change as a
        HarnessLifecycleEvent for real-time UI updates.

        Args:
            ctx: Workflow context for event emission.
        """
        if self._work_item_middleware is None:
            return

        events = self._work_item_middleware.drain_events()
        for event_data in events:
            await ctx.add_event(
                HarnessLifecycleEvent(
                    event_type="work_item_changed",
                    data=event_data,
                ),
            )
            logger.debug(
                "AgentTurnExecutor: Emitted work_item_changed event for tool %s",
                event_data.get("tool", "unknown"),
            )

    async def _sync_work_item_ledger(self, ctx: WorkflowContext[Any]) -> None:
        """Sync the work item ledger to SharedState after each turn.

        Args:
            ctx: Workflow context for state access.
        """
        if self._task_list is None:
            return

        ledger = self._task_list.ledger
        await ctx.set_shared_state(HARNESS_WORK_ITEM_LEDGER_KEY, ledger.to_dict())

        # Emit progress event whenever ledger has items (progress bar + deliverables + all items)
        if ledger.items:
            deliverables = ledger.get_deliverables()
            total_items = len(ledger.items)
            done_items = sum(1 for item in ledger.items.values() if item.status.value in ("done", "skipped"))
            await ctx.add_event(
                HarnessLifecycleEvent(
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
                        # Full work item list for Tasks tab in DevUI
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
            )

        # Layer 3: Runtime invariant - failed control audits must produce revisions
        invariant_prompt = self._check_control_invariants(ledger)
        if invariant_prompt:
            from .._types import ChatMessage

            self._cache.append(ChatMessage(role="user", text=invariant_prompt))
            await self._append_event(
                ctx,
                HarnessEvent(
                    event_type="control_invariant_violation",
                    data={"prompt": invariant_prompt},
                ),
            )
            logger.info("AgentTurnExecutor: Control invariant violated, injected continuation prompt")

    def _check_control_invariants(self, ledger: "WorkItemLedger") -> str | None:
        """Check if control artifacts with verdict=fail have corresponding revisions.

        Returns:
            A continuation prompt string if the invariant is violated, None otherwise.
        """
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

        # Check if any revision items exist in the ledger
        has_revisions = any(item.revision_of for item in ledger.items.values())

        if has_revisions:
            return None  # Revisions exist, invariant satisfied

        # Build continuation prompt
        check_names: list[str] = []
        for _, checks in failed_control_items:
            check_names.extend(c.get("name", "unnamed") for c in checks)

        return (
            "Your control audit reported a 'fail' verdict with failed checks: "
            f"{', '.join(check_names)}. "
            "You must call flag_revision on the work items that need correction, "
            "then complete the revision items with corrected artifacts before finishing."
        )

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
