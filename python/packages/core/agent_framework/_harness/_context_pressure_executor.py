# Copyright (c) Microsoft. All rights reserved.

"""ContextPressureExecutor for managing context window utilization."""

import logging
from dataclasses import dataclass
from typing import Any, cast

from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_SOFT_THRESHOLD_PERCENT,
    HARNESS_CONTEXT_EDIT_HISTORY_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
)
from ._context_pressure import (
    ClearEdit,
    CompactEdit,
    ContextEdit,
    ContextEditPlan,
    ContextPressureStrategy,
    DropEdit,
    TokenBudget,
    estimate_transcript_tokens,
    get_default_strategies,
)
from ._state import HarnessEvent, RepairComplete

logger = logging.getLogger(__name__)


@dataclass
class ContextPressureComplete(RepairComplete):
    """Message indicating context pressure check is complete.

    Extends RepairComplete to maintain compatibility with existing flow.

    Attributes:
        edits_applied: Number of context edits applied.
        tokens_freed: Estimated tokens freed by edits.
    """

    edits_applied: int = 0
    tokens_freed: int = 0


class ContextPressureExecutor(Executor):
    """Executor that manages context window pressure.

    This executor runs before each agent turn to check if the conversation
    context is approaching token limits. If pressure is detected, it applies
    strategies in order from least to most aggressive:

    1. Clear tool results - Replace older tool results with placeholders
    2. Compact conversation - Summarize older messages
    3. Drop oldest - Remove oldest content entirely (last resort)

    The executor maintains a token budget estimate and applies strategies
    when the soft threshold (default 85%) is exceeded.
    """

    def __init__(
        self,
        *,
        strategies: list[ContextPressureStrategy] | None = None,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        soft_threshold_percent: float = DEFAULT_SOFT_THRESHOLD_PERCENT,
        id: str = "harness_context_pressure",
    ):
        """Initialize the ContextPressureExecutor.

        Args:
            strategies: List of pressure strategies to use. Defaults to standard set.
            max_input_tokens: Maximum allowed input tokens.
            soft_threshold_percent: Percentage at which to trigger strategies (0-1).
            id: Unique identifier for this executor.
        """
        super().__init__(id)
        self._strategies = strategies or get_default_strategies()
        self._max_input_tokens = max_input_tokens
        self._soft_threshold_percent = soft_threshold_percent

    @handler
    async def check_pressure(
        self, trigger: RepairComplete, ctx: WorkflowContext[ContextPressureComplete]
    ) -> None:
        """Check context pressure and apply strategies if needed.

        Args:
            trigger: The repair complete message indicating we can proceed.
            ctx: Workflow context for state access and message sending.
        """
        # 1. Initialize or load token budget
        budget = await self._get_or_create_budget(ctx)

        # 2. Get current transcript and estimate tokens
        transcript = await self._get_transcript(ctx)
        budget.current_estimate = estimate_transcript_tokens(transcript)

        # 3. Check if under pressure
        if not budget.is_under_pressure:
            logger.debug(
                f"ContextPressureExecutor: Under budget "
                f"({budget.current_estimate}/{budget.soft_threshold} tokens)"
            )
            await self._save_budget(ctx, budget)
            await ctx.send_message(
                ContextPressureComplete(repairs_made=trigger.repairs_made)
            )
            return

        logger.info(
            f"ContextPressureExecutor: Context pressure detected "
            f"({budget.current_estimate}/{budget.soft_threshold} tokens, "
            f"{budget.tokens_over_threshold} over threshold)"
        )

        # 4. Try strategies in order until pressure is relieved
        total_edits = 0
        total_tokens_freed = 0

        for strategy in self._strategies:
            if not strategy.is_applicable(budget, transcript):
                logger.debug(f"ContextPressureExecutor: Strategy {strategy.name} not applicable")
                continue

            plan = await strategy.propose(budget, transcript, ctx)
            if plan is None:
                continue

            logger.info(
                f"ContextPressureExecutor: Applying strategy {strategy.name} "
                f"(estimated {plan.estimated_token_reduction} tokens)"
            )

            # Apply the edit plan
            await self._apply_edit_plan(ctx, plan, transcript)

            total_edits += len(plan.edits)
            total_tokens_freed += plan.estimated_token_reduction

            # Record the edit in history
            await self._record_edit(ctx, plan, strategy.name)

            # Update budget estimate
            budget.current_estimate = max(0, budget.current_estimate - plan.estimated_token_reduction)

            # Check if we've relieved enough pressure
            if not budget.is_under_pressure:
                logger.info(
                    f"ContextPressureExecutor: Pressure relieved "
                    f"(now at {budget.current_estimate}/{budget.soft_threshold} tokens)"
                )
                break

        # 5. Save updated budget
        await self._save_budget(ctx, budget)

        # 6. Signal completion
        await ctx.send_message(
            ContextPressureComplete(
                repairs_made=trigger.repairs_made,
                edits_applied=total_edits,
                tokens_freed=total_tokens_freed,
            )
        )

    async def _get_or_create_budget(self, ctx: WorkflowContext[Any]) -> TokenBudget:
        """Get or create the token budget from shared state.

        Args:
            ctx: Workflow context for state access.

        Returns:
            The token budget.
        """
        try:
            budget_data = await ctx.get_shared_state(HARNESS_TOKEN_BUDGET_KEY)
            if budget_data and isinstance(budget_data, dict):
                return TokenBudget.from_dict(cast(dict[str, Any], budget_data))
        except KeyError:
            pass

        # Create default budget
        return TokenBudget(
            max_input_tokens=self._max_input_tokens,
            soft_threshold_percent=self._soft_threshold_percent,
        )

    async def _save_budget(self, ctx: WorkflowContext[Any], budget: TokenBudget) -> None:
        """Save the token budget to shared state.

        Args:
            ctx: Workflow context for state access.
            budget: The budget to save.
        """
        await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())

    async def _get_transcript(self, ctx: WorkflowContext[Any]) -> list[dict[str, Any]]:
        """Get the current transcript from shared state.

        Args:
            ctx: Workflow context for state access.

        Returns:
            The transcript as a list of event dictionaries.
        """
        try:
            transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            if transcript:
                return list(transcript)
        except KeyError:
            pass
        return []

    async def _apply_edit_plan(
        self,
        ctx: WorkflowContext[Any],
        plan: ContextEditPlan,
        transcript: list[dict[str, Any]],
    ) -> None:
        """Apply a context edit plan to the transcript.

        Args:
            ctx: Workflow context for state access.
            plan: The edit plan to apply.
            transcript: The current transcript (will be modified).
        """
        for edit in plan.edits:
            await self._apply_edit(ctx, edit, transcript)

        # Save the modified transcript
        await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, transcript)

    async def _apply_edit(
        self,
        ctx: WorkflowContext[Any],
        edit: ContextEdit,
        transcript: list[dict[str, Any]],
    ) -> None:
        """Apply a single context edit to the transcript.

        Args:
            ctx: Workflow context for state access.
            edit: The edit to apply.
            transcript: The transcript (will be modified in place).
        """
        if isinstance(edit, ClearEdit):
            await self._apply_clear_edit(edit, transcript)
        elif isinstance(edit, CompactEdit):
            await self._apply_compact_edit(edit, transcript)
        elif isinstance(edit, DropEdit):
            await self._apply_drop_edit(edit, transcript)
        # Note: ExternalizeEdit would require file system access - deferred to future

    async def _apply_clear_edit(
        self, edit: ClearEdit, transcript: list[dict[str, Any]]
    ) -> None:
        """Apply a clear edit to the transcript.

        Args:
            edit: The clear edit to apply.
            transcript: The transcript (modified in place).
        """
        start = edit.scope.start_index
        end = edit.scope.end_index or len(transcript)
        event_types = edit.scope.event_types

        for i in range(start, min(end, len(transcript))):
            event = transcript[i]

            # Filter by event type if specified
            if event_types and event.get("event_type") not in event_types:
                continue

            # Clear the data, keeping metadata
            turn_number = event.get("data", {}).get("turn_number", "?")
            placeholder = edit.placeholder_template.format(turn_number=turn_number)

            event["data"] = {
                "cleared": True,
                "original_type": event.get("event_type"),
                "placeholder": placeholder,
            }

    async def _apply_compact_edit(
        self, edit: CompactEdit, transcript: list[dict[str, Any]]
    ) -> None:
        """Apply a compact edit to the transcript.

        This creates a summary of the compacted content and replaces it
        with a single summary event.

        Args:
            edit: The compact edit to apply.
            transcript: The transcript (modified in place).
        """
        start = edit.scope.start_index
        end = edit.scope.end_index or len(transcript)

        if end <= start:
            return

        # Extract events to compact
        events_to_compact = transcript[start:end]

        # Generate summary based on type
        if edit.summary_type == "bullet":
            summary = self._generate_bullet_summary(events_to_compact)
        elif edit.summary_type == "state_json":
            summary = self._generate_state_summary(events_to_compact)
        else:  # narrative
            summary = self._generate_narrative_summary(events_to_compact)

        # Create summary event
        summary_event = HarnessEvent(
            event_type="repair",
            data={
                "kind": "compaction_summary",
                "compacted_count": len(events_to_compact),
                "summary": summary,
            },
        )

        # Replace compacted events with summary
        transcript[start:end] = [summary_event.to_dict()]

    async def _apply_drop_edit(
        self, edit: DropEdit, transcript: list[dict[str, Any]]
    ) -> None:
        """Apply a drop edit to the transcript.

        Args:
            edit: The drop edit to apply.
            transcript: The transcript (modified in place).
        """
        start = edit.scope.start_index
        end = edit.scope.end_index or len(transcript)

        if end <= start:
            return

        # Record what was dropped
        dropped_count = end - start
        drop_marker = HarnessEvent(
            event_type="repair",
            data={
                "kind": "content_dropped",
                "dropped_count": dropped_count,
                "message": f"Dropped {dropped_count} oldest events due to context pressure",
            },
        )

        # Replace dropped events with marker
        transcript[start:end] = [drop_marker.to_dict()]

    def _generate_bullet_summary(self, events: list[dict[str, Any]]) -> str:
        """Generate a bullet-point summary of events.

        Args:
            events: Events to summarize.

        Returns:
            Bullet-point summary string.
        """
        lines = ["Previous conversation summary:"]

        for event in events:
            event_type = event.get("event_type", "unknown")
            data = event.get("data", {})

            if event_type == "turn_start":
                lines.append(f"- Turn {data.get('turn_number', '?')} started")
            elif event_type == "agent_response":
                turn = data.get("turn_number", "?")
                lines.append(f"- Turn {turn}: Agent responded")
            elif event_type == "tool_call":
                tool = data.get("tool_name", "unknown")
                lines.append(f"- Tool called: {tool}")
            elif event_type == "tool_result":
                lines.append("- Tool result received")

        return "\n".join(lines)

    def _generate_state_summary(self, events: list[dict[str, Any]]) -> str:
        """Generate a JSON state summary of events.

        Args:
            events: Events to summarize.

        Returns:
            JSON summary string.
        """
        import json

        summary: dict[str, Any] = {
            "total_events": len(events),
            "event_types": {},
            "turns_covered": [],
        }

        for event in events:
            event_type = event.get("event_type", "unknown")
            summary["event_types"][event_type] = summary["event_types"].get(event_type, 0) + 1

            turn = event.get("data", {}).get("turn_number")
            if turn and turn not in summary["turns_covered"]:
                summary["turns_covered"].append(turn)

        return json.dumps(summary)

    def _generate_narrative_summary(self, events: list[dict[str, Any]]) -> str:
        """Generate a narrative summary of events.

        Args:
            events: Events to summarize.

        Returns:
            Narrative summary string.
        """
        if not events:
            return "No previous events."

        turns = set()
        tool_calls = []

        for event in events:
            data = event.get("data", {})
            turn = data.get("turn_number")
            if turn:
                turns.add(turn)

            if event.get("event_type") == "tool_call":
                tool_calls.append(data.get("tool_name", "unknown"))

        parts = []
        if turns:
            parts.append(f"Covered turns {min(turns)}-{max(turns)}")
        if tool_calls:
            unique_tools = list(set(tool_calls))[:5]  # Limit to 5
            parts.append(f"tools used: {', '.join(unique_tools)}")

        return ". ".join(parts) + "." if parts else "Previous conversation compacted."

    async def _record_edit(
        self, ctx: WorkflowContext[Any], plan: ContextEditPlan, strategy_name: str
    ) -> None:
        """Record a context edit in the history.

        Args:
            ctx: Workflow context for state access.
            plan: The edit plan that was applied.
            strategy_name: Name of the strategy that proposed it.
        """
        history: list[dict[str, Any]] = []
        try:
            stored = await ctx.get_shared_state(HARNESS_CONTEXT_EDIT_HISTORY_KEY)
            if stored:
                history = list(stored)
        except KeyError:
            pass

        from datetime import datetime, timezone

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy_name,
            "reason": plan.reason,
            "estimated_tokens_freed": plan.estimated_token_reduction,
            "edit_count": len(plan.edits),
        }

        history.append(record)
        await ctx.set_shared_state(HARNESS_CONTEXT_EDIT_HISTORY_KEY, history)
