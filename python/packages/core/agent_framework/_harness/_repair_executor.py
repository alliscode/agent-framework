# Copyright (c) Microsoft. All rights reserved.

"""RepairExecutor for fixing transcript invariants before each agent turn."""

import logging
from typing import Any, cast

from .._workflows._const import WORKFLOW_RUN_KWARGS_KEY
from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    DEFAULT_MAX_TURNS,
    HARNESS_COVERAGE_LEDGER_KEY,
    HARNESS_INITIAL_MESSAGE_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_PENDING_TOOL_CALLS_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_STOP_REASON_KEY,
    HARNESS_TASK_CONTRACT_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from ._state import HarnessEvent, HarnessLifecycleEvent, HarnessStatus, PendingToolCall, RepairComplete, RepairTrigger

logger = logging.getLogger(__name__)

# Key for harness config in kwargs
HARNESS_CONFIG_KEY = "harness.config"


class RepairExecutor(Executor):
    """Repairs transcript invariants before each agent turn.

    Phase 1 repairs:
    - Dangling tool calls (call recorded but no result)

    The repair executor runs at the start of each turn to ensure the harness
    state is consistent before the agent executes. Any invariant violations
    are fixed by inserting synthetic events into the transcript.

    On the first run, this executor also initializes harness state from
    configuration passed via workflow kwargs.
    """

    def __init__(self, id: str = "harness_repair"):
        """Initialize the RepairExecutor.

        Args:
            id: Unique identifier for this executor.
        """
        super().__init__(id)

    @handler
    async def repair(self, trigger: RepairTrigger, ctx: WorkflowContext[RepairComplete]) -> None:
        """Handle repair trigger and fix any transcript invariants.

        Args:
            trigger: The repair trigger message.
            ctx: Workflow context for state access and message sending.
        """
        # 0. Initialize harness state on first run
        await self._ensure_harness_state_initialized(ctx)

        repairs_made = 0

        # 1. Load pending tool calls from shared state
        pending_calls = await self._get_pending_tool_calls(ctx)

        # 2. Repair dangling tool calls
        if pending_calls:
            for call in pending_calls:
                logger.info(
                    f"RepairExecutor: Repairing dangling tool call {call.call_id} "
                    f"(tool: {call.tool_name}) from turn {call.turn_number}",
                )

                # Insert synthetic error result into transcript
                repair_event = HarnessEvent(
                    event_type="repair",
                    data={
                        "kind": "dangling_tool_call",
                        "call_id": call.call_id,
                        "tool_name": call.tool_name,
                        "turn_number": call.turn_number,
                        "message": "Tool call did not complete - execution was interrupted",
                    },
                )
                await self._append_event(ctx, repair_event)
                repairs_made += 1

            # Clear pending calls
            await ctx.set_shared_state(HARNESS_PENDING_TOOL_CALLS_KEY, [])

        if repairs_made > 0:
            logger.info("RepairExecutor: Made %s repair(s)", repairs_made)

        # 3. Signal completion
        await ctx.send_message(RepairComplete(repairs_made=repairs_made))

    async def _get_pending_tool_calls(self, ctx: WorkflowContext[Any]) -> list[PendingToolCall]:
        """Get pending tool calls from shared state.

        Args:
            ctx: Workflow context for state access.

        Returns:
            List of pending tool calls, or empty list if none.
        """
        try:
            pending_data = await ctx.get_shared_state(HARNESS_PENDING_TOOL_CALLS_KEY)
            if not pending_data:
                return []
            result: list[PendingToolCall] = []
            for p in pending_data:
                if isinstance(p, dict):
                    result.append(PendingToolCall.from_dict(cast("dict[str, Any]", p)))
                elif isinstance(p, PendingToolCall):
                    result.append(p)
            return result
        except KeyError:
            return []

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

    async def _ensure_harness_state_initialized(self, ctx: WorkflowContext[Any]) -> None:
        """Ensure harness state is initialized on first run.

        Reads configuration from workflow kwargs and initializes harness state
        if it doesn't already exist.

        Args:
            ctx: Workflow context for state access.
        """
        # Check if harness state already exists (e.g., from checkpoint restore)
        try:
            status = await ctx.get_shared_state(HARNESS_STATUS_KEY)
            if status is not None:
                # State already initialized
                return
        except KeyError:
            pass

        # Get harness config from workflow kwargs
        harness_config: dict[str, Any] = {}
        try:
            run_kwargs = await ctx.get_shared_state(WORKFLOW_RUN_KWARGS_KEY)
            if run_kwargs and isinstance(run_kwargs, dict):
                config = cast("dict[str, Any]", run_kwargs).get(HARNESS_CONFIG_KEY)
                if config and isinstance(config, dict):
                    harness_config = cast("dict[str, Any]", config)
        except KeyError:
            pass

        max_turns: int = int(harness_config.get("max_turns", DEFAULT_MAX_TURNS))

        # Get initial message from config
        initial_message = harness_config.get("initial_message")

        logger.info("RepairExecutor: Initializing harness state with max_turns=%s", max_turns)

        # Initialize harness state
        await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, [])
        await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 0)
        await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, max_turns)
        await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
        await ctx.set_shared_state(HARNESS_STOP_REASON_KEY, None)
        await ctx.set_shared_state(HARNESS_PENDING_TOOL_CALLS_KEY, [])
        await ctx.set_shared_state(HARNESS_INITIAL_MESSAGE_KEY, initial_message)

        # Emit lifecycle event for DevUI
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="harness_started",
                max_turns=max_turns,
                data={"initial_message_preview": str(initial_message)[:100] if initial_message else None},
            ),
        )

        # Initialize task contract if provided (Phase 3)
        task_contract_data = harness_config.get("task_contract")
        if task_contract_data and isinstance(task_contract_data, dict):
            from ._task_contract import CoverageLedger, TaskContract

            contract = TaskContract.from_dict(task_contract_data)
            ledger = CoverageLedger.for_contract(contract)

            await ctx.set_shared_state(HARNESS_TASK_CONTRACT_KEY, contract.to_dict())
            await ctx.set_shared_state(HARNESS_COVERAGE_LEDGER_KEY, ledger.to_dict())

            logger.info(f"RepairExecutor: Initialized task contract with {len(contract.required_outputs)} requirements")
