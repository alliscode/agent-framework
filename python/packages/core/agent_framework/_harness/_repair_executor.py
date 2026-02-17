# Copyright (c) Microsoft. All rights reserved.

"""RepairExecutor for fixing transcript invariants before each agent turn."""

import logging
from typing import Any, cast

from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._compaction_owner_mode import normalize_compaction_owner_mode
from ._constants import (
    DEFAULT_COMPACTION_OWNER_MODE,
    DEFAULT_MAX_TURNS,
    DEFAULT_STOP_POLICY_PROFILE,
)
from ._state import HarnessEvent, HarnessLifecycleEvent, HarnessStatus, PendingToolCall, RepairComplete, RepairTrigger
from ._state_store import HarnessStateStore

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
        self._state_store = HarnessStateStore()

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
            await self._state_store.set_pending_tool_calls(ctx, [])

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
        pending_data = await self._state_store.get_pending_tool_calls(ctx)
        if not pending_data:
            return []
        result: list[PendingToolCall] = []
        for p in pending_data:
            if isinstance(p, dict):
                result.append(PendingToolCall.from_dict(cast("dict[str, Any]", p)))
            elif isinstance(p, PendingToolCall):
                result.append(p)
        return result

    async def _append_event(self, ctx: WorkflowContext[Any], event: HarnessEvent) -> None:
        """Append an event to the transcript.

        Args:
            ctx: Workflow context for state access.
            event: The event to append.
        """
        await self._state_store.append_transcript_event(ctx, event)

    async def _ensure_harness_state_initialized(self, ctx: WorkflowContext[Any]) -> None:
        """Ensure harness state is initialized on first run.

        Reads configuration from workflow kwargs and initializes harness state
        if it doesn't already exist.

        Args:
            ctx: Workflow context for state access.
        """
        # Check if harness state already exists (e.g., from checkpoint restore)
        status = await self._state_store.get_status(ctx)
        if status is not None:
            # State already initialized
            return

        # Get harness config from workflow kwargs
        harness_config: dict[str, Any] = {}
        run_kwargs = await self._state_store.get_run_kwargs(ctx)
        if run_kwargs:
            config = cast("dict[str, Any]", run_kwargs).get(HARNESS_CONFIG_KEY)
            if config and isinstance(config, dict):
                harness_config = cast("dict[str, Any]", config)

        max_turns: int = int(harness_config.get("max_turns", DEFAULT_MAX_TURNS))
        compaction_config = harness_config.get("compaction")
        owner_mode = DEFAULT_COMPACTION_OWNER_MODE
        if isinstance(compaction_config, dict):
            raw_mode = compaction_config.get("owner_mode", DEFAULT_COMPACTION_OWNER_MODE)
            owner_mode = normalize_compaction_owner_mode(raw_mode)

        # Get initial message from config
        initial_message = harness_config.get("initial_message")
        stop_policy_profile = harness_config.get("stop_policy_profile", DEFAULT_STOP_POLICY_PROFILE)
        if stop_policy_profile not in ("interactive", "strict_automation"):
            stop_policy_profile = DEFAULT_STOP_POLICY_PROFILE

        logger.info(
            "RepairExecutor: Initializing harness state with max_turns=%s, compaction_owner_mode=%s, stop_policy=%s",
            max_turns,
            owner_mode,
            stop_policy_profile,
        )

        # Initialize harness state
        await self._state_store.set_transcript_data(ctx, [])
        await self._state_store.set_turn_count(ctx, 0)
        await self._state_store.set_max_turns(ctx, max_turns)
        await self._state_store.set_status(ctx, HarnessStatus.RUNNING.value)
        await self._state_store.set_stop_reason(ctx, None)
        await self._state_store.set_pending_tool_calls(ctx, [])
        await self._state_store.set_initial_message(ctx, initial_message)
        await self._state_store.set_compaction_owner_mode(ctx, owner_mode)
        await self._state_store.set_stop_policy_profile(ctx, stop_policy_profile)

        # Emit lifecycle event for DevUI
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="harness_started",
                max_turns=max_turns,
                data={
                    "initial_message_preview": str(initial_message)[:100] if initial_message else None,
                    "compaction_owner_mode": owner_mode,
                    "stop_policy_profile": stop_policy_profile,
                },
            ),
        )

        # Initialize task contract if provided (Phase 3)
        task_contract_data = harness_config.get("task_contract")
        if task_contract_data and isinstance(task_contract_data, dict):
            from ._task_contract import CoverageLedger, TaskContract

            contract = TaskContract.from_dict(task_contract_data)
            ledger = CoverageLedger.for_contract(contract)

            await self._state_store.set_task_contract_data(ctx, contract.to_dict())
            await self._state_store.set_coverage_ledger_data(ctx, ledger.to_dict())

            logger.info(f"RepairExecutor: Initialized task contract with {len(contract.required_outputs)} requirements")
