# Copyright (c) Microsoft. All rights reserved.

"""StopDecisionExecutor for evaluating stop conditions after each agent turn."""

import logging
from typing import TYPE_CHECKING, Any, cast

from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_WORK_COMPLETE_MAX_RETRIES,
)
from ._state_store import HarnessStateStore
from ._state import (
    HarnessEvent,
    HarnessLifecycleEvent,
    HarnessResult,
    HarnessStatus,
    RepairTrigger,
    StopReason,
    TurnComplete,
)

if TYPE_CHECKING:
    from ._hooks import HarnessHooks

logger = logging.getLogger(__name__)


class StopDecisionExecutor(Executor):
    """Evaluates stop conditions after each agent turn.

    This executor implements a layered stop strategy:

    Layer 1: Hard stops (workflow-owned)
    - Maximum turns reached
    - Agent encountered an error

    Layer 2: Contract verification (Phase 3)
    - When agent signals done, verify contract requirements
    - If contract not satisfied, generate gap report and continue

    Layer 3: Stall detection (Phase 3)
    - Track progress fingerprints
    - Stop with "stalled" if no progress for multiple turns

    Layer 4: Agent signals (agent-owned)
    - Agent indicates task completion (no more tool calls)

    If no stop condition is met, the executor triggers another turn
    by sending a RepairTrigger message back to the RepairExecutor.
    """

    def __init__(
        self,
        *,
        enable_contract_verification: bool = False,
        enable_stall_detection: bool = False,
        enable_work_item_verification: bool = False,
        require_work_complete: bool = True,
        work_complete_max_retries: int = DEFAULT_WORK_COMPLETE_MAX_RETRIES,
        accept_done_after_retries_exhausted: bool = True,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        hooks: "HarnessHooks | None" = None,
        id: str = "harness_stop_decision",
    ):
        """Initialize the StopDecisionExecutor.

        Args:
            enable_contract_verification: Whether to verify contracts before stopping.
            enable_stall_detection: Whether to detect stalled progress.
            enable_work_item_verification: Whether to verify work items before stopping.
            require_work_complete: Whether to require explicit work_complete call before stopping.
            work_complete_max_retries: Max times to retry when agent signals done without
                calling work_complete. After this many retries, accept the done signal anyway
                to prevent infinite loops. Default is 3.
            accept_done_after_retries_exhausted: Whether to accept an agent_done signal after
                work_complete retries are exhausted. Set False for strict automation mode.
            stall_threshold: Number of unchanged turns before stall detection.
            hooks: Optional harness hooks for agent-stop interception.
            id: Unique identifier for this executor.
        """
        super().__init__(id)
        self._enable_contract_verification = enable_contract_verification
        self._enable_stall_detection = enable_stall_detection
        self._enable_work_item_verification = enable_work_item_verification
        self._require_work_complete = require_work_complete
        self._work_complete_max_retries = work_complete_max_retries
        self._accept_done_after_retries_exhausted = accept_done_after_retries_exhausted
        self._stall_threshold = stall_threshold
        self._hooks = hooks
        self._state_store = HarnessStateStore()

    @handler
    async def evaluate(self, turn_result: TurnComplete, ctx: WorkflowContext[RepairTrigger, HarnessResult]) -> None:
        """Evaluate stop conditions and decide whether to continue or stop.

        Args:
            turn_result: The result from the completed agent turn.
            ctx: Workflow context for state access and message/output sending.
        """
        # Get current state
        turn_count = await self._state_store.get_turn_count(ctx)
        max_turns = await self._state_store.get_max_turns(ctx)

        logger.debug("StopDecisionExecutor: Evaluating turn %s/%s", turn_count, max_turns)

        # Layer 1: Hard stops - check for errors first
        if turn_result.error:
            reason = StopReason(
                kind="failed",
                message=f"Agent turn failed: {turn_result.error}",
                details={"turn": turn_count, "error": turn_result.error},
            )
            await self._stop(ctx, HarnessStatus.FAILED, reason, turn_count)
            return

        # Layer 1: Hard stops - check max turns
        if turn_count >= max_turns:
            reason = StopReason(
                kind="max_turns",
                message=f"Reached maximum of {max_turns} turns",
                details={"turn": turn_count, "max_turns": max_turns},
            )
            await self._stop(ctx, HarnessStatus.DONE, reason, turn_count)
            return

        # Layer 2: Stall detection (Phase 3)
        if self._enable_stall_detection:
            is_stalled, stall_turns = await self._check_stall(ctx, turn_count)
            if is_stalled:
                # Emit lifecycle event for stall detection
                await ctx.add_event(
                    HarnessLifecycleEvent(
                        event_type="stall_detected",
                        turn_number=turn_count,
                        max_turns=max_turns,
                        data={"stall_turns": stall_turns},
                    ),
                )
                reason = StopReason(
                    kind="stalled",
                    message=f"No progress detected for {stall_turns} turns",
                    details={"turn": turn_count, "stall_turns": stall_turns},
                )
                await self._stop(ctx, HarnessStatus.STALLED, reason, turn_count)
                return

        # Layer 3: Agent signals done - with optional contract verification
        if turn_result.agent_done:
            # 3-pre: Require explicit work_complete call (with retry cap)
            if self._require_work_complete and not turn_result.called_work_complete:
                # Track how many times we've retried without work_complete
                retry_count = await self._state_store.get_work_complete_retry_count(ctx)
                retry_count += 1
                await self._state_store.set_work_complete_retry_count(ctx, retry_count)

                if retry_count <= self._work_complete_max_retries:
                    logger.info(
                        "StopDecisionExecutor: Agent signaled done but did not call work_complete (retry %d/%d)",
                        retry_count,
                        self._work_complete_max_retries,
                    )
                    await self._append_event(
                        ctx,
                        HarnessEvent(
                            event_type="stop_decision",
                            data={
                                "decision": "continue",
                                "reason": "work_complete_not_called",
                                "turn": turn_count,
                                "retry": retry_count,
                                "max_retries": self._work_complete_max_retries,
                            },
                        ),
                    )
                    await ctx.send_message(RepairTrigger())
                    return
                if not self._accept_done_after_retries_exhausted:
                    logger.info(
                        "StopDecisionExecutor: Agent did not call work_complete after %d retries; "
                        "strict policy continues execution",
                        self._work_complete_max_retries,
                    )
                    await self._append_event(
                        ctx,
                        HarnessEvent(
                            event_type="stop_decision",
                            data={
                                "decision": "continue",
                                "reason": "work_complete_required_strict",
                                "turn": turn_count,
                                "retries": retry_count,
                                "max_retries": self._work_complete_max_retries,
                            },
                        ),
                    )
                    await ctx.send_message(RepairTrigger())
                    return

                # Max retries exceeded — accept the done signal to prevent infinite loops.
                logger.warning(
                    "StopDecisionExecutor: Agent never called work_complete after %d retries, "
                    "accepting done signal to prevent infinite loop",
                    self._work_complete_max_retries,
                )
                await self._append_event(
                    ctx,
                    HarnessEvent(
                        event_type="stop_decision",
                        data={
                            "decision": "stop",
                            "reason": "work_complete_retries_exhausted",
                            "turn": turn_count,
                            "retries": retry_count,
                        },
                    ),
                )

            if self._enable_contract_verification:
                # Verify contract before accepting done signal
                contract_satisfied, gap_info = await self._verify_contract(ctx)

                if not contract_satisfied:
                    # Contract not satisfied - log gap and continue
                    logger.info(
                        "StopDecisionExecutor: Agent signaled done but contract not satisfied. Gaps: %s",
                        gap_info,
                    )
                    # Record gap event
                    await self._append_event(
                        ctx,
                        HarnessEvent(
                            event_type="stop_decision",
                            data={
                                "decision": "continue",
                                "reason": "contract_not_satisfied",
                                "gaps": gap_info,
                                "turn": turn_count,
                            },
                        ),
                    )
                    # Continue execution
                    await ctx.send_message(RepairTrigger())
                    return

            # Layer 3b: Work item verification (if enabled)
            if self._enable_work_item_verification:
                work_items_complete = await self._verify_work_items(ctx)
                if not work_items_complete:
                    logger.info("StopDecisionExecutor: Agent signaled done but work items incomplete")
                    await self._append_event(
                        ctx,
                        HarnessEvent(
                            event_type="stop_decision",
                            data={
                                "decision": "continue",
                                "reason": "work_items_incomplete",
                                "turn": turn_count,
                            },
                        ),
                    )
                    await ctx.send_message(RepairTrigger())
                    return

            # All verification passed — run agent-stop hooks before accepting
            if self._hooks and self._hooks.agent_stop:
                blocked = await self._run_agent_stop_hooks(ctx, turn_count, turn_result)
                if blocked:
                    return

            # All verification passed - accept done signal
            details: dict[str, Any] = {"turn": turn_count}
            if self._enable_contract_verification:
                details["contract_verified"] = True
            if self._enable_work_item_verification:
                details["work_items_verified"] = True

            reason = StopReason(
                kind="agent_done",
                message="Agent completed task",
                details=details,
            )
            await self._stop(ctx, HarnessStatus.DONE, reason, turn_count)
            return

        # No stop condition met - continue to next turn
        logger.info(f"StopDecisionExecutor: Continuing to turn {turn_count + 1}")
        await ctx.send_message(RepairTrigger())

    async def _check_stall(self, ctx: WorkflowContext[Any, Any], turn_count: int) -> tuple[bool, int]:
        """Check if progress has stalled.

        Args:
            ctx: Workflow context for state access.
            turn_count: Current turn number.

        Returns:
            Tuple of (is_stalled, stall_turns).
        """
        from ._task_contract import ProgressFingerprint

        # Get or create progress tracker
        tracker = await self._get_progress_tracker(ctx)

        # Get transcript length for fingerprint
        transcript = await self._get_transcript(ctx)

        # Get coverage ledger if available
        ledger = await self._get_coverage_ledger(ctx)

        # Get work item statuses if available
        work_item_statuses = await self._get_work_item_statuses(ctx)

        # Compute and add fingerprint
        fingerprint = ProgressFingerprint.compute(
            turn_number=turn_count,
            ledger=ledger,
            transcript_length=len(transcript),
            work_item_statuses=work_item_statuses,
        )
        tracker.add_fingerprint(fingerprint)

        # Save updated tracker
        await self._state_store.set_progress_tracker_data(ctx, tracker.to_dict())

        # Check for stall
        is_stalled = tracker.is_stalled()
        stall_duration = tracker.get_stall_duration()

        return is_stalled, stall_duration

    async def _get_work_item_statuses(self, ctx: WorkflowContext[Any, Any]) -> dict[str, str] | None:
        """Get work item statuses for progress fingerprint."""
        from ._work_items import WorkItemLedger

        ledger_data = await self._state_store.get_work_item_ledger_data(ctx)
        if ledger_data and isinstance(ledger_data, dict):
            ledger = WorkItemLedger.from_dict(cast("dict[str, Any]", ledger_data))
            if ledger.items:
                return {item_id: item.status.value for item_id, item in ledger.items.items()}
        return None

    async def _get_progress_tracker(self, ctx: WorkflowContext[Any, Any]) -> Any:
        """Get or create the progress tracker."""
        from ._task_contract import ProgressTracker

        tracker_data = await self._state_store.get_progress_tracker_data(ctx)
        if tracker_data and isinstance(tracker_data, dict):
            return ProgressTracker.from_dict(cast("dict[str, Any]", tracker_data))

        return ProgressTracker(stall_threshold=self._stall_threshold)

    async def _get_coverage_ledger(self, ctx: WorkflowContext[Any, Any]) -> Any:
        """Get the coverage ledger if available."""
        from ._task_contract import CoverageLedger

        ledger_data = await self._state_store.get_coverage_ledger_data(ctx)
        if ledger_data and isinstance(ledger_data, dict):
            return CoverageLedger.from_dict(cast("dict[str, Any]", ledger_data))

        return None

    async def _get_transcript(self, ctx: WorkflowContext[Any, Any]) -> list[dict[str, Any]]:
        """Get the current transcript."""
        try:
            return await self._state_store.get_transcript_data(ctx)
        except Exception:
            return []

    async def _verify_contract(self, ctx: WorkflowContext[Any, Any]) -> tuple[bool, dict[str, Any]]:
        """Verify the task contract is satisfied.

        Args:
            ctx: Workflow context for state access.

        Returns:
            Tuple of (satisfied, gap_info).
        """
        from ._contract_verifier import ContractVerifier
        from ._task_contract import CoverageLedger, GapReport

        # Get contract
        contract = await self._get_task_contract(ctx)
        if contract is None:
            # No contract - treat as satisfied
            return True, {}

        # Get or create ledger
        ledger = await self._get_coverage_ledger(ctx)
        if ledger is None:
            ledger = CoverageLedger.for_contract(contract)

        # Get transcript for verification context
        transcript = await self._get_transcript(ctx)

        # Create verifier and verify
        verifier = ContractVerifier(transcript=transcript)
        result = verifier.verify_contract(contract, ledger)

        # Save updated ledger
        await self._state_store.set_coverage_ledger_data(ctx, ledger.to_dict())

        if result.satisfied:
            return True, {}

        # Generate gap report
        gap_report = GapReport.from_contract_and_ledger(contract, ledger)

        return False, gap_report.to_dict()

    async def _verify_work_items(self, ctx: WorkflowContext[Any, Any]) -> bool:
        """Verify that all work items are complete.

        Args:
            ctx: Workflow context for state access.

        Returns:
            True if all work items are complete (or no items exist).
        """
        from ._work_items import WorkItemLedger

        ledger_data = await self._state_store.get_work_item_ledger_data(ctx)
        if ledger_data and isinstance(ledger_data, dict):
            ledger = WorkItemLedger.from_dict(cast("dict[str, Any]", ledger_data))
            return ledger.is_all_complete()

        # No ledger means no items to verify
        return True

    async def _get_task_contract(self, ctx: WorkflowContext[Any, Any]) -> Any:
        """Get the task contract if available."""
        from ._task_contract import TaskContract

        contract_data = await self._state_store.get_task_contract_data(ctx)
        if contract_data and isinstance(contract_data, dict):
            return TaskContract.from_dict(cast("dict[str, Any]", contract_data))

        return None

    async def _run_agent_stop_hooks(
        self,
        ctx: WorkflowContext[Any, Any],
        turn_count: int,
        turn_result: TurnComplete,
    ) -> bool:
        """Run agent-stop hooks and return True if any hook blocked the stop.

        Args:
            ctx: Workflow context for state access and messaging.
            turn_count: Current turn number.
            turn_result: The turn completion result.

        Returns:
            True if a hook blocked the stop (RepairTrigger sent), False otherwise.
        """
        from ._hooks import AgentStopEvent

        event = AgentStopEvent(
            turn_count=turn_count,
            called_work_complete=turn_result.called_work_complete,
        )

        for hook in self._hooks.agent_stop:  # type: ignore[union-attr]
            try:
                result = await hook(event)
                if result and result.decision == "block":
                    logger.info(
                        "StopDecisionExecutor: agent_stop hook blocked stop: %s",
                        result.reason,
                    )
                    await self._append_event(
                        ctx,
                        HarnessEvent(
                            event_type="stop_decision",
                            data={
                                "decision": "continue",
                                "reason": "agent_stop_hook_blocked",
                                "hook_reason": result.reason,
                                "turn": turn_count,
                            },
                        ),
                    )
                    await ctx.send_message(RepairTrigger())
                    return True
            except Exception:
                logger.exception("StopDecisionExecutor: agent_stop hook error")

        return False

    async def _stop(
        self,
        ctx: WorkflowContext[Any, HarnessResult],
        status: HarnessStatus,
        reason: StopReason,
        turn_count: int,
    ) -> None:
        """Stop the harness and yield the final result.

        Args:
            ctx: Workflow context for state updates and output.
            status: The final harness status.
            reason: The reason for stopping.
            turn_count: The total number of turns executed.
        """
        logger.info(f"StopDecisionExecutor: Stopping with status={status.value}, reason={reason.kind}")

        # Record stop decision event
        await self._append_event(
            ctx,
            HarnessEvent(
                event_type="stop_decision",
                data={
                    "status": status.value,
                    "reason": reason.to_dict(),
                    "turn_count": turn_count,
                },
            ),
        )

        # Update shared state
        await self._state_store.set_completion_state(
            ctx,
            status=status.value,
            stop_reason=reason.to_dict(),
        )

        # Get max_turns for lifecycle event
        max_turns = await self._state_store.get_max_turns(ctx)

        # Emit lifecycle event for DevUI
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="harness_completed",
                turn_number=turn_count,
                max_turns=max_turns,
                data={
                    "status": status.value,
                    "reason_kind": reason.kind,
                    "reason_message": reason.message,
                },
            ),
        )

        # Get transcript for final result
        transcript_data = await self._state_store.get_transcript_data(ctx)
        transcript = [HarnessEvent.from_dict(e) for e in transcript_data] if transcript_data else []

        # Collect deliverable artifacts from work item ledger
        deliverables: list[dict[str, Any]] = []
        ledger_data = await self._state_store.get_work_item_ledger_data(ctx)
        if ledger_data and isinstance(ledger_data, dict):
            from ._work_items import WorkItemLedger

            ledger = WorkItemLedger.from_dict(cast("dict[str, Any]", ledger_data))
            for item in ledger.get_deliverables():
                deliverables.append({
                    "item_id": item.id,
                    "title": item.title,
                    "content": item.artifact,
                })

        # Yield final result
        result = HarnessResult(
            status=status,
            reason=reason,
            transcript=transcript,
            turn_count=turn_count,
            deliverables=deliverables,
        )
        await ctx.yield_output(result)

    async def _append_event(self, ctx: WorkflowContext[Any, Any], event: HarnessEvent) -> None:
        """Append an event to the transcript.

        Args:
            ctx: Workflow context for state access.
            event: The event to append.
        """
        await self._state_store.append_transcript_event(ctx, event)
