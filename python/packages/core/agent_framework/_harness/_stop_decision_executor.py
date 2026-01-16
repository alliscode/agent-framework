# Copyright (c) Microsoft. All rights reserved.

"""StopDecisionExecutor for evaluating stop conditions after each agent turn."""

import logging
from typing import Any, cast

from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    DEFAULT_STALL_THRESHOLD,
    HARNESS_COVERAGE_LEDGER_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_PROGRESS_TRACKER_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_STOP_REASON_KEY,
    HARNESS_TASK_CONTRACT_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from ._state import (
    HarnessEvent,
    HarnessLifecycleEvent,
    HarnessResult,
    HarnessStatus,
    RepairTrigger,
    StopReason,
    TurnComplete,
)

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
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        id: str = "harness_stop_decision",
    ):
        """Initialize the StopDecisionExecutor.

        Args:
            enable_contract_verification: Whether to verify contracts before stopping.
            enable_stall_detection: Whether to detect stalled progress.
            stall_threshold: Number of unchanged turns before stall detection.
            id: Unique identifier for this executor.
        """
        super().__init__(id)
        self._enable_contract_verification = enable_contract_verification
        self._enable_stall_detection = enable_stall_detection
        self._stall_threshold = stall_threshold

    @handler
    async def evaluate(self, turn_result: TurnComplete, ctx: WorkflowContext[RepairTrigger, HarnessResult]) -> None:
        """Evaluate stop conditions and decide whether to continue or stop.

        Args:
            turn_result: The result from the completed agent turn.
            ctx: Workflow context for state access and message/output sending.
        """
        # Get current state
        turn_count = await self._get_turn_count(ctx)
        max_turns = await self._get_max_turns(ctx)

        logger.debug(f"StopDecisionExecutor: Evaluating turn {turn_count}/{max_turns}")

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
                    )
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
            if self._enable_contract_verification:
                # Verify contract before accepting done signal
                contract_satisfied, gap_info = await self._verify_contract(ctx)

                if contract_satisfied:
                    reason = StopReason(
                        kind="agent_done",
                        message="Agent completed task (contract verified)",
                        details={"turn": turn_count, "contract_verified": True},
                    )
                    await self._stop(ctx, HarnessStatus.DONE, reason, turn_count)
                    return
                else:
                    # Contract not satisfied - log gap and continue
                    logger.info(
                        f"StopDecisionExecutor: Agent signaled done but contract not satisfied. "
                        f"Gaps: {gap_info}"
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
            else:
                # No contract verification - accept done signal
                reason = StopReason(
                    kind="agent_done",
                    message="Agent completed task",
                    details={"turn": turn_count},
                )
                await self._stop(ctx, HarnessStatus.DONE, reason, turn_count)
                return

        # No stop condition met - continue to next turn
        logger.info(f"StopDecisionExecutor: Continuing to turn {turn_count + 1}")
        await ctx.send_message(RepairTrigger())

    async def _get_turn_count(self, ctx: WorkflowContext[Any, Any]) -> int:
        """Get the current turn count."""
        try:
            return int(await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0)
        except KeyError:
            return 0

    async def _get_max_turns(self, ctx: WorkflowContext[Any, Any]) -> int:
        """Get the maximum turns limit."""
        try:
            return int(await ctx.get_shared_state(HARNESS_MAX_TURNS_KEY) or 50)
        except KeyError:
            return 50

    async def _check_stall(
        self, ctx: WorkflowContext[Any, Any], turn_count: int
    ) -> tuple[bool, int]:
        """Check if progress has stalled.

        Args:
            ctx: Workflow context for state access.
            turn_count: Current turn number.

        Returns:
            Tuple of (is_stalled, stall_turns).
        """
        from ._task_contract import CoverageLedger, ProgressFingerprint, ProgressTracker

        # Get or create progress tracker
        tracker = await self._get_progress_tracker(ctx)

        # Get transcript length for fingerprint
        transcript = await self._get_transcript(ctx)

        # Get coverage ledger if available
        ledger = await self._get_coverage_ledger(ctx)

        # Compute and add fingerprint
        fingerprint = ProgressFingerprint.compute(
            turn_number=turn_count,
            ledger=ledger,
            transcript_length=len(transcript),
        )
        tracker.add_fingerprint(fingerprint)

        # Save updated tracker
        await ctx.set_shared_state(HARNESS_PROGRESS_TRACKER_KEY, tracker.to_dict())

        # Check for stall
        is_stalled = tracker.is_stalled()
        stall_duration = tracker.get_stall_duration()

        return is_stalled, stall_duration

    async def _get_progress_tracker(self, ctx: WorkflowContext[Any, Any]) -> Any:
        """Get or create the progress tracker."""
        from ._task_contract import ProgressTracker

        try:
            tracker_data = await ctx.get_shared_state(HARNESS_PROGRESS_TRACKER_KEY)
            if tracker_data and isinstance(tracker_data, dict):
                return ProgressTracker.from_dict(cast(dict[str, Any], tracker_data))
        except KeyError:
            pass

        return ProgressTracker(stall_threshold=self._stall_threshold)

    async def _get_coverage_ledger(self, ctx: WorkflowContext[Any, Any]) -> Any:
        """Get the coverage ledger if available."""
        from ._task_contract import CoverageLedger

        try:
            ledger_data = await ctx.get_shared_state(HARNESS_COVERAGE_LEDGER_KEY)
            if ledger_data and isinstance(ledger_data, dict):
                return CoverageLedger.from_dict(cast(dict[str, Any], ledger_data))
        except KeyError:
            pass

        return None

    async def _get_transcript(self, ctx: WorkflowContext[Any, Any]) -> list[dict[str, Any]]:
        """Get the current transcript."""
        try:
            transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            if transcript:
                return list(transcript)
        except KeyError:
            pass
        return []

    async def _verify_contract(
        self, ctx: WorkflowContext[Any, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Verify the task contract is satisfied.

        Args:
            ctx: Workflow context for state access.

        Returns:
            Tuple of (satisfied, gap_info).
        """
        from ._contract_verifier import ContractVerifier
        from ._task_contract import CoverageLedger, GapReport, TaskContract

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
        await ctx.set_shared_state(HARNESS_COVERAGE_LEDGER_KEY, ledger.to_dict())

        if result.satisfied:
            return True, {}

        # Generate gap report
        gap_report = GapReport.from_contract_and_ledger(contract, ledger)

        return False, gap_report.to_dict()

    async def _get_task_contract(self, ctx: WorkflowContext[Any, Any]) -> Any:
        """Get the task contract if available."""
        from ._task_contract import TaskContract

        try:
            contract_data = await ctx.get_shared_state(HARNESS_TASK_CONTRACT_KEY)
            if contract_data and isinstance(contract_data, dict):
                return TaskContract.from_dict(cast(dict[str, Any], contract_data))
        except KeyError:
            pass

        return None

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
        await ctx.set_shared_state(HARNESS_STATUS_KEY, status.value)
        await ctx.set_shared_state(HARNESS_STOP_REASON_KEY, reason.to_dict())

        # Get max_turns for lifecycle event
        max_turns = await self._get_max_turns(ctx)

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
            )
        )

        # Get transcript for final result
        try:
            transcript_data = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY)
            transcript = [HarnessEvent.from_dict(e) for e in transcript_data] if transcript_data else []
        except KeyError:
            transcript = []

        # Yield final result
        result = HarnessResult(
            status=status,
            reason=reason,
            transcript=transcript,
            turn_count=turn_count,
        )
        await ctx.yield_output(result)

    async def _append_event(self, ctx: WorkflowContext[Any, Any], event: HarnessEvent) -> None:
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
