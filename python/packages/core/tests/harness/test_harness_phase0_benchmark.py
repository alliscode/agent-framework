# Copyright (c) Microsoft. All rights reserved.

"""Phase 0 benchmark regression test for harness stability.

This test is intended to be the refactor phase gate sentinel:
- Configure the harness with a "full" profile similar to harness_repl.py
- Run a deterministic multi-turn scenario that grows context and triggers compaction
- Grade output on stability dimensions for long-running harness behavior
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import pytest

from agent_framework import (
    AgentRunResponse,
    AgentRunResponseUpdate,
    BaseAgent,
    ChatMessage,
    FunctionCallContent,
    FunctionResultContent,
    UsageDetails,
    WorkflowOutputEvent,
)
from agent_framework._harness import (
    AgentHarness,
    AgentStopEvent,
    AgentStopResult,
    HarnessHooks,
    HarnessLifecycleEvent,
    HarnessResult,
    HarnessStatus,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    TaskContract,
    WorkItemTaskList,
)
from agent_framework._harness._compaction import SpanReference, StructuredSummary


class _DummySummarizer:
    """Deterministic summarizer for compaction tests."""

    async def summarize(
        self,
        messages: list[ChatMessage],
        *,
        target_token_ratio: float = 0.25,
        preserve_facts: list[str] | None = None,
    ) -> StructuredSummary:
        message_ids = [m.message_id for m in messages if m.message_id]
        summary_facts = ["Compacted historical context"]
        if preserve_facts:
            summary_facts.extend(preserve_facts)

        return StructuredSummary(
            span=SpanReference(
                message_ids=message_ids or ["synthetic-span-id"],
                first_turn=0,
                last_turn=max(0, len(messages) - 1),
            ),
            facts=summary_facts,
            decisions=[],
            open_items=[],
            artifacts=[],
            tool_outcomes=[],
            current_task="Continue solving user task",
            current_plan=["Proceed to next turn with compacted context"],
        )


class _ScriptedLongRunAgent(BaseAgent):
    """Deterministic multi-turn scripted agent.

    Script:
    - Turns 1-3: emits large assistant text + non-work_complete tool call + large tool result.
    - Turn 4: emits large assistant text + work_complete call to stop.
    """

    def __init__(self) -> None:
        super().__init__(name="scripted_long_run_agent")
        self._turn = 0
        self._payload = "X" * 2600  # Large content to pressure context budget
        self._tool_payload = "R" * 3200  # Large tool outputs that compaction can clear

    async def run(
        self,
        messages: list[Any],
        thread: Any | None = None,
        **kwargs: Any,
    ) -> AgentRunResponse:
        self._turn += 1

        if self._turn < 4:
            assistant_msg = ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(
                        call_id=f"call-{self._turn}",
                        name="read_file",
                        arguments={"path": f"file_{self._turn}.txt"},
                    )
                ],
                text=f"Turn {self._turn} analysis chunk: {self._payload}",
            )
            tool_msg = ChatMessage(
                role="tool",
                contents=[
                    FunctionResultContent(
                        call_id=f"call-{self._turn}",
                        result=f"Tool output chunk {self._turn}: {self._tool_payload}",
                    )
                ],
            )
            messages = [assistant_msg, tool_msg]
        else:
            messages = [
                ChatMessage(
                role="assistant",
                contents=[
                    FunctionCallContent(
                        call_id=f"call-{self._turn}",
                        name="work_complete",
                        arguments={"summary": "Completed benchmark scenario successfully"},
                    )
                ],
                    text=f"Final synthesis chunk: {self._payload}",
                )
            ]

        usage = UsageDetails(
            input_token_count=3000 + self._turn * 500,
            output_token_count=900 + self._turn * 100,
            total_token_count=3900 + self._turn * 600,
        )
        return AgentRunResponse(messages=messages, usage_details=usage)

    async def run_stream(
        self,
        messages: list[Any],
        thread: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentRunResponseUpdate]:
        response = await self.run(messages=messages, thread=thread, **kwargs)
        for message in response.messages:
            yield AgentRunResponseUpdate(
                contents=message.contents,
                role=message.role,
                message_id=message.message_id,
            )


@dataclass
class HarnessBenchmarkScore:
    """Rubric-style scoring for harness stability behavior."""

    stop_correctness: bool
    turn_efficiency: bool
    compaction_correctness: bool
    completion_integrity: bool
    state_invariants: bool
    details: dict[str, Any]

    @property
    def total_passed(self) -> int:
        checks = [
            self.stop_correctness,
            self.turn_efficiency,
            self.compaction_correctness,
            self.completion_integrity,
            self.state_invariants,
        ]
        return sum(1 for c in checks if c)


@pytest.mark.asyncio
async def test_phase0_harness_benchmark_regression_gate() -> None:
    """Benchmark gate for harness refactor phases.

    Dimensions:
    - stop correctness
    - turn efficiency
    - compaction correctness
    - completion integrity
    - state/event invariants
    """

    tmp_path = Path(tempfile.mkdtemp(prefix="harness-benchmark-"))

    # Track agent_stop hook execution
    stop_hook_calls: list[AgentStopEvent] = []

    async def _allow_stop(event: AgentStopEvent) -> AgentStopResult | None:
        stop_hook_calls.append(event)
        return AgentStopResult(decision="allow", reason="benchmark-allow")

    hooks = HarnessHooks(agent_stop=[_allow_stop])
    task_list = WorkItemTaskList()
    agent = _ScriptedLongRunAgent()

    harness = AgentHarness(
        agent,
        max_turns=12,
        enable_stall_detection=True,
        stall_threshold=4,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        task_list=task_list,
        enable_compaction=True,
        compaction_store=InMemoryCompactionStore(),
        artifact_store=InMemoryArtifactStore(),
        summary_cache=InMemorySummaryCache(max_entries=50),
        summarizer=_DummySummarizer(),
        max_input_tokens=1200,
        soft_threshold_percent=0.40,
        task_contract=TaskContract.simple(
            "Complete a multi-turn benchmark task",
            "Perform iterative progress across multiple turns",
            "Signal completion explicitly",
        ),
        sandbox_path=str(tmp_path),
        hooks=hooks,
    )

    prompt = (
        "Perform a deep multi-step analysis of this workspace. "
        "Iterate across several steps, preserve context under pressure, and finish cleanly."
    )

    lifecycle_events: list[HarnessLifecycleEvent] = []
    final_result: HarnessResult | None = None

    async for event in harness.run_stream(prompt):
        if isinstance(event, HarnessLifecycleEvent):
            lifecycle_events.append(event)
        if isinstance(event, WorkflowOutputEvent) and isinstance(event.data, HarnessResult):
            final_result = event.data

    assert final_result is not None, "Harness did not produce a final result"

    compaction_started = [e for e in lifecycle_events if e.event_type == "compaction_started"]
    compaction_completed = [e for e in lifecycle_events if e.event_type == "compaction_completed"]

    any_tokens_freed = any((e.data or {}).get("tokens_freed", 0) > 0 for e in compaction_completed)
    turn_started = [e for e in lifecycle_events if e.event_type == "turn_started"]
    turn_completed = [e for e in lifecycle_events if e.event_type == "turn_completed"]

    transcript_event_types = [e.event_type for e in final_result.transcript]
    turn_start_count = sum(1 for t in transcript_event_types if t == "turn_start")
    stop_decision_count = sum(1 for t in transcript_event_types if t == "stop_decision")

    score = HarnessBenchmarkScore(
        stop_correctness=(
            final_result.status == HarnessStatus.DONE
            and final_result.reason is not None
            and final_result.reason.kind == "agent_done"
        ),
        turn_efficiency=(3 <= final_result.turn_count <= 8),
        compaction_correctness=(len(compaction_started) >= 1 and len(compaction_completed) >= 1 and any_tokens_freed),
        completion_integrity=(len(stop_hook_calls) >= 1 and stop_hook_calls[-1].called_work_complete),
        state_invariants=(
            len(turn_started) == len(turn_completed) == final_result.turn_count
            and turn_start_count == final_result.turn_count
            and stop_decision_count >= 1
        ),
        details={
            "turn_count": final_result.turn_count,
            "status": final_result.status.value,
            "reason": final_result.reason.kind if final_result.reason else None,
            "compaction_started": len(compaction_started),
            "compaction_completed": len(compaction_completed),
            "tokens_freed_positive": any_tokens_freed,
            "stop_hook_calls": len(stop_hook_calls),
            "turn_start_events": turn_start_count,
            "stop_decisions": stop_decision_count,
        },
    )

    assert score.total_passed == 5, f"Benchmark rubric failed: {score.details}"


@pytest.mark.asyncio
async def test_phase0_harness_shadow_parity_gate() -> None:
    """Shadow mode gate for candidate-vs-actual compaction parity signals.

    This does not switch ownership. It validates that shadow telemetry is present,
    buffer snapshots stay in parity, and divergence stays within an acceptable
    bound for the benchmark scenario.
    """

    tmp_path = Path(tempfile.mkdtemp(prefix="harness-benchmark-shadow-"))

    async def _allow_stop(_: AgentStopEvent) -> AgentStopResult | None:
        return AgentStopResult(decision="allow", reason="benchmark-allow")

    harness = AgentHarness(
        _ScriptedLongRunAgent(),
        max_turns=12,
        enable_stall_detection=True,
        stall_threshold=4,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        task_list=WorkItemTaskList(),
        enable_compaction=True,
        compaction_owner_mode="shadow",
        compaction_store=InMemoryCompactionStore(),
        artifact_store=InMemoryArtifactStore(),
        summary_cache=InMemorySummaryCache(max_entries=50),
        summarizer=_DummySummarizer(),
        max_input_tokens=1200,
        soft_threshold_percent=0.40,
        task_contract=TaskContract.simple(
            "Complete a multi-turn benchmark task",
            "Perform iterative progress across multiple turns",
            "Signal completion explicitly",
        ),
        sandbox_path=str(tmp_path),
        hooks=HarnessHooks(agent_stop=[_allow_stop]),
    )

    lifecycle_events: list[HarnessLifecycleEvent] = []
    final_result: HarnessResult | None = None

    async for event in harness.run_stream("Run shadow parity benchmark scenario"):
        if isinstance(event, HarnessLifecycleEvent):
            lifecycle_events.append(event)
        if isinstance(event, WorkflowOutputEvent) and isinstance(event.data, HarnessResult):
            final_result = event.data

    assert final_result is not None, "Harness did not produce a final result"
    assert final_result.status == HarnessStatus.DONE

    shadow_compare_payloads = [
        (e.data or {}).get("shadow_compare")
        for e in lifecycle_events
        if e.event_type == "context_pressure" and isinstance(e.data, dict) and "shadow_compare" in e.data
    ]
    shadow_compare_payloads = [p for p in shadow_compare_payloads if isinstance(p, dict)]

    assert len(shadow_compare_payloads) >= 1, "Missing shadow comparison telemetry"

    buffer_matches = [bool(p.get("buffer_parity", {}).get("match", False)) for p in shadow_compare_payloads]
    assert all(buffer_matches), f"Buffer parity mismatch in shadow compare payloads: {shadow_compare_payloads}"

    divergences = [bool(p.get("diverged", False)) for p in shadow_compare_payloads]
    divergence_rate = sum(1 for d in divergences if d) / len(divergences)

    # Gate threshold for this deterministic scenario.
    assert divergence_rate <= 0.50, (
        f"Shadow parity divergence too high: rate={divergence_rate:.2f}, "
        f"events={len(shadow_compare_payloads)}, payloads={shadow_compare_payloads}"
    )


@pytest.mark.asyncio
async def test_phase0_harness_compaction_executor_canary_gate() -> None:
    """Canary gate for compaction executor ownership path.

    Validates that owner-path compaction is exercised, fallback stays bounded,
    and core harness invariants remain stable.
    """

    tmp_path = Path(tempfile.mkdtemp(prefix="harness-benchmark-canary-"))

    async def _allow_stop(_: AgentStopEvent) -> AgentStopResult | None:
        return AgentStopResult(decision="allow", reason="benchmark-allow")

    harness = AgentHarness(
        _ScriptedLongRunAgent(),
        max_turns=12,
        enable_stall_detection=True,
        stall_threshold=4,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        task_list=WorkItemTaskList(),
        enable_compaction=True,
        compaction_owner_mode="compaction_executor",
        compaction_store=InMemoryCompactionStore(),
        artifact_store=InMemoryArtifactStore(),
        summary_cache=InMemorySummaryCache(max_entries=50),
        summarizer=_DummySummarizer(),
        max_input_tokens=1200,
        soft_threshold_percent=0.40,
        task_contract=TaskContract.simple(
            "Complete a multi-turn benchmark task",
            "Perform iterative progress across multiple turns",
            "Signal completion explicitly",
        ),
        sandbox_path=str(tmp_path),
        hooks=HarnessHooks(agent_stop=[_allow_stop]),
    )

    lifecycle_events: list[HarnessLifecycleEvent] = []
    final_result: HarnessResult | None = None

    async for event in harness.run_stream("Run compaction owner canary benchmark scenario"):
        if isinstance(event, HarnessLifecycleEvent):
            lifecycle_events.append(event)
        if isinstance(event, WorkflowOutputEvent) and isinstance(event.data, HarnessResult):
            final_result = event.data

    assert final_result is not None, "Harness did not produce a final result"
    assert final_result.status == HarnessStatus.DONE

    owner_path_completions = [
        e
        for e in lifecycle_events
        if e.event_type == "compaction_completed"
        and isinstance(e.data, dict)
        and (e.data or {}).get("compaction_owner_mode") == "compaction_executor"
        and bool((e.data or {}).get("owner_path_applied", False))
    ]
    fallback_starts = [
        e
        for e in lifecycle_events
        if e.event_type == "compaction_started"
        and isinstance(e.data, dict)
        and (e.data or {}).get("compaction_owner_mode") == "compaction_executor"
    ]
    pressure_events = [
        e
        for e in lifecycle_events
        if e.event_type == "context_pressure"
        and isinstance(e.data, dict)
        and (e.data or {}).get("compaction_owner_mode") == "compaction_executor"
    ]

    # Owner-mode canary expectations:
    # - owner path must be exercised
    # - fallback should be bootstrap-only (at most one pre-turn check)
    owner_path_turns = {e.turn_number for e in owner_path_completions}
    fallback_turns = {e.turn_number for e in fallback_starts}
    assert owner_path_turns, (
        "Compaction owner canary did not exercise owner path. "
        f"events={[e.event_type for e in lifecycle_events]}"
    )
    assert len(fallback_starts) <= 1, (
        "Unexpected fallback frequency for deterministic canary scenario. "
        f"fallback_count={len(fallback_starts)}"
    )

    compaction_completions_owner_mode = [
        e
        for e in lifecycle_events
        if e.event_type == "compaction_completed"
        and isinstance(e.data, dict)
        and (e.data or {}).get("compaction_owner_mode") == "compaction_executor"
    ]
    non_owner_completions = [
        e for e in compaction_completions_owner_mode if not bool((e.data or {}).get("owner_path_applied", False))
    ]
    non_owner_completion_turns = {e.turn_number for e in non_owner_completions}
    fallback_effect_turns = {t + 1 for t in fallback_turns}
    assert non_owner_completion_turns.issubset(fallback_effect_turns), (
        "Compaction completion without owner-path flag must correspond to explicit fallback signaling "
        "from the preceding compaction check. "
        f"non_owner_turns={sorted(non_owner_completion_turns)}, "
        f"fallback_turns={sorted(fallback_turns)}, fallback_effect_turns={sorted(fallback_effect_turns)}"
    )

    # Workflow stability invariants should still hold.
    turn_started = [e for e in lifecycle_events if e.event_type == "turn_started"]
    turn_completed = [e for e in lifecycle_events if e.event_type == "turn_completed"]
    transcript_event_types = [e.event_type for e in final_result.transcript]
    turn_start_count = sum(1 for t in transcript_event_types if t == "turn_start")
    stop_decision_count = sum(1 for t in transcript_event_types if t == "stop_decision")

    assert 3 <= final_result.turn_count <= 8
    assert final_result.reason is not None and final_result.reason.kind == "agent_done"
    assert len(turn_started) == len(turn_completed) == final_result.turn_count
    assert turn_start_count == final_result.turn_count
    assert stop_decision_count >= 1

    # If fallback occurs, context_pressure is the expected signal carrier.
    if fallback_starts:
        assert len(pressure_events) >= 1
        fallback_reasons = {
            (e.data or {}).get("owner_fallback_reason")
            for e in fallback_starts
            if isinstance(e.data, dict)
        }
        fallback_reasons.discard(None)
        assert fallback_reasons, "Missing owner_fallback_reason in fallback lifecycle telemetry"
