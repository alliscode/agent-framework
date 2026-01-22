# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness Task Contract module (Phase 3)."""

import os
import tempfile

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework._harness import (
    AcceptabilityCriteria,
    CompletionReport,
    ContractVerifier,
    CoverageLedger,
    Evidence,
    GapReport,
    HarnessResult,
    HarnessStatus,
    Predicate,
    PredicateType,
    ProgressFingerprint,
    ProgressTracker,
    RepairComplete,
    RepairExecutor,
    RepairTrigger,
    RequiredOutput,
    RequirementStatus,
    StopDecisionExecutor,
    TaskContract,
    TurnComplete,
    UserQuestion,
)
from agent_framework._harness._constants import (
    HARNESS_MAX_TURNS_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_TASK_CONTRACT_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from agent_framework._harness._repair_executor import HARNESS_CONFIG_KEY

# Test TaskContract


def test_task_contract_creation() -> None:
    """Test creating a TaskContract."""
    contract = TaskContract(
        goal="Complete the task",
        required_outputs=[
            RequiredOutput(id="R1", description="Create a report"),
            RequiredOutput(id="R2", description="Run tests", optional=True),
        ],
    )

    assert contract.goal == "Complete the task"
    assert len(contract.required_outputs) == 2
    assert contract.get_required_ids() == ["R1"]  # R2 is optional


def test_task_contract_simple_factory() -> None:
    """Test TaskContract.simple() factory method."""
    contract = TaskContract.simple(
        "Build a feature",
        "Write the code",
        "Add tests",
        "Update documentation",
    )

    assert contract.goal == "Build a feature"
    assert len(contract.required_outputs) == 3
    assert contract.required_outputs[0].id == "R1"
    assert contract.required_outputs[0].description == "Write the code"


def test_task_contract_serialization() -> None:
    """Test TaskContract serialization round-trip."""
    contract = TaskContract(
        goal="Test goal",
        required_outputs=[
            RequiredOutput(
                id="R1",
                description="Test requirement",
                predicate=Predicate.file_exists("output.txt"),
            ),
        ],
        acceptability=AcceptabilityCriteria(min_confidence=0.8),
    )

    serialized = contract.to_dict()
    restored = TaskContract.from_dict(serialized)

    assert restored.goal == "Test goal"
    assert len(restored.required_outputs) == 1
    assert restored.required_outputs[0].id == "R1"
    assert restored.acceptability.min_confidence == 0.8


# Test Predicate


def test_predicate_factories() -> None:
    """Test Predicate factory methods."""
    # file_exists
    pred1 = Predicate.file_exists("/path/to/file.txt")
    assert pred1.type == PredicateType.FILE_EXISTS
    assert pred1.args["path"] == "/path/to/file.txt"

    # contains_text
    pred2 = Predicate.contains_text("success", in_field="response")
    assert pred2.type == PredicateType.CONTAINS_TEXT
    assert pred2.args["pattern"] == "success"

    # tool_success
    pred3 = Predicate.tool_success("run_tests")
    assert pred3.type == PredicateType.TOOL_RESULT_SUCCESS
    assert pred3.args["tool_name"] == "run_tests"

    # always_true
    pred4 = Predicate.always_true("Model says it's done")
    assert pred4.type == PredicateType.ALWAYS_TRUE


def test_predicate_serialization() -> None:
    """Test Predicate serialization round-trip."""
    pred = Predicate.file_exists("/path/file.txt")

    serialized = pred.to_dict()
    restored = Predicate.from_dict(serialized)

    assert restored.type == PredicateType.FILE_EXISTS
    assert restored.args["path"] == "/path/file.txt"


# Test CoverageLedger


def test_coverage_ledger_creation() -> None:
    """Test creating a CoverageLedger."""
    contract = TaskContract.simple("Test", "Req 1", "Req 2")
    ledger = CoverageLedger.for_contract(contract)

    assert ledger.contract_id == contract.id
    assert len(ledger.coverage) == 2
    assert ledger.coverage["R1"].status == RequirementStatus.UNMET


def test_coverage_ledger_mark_met() -> None:
    """Test marking requirements as met."""
    contract = TaskContract.simple("Test", "Req 1", "Req 2")
    ledger = CoverageLedger.for_contract(contract)

    evidence = Evidence(
        event_ref="event-123",
        kind="artifact",
        value="output.txt",
    )
    ledger.mark_met("R1", evidence)

    assert ledger.coverage["R1"].status == RequirementStatus.MET
    assert len(ledger.coverage["R1"].evidence) == 1


def test_coverage_ledger_completion_check() -> None:
    """Test checking if contract is satisfied."""
    contract = TaskContract.simple("Test", "Req 1", "Req 2")
    ledger = CoverageLedger.for_contract(contract)

    # Not satisfied initially
    assert not ledger.is_contract_satisfied(contract)
    assert ledger.get_completion_percentage(contract) == 0.0

    # Mark one met
    evidence = Evidence(event_ref="e1", kind="response", value="done")
    ledger.mark_met("R1", evidence)
    assert not ledger.is_contract_satisfied(contract)
    assert ledger.get_completion_percentage(contract) == 50.0

    # Mark both met
    ledger.mark_met("R2", evidence)
    assert ledger.is_contract_satisfied(contract)
    assert ledger.get_completion_percentage(contract) == 100.0


def test_coverage_ledger_serialization() -> None:
    """Test CoverageLedger serialization round-trip."""
    contract = TaskContract.simple("Test", "Req 1")
    ledger = CoverageLedger.for_contract(contract)
    ledger.mark_met("R1", Evidence(event_ref="e1", kind="response", value="done"))

    serialized = ledger.to_dict()
    restored = CoverageLedger.from_dict(serialized)

    assert restored.contract_id == contract.id
    assert restored.coverage["R1"].status == RequirementStatus.MET


# Test ProgressFingerprint


def test_progress_fingerprint_compute() -> None:
    """Test computing progress fingerprints."""
    fp1 = ProgressFingerprint.compute(turn_number=1, transcript_length=5)
    fp2 = ProgressFingerprint.compute(turn_number=2, transcript_length=5)
    fp3 = ProgressFingerprint.compute(turn_number=3, transcript_length=10)

    # Same transcript length = same fingerprint (different turn numbers don't matter)
    assert fp1.fingerprint == fp2.fingerprint

    # Different transcript length = different fingerprint
    assert fp1.fingerprint != fp3.fingerprint


def test_progress_fingerprint_with_ledger() -> None:
    """Test fingerprint changes with ledger updates."""
    contract = TaskContract.simple("Test", "Req 1")
    ledger = CoverageLedger.for_contract(contract)

    fp1 = ProgressFingerprint.compute(turn_number=1, ledger=ledger, transcript_length=5)

    # Mark requirement met
    ledger.mark_met("R1", Evidence(event_ref="e1", kind="response", value="done"))

    fp2 = ProgressFingerprint.compute(turn_number=2, ledger=ledger, transcript_length=5)

    # Fingerprint should change when ledger status changes
    assert fp1.fingerprint != fp2.fingerprint


# Test ProgressTracker


def test_progress_tracker_stall_detection() -> None:
    """Test stall detection in ProgressTracker."""
    tracker = ProgressTracker(stall_threshold=3)

    # Add same fingerprint multiple times
    for i in range(5):
        fp = ProgressFingerprint.compute(turn_number=i + 1, transcript_length=10)
        tracker.add_fingerprint(fp)

    # Should detect stall after threshold
    assert tracker.is_stalled()
    assert tracker.get_stall_duration() == 5


def test_progress_tracker_no_stall_with_progress() -> None:
    """Test that progress prevents stall detection."""
    tracker = ProgressTracker(stall_threshold=3)

    # Add fingerprints with changing transcript length
    for i in range(5):
        fp = ProgressFingerprint.compute(turn_number=i + 1, transcript_length=10 + i)
        tracker.add_fingerprint(fp)

    # Should not detect stall
    assert not tracker.is_stalled()


# Test ContractVerifier


def test_verifier_always_true() -> None:
    """Test verifier with always_true predicate."""
    verifier = ContractVerifier()
    pred = Predicate.always_true("Test")

    result = verifier.verify_predicate(pred)

    assert result.success
    assert result.evidence is not None


def test_verifier_file_exists() -> None:
    """Test verifier with file_exists predicate."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name

    try:
        verifier = ContractVerifier()

        # File exists
        pred1 = Predicate.file_exists(temp_path)
        result1 = verifier.verify_predicate(pred1)
        assert result1.success

        # File doesn't exist
        pred2 = Predicate.file_exists("/nonexistent/file.txt")
        result2 = verifier.verify_predicate(pred2)
        assert not result2.success
    finally:
        os.unlink(temp_path)


def test_verifier_contains_text() -> None:
    """Test verifier with contains_text predicate."""
    transcript = [
        {"event_type": "agent_response", "data": {"message": "The task is complete and successful."}},
    ]
    verifier = ContractVerifier(transcript=transcript)

    # Pattern found
    pred1 = Predicate.contains_text("successful")
    result1 = verifier.verify_predicate(pred1)
    assert result1.success

    # Pattern not found
    pred2 = Predicate.contains_text("failed")
    result2 = verifier.verify_predicate(pred2)
    assert not result2.success


def test_verifier_tool_success() -> None:
    """Test verifier with tool_result_success predicate."""
    transcript = [
        {"event_type": "tool_result", "event_id": "t1", "data": {"tool_name": "run_tests"}},
        {"event_type": "tool_result", "event_id": "t2", "data": {"tool_name": "deploy", "error": "Failed"}},
    ]
    verifier = ContractVerifier(transcript=transcript)

    # Tool succeeded
    pred1 = Predicate.tool_success("run_tests")
    result1 = verifier.verify_predicate(pred1)
    assert result1.success

    # Tool failed
    pred2 = Predicate.tool_success("deploy")
    result2 = verifier.verify_predicate(pred2)
    assert not result2.success


def test_verifier_contract_verification() -> None:
    """Test full contract verification."""
    contract = TaskContract(
        goal="Test task",
        required_outputs=[
            RequiredOutput(id="R1", description="Soft req", predicate=Predicate.always_true()),
            RequiredOutput(id="R2", description="Hard req", predicate=Predicate.contains_text("done")),
        ],
    )

    transcript = [
        {"event_type": "agent_response", "data": {"message": "Task is done!"}},
    ]
    verifier = ContractVerifier(transcript=transcript)
    ledger = CoverageLedger.for_contract(contract)

    result = verifier.verify_contract(contract, ledger)

    assert result.satisfied
    assert ledger.coverage["R1"].status == RequirementStatus.MET
    assert ledger.coverage["R2"].status == RequirementStatus.MET


# Test GapReport


def test_gap_report_generation() -> None:
    """Test generating gap reports."""
    contract = TaskContract.simple("Test", "Req 1", "Req 2", "Req 3")
    ledger = CoverageLedger.for_contract(contract)

    # Mark one met
    ledger.mark_met("R1", Evidence(event_ref="e1", kind="response", value="done"))

    gap_report = GapReport.from_contract_and_ledger(contract, ledger)

    assert len(gap_report.unmet_requirements) == 2
    assert gap_report.unmet_requirements[0]["id"] == "R2"
    assert gap_report.unmet_requirements[1]["id"] == "R3"


# Test CompletionReport


def test_completion_report_creation() -> None:
    """Test creating a CompletionReport."""
    report = CompletionReport(
        recommendation="done",
        deliverables=[{"type": "file", "path": "output.txt"}],
        confidence=0.95,
        summary="Task completed successfully",
    )

    assert report.recommendation == "done"
    assert report.confidence == 0.95
    assert len(report.deliverables) == 1


def test_completion_report_serialization() -> None:
    """Test CompletionReport serialization round-trip."""
    report = CompletionReport(
        recommendation="need_user",
        open_questions=["What format?"],
        confidence=0.6,
    )

    serialized = report.to_dict()
    restored = CompletionReport.from_dict(serialized)

    assert restored.recommendation == "need_user"
    assert restored.open_questions == ["What format?"]


# Test StopDecisionExecutor with contract verification


async def test_stop_decision_with_contract_verification_satisfied() -> None:
    """Test StopDecisionExecutor accepts done when contract is satisfied."""

    class TestAgentTurnExecutor(Executor):
        @handler
        async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
            turn = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
            turn = (turn or 0) + 1
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn)
            # Signal done
            await ctx.send_message(TurnComplete(agent_done=True))

    class SetupExecutor(Executor):
        @handler
        async def setup(self, msg: str, ctx: WorkflowContext[RepairTrigger]) -> None:
            # Set up contract with always_true predicate (will be satisfied)
            contract = TaskContract.simple("Test", "Do something")
            await ctx.set_shared_state(HARNESS_TASK_CONTRACT_KEY, contract.to_dict())
            await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, [])
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, 0)
            await ctx.set_shared_state(HARNESS_MAX_TURNS_KEY, 50)
            await ctx.set_shared_state(HARNESS_STATUS_KEY, HarnessStatus.RUNNING.value)
            await ctx.send_message(RepairTrigger())

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: SetupExecutor(id="setup"), name="setup")
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: TestAgentTurnExecutor(id="agent"), name="agent")
        .register_executor(
            lambda: StopDecisionExecutor(
                enable_contract_verification=True,
                id="stop",
            ),
            name="stop",
        )
        .add_edge("setup", "repair")
        .add_edge("repair", "agent")
        .add_edge("agent", "stop")
        .add_edge("stop", "repair")
        .set_start_executor("setup")
        .set_max_iterations(50)
        .build()
    )

    result = await workflow.run("start")
    outputs = result.get_outputs()

    assert len(outputs) == 1
    harness_result = outputs[0]
    assert isinstance(harness_result, HarnessResult)
    assert harness_result.status == HarnessStatus.DONE
    assert harness_result.reason is not None
    assert harness_result.reason.kind == "agent_done"


async def test_stop_decision_with_stall_detection() -> None:
    """Test StopDecisionExecutor detects stalled progress."""

    class StuckAgentExecutor(Executor):
        """Agent that never makes progress."""

        @handler
        async def handle(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
            turn = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
            turn = (turn or 0) + 1
            await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn)
            # Never signal done, transcript stays same length
            await ctx.send_message(TurnComplete(agent_done=False))

    workflow = (
        WorkflowBuilder()
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: StuckAgentExecutor(id="agent"), name="agent")
        .register_executor(
            lambda: StopDecisionExecutor(
                enable_stall_detection=True,
                stall_threshold=3,
                id="stop",
            ),
            name="stop",
        )
        .add_edge("repair", "agent")
        .add_edge("agent", "stop")
        .add_edge("stop", "repair")
        .set_start_executor("repair")
        .set_max_iterations(100)
        .build()
    )

    result = await workflow.run(
        RepairTrigger(),
        **{HARNESS_CONFIG_KEY: {"max_turns": 20}},
    )
    outputs = result.get_outputs()

    assert len(outputs) == 1
    harness_result = outputs[0]
    assert isinstance(harness_result, HarnessResult)
    assert harness_result.status == HarnessStatus.STALLED
    assert harness_result.reason is not None
    assert harness_result.reason.kind == "stalled"


# Test UserQuestion


def test_user_question_creation() -> None:
    """Test creating UserQuestion."""
    question = UserQuestion(
        id="Q1",
        question="What format do you want?",
        required_for=["R1", "R2"],
    )

    assert question.id == "Q1"
    assert not question.asked
    assert question.answer is None


def test_user_question_serialization() -> None:
    """Test UserQuestion serialization round-trip."""
    question = UserQuestion(
        id="Q1",
        question="What format?",
        asked=True,
        answer="JSON",
    )

    serialized = question.to_dict()
    restored = UserQuestion.from_dict(serialized)

    assert restored.asked
    assert restored.answer == "JSON"


# Test AcceptabilityCriteria


def test_acceptability_criteria_defaults() -> None:
    """Test AcceptabilityCriteria defaults."""
    criteria = AcceptabilityCriteria()

    assert criteria.max_known_unknowns == 0
    assert criteria.min_confidence == 0.6
    assert not criteria.allow_partial


def test_acceptability_criteria_serialization() -> None:
    """Test AcceptabilityCriteria serialization round-trip."""
    criteria = AcceptabilityCriteria(
        max_known_unknowns=2,
        min_confidence=0.8,
        allow_partial=True,
    )

    serialized = criteria.to_dict()
    restored = AcceptabilityCriteria.from_dict(serialized)

    assert restored.max_known_unknowns == 2
    assert restored.min_confidence == 0.8
    assert restored.allow_partial
