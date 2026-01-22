# Copyright (c) Microsoft. All rights reserved.

"""Task Contract types for Agent Harness.

Task Contracts define explicit completion criteria for agent tasks.
The harness uses these to verify that work is actually complete,
not just that the agent claims to be done.

This module provides:
- TaskContract: Defines what "done" means for a task
- CoverageLedger: Tracks which requirements have been satisfied
- CompletionReport: What the agent submits when proposing completion
- ProgressFingerprint: For detecting stalled execution
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4


class RequirementStatus(Enum):
    """Status of a requirement in the coverage ledger."""

    UNMET = "unmet"
    MET = "met"
    PARTIALLY_MET = "partially_met"
    SKIPPED = "skipped"  # Explicitly skipped by user


class PredicateType(Enum):
    """Types of predicates that can verify requirements."""

    FILE_EXISTS = "file_exists"
    JSON_SCHEMA_VALID = "json_schema_valid"
    CONTAINS_TEXT = "contains_text"
    TOOL_RESULT_SUCCESS = "tool_result_success"
    CUSTOM = "custom"
    ALWAYS_TRUE = "always_true"  # For requirements verified by model judgment


@dataclass
class Predicate:
    """A verifiable condition for a requirement.

    Attributes:
        type: The type of predicate.
        args: Arguments for the predicate (type-specific).
        description: Human-readable description of what this checks.
    """

    type: PredicateType
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type.value,
            "args": self.args,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Predicate":
        """Deserialize from dictionary."""
        return cls(
            type=PredicateType(data["type"]),
            args=data.get("args", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def file_exists(cls, path: str) -> "Predicate":
        """Create a file_exists predicate."""
        return cls(
            type=PredicateType.FILE_EXISTS,
            args={"path": path},
            description=f"File exists: {path}",
        )

    @classmethod
    def contains_text(cls, pattern: str, in_field: str = "response") -> "Predicate":
        """Create a contains_text predicate."""
        return cls(
            type=PredicateType.CONTAINS_TEXT,
            args={"pattern": pattern, "field": in_field},
            description=f"Contains: {pattern}",
        )

    @classmethod
    def tool_success(cls, tool_name: str) -> "Predicate":
        """Create a tool_result_success predicate."""
        return cls(
            type=PredicateType.TOOL_RESULT_SUCCESS,
            args={"tool_name": tool_name},
            description=f"Tool succeeded: {tool_name}",
        )

    @classmethod
    def always_true(cls, description: str = "Verified by model judgment") -> "Predicate":
        """Create an always_true predicate (for soft requirements)."""
        return cls(
            type=PredicateType.ALWAYS_TRUE,
            args={},
            description=description,
        )


@dataclass
class RequiredOutput:
    """A single required output for task completion.

    Attributes:
        id: Unique identifier for this requirement.
        description: Human-readable description of what's required.
        type: Type of output (artifact, response, or action).
        predicate: How to verify this requirement is met.
        optional: If True, task can complete without this.
    """

    id: str
    description: str
    type: Literal["artifact", "response", "action"] = "response"
    predicate: Predicate = field(default_factory=lambda: Predicate.always_true())
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "predicate": self.predicate.to_dict(),
            "optional": self.optional,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequiredOutput":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            type=data.get("type", "response"),
            predicate=Predicate.from_dict(data["predicate"]) if "predicate" in data else Predicate.always_true(),
            optional=data.get("optional", False),
        )


@dataclass
class UserQuestion:
    """A question that must be asked if information is missing.

    Attributes:
        id: Unique identifier for this question.
        question: The question to ask.
        required_for: List of requirement IDs this question helps satisfy.
        asked: Whether the question has been asked.
        answer: The user's answer, if provided.
    """

    id: str
    question: str
    required_for: list[str] = field(default_factory=list)
    asked: bool = False
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "required_for": self.required_for,
            "asked": self.asked,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserQuestion":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            required_for=data.get("required_for", []),
            asked=data.get("asked", False),
            answer=data.get("answer"),
        )


@dataclass
class AcceptabilityCriteria:
    """Criteria for when a task is acceptably complete.

    Attributes:
        max_known_unknowns: Maximum number of unresolved uncertainties allowed.
        min_confidence: Minimum confidence level (0-1) for completion.
        allow_partial: Whether partial completion is acceptable.
    """

    max_known_unknowns: int = 0
    min_confidence: float = 0.6
    allow_partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_known_unknowns": self.max_known_unknowns,
            "min_confidence": self.min_confidence,
            "allow_partial": self.allow_partial,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcceptabilityCriteria":
        """Deserialize from dictionary."""
        return cls(
            max_known_unknowns=data.get("max_known_unknowns", 0),
            min_confidence=data.get("min_confidence", 0.6),
            allow_partial=data.get("allow_partial", False),
        )


@dataclass
class TaskContract:
    """Contract defining completion criteria for a task.

    The TaskContract is the authoritative definition of what "done" means.
    It should be created early in the task and persisted throughout execution.

    Attributes:
        id: Unique identifier for this contract.
        goal: High-level description of the task goal.
        required_outputs: List of required outputs for completion.
        questions: Questions to ask if information is missing.
        acceptability: Criteria for acceptable completion.
        created_at: When the contract was created.
    """

    goal: str
    required_outputs: list[RequiredOutput] = field(default_factory=list)
    questions: list[UserQuestion] = field(default_factory=list)
    acceptability: AcceptabilityCriteria = field(default_factory=AcceptabilityCriteria)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_required_ids(self) -> list[str]:
        """Get IDs of all non-optional requirements."""
        return [r.id for r in self.required_outputs if not r.optional]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "goal": self.goal,
            "required_outputs": [r.to_dict() for r in self.required_outputs],
            "questions": [q.to_dict() for q in self.questions],
            "acceptability": self.acceptability.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskContract":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            goal=data["goal"],
            required_outputs=[RequiredOutput.from_dict(r) for r in data.get("required_outputs", [])],
            questions=[UserQuestion.from_dict(q) for q in data.get("questions", [])],
            acceptability=AcceptabilityCriteria.from_dict(data.get("acceptability", {})),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def simple(cls, goal: str, *requirements: str) -> "TaskContract":
        """Create a simple contract with basic requirements.

        Args:
            goal: The task goal.
            *requirements: Descriptions of required outputs.

        Returns:
            A TaskContract with the specified requirements.
        """
        outputs = [
            RequiredOutput(
                id=f"R{i + 1}",
                description=req,
                predicate=Predicate.always_true(f"Requirement: {req}"),
            )
            for i, req in enumerate(requirements)
        ]
        return cls(goal=goal, required_outputs=outputs)


@dataclass
class Evidence:
    """Evidence that a requirement has been satisfied.

    Attributes:
        event_ref: Reference to the event that provides evidence.
        kind: Type of evidence.
        value: The evidence value (path, content, etc.).
        timestamp: When the evidence was recorded.
    """

    event_ref: str
    kind: Literal["artifact", "response", "tool_result", "user_confirmation"]
    value: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_ref": self.event_ref,
            "kind": self.kind,
            "value": self.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        """Deserialize from dictionary."""
        return cls(
            event_ref=data["event_ref"],
            kind=data["kind"],
            value=data["value"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class RequirementCoverage:
    """Coverage status for a single requirement.

    Attributes:
        requirement_id: ID of the requirement.
        status: Current status.
        evidence: List of evidence supporting the status.
        notes: Optional notes about the coverage.
    """

    requirement_id: str
    status: RequirementStatus = RequirementStatus.UNMET
    evidence: list[Evidence] = field(default_factory=list)
    notes: str = ""

    def mark_met(self, evidence: Evidence) -> None:
        """Mark this requirement as met with evidence."""
        self.status = RequirementStatus.MET
        self.evidence.append(evidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "requirement_id": self.requirement_id,
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequirementCoverage":
        """Deserialize from dictionary."""
        return cls(
            requirement_id=data["requirement_id"],
            status=RequirementStatus(data.get("status", "unmet")),
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            notes=data.get("notes", ""),
        )


@dataclass
class CoverageLedger:
    """Ledger tracking requirement coverage for a task.

    The CoverageLedger is the authoritative record of which requirements
    have been satisfied and what evidence supports that.

    Attributes:
        contract_id: ID of the associated TaskContract.
        coverage: Coverage status for each requirement.
        last_updated: When the ledger was last updated.
    """

    contract_id: str
    coverage: dict[str, RequirementCoverage] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_coverage(self, requirement_id: str) -> RequirementCoverage:
        """Get coverage for a requirement, creating if needed."""
        if requirement_id not in self.coverage:
            self.coverage[requirement_id] = RequirementCoverage(requirement_id=requirement_id)
        return self.coverage[requirement_id]

    def mark_met(self, requirement_id: str, evidence: Evidence) -> None:
        """Mark a requirement as met with evidence."""
        coverage = self.get_coverage(requirement_id)
        coverage.mark_met(evidence)
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def get_unmet_requirements(self, contract: TaskContract) -> list[str]:
        """Get IDs of unmet non-optional requirements."""
        unmet = []
        for req in contract.required_outputs:
            if req.optional:
                continue
            coverage = self.coverage.get(req.id)
            if coverage is None or coverage.status != RequirementStatus.MET:
                unmet.append(req.id)
        return unmet

    def is_contract_satisfied(self, contract: TaskContract) -> bool:
        """Check if all non-optional requirements are met."""
        return len(self.get_unmet_requirements(contract)) == 0

    def get_completion_percentage(self, contract: TaskContract) -> float:
        """Calculate percentage of requirements met."""
        required_ids = contract.get_required_ids()
        if not required_ids:
            return 100.0

        met_count = sum(
            1 for req_id in required_ids
            if self.coverage.get(req_id, RequirementCoverage(req_id)).status == RequirementStatus.MET
        )
        return (met_count / len(required_ids)) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "contract_id": self.contract_id,
            "coverage": {k: v.to_dict() for k, v in self.coverage.items()},
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoverageLedger":
        """Deserialize from dictionary."""
        return cls(
            contract_id=data["contract_id"],
            coverage={k: RequirementCoverage.from_dict(v) for k, v in data.get("coverage", {}).items()},
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def for_contract(cls, contract: TaskContract) -> "CoverageLedger":
        """Create a new ledger for a contract."""
        ledger = cls(contract_id=contract.id)
        # Initialize coverage for all requirements
        for req in contract.required_outputs:
            ledger.coverage[req.id] = RequirementCoverage(requirement_id=req.id)
        return ledger


@dataclass
class CompletionReport:
    """Report submitted by agent when proposing task completion.

    Attributes:
        recommendation: The agent's recommendation (done, need_user, continue).
        deliverables: List of deliverables produced.
        requirements_coverage: Agent's assessment of requirement coverage.
        open_questions: Any unresolved questions.
        confidence: Agent's confidence in completion (0-1).
        summary: Brief summary of what was accomplished.
    """

    recommendation: Literal["done", "need_user", "continue"]
    deliverables: list[dict[str, str]] = field(default_factory=list)
    requirements_coverage: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    confidence: float = 0.8
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "recommendation": self.recommendation,
            "deliverables": self.deliverables,
            "requirements_coverage": self.requirements_coverage,
            "open_questions": self.open_questions,
            "confidence": self.confidence,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletionReport":
        """Deserialize from dictionary."""
        return cls(
            recommendation=data.get("recommendation", "continue"),
            deliverables=data.get("deliverables", []),
            requirements_coverage=data.get("requirements_coverage", []),
            open_questions=data.get("open_questions", []),
            confidence=data.get("confidence", 0.8),
            summary=data.get("summary", ""),
        )


@dataclass
class ProgressFingerprint:
    """Fingerprint for detecting stalled progress.

    The fingerprint is a hash of key state that should change if
    progress is being made. If it stays the same across multiple
    turns, the agent may be stuck.

    Attributes:
        fingerprint: The hash value.
        turn_number: Turn when this fingerprint was taken.
        timestamp: When the fingerprint was taken.
    """

    fingerprint: str
    turn_number: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fingerprint": self.fingerprint,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressFingerprint":
        """Deserialize from dictionary."""
        return cls(
            fingerprint=data["fingerprint"],
            turn_number=data["turn_number"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def compute(
        cls,
        turn_number: int,
        ledger: CoverageLedger | None = None,
        transcript_length: int = 0,
        artifacts: list[str] | None = None,
        work_item_statuses: dict[str, str] | None = None,
    ) -> "ProgressFingerprint":
        """Compute a progress fingerprint from current state.

        Args:
            turn_number: Current turn number.
            ledger: Coverage ledger (if available).
            transcript_length: Length of transcript.
            artifacts: List of artifact identifiers.
            work_item_statuses: Mapping of work item IDs to status values.
                When provided, work item changes count as progress
                (prevents false stall detection).

        Returns:
            A new ProgressFingerprint.
        """
        # Build state to hash
        state: dict[str, Any] = {
            "transcript_length": transcript_length,
            "artifacts": sorted(artifacts or []),
        }

        if ledger:
            # Include coverage status in fingerprint
            state["coverage"] = {
                req_id: cov.status.value
                for req_id, cov in ledger.coverage.items()
            }

        if work_item_statuses:
            # Include work item statuses so changes count as progress
            state["work_item_statuses"] = dict(sorted(work_item_statuses.items()))

        # Compute hash
        state_str = json.dumps(state, sort_keys=True)
        fingerprint = hashlib.sha256(state_str.encode()).hexdigest()[:16]

        return cls(fingerprint=fingerprint, turn_number=turn_number)


@dataclass
class ProgressTracker:
    """Tracks progress fingerprints to detect stalls.

    Attributes:
        fingerprints: History of fingerprints.
        stall_threshold: Number of unchanged fingerprints before stall.
    """

    fingerprints: list[ProgressFingerprint] = field(default_factory=list)
    stall_threshold: int = 3

    def add_fingerprint(self, fingerprint: ProgressFingerprint) -> None:
        """Add a new fingerprint to the history."""
        self.fingerprints.append(fingerprint)
        # Keep only recent history
        if len(self.fingerprints) > self.stall_threshold * 2:
            self.fingerprints = self.fingerprints[-self.stall_threshold * 2:]

    def is_stalled(self) -> bool:
        """Check if progress has stalled.

        Returns True if the last stall_threshold fingerprints are identical.
        """
        if len(self.fingerprints) < self.stall_threshold:
            return False

        recent = self.fingerprints[-self.stall_threshold:]
        return all(fp.fingerprint == recent[0].fingerprint for fp in recent)

    def get_stall_duration(self) -> int:
        """Get number of turns with unchanged fingerprint."""
        if not self.fingerprints:
            return 0

        current = self.fingerprints[-1].fingerprint
        count = 0
        for fp in reversed(self.fingerprints):
            if fp.fingerprint == current:
                count += 1
            else:
                break
        return count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fingerprints": [fp.to_dict() for fp in self.fingerprints],
            "stall_threshold": self.stall_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProgressTracker":
        """Deserialize from dictionary."""
        return cls(
            fingerprints=[ProgressFingerprint.from_dict(fp) for fp in data.get("fingerprints", [])],
            stall_threshold=data.get("stall_threshold", 3),
        )


@dataclass
class GapReport:
    """Report of gaps between current state and contract satisfaction.

    Generated when agent proposes completion but contract is not satisfied.

    Attributes:
        unmet_requirements: List of unmet requirement IDs and descriptions.
        unanswered_questions: Questions that need answers.
        suggestions: Suggested next steps to close gaps.
    """

    unmet_requirements: list[dict[str, str]] = field(default_factory=list)
    unanswered_questions: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "unmet_requirements": self.unmet_requirements,
            "unanswered_questions": self.unanswered_questions,
            "suggestions": self.suggestions,
        }

    @classmethod
    def from_contract_and_ledger(
        cls,
        contract: TaskContract,
        ledger: CoverageLedger,
    ) -> "GapReport":
        """Generate a gap report from contract and ledger.

        Args:
            contract: The task contract.
            ledger: The coverage ledger.

        Returns:
            A GapReport describing what's missing.
        """
        unmet = []
        for req_id in ledger.get_unmet_requirements(contract):
            # Find the requirement
            req = next((r for r in contract.required_outputs if r.id == req_id), None)
            if req:
                unmet.append({"id": req_id, "description": req.description})

        unanswered = [
            q.question for q in contract.questions
            if not q.asked or q.answer is None
        ]

        suggestions = []
        if unmet:
            suggestions.append(f"Complete {len(unmet)} remaining requirement(s)")
        if unanswered:
            suggestions.append(f"Ask {len(unanswered)} clarifying question(s)")

        return cls(
            unmet_requirements=unmet,
            unanswered_questions=unanswered,
            suggestions=suggestions,
        )
