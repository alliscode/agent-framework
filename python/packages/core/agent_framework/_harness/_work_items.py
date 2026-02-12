# Copyright (c) Microsoft. All rights reserved.

"""Work Item tracking for Agent Harness self-critique loop.

This module provides the work item tracking system that enables agents to:
1. Plan subtasks by adding work items
2. Track progress by updating item status
3. Self-critique by listing items and checking completeness

The harness auto-injects work item tools at runtime and verifies
completeness before accepting the agent's done signal.
"""

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

from .._middleware import FunctionInvocationContext, FunctionMiddleware
from .._tools import ai_function

if TYPE_CHECKING:
    from .._tools import AIFunction

# ============================================================
# Artifact Contamination Detection
# ============================================================


class ArtifactContaminationLevel(Enum):
    """Classification of narrative contamination in artifact content."""

    CLEAN = "clean"
    LIGHT = "light_contamination"
    HEAVY = "heavy_contamination"


@dataclass
class ArtifactValidationResult:
    """Result of artifact content validation.

    Attributes:
        level: The contamination classification.
        cleaned_content: Content with boundary narration stripped (same as input if CLEAN).
        removed_lines: Lines that were removed during cleaning.
        message: Human-readable message describing the result.
    """

    level: ArtifactContaminationLevel
    cleaned_content: str
    removed_lines: list[str]
    message: str


# Preamble patterns - detected case-insensitively at line starts (first 5 lines)
_PREAMBLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*I will now\b", re.IGNORECASE),
    re.compile(r"^\s*I'll (create|write|document|generate|produce|build|prepare)\b", re.IGNORECASE),
    re.compile(r"^\s*Let me\b", re.IGNORECASE),
    re.compile(r"^\s*Now I'll\b", re.IGNORECASE),
    re.compile(r"^\s*Here is the\b", re.IGNORECASE),
    re.compile(r"^\s*Here's the\b", re.IGNORECASE),
    re.compile(r"^\s*Below is\b", re.IGNORECASE),
    re.compile(r"^\s*I'm going to\b", re.IGNORECASE),
    re.compile(r"^\s*The following is\b", re.IGNORECASE),
]

# Postamble patterns - detected case-insensitively at line starts (last 5 lines)
_POSTAMBLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*I've (stored|saved|recorded|created|written|documented)\b", re.IGNORECASE),
    re.compile(r"^\s*I'll now (move|proceed|continue)\b", re.IGNORECASE),
    re.compile(r"^\s*Next,? I'll\b", re.IGNORECASE),
    re.compile(r"^\s*Moving on to\b", re.IGNORECASE),
    re.compile(r"^\s*This has been (stored|saved|recorded)\b", re.IGNORECASE),
    re.compile(r"^\s*The above has been\b", re.IGNORECASE),
]

# Boundary zone size (number of lines at start/end to check for preamble/postamble)
_BOUNDARY_ZONE_SIZE = 5


def _is_narration_line(line: str, patterns: list[re.Pattern[str]]) -> bool:
    """Check if a line matches any of the given narration patterns."""
    return any(p.search(line) for p in patterns)


def _find_narration_lines(
    lines: list[str],
) -> tuple[list[int], list[int], list[int]]:
    """Find indices of narration lines in preamble and postamble boundary zones.

    Only checks the first/last N lines (boundary zones) for known narration
    patterns. Interior lines are never flagged — the agent's content is
    accepted as-is. This avoids false positives when artifact content
    legitimately discusses tools, work items, or other harness concepts.

    Returns:
        Tuple of (preamble_indices, postamble_indices, interior_indices).
        interior_indices is always empty (kept for API compatibility).
    """
    preamble_indices: list[int] = []
    postamble_indices: list[int] = []

    total = len(lines)
    preamble_end = min(_BOUNDARY_ZONE_SIZE, total)
    postamble_start = max(0, total - _BOUNDARY_ZONE_SIZE)

    # Scan preamble zone
    for i in range(preamble_end):
        if not lines[i].strip():
            continue
        if _is_narration_line(lines[i], _PREAMBLE_PATTERNS):
            preamble_indices.append(i)

    # Scan postamble zone
    for i in range(postamble_start, total):
        if not lines[i].strip():
            continue
        # Skip if already counted as preamble (overlapping zones in short content)
        if i in preamble_indices:
            continue
        if _is_narration_line(lines[i], _POSTAMBLE_PATTERNS):
            postamble_indices.append(i)

    return preamble_indices, postamble_indices, []


def validate_artifact_content(artifact: str) -> ArtifactValidationResult:
    """Validate artifact content for boundary narration and auto-strip it.

    Classification:
    - CLEAN: No narration lines detected in boundary zones.
    - LIGHT: Narration detected in boundary zones (first/last 5 lines).
             Auto-strips boundary narration and stores cleaned content.

    Interior content is never rejected. This avoids false positives when
    the artifact legitimately discusses tools, work items, or harness
    concepts, and prevents rejection loops that degrade agent performance.

    Args:
        artifact: The artifact content to validate.

    Returns:
        ArtifactValidationResult with classification and cleaned content.
    """
    lines = artifact.split("\n")

    preamble_idx, postamble_idx, _interior_idx = _find_narration_lines(lines)

    total_narration = len(preamble_idx) + len(postamble_idx)

    # CLEAN: no narration detected
    if total_narration == 0:
        return ArtifactValidationResult(
            level=ArtifactContaminationLevel.CLEAN,
            cleaned_content=artifact,
            removed_lines=[],
            message="",
        )

    # LIGHT: boundary narration detected - auto-strip and store
    boundary_indices = sorted(set(preamble_idx + postamble_idx))
    removed = [lines[i] for i in boundary_indices]

    # Strip narration lines and clean up leading/trailing blank lines
    boundary_set = set(boundary_indices)
    cleaned_lines = [line for i, line in enumerate(lines) if i not in boundary_set]

    # Strip leading blank lines
    while cleaned_lines and not cleaned_lines[0].strip():
        cleaned_lines.pop(0)

    # Strip trailing blank lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    cleaned_content = "\n".join(cleaned_lines)

    return ArtifactValidationResult(
        level=ArtifactContaminationLevel.LIGHT,
        cleaned_content=cleaned_content,
        removed_lines=removed,
        message=(
            f"Artifact stored (auto-cleaned): removed {len(removed)} "
            f"narration line(s) from boundaries. "
            f"Reminder: artifact content should be pure deliverable data. "
            f"Place process commentary in your response text instead."
        ),
    )


def validate_control_artifact(artifact: str) -> tuple[bool, str]:
    """Validate that a control artifact has the required structured format.

    Control artifacts must be valid JSON with:
    - verdict: "pass" or "fail"
    - checks: list of {name, result, detail} objects
    - summary: string explaining the overall verdict

    Args:
        artifact: The artifact content to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    import json

    try:
        data = json.loads(artifact)
    except (json.JSONDecodeError, ValueError) as e:
        return False, f"Control artifact must be valid JSON: {e}"

    if not isinstance(data, dict):
        return False, "Control artifact must be a JSON object"

    if "verdict" not in data:
        return False, "Missing required field 'verdict' (must be 'pass' or 'fail')"
    if data["verdict"] not in ("pass", "fail"):
        return False, f"'verdict' must be 'pass' or 'fail', got '{data['verdict']}'"

    if "checks" not in data:
        return False, "Missing required field 'checks' (array of check objects)"
    if not isinstance(data["checks"], list):
        return False, "'checks' must be an array"
    if len(data["checks"]) == 0:
        return False, "'checks' must contain at least one check"

    for i, check in enumerate(data["checks"]):
        if not isinstance(check, dict):
            return False, f"checks[{i}] must be an object"
        for required_field in ("name", "result", "detail"):
            if required_field not in check:
                return False, f"checks[{i}] missing required field '{required_field}'"
        if check["result"] not in ("pass", "fail"):
            return False, f"checks[{i}].result must be 'pass' or 'fail', got '{check['result']}'"

    if "summary" not in data:
        return False, "Missing required field 'summary'"
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return False, "'summary' must be a non-empty string"

    return True, ""


class WorkItemStatus(Enum):
    """Status of a work item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


class WorkItemPriority(Enum):
    """Priority level for a work item."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ArtifactRole(Enum):
    """Role classification for work item artifacts."""

    DELIVERABLE = "deliverable"
    WORKING = "working"
    CONTROL = "control"


@dataclass
class WorkItem:
    """A single work item in the ledger.

    Attributes:
        id: Auto-generated 8-char ID.
        title: Brief description of the work item.
        status: Current status of the item.
        priority: Priority level.
        notes: Optional context or notes.
        artifact: Structured output produced by this step.
        requires_revision: Whether this item's output needs correction.
        revision_of: ID of the parent item this is a revision of.
        created_at: ISO timestamp when created.
        updated_at: ISO timestamp when last updated.
    """

    id: str
    title: str
    status: WorkItemStatus = WorkItemStatus.PENDING
    priority: WorkItemPriority = WorkItemPriority.MEDIUM
    notes: str = ""
    artifact: str = ""
    artifact_role: ArtifactRole = ArtifactRole.WORKING
    requires_revision: bool = False
    revision_of: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "priority": self.priority.value,
            "notes": self.notes,
            "artifact": self.artifact,
            "artifact_role": self.artifact_role.value,
            "requires_revision": self.requires_revision,
            "revision_of": self.revision_of,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkItem":
        """Deserialize from dictionary."""
        try:
            artifact_role = ArtifactRole(data.get("artifact_role", "working"))
        except ValueError:
            artifact_role = ArtifactRole.WORKING
        return cls(
            id=data["id"],
            title=data["title"],
            status=WorkItemStatus(data.get("status", "pending")),
            priority=WorkItemPriority(data.get("priority", "medium")),
            notes=data.get("notes", ""),
            artifact=data.get("artifact", ""),
            artifact_role=artifact_role,
            requires_revision=data.get("requires_revision", False),
            revision_of=data.get("revision_of", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class WorkItemLedger:
    """Ledger of all work items for a task.

    Attributes:
        items: Dictionary mapping item IDs to WorkItem objects.
    """

    items: dict[str, WorkItem] = field(default_factory=dict)

    def add_item(self, item: WorkItem) -> None:
        """Add an item to the ledger."""
        self.items[item.id] = item

    def update_status(self, item_id: str, status: WorkItemStatus, notes: str = "") -> bool:
        """Update the status of an item.

        Args:
            item_id: ID of the item to update.
            status: New status.
            notes: Optional notes to append.

        Returns:
            True if the item was found and updated.
        """
        item = self.items.get(item_id)
        if item is None:
            return False
        item.status = status
        item.updated_at = datetime.now(timezone.utc).isoformat()
        if notes:
            item.notes = notes
        return True

    def get_incomplete_items(self) -> list[WorkItem]:
        """Get all items that are not done or skipped, plus unresolved revisions."""
        incomplete = [
            item for item in self.items.values() if item.status not in (WorkItemStatus.DONE, WorkItemStatus.SKIPPED)
        ]
        # Also include done items that still require revision without a completed fix
        for item in self.items.values():
            if item.requires_revision and not self._has_completed_revision(item.id) and item not in incomplete:
                incomplete.append(item)
        return incomplete

    def _has_completed_revision(self, item_id: str) -> bool:
        """Check if an item has a completed revision child."""
        for item in self.items.values():
            if item.revision_of == item_id and item.status in (WorkItemStatus.DONE, WorkItemStatus.SKIPPED):
                return True
        return False

    def get_items_needing_revision(self) -> list[WorkItem]:
        """Get items flagged for revision that don't have a completed fix."""
        return [
            item for item in self.items.values() if item.requires_revision and not self._has_completed_revision(item.id)
        ]

    def get_deliverables(self) -> list[WorkItem]:
        """Get all items with deliverable role that have artifact content."""
        return [
            item for item in self.items.values() if item.artifact_role == ArtifactRole.DELIVERABLE and item.artifact
        ]

    def is_all_complete(self) -> bool:
        """Check if all items are done/skipped and no unresolved revisions."""
        if not self.items:
            return True
        return len(self.get_incomplete_items()) == 0

    def get_completion_percentage(self) -> float:
        """Calculate percentage of items completed."""
        if not self.items:
            return 100.0
        completed = sum(
            1 for item in self.items.values() if item.status in (WorkItemStatus.DONE, WorkItemStatus.SKIPPED)
        )
        return (completed / len(self.items)) * 100.0

    def get_status_summary(self) -> dict[str, int]:
        """Get count of items in each status."""
        summary: dict[str, int] = {}
        for item in self.items.values():
            summary[item.status.value] = summary.get(item.status.value, 0) + 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "items": {item_id: item.to_dict() for item_id, item in self.items.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkItemLedger":
        """Deserialize from dictionary."""
        items = {item_id: WorkItem.from_dict(item_data) for item_id, item_data in data.get("items", {}).items()}
        return cls(items=items)


def _generate_item_id() -> str:
    """Generate an 8-character work item ID."""
    return uuid.uuid4().hex[:8]


@runtime_checkable
class WorkItemTaskListProtocol(Protocol):
    """Protocol for custom task list implementations."""

    @property
    def ledger(self) -> WorkItemLedger:
        """Get the work item ledger."""
        ...

    def get_tools(self) -> "list[AIFunction[Any, str]]":
        """Get tool closures for the agent."""
        ...


class WorkItemTaskList:
    """Default implementation of work item tracking.

    Owns a WorkItemLedger and provides tool closures that are
    bound to the ledger. Tools are injected into the agent at
    runtime via agent.run(tools=...).
    """

    def __init__(self) -> None:
        """Initialize with an empty ledger."""
        self._ledger = WorkItemLedger()

    @property
    def ledger(self) -> WorkItemLedger:
        """Get the work item ledger."""
        return self._ledger

    def get_tools(self) -> "list[AIFunction[Any, str]]":
        """Get tool closures bound to the ledger.

        Returns:
            List of tool functions for work item management.
        """
        return [
            self._make_add_tool(),
            self._make_update_tool(),
            self._make_list_tool(),
            self._make_set_artifact_tool(),
            self._make_flag_revision_tool(),
        ]

    def _make_add_tool(self) -> "AIFunction[Any, str]":
        """Create the work_item_add tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_add", approval_mode="never_require")
        def work_item_add(
            title: Annotated[str, "Brief description of the work item"],
            role: Annotated[str, "Artifact role: deliverable, working, or control"],
            priority: Annotated[str, "Priority level: high, medium, or low"] = "medium",
            notes: Annotated[str, "Optional context or notes"] = "",
        ) -> str:
            """Add a new work item to track a subtask.

            Use this to plan out the steps needed to complete the overall task.
            Each work item represents a discrete piece of work that should be
            completed and checked off. You must classify each item's role:
            - deliverable: User-facing output the user will receive directly.
            - working: Internal scratchpad, drafts, or intermediate data.
            - control: Validation checks, audits, quality gates.

            Args:
                title: Brief description of the work item.
                role: Artifact role classification.
                priority: Priority level (high, medium, low).
                notes: Optional context or notes.

            Returns:
                Confirmation with the item ID.
            """
            try:
                priority_enum = WorkItemPriority(priority)
            except ValueError:
                priority_enum = WorkItemPriority.MEDIUM

            try:
                role_enum = ArtifactRole(role)
            except ValueError:
                role_enum = ArtifactRole.WORKING

            item = WorkItem(
                id=_generate_item_id(),
                title=title,
                status=WorkItemStatus.PENDING,
                priority=priority_enum,
                artifact_role=role_enum,
                notes=notes,
            )
            ledger.add_item(item)
            return f"Added [{item.id}]: {title} (role: {role_enum.value}, priority: {priority_enum.value})"

        return work_item_add

    def _make_update_tool(self) -> "AIFunction[Any, str]":
        """Create the work_item_update tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_update", approval_mode="never_require")
        def work_item_update(
            item_id: Annotated[str, "ID of the work item to update"],
            status: Annotated[str, "New status: pending, in_progress, done, or skipped"],
            notes: Annotated[str, "Optional notes about the update"] = "",
        ) -> str:
            """Update the status of a work item.

            Call this as you make progress on work items to track completion.
            Mark items as 'done' when complete, or 'skipped' if no longer needed.

            Args:
                item_id: ID of the work item to update.
                status: New status value.
                notes: Optional notes about the update.

            Returns:
                Confirmation with remaining item count.
            """
            try:
                status_enum = WorkItemStatus(status)
            except ValueError:
                return f"Error: Invalid status '{status}'. Use: pending, in_progress, done, skipped"

            if not ledger.update_status(item_id, status_enum, notes):
                return f"Error: Work item [{item_id}] not found"

            remaining = len(ledger.get_incomplete_items())
            pct = ledger.get_completion_percentage()
            result = f"Updated [{item_id}] -> {status_enum.value} ({remaining} remaining, {pct:.0f}% complete)"
            if remaining == 0:
                result += " — all items done, call work_complete now."
            return result

        return work_item_update

    def _make_list_tool(self) -> "AIFunction[Any, str]":
        """Create the work_item_list tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_list", approval_mode="never_require")
        def work_item_list(
            filter_status: Annotated[str, "Filter by status: pending, in_progress, done, skipped, or 'all'"] = "all",
        ) -> str:
            """List current work items with their status.

            Use this to review your progress and identify what still needs to be done.
            This is your self-critique checkpoint - review items before signaling done.

            Args:
                filter_status: Optional status filter.

            Returns:
                Formatted checklist with completion percentage.
            """
            if not ledger.items:
                return "No work items tracked. Use work_item_add to plan subtasks."

            # Filter items
            if filter_status == "all":
                items = list(ledger.items.values())
            else:
                try:
                    status_enum = WorkItemStatus(filter_status)
                    items = [i for i in ledger.items.values() if i.status == status_enum]
                except ValueError:
                    items = list(ledger.items.values())

            # Format output
            status_icons = {
                WorkItemStatus.PENDING: "[ ]",
                WorkItemStatus.IN_PROGRESS: "[~]",
                WorkItemStatus.DONE: "[x]",
                WorkItemStatus.SKIPPED: "[-]",
            }

            lines = []
            pct = ledger.get_completion_percentage()
            lines.append(f"Work Items ({pct:.0f}% complete):")
            lines.append("")

            for item in items:
                icon = status_icons.get(item.status, "[ ]")
                priority_tag = f" ({item.priority.value})" if item.priority != WorkItemPriority.MEDIUM else ""
                revision_tag = " [NEEDS REVISION]" if item.requires_revision else ""
                parent_tag = f" (revision of {item.revision_of})" if item.revision_of else ""
                lines.append(f"  {icon} [{item.id}] {item.title}{priority_tag}{revision_tag}{parent_tag}")
                if item.notes:
                    lines.append(f"      Notes: {item.notes}")
                if item.artifact:
                    preview = item.artifact[:60] + "..." if len(item.artifact) > 60 else item.artifact
                    lines.append(f"      Artifact: {preview}")

            # Summary
            summary = ledger.get_status_summary()
            summary_parts = [f"{v} {k}" for k, v in sorted(summary.items())]
            lines.append("")
            lines.append(f"Summary: {', '.join(summary_parts)}")

            return "\n".join(lines)

        return work_item_list

    def _make_set_artifact_tool(self) -> "AIFunction[Any, str]":
        """Create the work_item_set_artifact tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_set_artifact", approval_mode="never_require")
        def work_item_set_artifact(
            item_id: Annotated[str, "ID of the work item"],
            artifact: Annotated[str, "The structured output/artifact produced by this step"],
            role: Annotated[str, "Artifact role: deliverable, working, or control"] = "working",
        ) -> str:
            """Store the structured output (artifact) for a completed work item.

            Use this to record the concrete deliverable of a step. For audit
            or review steps, store findings as structured data (JSON preferred).
            Stored artifacts enable cross-step validation and revision tracking.

            Args:
                item_id: ID of the work item.
                artifact: The output content (text, JSON, etc.).
                role: Artifact role classification.

            Returns:
                Confirmation message.
            """
            item = ledger.items.get(item_id)
            if item is None:
                return f"Error: Work item [{item_id}] not found"

            # Parse role to enum
            try:
                role_enum = ArtifactRole(role)
            except ValueError:
                role_enum = ArtifactRole.WORKING

            # Validate control artifact structure
            if role_enum == ArtifactRole.CONTROL:
                valid, error = validate_control_artifact(artifact)
                if not valid:
                    return (
                        f"Error: Invalid control artifact format. {error}\n\n"
                        f"Required format:\n"
                        f'{{"verdict": "pass"|"fail", '
                        f'"checks": [{{"name": "...", "result": "pass"|"fail", "detail": "..."}}], '
                        f'"summary": "..."}}'
                    )

            # Validate artifact content for boundary narration
            validation = validate_artifact_content(artifact)

            if validation.level == ArtifactContaminationLevel.LIGHT:
                item.artifact = validation.cleaned_content
                item.artifact_role = role_enum
                item.updated_at = datetime.now(timezone.utc).isoformat()
                cleaned = validation.cleaned_content
                preview = cleaned[:80] + "..." if len(cleaned) > 80 else cleaned
                return f"{validation.message}\nStored for [{item_id}]: {preview}"

            # CLEAN
            item.artifact = artifact
            item.artifact_role = role_enum
            item.updated_at = datetime.now(timezone.utc).isoformat()
            preview = artifact[:80] + "..." if len(artifact) > 80 else artifact
            return f"Artifact stored for [{item_id}]: {preview}"

        return work_item_set_artifact

    def _make_flag_revision_tool(self) -> "AIFunction[Any, str]":
        """Create the work_item_flag_revision tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_flag_revision", approval_mode="never_require")
        def work_item_flag_revision(
            item_id: Annotated[str, "ID of the work item that needs revision"],
            reason: Annotated[str, "What needs to be fixed in this item's output"],
        ) -> str:
            """Flag a completed work item as needing revision.

            Use this during audit/review steps when you find issues in a prior
            step's output. This creates a new revision work item that must be
            completed before the task can finish. The revision item should
            produce a corrected version of the original artifact.

            Args:
                item_id: ID of the item whose output needs correction.
                reason: Description of what needs fixing.

            Returns:
                Confirmation with the new revision item ID.
            """
            item = ledger.items.get(item_id)
            if item is None:
                return f"Error: Work item [{item_id}] not found"

            # Flag the original item
            item.requires_revision = True
            item.updated_at = datetime.now(timezone.utc).isoformat()

            # Create a revision child item (inherits artifact_role from parent)
            revision_item = WorkItem(
                id=_generate_item_id(),
                title=f"Revise: {item.title}",
                status=WorkItemStatus.PENDING,
                priority=WorkItemPriority.HIGH,
                notes=f"Revision needed: {reason}",
                artifact_role=item.artifact_role,
                revision_of=item_id,
            )
            ledger.add_item(revision_item)

            return (
                f"Flagged [{item_id}] for revision. "
                f"Created revision item [{revision_item.id}]: "
                f"Revise: {item.title}\n"
                f"Reason: {reason}"
            )

        return work_item_flag_revision


def format_work_item_reminder(ledger: WorkItemLedger) -> str:
    """Format a reminder message for incomplete work items.

    Used when the agent's done signal is rejected due to incomplete items.

    Args:
        ledger: The work item ledger.

    Returns:
        Formatted reminder message.
    """
    incomplete = ledger.get_incomplete_items()
    if not incomplete:
        return ""

    status_icons = {
        WorkItemStatus.PENDING: "[ ]",
        WorkItemStatus.IN_PROGRESS: "[~]",
        WorkItemStatus.DONE: "[x]",  # shown for items needing revision
    }

    # Separate revision-flagged items from regular incomplete
    needs_revision = [i for i in incomplete if i.requires_revision]
    regular_incomplete = [i for i in incomplete if not i.requires_revision]

    lines = [
        "You indicated you are done, but you have unresolved work items:",
    ]

    if needs_revision:
        lines.append("")
        lines.append("Items requiring revision (use work_item_flag_revision findings):")
        for item in needs_revision:
            icon = status_icons.get(item.status, "[ ]")
            lines.append(f"  {icon} [{item.id}] {item.title} [NEEDS REVISION]")

    if regular_incomplete:
        lines.append("")
        lines.append("Incomplete items:")
        for item in regular_incomplete:
            icon = status_icons.get(item.status, "[ ]")
            priority_tag = f" ({item.priority.value})" if item.priority != WorkItemPriority.MEDIUM else ""
            lines.append(f"  {icon} [{item.id}] {item.title}{priority_tag}")

    lines.append("")
    lines.append(
        "Complete remaining items, apply revisions (producing corrected artifacts), "
        "or mark items as skipped if no longer needed. Then signal done again.",
    )

    return "\n".join(lines)


class WorkItemEventMiddleware(FunctionMiddleware):
    """Middleware that queues events when work items change.

    This middleware intercepts work item tool calls and queues event data
    after each modification. The harness drains this queue to emit events
    for real-time UI updates.

    Attributes:
        WORK_ITEM_TOOLS: Set of tool names that modify work items.
    """

    WORK_ITEM_TOOLS = frozenset({
        "work_item_add",
        "work_item_update",
        "work_item_set_artifact",
        "work_item_flag_revision",
    })

    def __init__(self, ledger: WorkItemLedger) -> None:
        """Initialize the middleware with a ledger reference.

        Args:
            ledger: The WorkItemLedger to read state from after tool calls.
        """
        self._ledger = ledger
        self._pending_events: list[dict[str, Any]] = []

    @property
    def pending_events(self) -> list[dict[str, Any]]:
        """Get the list of pending events."""
        return self._pending_events

    def drain_events(self) -> list[dict[str, Any]]:
        """Drain and return all pending events.

        Returns:
            List of event data dictionaries. The list is cleared after draining.
        """
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Any],
    ) -> None:
        """Process a function invocation, queueing events for work item tools.

        Args:
            context: The function invocation context.
            next: The next handler in the middleware chain.
        """
        # Execute the tool
        await next(context)

        # If it was a work item tool, queue an event
        if context.function.name in self.WORK_ITEM_TOOLS:
            self._pending_events.append({
                "tool": context.function.name,
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
                    for item in self._ledger.items.values()
                ],
            })
