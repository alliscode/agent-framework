# Copyright (c) Microsoft. All rights reserved.

"""Work Item tracking for Agent Harness self-critique loop.

This module provides the work item tracking system that enables agents to:
1. Plan subtasks by adding work items
2. Track progress by updating item status
3. Self-critique by listing items and checking completeness

The harness auto-injects work item tools at runtime and verifies
completeness before accepting the agent's done signal.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Callable, Protocol, runtime_checkable

from .._tools import ai_function


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


@dataclass
class WorkItem:
    """A single work item in the ledger.

    Attributes:
        id: Auto-generated 8-char ID.
        title: Brief description of the work item.
        status: Current status of the item.
        priority: Priority level.
        notes: Optional context or notes.
        created_at: ISO timestamp when created.
        updated_at: ISO timestamp when last updated.
    """

    id: str
    title: str
    status: WorkItemStatus = WorkItemStatus.PENDING
    priority: WorkItemPriority = WorkItemPriority.MEDIUM
    notes: str = ""
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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkItem":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            status=WorkItemStatus(data.get("status", "pending")),
            priority=WorkItemPriority(data.get("priority", "medium")),
            notes=data.get("notes", ""),
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
        """Get all items that are not done or skipped."""
        return [
            item for item in self.items.values()
            if item.status not in (WorkItemStatus.DONE, WorkItemStatus.SKIPPED)
        ]

    def is_all_complete(self) -> bool:
        """Check if all items are done or skipped."""
        if not self.items:
            return True
        return len(self.get_incomplete_items()) == 0

    def get_completion_percentage(self) -> float:
        """Calculate percentage of items completed."""
        if not self.items:
            return 100.0
        completed = sum(
            1 for item in self.items.values()
            if item.status in (WorkItemStatus.DONE, WorkItemStatus.SKIPPED)
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
        items = {
            item_id: WorkItem.from_dict(item_data)
            for item_id, item_data in data.get("items", {}).items()
        }
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

    def get_tools(self) -> list[Callable[..., str]]:
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

    def get_tools(self) -> list[Callable[..., str]]:
        """Get tool closures bound to the ledger.

        Returns:
            List of tool functions: [work_item_add, work_item_update, work_item_list]
        """
        return [self._make_add_tool(), self._make_update_tool(), self._make_list_tool()]

    def _make_add_tool(self) -> Callable[..., str]:
        """Create the work_item_add tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_add", approval_mode="never_require")
        def work_item_add(
            title: Annotated[str, "Brief description of the work item"],
            priority: Annotated[str, "Priority level: high, medium, or low"] = "medium",
            notes: Annotated[str, "Optional context or notes"] = "",
        ) -> str:
            """Add a new work item to track a subtask.

            Use this to plan out the steps needed to complete the overall task.
            Each work item represents a discrete piece of work that should be
            completed and checked off.

            Args:
                title: Brief description of the work item.
                priority: Priority level (high, medium, low).
                notes: Optional context or notes.

            Returns:
                Confirmation with the item ID.
            """
            try:
                priority_enum = WorkItemPriority(priority)
            except ValueError:
                priority_enum = WorkItemPriority.MEDIUM

            item = WorkItem(
                id=_generate_item_id(),
                title=title,
                status=WorkItemStatus.PENDING,
                priority=priority_enum,
                notes=notes,
            )
            ledger.add_item(item)
            return f"Added [{item.id}]: {title} (priority: {priority_enum.value})"

        return work_item_add

    def _make_update_tool(self) -> Callable[..., str]:
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
            return f"Updated [{item_id}] -> {status_enum.value} ({remaining} remaining, {pct:.0f}% complete)"

        return work_item_update

    def _make_list_tool(self) -> Callable[..., str]:
        """Create the work_item_list tool closure."""
        ledger = self._ledger

        @ai_function(name="work_item_list", approval_mode="never_require")
        def work_item_list(
            filter_status: Annotated[
                str, "Filter by status: pending, in_progress, done, skipped, or 'all'"
            ] = "all",
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
                lines.append(f"  {icon} [{item.id}] {item.title}{priority_tag}")
                if item.notes:
                    lines.append(f"      Notes: {item.notes}")

            # Summary
            summary = ledger.get_status_summary()
            summary_parts = [f"{v} {k}" for k, v in sorted(summary.items())]
            lines.append("")
            lines.append(f"Summary: {', '.join(summary_parts)}")

            return "\n".join(lines)

        return work_item_list


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
    }

    lines = [
        "You indicated you are done, but you have incomplete work items:",
    ]
    for item in incomplete:
        icon = status_icons.get(item.status, "[ ]")
        priority_tag = f" ({item.priority.value})" if item.priority != WorkItemPriority.MEDIUM else ""
        lines.append(f"  {icon} [{item.id}] {item.title}{priority_tag}")

    lines.append("")
    lines.append(
        "Please complete these items or mark them as skipped if no longer needed, "
        "then signal done again."
    )

    return "\n".join(lines)
