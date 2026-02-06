# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness Work Item tracking module."""

import pytest

from agent_framework._harness import (
    ArtifactContaminationLevel,
    ArtifactRole,
    HarnessResult,
    HarnessStatus,
    ProgressFingerprint,
    RepairTrigger,
    StopDecisionExecutor,
    TurnComplete,
    WorkItem,
    WorkItemLedger,
    WorkItemPriority,
    WorkItemStatus,
    WorkItemTaskList,
    WorkItemTaskListProtocol,
    format_work_item_reminder,
    validate_artifact_content,
    validate_control_artifact,
)
from agent_framework._harness._constants import (
    HARNESS_MAX_TURNS_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
    HARNESS_WORK_ITEM_LEDGER_KEY,
)
from agent_framework._harness._state import HarnessLifecycleEvent

# ============================================================
# WorkItem Tests
# ============================================================


class TestWorkItem:
    """Tests for WorkItem dataclass."""

    def test_create_work_item(self) -> None:
        """Test creating a basic work item."""
        item = WorkItem(id="abc12345", title="Implement feature")
        assert item.id == "abc12345"
        assert item.title == "Implement feature"
        assert item.status == WorkItemStatus.PENDING
        assert item.priority == WorkItemPriority.MEDIUM
        assert item.notes == ""
        assert item.created_at != ""
        assert item.updated_at != ""

    def test_create_work_item_with_all_fields(self) -> None:
        """Test creating a work item with all fields specified."""
        item = WorkItem(
            id="def67890",
            title="Fix bug",
            status=WorkItemStatus.IN_PROGRESS,
            priority=WorkItemPriority.HIGH,
            notes="Critical issue",
        )
        assert item.status == WorkItemStatus.IN_PROGRESS
        assert item.priority == WorkItemPriority.HIGH
        assert item.notes == "Critical issue"

    def test_serialization_round_trip(self) -> None:
        """Test WorkItem serialization and deserialization."""
        item = WorkItem(
            id="abc12345",
            title="Test task",
            status=WorkItemStatus.DONE,
            priority=WorkItemPriority.LOW,
            notes="Completed successfully",
        )
        data = item.to_dict()
        restored = WorkItem.from_dict(data)

        assert restored.id == item.id
        assert restored.title == item.title
        assert restored.status == item.status
        assert restored.priority == item.priority
        assert restored.notes == item.notes
        assert restored.created_at == item.created_at
        assert restored.updated_at == item.updated_at

    def test_create_work_item_with_artifact_fields(self) -> None:
        """Test creating a work item with artifact and revision fields."""
        item = WorkItem(
            id="abc12345",
            title="Generate report",
            artifact="## Report\nContent here",
            requires_revision=True,
            revision_of="parent123",
        )
        assert item.artifact == "## Report\nContent here"
        assert item.requires_revision is True
        assert item.revision_of == "parent123"

    def test_serialization_with_artifact_fields(self) -> None:
        """Test serialization round-trip for artifact/revision fields."""
        item = WorkItem(
            id="abc12345",
            title="Generate report",
            artifact='{"findings": ["issue1"]}',
            requires_revision=True,
            revision_of="parent123",
        )
        data = item.to_dict()
        assert data["artifact"] == '{"findings": ["issue1"]}'
        assert data["requires_revision"] is True
        assert data["revision_of"] == "parent123"

        restored = WorkItem.from_dict(data)
        assert restored.artifact == item.artifact
        assert restored.requires_revision == item.requires_revision
        assert restored.revision_of == item.revision_of

    def test_serialization_format(self) -> None:
        """Test that serialized format uses string values for enums."""
        item = WorkItem(id="x", title="t", status=WorkItemStatus.IN_PROGRESS, priority=WorkItemPriority.HIGH)
        data = item.to_dict()
        assert data["status"] == "in_progress"
        assert data["priority"] == "high"


# ============================================================
# WorkItemLedger Tests
# ============================================================


class TestWorkItemLedger:
    """Tests for WorkItemLedger."""

    def test_empty_ledger(self) -> None:
        """Test empty ledger state."""
        ledger = WorkItemLedger()
        assert len(ledger.items) == 0
        assert ledger.is_all_complete()
        assert ledger.get_completion_percentage() == 100.0
        assert ledger.get_incomplete_items() == []

    def test_add_item(self) -> None:
        """Test adding items to the ledger."""
        ledger = WorkItemLedger()
        item = WorkItem(id="abc", title="Task 1")
        ledger.add_item(item)

        assert len(ledger.items) == 1
        assert ledger.items["abc"] == item

    def test_update_status(self) -> None:
        """Test updating item status."""
        ledger = WorkItemLedger()
        item = WorkItem(id="abc", title="Task 1")
        ledger.add_item(item)

        result = ledger.update_status("abc", WorkItemStatus.IN_PROGRESS)
        assert result is True
        assert ledger.items["abc"].status == WorkItemStatus.IN_PROGRESS

        result = ledger.update_status("abc", WorkItemStatus.DONE, notes="All done")
        assert result is True
        assert ledger.items["abc"].status == WorkItemStatus.DONE
        assert ledger.items["abc"].notes == "All done"

    def test_update_nonexistent_item(self) -> None:
        """Test updating a non-existent item returns False."""
        ledger = WorkItemLedger()
        result = ledger.update_status("nonexistent", WorkItemStatus.DONE)
        assert result is False

    def test_get_incomplete_items(self) -> None:
        """Test getting incomplete items."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.PENDING))
        ledger.add_item(WorkItem(id="b", title="T2", status=WorkItemStatus.IN_PROGRESS))
        ledger.add_item(WorkItem(id="c", title="T3", status=WorkItemStatus.DONE))
        ledger.add_item(WorkItem(id="d", title="T4", status=WorkItemStatus.SKIPPED))

        incomplete = ledger.get_incomplete_items()
        assert len(incomplete) == 2
        ids = {i.id for i in incomplete}
        assert ids == {"a", "b"}

    def test_is_all_complete(self) -> None:
        """Test completion check."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE))
        ledger.add_item(WorkItem(id="b", title="T2", status=WorkItemStatus.SKIPPED))
        assert ledger.is_all_complete()

        ledger.add_item(WorkItem(id="c", title="T3", status=WorkItemStatus.PENDING))
        assert not ledger.is_all_complete()

    def test_completion_percentage(self) -> None:
        """Test completion percentage calculation."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE))
        ledger.add_item(WorkItem(id="b", title="T2", status=WorkItemStatus.PENDING))
        ledger.add_item(WorkItem(id="c", title="T3", status=WorkItemStatus.IN_PROGRESS))
        ledger.add_item(WorkItem(id="d", title="T4", status=WorkItemStatus.SKIPPED))

        # 2 of 4 are done/skipped = 50%
        assert ledger.get_completion_percentage() == 50.0

    def test_status_summary(self) -> None:
        """Test status summary counts."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.PENDING))
        ledger.add_item(WorkItem(id="b", title="T2", status=WorkItemStatus.PENDING))
        ledger.add_item(WorkItem(id="c", title="T3", status=WorkItemStatus.DONE))

        summary = ledger.get_status_summary()
        assert summary == {"pending": 2, "done": 1}

    def test_get_items_needing_revision(self) -> None:
        """Test getting items flagged for revision without completed fixes."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE, requires_revision=True))
        ledger.add_item(WorkItem(id="b", title="T2", status=WorkItemStatus.DONE))
        ledger.add_item(WorkItem(id="c", title="T3", status=WorkItemStatus.DONE, requires_revision=True))

        needing = ledger.get_items_needing_revision()
        assert len(needing) == 2
        ids = {i.id for i in needing}
        assert ids == {"a", "c"}

    def test_revision_resolved_by_child(self) -> None:
        """Test that a completed revision child resolves the parent."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE, requires_revision=True))
        # Add a completed revision child
        ledger.add_item(
            WorkItem(
                id="rev_a",
                title="Revise: T1",
                status=WorkItemStatus.DONE,
                revision_of="a",
            ),
        )

        needing = ledger.get_items_needing_revision()
        assert len(needing) == 0
        assert ledger.is_all_complete()

    def test_incomplete_includes_unresolved_revisions(self) -> None:
        """Test that get_incomplete_items includes items needing revision."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE, requires_revision=True))
        # No revision child yet - should count as incomplete
        incomplete = ledger.get_incomplete_items()
        assert len(incomplete) == 1
        assert incomplete[0].id == "a"
        assert not ledger.is_all_complete()

    def test_skipped_revision_resolves_parent(self) -> None:
        """Test that a skipped revision child also resolves the parent."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE, requires_revision=True))
        ledger.add_item(
            WorkItem(
                id="rev_a",
                title="Revise: T1",
                status=WorkItemStatus.SKIPPED,
                revision_of="a",
            ),
        )

        assert ledger.is_all_complete()

    def test_pending_revision_child_is_incomplete(self) -> None:
        """Test that a pending revision child keeps parent unresolved."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE, requires_revision=True))
        ledger.add_item(
            WorkItem(
                id="rev_a",
                title="Revise: T1",
                status=WorkItemStatus.PENDING,
                revision_of="a",
            ),
        )

        incomplete = ledger.get_incomplete_items()
        # Both the parent (needs revision) and the child (pending) are incomplete
        ids = {i.id for i in incomplete}
        assert "a" in ids
        assert "rev_a" in ids
        assert not ledger.is_all_complete()

    def test_serialization_round_trip(self) -> None:
        """Test ledger serialization."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE))
        ledger.add_item(WorkItem(id="b", title="T2", priority=WorkItemPriority.HIGH))

        data = ledger.to_dict()
        restored = WorkItemLedger.from_dict(data)

        assert len(restored.items) == 2
        assert restored.items["a"].title == "T1"
        assert restored.items["a"].status == WorkItemStatus.DONE
        assert restored.items["b"].priority == WorkItemPriority.HIGH


# ============================================================
# WorkItemTaskList Tests
# ============================================================


class TestWorkItemTaskList:
    """Tests for WorkItemTaskList and its tool closures."""

    def test_implements_protocol(self) -> None:
        """Test that WorkItemTaskList satisfies the protocol."""
        task_list = WorkItemTaskList()
        assert isinstance(task_list, WorkItemTaskListProtocol)

    def test_get_tools_returns_five_tools(self) -> None:
        """Test that get_tools returns five tool functions."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        assert len(tools) == 5

    def test_add_tool(self) -> None:
        """Test the work_item_add tool closure."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        result = add_tool(title="Write tests", role="working", priority="high", notes="Important")
        assert "Added [" in result
        assert "Write tests" in result
        assert "high" in result

        # Verify the ledger was updated
        assert len(task_list.ledger.items) == 1
        item = list(task_list.ledger.items.values())[0]
        assert item.title == "Write tests"
        assert item.priority == WorkItemPriority.HIGH
        assert item.notes == "Important"

    def test_add_tool_default_priority(self) -> None:
        """Test that add tool defaults to medium priority."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        add_tool(title="Simple task", role="working")
        item = list(task_list.ledger.items.values())[0]
        assert item.priority == WorkItemPriority.MEDIUM

    def test_add_tool_invalid_priority(self) -> None:
        """Test that invalid priority falls back to medium."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        add_tool(title="Task", role="working", priority="invalid")
        item = list(task_list.ledger.items.values())[0]
        assert item.priority == WorkItemPriority.MEDIUM

    def test_add_tool_sets_role(self) -> None:
        """Test that add tool sets the artifact role on creation."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        add_tool(title="User report", role="deliverable")
        item = list(task_list.ledger.items.values())[0]
        assert item.artifact_role == ArtifactRole.DELIVERABLE

        task_list2 = WorkItemTaskList()
        tools2 = task_list2.get_tools()
        tools2[0](title="Audit gate", role="control")
        item2 = list(task_list2.ledger.items.values())[0]
        assert item2.artifact_role == ArtifactRole.CONTROL

    def test_add_tool_invalid_role_falls_back(self) -> None:
        """Test that invalid role in add tool falls back to working."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        add_tool(title="Mystery", role="nonexistent")
        item = list(task_list.ledger.items.values())[0]
        assert item.artifact_role == ArtifactRole.WORKING

    def test_add_tool_role_in_response(self) -> None:
        """Test that add tool response includes the role."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        result = add_tool(title="Report", role="deliverable")
        assert "role: deliverable" in result

    def test_update_tool(self) -> None:
        """Test the work_item_update tool closure."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool, update_tool = tools[0], tools[1]

        # Add an item first
        add_tool(title="Build feature", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        # Update it
        result = update_tool(item_id=item_id, status="in_progress")
        assert f"Updated [{item_id}]" in result
        assert "in_progress" in result
        assert task_list.ledger.items[item_id].status == WorkItemStatus.IN_PROGRESS

        # Mark done
        result = update_tool(item_id=item_id, status="done", notes="Finished!")
        assert "done" in result
        assert "0 remaining" in result
        assert task_list.ledger.items[item_id].notes == "Finished!"

    def test_update_tool_invalid_status(self) -> None:
        """Test update tool with invalid status."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        update_tool = tools[1]

        result = update_tool(item_id="abc", status="invalid_status")
        assert "Error" in result
        assert "Invalid status" in result

    def test_update_tool_nonexistent_id(self) -> None:
        """Test update tool with non-existent item ID."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        update_tool = tools[1]

        result = update_tool(item_id="nonexistent", status="done")
        assert "Error" in result
        assert "not found" in result

    def test_list_tool_empty(self) -> None:
        """Test list tool with no items."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        list_tool = tools[2]

        result = list_tool()
        assert "No work items tracked" in result

    def test_list_tool_with_items(self) -> None:
        """Test list tool with items."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool, update_tool, list_tool = tools[0], tools[1], tools[2]

        add_tool(title="Task A", role="working", priority="high")
        add_tool(title="Task B", role="working")
        item_ids = list(task_list.ledger.items.keys())
        update_tool(item_id=item_ids[0], status="done")

        result = list_tool()
        assert "50%" in result  # 1 of 2 done
        assert "[x]" in result  # done item
        assert "[ ]" in result  # pending item
        assert "Task A" in result
        assert "Task B" in result
        assert "(high)" in result  # priority tag for high

    def test_list_tool_with_filter(self) -> None:
        """Test list tool with status filter."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool, update_tool, list_tool = tools[0], tools[1], tools[2]

        add_tool(title="Done task", role="working")
        add_tool(title="Pending task", role="working")
        item_ids = list(task_list.ledger.items.keys())
        update_tool(item_id=item_ids[0], status="done")

        result = list_tool(filter_status="done")
        assert "Done task" in result
        assert "Pending task" not in result

    def test_set_artifact_tool(self) -> None:
        """Test the work_item_set_artifact tool closure."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Generate report", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        result = set_artifact_tool(item_id=item_id, artifact='{"data": [1,2,3]}')
        assert "Artifact stored" in result
        assert item_id in result
        assert task_list.ledger.items[item_id].artifact == '{"data": [1,2,3]}'

    def test_set_artifact_tool_nonexistent_id(self) -> None:
        """Test set_artifact tool with non-existent item."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        set_artifact_tool = tools[3]

        result = set_artifact_tool(item_id="nonexistent", artifact="data")
        assert "Error" in result
        assert "not found" in result

    def test_set_artifact_tool_long_preview(self) -> None:
        """Test that long artifacts are truncated in response."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Big output", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        long_artifact = "x" * 200
        result = set_artifact_tool(item_id=item_id, artifact=long_artifact)
        assert "..." in result  # truncated in response
        # But full artifact is stored
        assert task_list.ledger.items[item_id].artifact == long_artifact

    def test_flag_revision_tool(self) -> None:
        """Test the work_item_flag_revision tool closure."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        update_tool = tools[1]
        flag_revision_tool = tools[4]

        add_tool(title="Write timeline", role="working")
        item_id = list(task_list.ledger.items.keys())[0]
        update_tool(item_id=item_id, status="done")

        result = flag_revision_tool(item_id=item_id, reason="Dates are wrong")
        assert "Flagged" in result
        assert item_id in result
        assert "revision item" in result
        assert "Dates are wrong" in result

        # Original item should be flagged
        assert task_list.ledger.items[item_id].requires_revision is True

        # A new revision child should exist
        assert len(task_list.ledger.items) == 2
        revision_items = [i for i in task_list.ledger.items.values() if i.revision_of == item_id]
        assert len(revision_items) == 1
        assert revision_items[0].title.startswith("Revise:")
        assert revision_items[0].priority == WorkItemPriority.HIGH
        assert revision_items[0].status == WorkItemStatus.PENDING

    def test_flag_revision_tool_nonexistent_id(self) -> None:
        """Test flag_revision tool with non-existent item."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        flag_revision_tool = tools[4]

        result = flag_revision_tool(item_id="nonexistent", reason="issue")
        assert "Error" in result
        assert "not found" in result

    def test_flag_revision_blocks_completion(self) -> None:
        """Test that flagging revision blocks ledger completion."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        update_tool = tools[1]
        flag_revision_tool = tools[4]

        add_tool(title="Write report", role="working")
        item_id = list(task_list.ledger.items.keys())[0]
        update_tool(item_id=item_id, status="done")

        assert task_list.ledger.is_all_complete()

        # Flag for revision - now not complete
        flag_revision_tool(item_id=item_id, reason="Needs fixes")
        assert not task_list.ledger.is_all_complete()

        # Complete the revision child
        revision_id = [i.id for i in task_list.ledger.items.values() if i.revision_of == item_id][0]
        update_tool(item_id=revision_id, status="done")
        assert task_list.ledger.is_all_complete()

    def test_list_tool_shows_revision_status(self) -> None:
        """Test that list tool shows revision/artifact info."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        update_tool = tools[1]
        list_tool = tools[2]
        set_artifact_tool = tools[3]
        flag_revision_tool = tools[4]

        add_tool(title="Write section", role="working")
        item_id = list(task_list.ledger.items.keys())[0]
        update_tool(item_id=item_id, status="done")
        set_artifact_tool(item_id=item_id, artifact="Section content here")
        flag_revision_tool(item_id=item_id, reason="Typos")

        result = list_tool()
        assert "NEEDS REVISION" in result
        assert "Artifact:" in result
        assert "Section content here" in result
        assert "revision of" in result

    def test_tools_share_same_ledger(self) -> None:
        """Test that all tools operate on the same ledger."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool, update_tool, list_tool = tools[0], tools[1], tools[2]

        # Add via tool
        add_tool(title="Shared task", role="working")
        assert len(task_list.ledger.items) == 1

        # Update via tool
        item_id = list(task_list.ledger.items.keys())[0]
        update_tool(item_id=item_id, status="done")

        # List reflects the update
        result = list_tool()
        assert "100%" in result


# ============================================================
# format_work_item_reminder Tests
# ============================================================


class TestFormatWorkItemReminder:
    """Tests for the reminder message formatter."""

    def test_empty_ledger_returns_empty(self) -> None:
        """Test that empty ledger produces empty reminder."""
        ledger = WorkItemLedger()
        assert format_work_item_reminder(ledger) == ""

    def test_all_complete_returns_empty(self) -> None:
        """Test that complete ledger produces empty reminder."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE))
        assert format_work_item_reminder(ledger) == ""

    def test_incomplete_items_produce_reminder(self) -> None:
        """Test reminder format with incomplete items."""
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="abc123", title="Implement handler", priority=WorkItemPriority.HIGH))
        ledger.add_item(WorkItem(id="def456", title="Add tests", status=WorkItemStatus.IN_PROGRESS))
        ledger.add_item(WorkItem(id="ghi789", title="Done already", status=WorkItemStatus.DONE))

        result = format_work_item_reminder(ledger)
        assert "unresolved work items" in result
        assert "Incomplete items:" in result
        assert "[abc123]" in result
        assert "Implement handler" in result
        assert "(high)" in result
        assert "[def456]" in result
        assert "Add tests" in result
        assert "[~]" in result  # in_progress icon
        assert "ghi789" not in result  # done item should not appear
        assert "signal done again" in result

    def test_revision_items_in_reminder(self) -> None:
        """Test reminder format with revision-flagged items."""
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="abc123",
                title="Write timeline",
                status=WorkItemStatus.DONE,
                requires_revision=True,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="def456",
                title="Revise: Write timeline",
                status=WorkItemStatus.PENDING,
                revision_of="abc123",
            ),
        )

        result = format_work_item_reminder(ledger)
        assert "unresolved work items" in result
        assert "Items requiring revision" in result
        assert "[abc123]" in result
        assert "NEEDS REVISION" in result
        assert "Incomplete items:" in result
        assert "[def456]" in result


# ============================================================
# ProgressFingerprint with work_item_statuses Tests
# ============================================================


class TestProgressFingerprintWithWorkItems:
    """Tests for ProgressFingerprint extension with work item statuses."""

    def test_fingerprint_changes_with_work_item_status(self) -> None:
        """Test that changing work item status changes the fingerprint."""
        fp1 = ProgressFingerprint.compute(
            turn_number=1,
            transcript_length=10,
            work_item_statuses={"a": "pending", "b": "pending"},
        )
        fp2 = ProgressFingerprint.compute(
            turn_number=2,
            transcript_length=10,
            work_item_statuses={"a": "done", "b": "pending"},
        )
        assert fp1.fingerprint != fp2.fingerprint

    def test_fingerprint_same_without_changes(self) -> None:
        """Test that same work item statuses produce same fingerprint."""
        fp1 = ProgressFingerprint.compute(
            turn_number=1,
            transcript_length=10,
            work_item_statuses={"a": "pending"},
        )
        fp2 = ProgressFingerprint.compute(
            turn_number=2,
            transcript_length=10,
            work_item_statuses={"a": "pending"},
        )
        assert fp1.fingerprint == fp2.fingerprint

    def test_fingerprint_without_work_items_unchanged(self) -> None:
        """Test that fingerprint without work items still works."""
        fp1 = ProgressFingerprint.compute(
            turn_number=1,
            transcript_length=10,
        )
        fp2 = ProgressFingerprint.compute(
            turn_number=2,
            transcript_length=10,
        )
        # Same state = same fingerprint
        assert fp1.fingerprint == fp2.fingerprint

    def test_adding_work_items_changes_fingerprint(self) -> None:
        """Test that adding work items changes the fingerprint vs none."""
        fp_no_items = ProgressFingerprint.compute(
            turn_number=1,
            transcript_length=10,
        )
        fp_with_items = ProgressFingerprint.compute(
            turn_number=1,
            transcript_length=10,
            work_item_statuses={"a": "pending"},
        )
        assert fp_no_items.fingerprint != fp_with_items.fingerprint


# ============================================================
# StopDecisionExecutor Work Item Verification Tests
# ============================================================


class TestStopDecisionWorkItemVerification:
    """Tests for work item verification in StopDecisionExecutor."""

    @pytest.fixture
    def mock_ctx(self):
        """Create a mock workflow context."""

        class MockContext:
            def __init__(self):
                self._state: dict[str, object] = {
                    HARNESS_TURN_COUNT_KEY: 1,
                    HARNESS_MAX_TURNS_KEY: 20,
                    HARNESS_TRANSCRIPT_KEY: [],
                }
                self._messages: list = []
                self._events: list = []
                self._outputs: list = []

            async def get_shared_state(self, key: str):
                if key not in self._state:
                    raise KeyError(key)
                return self._state[key]

            async def set_shared_state(self, key: str, value: object):
                self._state[key] = value

            async def send_message(self, msg):
                self._messages.append(msg)

            async def add_event(self, event):
                self._events.append(event)

            async def yield_output(self, output):
                self._outputs.append(output)

            def is_streaming(self):
                return False

        return MockContext()

    @pytest.mark.asyncio
    async def test_work_items_complete_allows_done(self, mock_ctx) -> None:
        """Test that complete work items allow done signal."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        # Set up complete ledger
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE))
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have stopped (yielded output)
        assert len(mock_ctx._outputs) == 1
        result = mock_ctx._outputs[0]
        assert isinstance(result, HarnessResult)
        assert result.status == HarnessStatus.DONE

    @pytest.mark.asyncio
    async def test_work_items_incomplete_rejects_done(self, mock_ctx) -> None:
        """Test that incomplete work items reject done signal."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        # Set up incomplete ledger
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.PENDING))
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have sent RepairTrigger (continue)
        assert len(mock_ctx._outputs) == 0
        assert len(mock_ctx._messages) == 1
        assert isinstance(mock_ctx._messages[0], RepairTrigger)

        # Check that a stop_decision event was recorded
        transcript = mock_ctx._state[HARNESS_TRANSCRIPT_KEY]
        stop_events = [e for e in transcript if e["event_type"] == "stop_decision"]
        assert len(stop_events) == 1
        assert stop_events[0]["data"]["reason"] == "work_items_incomplete"

    @pytest.mark.asyncio
    async def test_no_ledger_allows_done(self, mock_ctx) -> None:
        """Test that no ledger (no items) allows done signal."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        # No ledger in state
        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have stopped
        assert len(mock_ctx._outputs) == 1

    @pytest.mark.asyncio
    async def test_verification_disabled_ignores_work_items(self, mock_ctx) -> None:
        """Test that disabled verification ignores incomplete items."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=False,
        )

        # Set up incomplete ledger
        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.PENDING))
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have stopped regardless
        assert len(mock_ctx._outputs) == 1

    @pytest.mark.asyncio
    async def test_skipped_items_count_as_complete(self, mock_ctx) -> None:
        """Test that skipped items are treated as complete."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        ledger = WorkItemLedger()
        ledger.add_item(WorkItem(id="a", title="T1", status=WorkItemStatus.DONE))
        ledger.add_item(WorkItem(id="b", title="T2", status=WorkItemStatus.SKIPPED))
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have stopped
        assert len(mock_ctx._outputs) == 1

    @pytest.mark.asyncio
    async def test_unresolved_revision_rejects_done(self, mock_ctx) -> None:
        """Test that unresolved revision-flagged items reject done signal."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        ledger = WorkItemLedger()
        # Item is done but flagged for revision, no revision child completed
        ledger.add_item(
            WorkItem(
                id="a",
                title="T1",
                status=WorkItemStatus.DONE,
                requires_revision=True,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="rev_a",
                title="Revise: T1",
                status=WorkItemStatus.PENDING,
                revision_of="a",
            ),
        )
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have rejected (sent RepairTrigger)
        assert len(mock_ctx._outputs) == 0
        assert len(mock_ctx._messages) == 1
        assert isinstance(mock_ctx._messages[0], RepairTrigger)

    @pytest.mark.asyncio
    async def test_resolved_revision_allows_done(self, mock_ctx) -> None:
        """Test that resolved revisions allow done signal."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="a",
                title="T1",
                status=WorkItemStatus.DONE,
                requires_revision=True,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="rev_a",
                title="Revise: T1",
                status=WorkItemStatus.DONE,
                revision_of="a",
            ),
        )
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        # Should have stopped (resolved)
        assert len(mock_ctx._outputs) == 1
        result = mock_ctx._outputs[0]
        assert isinstance(result, HarnessResult)
        assert result.status == HarnessStatus.DONE


# ============================================================
# HarnessWorkflowBuilder Integration Tests
# ============================================================


class TestHarnessBuilderWorkItems:
    """Tests for work item integration in HarnessWorkflowBuilder."""

    def test_enable_work_items_flag(self) -> None:
        """Test that enable_work_items creates a default task list."""
        from unittest.mock import MagicMock

        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        builder = HarnessWorkflowBuilder(agent, enable_work_items=True)
        assert builder._task_list is not None
        assert isinstance(builder._task_list, WorkItemTaskList)

    def test_custom_task_list(self) -> None:
        """Test that custom task_list is used when provided."""
        from unittest.mock import MagicMock

        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        custom_list = WorkItemTaskList()
        builder = HarnessWorkflowBuilder(agent, task_list=custom_list)
        assert builder._task_list is custom_list

    def test_task_list_overrides_enable_flag(self) -> None:
        """Test that explicit task_list takes priority over enable_work_items."""
        from unittest.mock import MagicMock

        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        custom_list = WorkItemTaskList()
        builder = HarnessWorkflowBuilder(agent, enable_work_items=True, task_list=custom_list)
        assert builder._task_list is custom_list

    def test_disabled_by_default(self) -> None:
        """Test that work items are disabled by default."""
        from unittest.mock import MagicMock

        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        builder = HarnessWorkflowBuilder(agent)
        assert builder._task_list is None


# ============================================================
# AgentTurnExecutor Work Item Integration Tests
# ============================================================


class TestAgentTurnExecutorWorkItems:
    """Tests for work item integration in AgentTurnExecutor."""

    def test_task_list_stored(self) -> None:
        """Test that task_list is stored in executor."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        executor = AgentTurnExecutor(agent, task_list=task_list)
        assert executor._task_list is task_list

    def test_no_task_list_by_default(self) -> None:
        """Test that task_list is None by default."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        executor = AgentTurnExecutor(agent)
        assert executor._task_list is None


# ============================================================
# Artifact Validation Tests
# ============================================================


class TestArtifactValidation:
    """Tests for validate_artifact_content function."""

    def test_clean_plain_text(self) -> None:
        """Test that plain deliverable content is classified as CLEAN."""
        result = validate_artifact_content("## Incident Timeline\n09:00 - Deploy\n09:15 - Errors")
        assert result.level == ArtifactContaminationLevel.CLEAN
        assert result.cleaned_content == "## Incident Timeline\n09:00 - Deploy\n09:15 - Errors"
        assert result.removed_lines == []
        assert result.message == ""

    def test_clean_json(self) -> None:
        """Test that pure JSON is classified as CLEAN."""
        json_content = '{"findings": ["issue1", "issue2"], "severity": "high"}'
        result = validate_artifact_content(json_content)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_clean_code_block(self) -> None:
        """Test that code content is classified as CLEAN."""
        code = "def process_data(items):\n    for item in items:\n        yield item.transform()"
        result = validate_artifact_content(code)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_clean_content_with_non_narration_will(self) -> None:
        """Test that 'will' in non-narration context is CLEAN."""
        content = "The system will now process the data.\nResults are stored in the database."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_clean_content_stored_in_database(self) -> None:
        """Test that 'stored' without first-person subject is CLEAN."""
        content = "Data is stored in the primary database.\nBackups are saved nightly."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_light_single_preamble(self) -> None:
        """Test single preamble line is classified as LIGHT."""
        content = "I will now document the timeline.\n\n## Timeline\n09:00 - Start\n09:15 - End"
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Timeline" in result.cleaned_content
        assert "I will now" not in result.cleaned_content
        assert len(result.removed_lines) == 1

    def test_light_single_postamble(self) -> None:
        """Test single postamble line is classified as LIGHT."""
        content = "## Report\nFindings here.\n\nI've stored this as the report artifact."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Report" in result.cleaned_content
        assert "I've stored" not in result.cleaned_content
        assert len(result.removed_lines) == 1

    def test_light_preamble_and_postamble(self) -> None:
        """Test preamble + postamble (≤3 total) is LIGHT."""
        content = (
            "Here is the incident report.\n\n"
            "## Incident\nServer crashed at 09:00.\n"
            "Root cause: memory leak.\n\n"
            "I'll now move on to the next step."
        )
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Incident" in result.cleaned_content
        assert "Here is the" not in result.cleaned_content
        assert "I'll now move" not in result.cleaned_content
        assert len(result.removed_lines) == 2

    def test_light_let_me_preamble(self) -> None:
        """Test 'Let me' preamble pattern."""
        content = "Let me write the analysis.\n\n## Analysis\nKey findings."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Let me" not in result.cleaned_content

    def test_light_heres_the_preamble(self) -> None:
        """Test 'Here's the' preamble pattern."""
        content = "Here's the completed document.\n\n## Document\nContent."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Here's the" not in result.cleaned_content

    def test_light_below_is_preamble(self) -> None:
        """Test 'Below is' preamble pattern."""
        content = "Below is the timeline.\n\n## Timeline\nEvents listed."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Below is" not in result.cleaned_content

    def test_light_moving_on_postamble(self) -> None:
        """Test 'Moving on to' postamble pattern."""
        content = "## Data\nResults.\n\nMoving on to the next analysis."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Moving on" not in result.cleaned_content

    def test_light_next_ill_postamble(self) -> None:
        """Test 'Next, I'll' postamble pattern."""
        content = "## Summary\nKey points.\n\nNext, I'll review the logs."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "Next, I'll" not in result.cleaned_content

    def test_light_cleans_leading_trailing_blanks(self) -> None:
        """Test that cleaning also strips resulting leading/trailing blanks."""
        content = "I will now write it.\n\n\n## Content\nData here.\n\n\nI've saved it."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert result.cleaned_content.startswith("## Content")
        assert result.cleaned_content.endswith("Data here.")

    def test_heavy_interior_meta_reference(self) -> None:
        """Test meta-reference in body interior triggers HEAVY."""
        content = (
            "## Timeline\n"
            "09:00 - Deployment started\n"
            "09:15 - Errors detected\n"
            "I've stored this artifact and will now move on.\n"
            "09:30 - Rollback initiated\n"
            "10:00 - Recovery complete"
        )
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY
        assert "rejected" in result.message

    def test_heavy_interleaved_narration(self) -> None:
        """Test narration in the middle of content triggers HEAVY."""
        lines = [
            "## Report",
            "Finding 1: issue detected.",
            "",
            "Now I'll document the next section.",
            "",
            "Finding 2: another issue.",
            "",
            "I will now summarize.",
            "",
            "## Summary",
            "All clear.",
        ]
        content = "\n".join(lines)
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY

    def test_heavy_too_many_boundary_lines(self) -> None:
        """Test >3 boundary narration lines triggers HEAVY."""
        content = (
            "I will now write the report.\n"
            "Let me document everything.\n"
            "Here is the content.\n"
            "I'm going to be thorough.\n"
            "## Report\nData.\n"
        )
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY

    def test_heavy_work_item_reference_interior(self) -> None:
        """Test work item references in body interior trigger HEAVY."""
        content = (
            "## Analysis\n"
            "The system had issues.\n"
            "I'll update the work item with these findings.\n"
            "Root cause identified."
        )
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY

    def test_heavy_artifact_stored_interior(self) -> None:
        """Test 'artifact stored' meta-reference in interior triggers HEAVY."""
        content = "## Timeline\n09:00 - Start\nThe artifact stored above is incomplete.\n09:15 - End"
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY

    def test_heavy_task_complete_reference(self) -> None:
        """Test task_complete reference triggers HEAVY."""
        content = "## Report\nData here.\nAfter this I will call task_complete.\nMore data."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY

    def test_heavy_work_item_set_artifact_reference(self) -> None:
        """Test work_item_set_artifact reference triggers HEAVY."""
        content = "## Output\nUsing work_item_set_artifact to store this.\nContent follows."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY

    def test_heavy_message_includes_resubmit_instruction(self) -> None:
        """Test that HEAVY message instructs resubmission."""
        content = "I will now write.\n## Data\nContent.\nI've stored the artifact saved above.\nDone."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY
        assert "resubmit" in result.message.lower()

    def test_light_message_includes_reminder(self) -> None:
        """Test that LIGHT message includes a reminder."""
        content = "Here is the output.\n\n## Data\nContent here."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert "auto-cleaned" in result.message
        assert "response text" in result.message

    def test_empty_artifact(self) -> None:
        """Test that empty artifact is CLEAN."""
        result = validate_artifact_content("")
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_single_line_clean(self) -> None:
        """Test single line of clean content."""
        result = validate_artifact_content("Just a result value: 42")
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_case_insensitive_matching(self) -> None:
        """Test that patterns match case-insensitively."""
        content = "I WILL NOW document this.\n\n## Data\nContent."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_indented_narration_detected(self) -> None:
        """Test that indented narration is still detected."""
        content = "  I will now write the report.\n\n## Report\nFindings."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_im_going_to_preamble(self) -> None:
        """Test 'I'm going to' preamble pattern."""
        content = "I'm going to document everything.\n\n## Doc\nContent."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_the_following_is_preamble(self) -> None:
        """Test 'The following is' preamble pattern."""
        content = "The following is the complete analysis.\n\n## Analysis\nData."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_this_has_been_stored_postamble(self) -> None:
        """Test 'This has been stored' postamble pattern."""
        content = "## Output\nResults.\n\nThis has been stored for review."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_the_above_has_been_postamble(self) -> None:
        """Test 'The above has been' postamble pattern."""
        content = "## Data\nNumbers.\n\nThe above has been recorded."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_now_ill_preamble(self) -> None:
        """Test 'Now I'll' preamble pattern."""
        content = "Now I'll create the summary.\n\n## Summary\nPoints."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_meta_reference_in_boundary_counts_as_boundary(self) -> None:
        """Test that meta-references in boundary zones count as boundary narration."""
        content = "I'll update the work item with this.\n\n## Data\nContent here."
        result = validate_artifact_content(content)
        # "work item" in first 5 lines = boundary meta-reference
        # Also matches preamble pattern "I'll ..."
        assert result.level == ArtifactContaminationLevel.LIGHT

    def test_false_positive_data_will_be_processed(self) -> None:
        """Test that 'The data will now be processed' is not a false positive."""
        content = "Step 1: Gather data.\nThe data will now be processed by the pipeline.\nStep 2: Output."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_false_positive_stored_in_database(self) -> None:
        """Test that 'stored in the database' without first-person is not flagged."""
        content = "## Architecture\nData is stored in the primary database.\nReplicas are saved hourly."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_multiline_json_clean(self) -> None:
        """Test that multiline JSON is CLEAN."""
        content = (
            '{\n  "events": [\n    {"time": "09:00", "event": "start"},'
            '\n    {"time": "09:15", "event": "error"}\n  ]\n}'
        )
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.CLEAN

    def test_three_boundary_lines_is_light(self) -> None:
        """Test that exactly 3 boundary narration lines is still LIGHT."""
        content = "I will now write this.\nLet me be thorough.\n## Report\nContent here.\nI've stored this artifact."
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.LIGHT
        assert len(result.removed_lines) == 3

    def test_four_boundary_lines_is_heavy(self) -> None:
        """Test that 4 boundary narration lines triggers HEAVY."""
        content = (
            "I will now write this.\n"
            "Let me be thorough.\n"
            "Here is the output.\n"
            "## Report\n"
            "Content here.\n"
            "I've stored this result."
        )
        result = validate_artifact_content(content)
        assert result.level == ArtifactContaminationLevel.HEAVY


# ============================================================
# Set Artifact Tool Validation Tests
# ============================================================


class TestSetArtifactToolValidation:
    """Tests for artifact validation integration in set_artifact tool."""

    def test_clean_artifact_stored_normally(self) -> None:
        """Test that clean artifacts are stored without modification."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Generate report", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = "## Report\n- Finding 1\n- Finding 2"
        result = set_artifact_tool(item_id=item_id, artifact=artifact)
        assert "Artifact stored" in result
        assert task_list.ledger.items[item_id].artifact == artifact

    def test_light_contamination_auto_cleaned(self) -> None:
        """Test that light contamination is auto-cleaned and stored."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Write timeline", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = "I will now document the timeline.\n\n## Timeline\n09:00 - Start\n09:15 - End"
        result = set_artifact_tool(item_id=item_id, artifact=artifact)
        assert "auto-cleaned" in result
        stored = task_list.ledger.items[item_id].artifact
        assert "I will now" not in stored
        assert "## Timeline" in stored
        assert "09:00" in stored

    def test_heavy_contamination_rejected(self) -> None:
        """Test that heavy contamination is rejected and artifact not stored."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Write analysis", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = "## Analysis\nFinding 1.\nI'll update the work item with this.\nFinding 2."
        result = set_artifact_tool(item_id=item_id, artifact=artifact)
        assert "Error" in result
        assert "rejected" in result
        # Artifact should NOT be stored
        assert task_list.ledger.items[item_id].artifact == ""

    def test_heavy_preserves_existing_artifact(self) -> None:
        """Test that rejection preserves any previously stored artifact."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Write report", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        # First: store a clean artifact
        set_artifact_tool(item_id=item_id, artifact="## Original\nClean content.")
        assert task_list.ledger.items[item_id].artifact == "## Original\nClean content."

        # Second: attempt contaminated artifact
        bad_artifact = "## Updated\nNew content.\nI stored the artifact saved above.\nMore content."
        result = set_artifact_tool(item_id=item_id, artifact=bad_artifact)
        assert "Error" in result
        # Original artifact preserved
        assert task_list.ledger.items[item_id].artifact == "## Original\nClean content."

    def test_light_contamination_with_postamble(self) -> None:
        """Test light contamination with postamble only."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Summary", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = "## Summary\nKey points here.\n\nMoving on to the next task."
        result = set_artifact_tool(item_id=item_id, artifact=artifact)
        assert "auto-cleaned" in result
        stored = task_list.ledger.items[item_id].artifact
        assert "Moving on" not in stored
        assert "Key points" in stored

    def test_nonexistent_item_still_returns_error(self) -> None:
        """Test that nonexistent item ID error takes priority."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        set_artifact_tool = tools[3]

        result = set_artifact_tool(item_id="nonexistent", artifact="data")
        assert "Error" in result
        assert "not found" in result

    def test_clean_json_artifact_stored(self) -> None:
        """Test that JSON artifacts pass validation cleanly."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Store data", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = '{"results": [1, 2, 3], "status": "complete"}'
        result = set_artifact_tool(item_id=item_id, artifact=artifact)
        assert "Artifact stored" in result
        assert task_list.ledger.items[item_id].artifact == artifact

    def test_light_cleaned_content_preview_in_response(self) -> None:
        """Test that the response for LIGHT includes a preview of cleaned content."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Output", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = "Here is the result.\n\n## Result\nValue: 42"
        result = set_artifact_tool(item_id=item_id, artifact=artifact)
        assert item_id in result
        assert "Result" in result or "Value" in result


# ============================================================
# ArtifactRole Tests
# ============================================================


class TestArtifactRole:
    """Tests for ArtifactRole enum and WorkItem integration."""

    def test_enum_values(self) -> None:
        """Test that ArtifactRole has the expected values."""
        assert ArtifactRole.DELIVERABLE.value == "deliverable"
        assert ArtifactRole.WORKING.value == "working"
        assert ArtifactRole.CONTROL.value == "control"

    def test_work_item_default_role(self) -> None:
        """Test that WorkItem defaults to WORKING role."""
        item = WorkItem(id="abc", title="Test")
        assert item.artifact_role == ArtifactRole.WORKING

    def test_work_item_explicit_role(self) -> None:
        """Test creating WorkItem with explicit role."""
        item = WorkItem(id="abc", title="Test", artifact_role=ArtifactRole.DELIVERABLE)
        assert item.artifact_role == ArtifactRole.DELIVERABLE

        item2 = WorkItem(id="def", title="Audit", artifact_role=ArtifactRole.CONTROL)
        assert item2.artifact_role == ArtifactRole.CONTROL

    def test_serialization_round_trip(self) -> None:
        """Test artifact_role survives serialization."""
        item = WorkItem(
            id="abc",
            title="Report",
            artifact="content",
            artifact_role=ArtifactRole.DELIVERABLE,
        )
        data = item.to_dict()
        assert data["artifact_role"] == "deliverable"

        restored = WorkItem.from_dict(data)
        assert restored.artifact_role == ArtifactRole.DELIVERABLE

    def test_serialization_missing_role_defaults_to_working(self) -> None:
        """Test that missing artifact_role in dict defaults to WORKING."""
        data = {"id": "abc", "title": "Test", "status": "pending", "priority": "medium"}
        item = WorkItem.from_dict(data)
        assert item.artifact_role == ArtifactRole.WORKING

    def test_serialization_invalid_role_defaults_to_working(self) -> None:
        """Test that invalid artifact_role in dict defaults to WORKING."""
        data = {
            "id": "abc",
            "title": "Test",
            "status": "pending",
            "priority": "medium",
            "artifact_role": "nonexistent_role",
        }
        item = WorkItem.from_dict(data)
        assert item.artifact_role == ArtifactRole.WORKING


# ============================================================
# Set Artifact Tool Role Tests
# ============================================================


class TestSetArtifactToolRole:
    """Tests for artifact role parameter in set_artifact tool."""

    def test_deliverable_role_via_tool(self) -> None:
        """Test setting deliverable role via tool."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Final report", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        set_artifact_tool(item_id=item_id, artifact="## Report\nContent", role="deliverable")
        assert task_list.ledger.items[item_id].artifact_role == ArtifactRole.DELIVERABLE

    def test_working_role_via_tool(self) -> None:
        """Test setting working role via tool."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Scratch notes", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        set_artifact_tool(item_id=item_id, artifact="notes here", role="working")
        assert task_list.ledger.items[item_id].artifact_role == ArtifactRole.WORKING

    def test_control_role_via_tool(self) -> None:
        """Test setting control role via tool."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Validation check", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        set_artifact_tool(item_id=item_id, artifact='{"pass": true}', role="control")
        assert task_list.ledger.items[item_id].artifact_role == ArtifactRole.CONTROL

    def test_default_role_is_working(self) -> None:
        """Test that omitting role defaults to working."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Something", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        set_artifact_tool(item_id=item_id, artifact="data")
        assert task_list.ledger.items[item_id].artifact_role == ArtifactRole.WORKING

    def test_invalid_role_falls_back_to_working(self) -> None:
        """Test that invalid role string falls back to working."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Test item", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        set_artifact_tool(item_id=item_id, artifact="data", role="bogus")
        assert task_list.ledger.items[item_id].artifact_role == ArtifactRole.WORKING

    def test_role_set_on_light_contamination(self) -> None:
        """Test that role is still set when artifact has light contamination."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Report", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = "Here is the output.\n\n## Data\nContent here."
        set_artifact_tool(item_id=item_id, artifact=artifact, role="deliverable")
        assert task_list.ledger.items[item_id].artifact_role == ArtifactRole.DELIVERABLE


# ============================================================
# Revision Role Inheritance Tests
# ============================================================


class TestRevisionRoleInheritance:
    """Tests that revision children inherit artifact_role from parent."""

    def test_inherits_deliverable_role(self) -> None:
        """Test that revision child inherits deliverable role."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]
        flag_revision_tool = tools[4]

        add_tool(title="Write report", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]
        set_artifact_tool(item_id=item_id, artifact="content", role="deliverable")

        flag_revision_tool(item_id=item_id, reason="Needs fixes")

        revision_items = [i for i in task_list.ledger.items.values() if i.revision_of == item_id]
        assert len(revision_items) == 1
        assert revision_items[0].artifact_role == ArtifactRole.DELIVERABLE

    def test_inherits_control_role(self) -> None:
        """Test that revision child inherits control role."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]
        flag_revision_tool = tools[4]

        add_tool(title="Audit check", role="control")
        item_id = list(task_list.ledger.items.keys())[0]
        set_artifact_tool(item_id=item_id, artifact='{"pass": false}', role="control")

        flag_revision_tool(item_id=item_id, reason="Incorrect check")

        revision_items = [i for i in task_list.ledger.items.values() if i.revision_of == item_id]
        assert len(revision_items) == 1
        assert revision_items[0].artifact_role == ArtifactRole.CONTROL

    def test_inherits_working_role(self) -> None:
        """Test that revision child inherits working role (default)."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]
        flag_revision_tool = tools[4]

        add_tool(title="Draft notes", role="working")
        item_id = list(task_list.ledger.items.keys())[0]
        set_artifact_tool(item_id=item_id, artifact="draft", role="working")

        flag_revision_tool(item_id=item_id, reason="Incomplete")

        revision_items = [i for i in task_list.ledger.items.values() if i.revision_of == item_id]
        assert len(revision_items) == 1
        assert revision_items[0].artifact_role == ArtifactRole.WORKING


# ============================================================
# Ledger get_deliverables Tests
# ============================================================


class TestLedgerGetDeliverables:
    """Tests for WorkItemLedger.get_deliverables()."""

    def test_empty_ledger(self) -> None:
        """Test that empty ledger returns no deliverables."""
        ledger = WorkItemLedger()
        assert ledger.get_deliverables() == []

    def test_only_deliverables_with_content(self) -> None:
        """Test that only items with DELIVERABLE role and artifact are returned."""
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="a",
                title="Report",
                artifact="report content",
                artifact_role=ArtifactRole.DELIVERABLE,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="b",
                title="Empty deliverable",
                artifact="",
                artifact_role=ArtifactRole.DELIVERABLE,
            ),
        )
        deliverables = ledger.get_deliverables()
        assert len(deliverables) == 1
        assert deliverables[0].id == "a"

    def test_mixed_roles(self) -> None:
        """Test filtering across mixed roles."""
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="a",
                title="Report",
                artifact="report content",
                artifact_role=ArtifactRole.DELIVERABLE,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="b",
                title="Notes",
                artifact="scratch data",
                artifact_role=ArtifactRole.WORKING,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="c",
                title="Check",
                artifact='{"pass": true}',
                artifact_role=ArtifactRole.CONTROL,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="d",
                title="Final doc",
                artifact="## Document\nContent",
                artifact_role=ArtifactRole.DELIVERABLE,
            ),
        )

        deliverables = ledger.get_deliverables()
        assert len(deliverables) == 2
        ids = {d.id for d in deliverables}
        assert ids == {"a", "d"}

    def test_working_items_not_returned(self) -> None:
        """Test that working-role items are not deliverables."""
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="a",
                title="Draft",
                artifact="content",
                artifact_role=ArtifactRole.WORKING,
            ),
        )
        assert ledger.get_deliverables() == []

    def test_control_items_not_returned(self) -> None:
        """Test that control-role items are not deliverables."""
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="a",
                title="Audit",
                artifact='{"valid": true}',
                artifact_role=ArtifactRole.CONTROL,
            ),
        )
        assert ledger.get_deliverables() == []


# ============================================================
# Deliverable Lifecycle Event Tests
# ============================================================


class TestDeliverableLifecycleEvent:
    """Tests for deliverables_updated lifecycle event emission."""

    @pytest.fixture
    def mock_ctx(self):
        """Create a mock workflow context."""

        class MockContext:
            def __init__(self):
                self._state: dict[str, object] = {
                    HARNESS_TURN_COUNT_KEY: 1,
                    HARNESS_MAX_TURNS_KEY: 20,
                    HARNESS_TRANSCRIPT_KEY: [],
                }
                self._messages: list = []
                self._events: list = []
                self._outputs: list = []

            async def get_shared_state(self, key: str):
                if key not in self._state:
                    raise KeyError(key)
                return self._state[key]

            async def set_shared_state(self, key: str, value: object):
                self._state[key] = value

            async def send_message(self, msg):
                self._messages.append(msg)

            async def add_event(self, event):
                self._events.append(event)

            async def yield_output(self, output):
                self._outputs.append(output)

            def is_streaming(self):
                return False

        return MockContext()

    @pytest.mark.asyncio
    async def test_event_emitted_with_deliverables(self, mock_ctx) -> None:
        """Test that deliverables_updated event is emitted when deliverables exist."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        # Add a deliverable work item
        tools = task_list.get_tools()
        tools[0](title="Final output", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]
        tools[3](item_id=item_id, artifact="## Report\nContent here", role="deliverable")

        executor = AgentTurnExecutor(agent, task_list=task_list)

        # Call _sync_work_item_ledger directly
        await executor._sync_work_item_ledger(mock_ctx)

        # Check that the deliverables_updated event was emitted
        lifecycle_events = [
            e
            for e in mock_ctx._events
            if isinstance(e, HarnessLifecycleEvent) and e.event_type == "deliverables_updated"
        ]
        assert len(lifecycle_events) == 1
        event = lifecycle_events[0]
        assert event.data["count"] == 1
        assert event.data["items"][0]["item_id"] == item_id
        assert event.data["items"][0]["title"] == "Final output"

    @pytest.mark.asyncio
    async def test_event_with_no_deliverables_shows_progress(self, mock_ctx) -> None:
        """Test that event is emitted with progress stats even without deliverables."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        # Add a working-role item (not deliverable)
        tools = task_list.get_tools()
        tools[0](title="Scratch", role="working")
        item_id = list(task_list.ledger.items.keys())[0]
        tools[3](item_id=item_id, artifact="notes", role="working")

        executor = AgentTurnExecutor(agent, task_list=task_list)

        await executor._sync_work_item_ledger(mock_ctx)

        lifecycle_events = [
            e
            for e in mock_ctx._events
            if isinstance(e, HarnessLifecycleEvent) and e.event_type == "deliverables_updated"
        ]
        # Event IS emitted (for progress bar) but with empty deliverables list
        assert len(lifecycle_events) == 1
        data = lifecycle_events[0].data
        assert data["count"] == 0
        assert data["total_items"] == 1
        assert data["done_items"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_no_event_with_empty_ledger(self, mock_ctx) -> None:
        """Test that no event is emitted when ledger has zero items."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        # Don't add any items - ledger is empty
        executor = AgentTurnExecutor(agent, task_list=task_list)

        await executor._sync_work_item_ledger(mock_ctx)

        lifecycle_events = [
            e
            for e in mock_ctx._events
            if isinstance(e, HarnessLifecycleEvent) and e.event_type == "deliverables_updated"
        ]
        assert len(lifecycle_events) == 0

    @pytest.mark.asyncio
    async def test_event_includes_full_content(self, mock_ctx) -> None:
        """Test that event includes full artifact content."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        tools[0](title="Big report", role="deliverable")
        item_id = list(task_list.ledger.items.keys())[0]
        long_content = "x" * 200
        tools[3](item_id=item_id, artifact=long_content, role="deliverable")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        await executor._sync_work_item_ledger(mock_ctx)

        lifecycle_events = [
            e
            for e in mock_ctx._events
            if isinstance(e, HarnessLifecycleEvent) and e.event_type == "deliverables_updated"
        ]
        assert len(lifecycle_events) == 1
        content = lifecycle_events[0].data["items"][0]["content"]
        assert content == long_content
        assert len(content) == 200

    @pytest.mark.asyncio
    async def test_event_includes_completion_stats(self, mock_ctx) -> None:
        """Test that event includes work item completion stats."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        tools[0](title="Report", role="deliverable")
        tools[0](title="Notes", role="working")
        item_id = list(task_list.ledger.items.keys())[0]
        tools[3](item_id=item_id, artifact="content", role="deliverable")
        # Mark one item done
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        await executor._sync_work_item_ledger(mock_ctx)

        lifecycle_events = [
            e
            for e in mock_ctx._events
            if isinstance(e, HarnessLifecycleEvent) and e.event_type == "deliverables_updated"
        ]
        assert len(lifecycle_events) == 1
        data = lifecycle_events[0].data
        assert data["total_items"] == 2
        assert data["done_items"] == 1


# ============================================================
# HarnessResult Deliverables Tests
# ============================================================


class TestHarnessResultDeliverables:
    """Tests for HarnessResult.deliverables field."""

    def test_default_empty(self) -> None:
        """Test that deliverables defaults to empty list."""
        result = HarnessResult(status=HarnessStatus.DONE)
        assert result.deliverables == []

    def test_populated_deliverables(self) -> None:
        """Test creating HarnessResult with deliverables."""
        deliverables = [
            {"item_id": "a", "title": "Report", "content": "## Report\nData"},
            {"item_id": "b", "title": "Summary", "content": "Key points"},
        ]
        result = HarnessResult(
            status=HarnessStatus.DONE,
            deliverables=deliverables,
        )
        assert len(result.deliverables) == 2
        assert result.deliverables[0]["item_id"] == "a"
        assert result.deliverables[1]["content"] == "Key points"

    def test_to_dict_includes_deliverables(self) -> None:
        """Test that to_dict includes deliverables."""
        deliverables = [{"item_id": "a", "title": "Report", "content": "data"}]
        result = HarnessResult(
            status=HarnessStatus.DONE,
            deliverables=deliverables,
        )
        data = result.to_dict()
        assert "deliverables" in data
        assert data["deliverables"] == deliverables

    def test_to_dict_empty_deliverables(self) -> None:
        """Test that to_dict includes empty deliverables list."""
        result = HarnessResult(status=HarnessStatus.DONE)
        data = result.to_dict()
        assert data["deliverables"] == []

    @pytest.fixture
    def mock_ctx(self):
        """Create a mock workflow context for stop decision tests."""

        class MockContext:
            def __init__(self):
                self._state: dict[str, object] = {
                    HARNESS_TURN_COUNT_KEY: 1,
                    HARNESS_MAX_TURNS_KEY: 20,
                    HARNESS_TRANSCRIPT_KEY: [],
                }
                self._messages: list = []
                self._events: list = []
                self._outputs: list = []

            async def get_shared_state(self, key: str):
                if key not in self._state:
                    raise KeyError(key)
                return self._state[key]

            async def set_shared_state(self, key: str, value: object):
                self._state[key] = value

            async def send_message(self, msg):
                self._messages.append(msg)

            async def add_event(self, event):
                self._events.append(event)

            async def yield_output(self, output):
                self._outputs.append(output)

            def is_streaming(self):
                return False

        return MockContext()

    @pytest.mark.asyncio
    async def test_stop_decision_collects_deliverables(self, mock_ctx) -> None:
        """Test that StopDecisionExecutor collects deliverables in result."""
        executor = StopDecisionExecutor(
            enable_work_item_verification=True,
        )

        # Set up ledger with a deliverable
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="abc",
                title="Final Report",
                status=WorkItemStatus.DONE,
                artifact="## Report\nImportant findings",
                artifact_role=ArtifactRole.DELIVERABLE,
            ),
        )
        ledger.add_item(
            WorkItem(
                id="def",
                title="Scratch notes",
                status=WorkItemStatus.DONE,
                artifact="internal notes",
                artifact_role=ArtifactRole.WORKING,
            ),
        )
        mock_ctx._state[HARNESS_WORK_ITEM_LEDGER_KEY] = ledger.to_dict()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        assert len(mock_ctx._outputs) == 1
        result = mock_ctx._outputs[0]
        assert isinstance(result, HarnessResult)
        assert len(result.deliverables) == 1
        assert result.deliverables[0]["item_id"] == "abc"
        assert result.deliverables[0]["title"] == "Final Report"
        assert result.deliverables[0]["content"] == "## Report\nImportant findings"

    @pytest.mark.asyncio
    async def test_stop_decision_no_ledger_empty_deliverables(self, mock_ctx) -> None:
        """Test that missing ledger results in empty deliverables."""
        executor = StopDecisionExecutor()

        turn_result = TurnComplete(agent_done=True)
        await executor.evaluate(turn_result, mock_ctx)

        assert len(mock_ctx._outputs) == 1
        result = mock_ctx._outputs[0]
        assert isinstance(result, HarnessResult)
        assert result.deliverables == []


# ============================================================
# Control Artifact Validation Tests
# ============================================================


class TestValidateControlArtifact:
    """Tests for validate_control_artifact function."""

    def test_valid_pass_artifact(self) -> None:
        """Test valid pass artifact is accepted."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "completeness", "result": "pass", "detail": "All sections present"}],
            "summary": "All checks passed successfully",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is True
        assert error == ""

    def test_valid_fail_artifact(self) -> None:
        """Test valid fail artifact is accepted."""
        import json

        artifact = json.dumps({
            "verdict": "fail",
            "checks": [
                {"name": "accuracy", "result": "fail", "detail": "Date is wrong"},
                {"name": "completeness", "result": "pass", "detail": "All sections present"},
            ],
            "summary": "Accuracy check failed due to incorrect date",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is True
        assert error == ""

    def test_missing_verdict(self) -> None:
        """Test missing verdict field is rejected."""
        import json

        artifact = json.dumps({
            "checks": [{"name": "x", "result": "pass", "detail": "y"}],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "verdict" in error

    def test_invalid_verdict(self) -> None:
        """Test invalid verdict value is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "maybe",
            "checks": [{"name": "x", "result": "pass", "detail": "y"}],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "'maybe'" in error

    def test_missing_checks(self) -> None:
        """Test missing checks field is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "checks" in error

    def test_empty_checks(self) -> None:
        """Test empty checks array is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "at least one check" in error

    def test_checks_not_array(self) -> None:
        """Test non-array checks is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": "not an array",
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "must be an array" in error

    def test_check_missing_name(self) -> None:
        """Test check missing name field is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"result": "pass", "detail": "y"}],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "checks[0]" in error
        assert "'name'" in error

    def test_check_missing_result(self) -> None:
        """Test check missing result field is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "x", "detail": "y"}],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "checks[0]" in error
        assert "'result'" in error

    def test_check_missing_detail(self) -> None:
        """Test check missing detail field is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "x", "result": "pass"}],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "checks[0]" in error
        assert "'detail'" in error

    def test_check_invalid_result(self) -> None:
        """Test check with invalid result value is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "x", "result": "unknown", "detail": "y"}],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "'unknown'" in error

    def test_check_not_object(self) -> None:
        """Test non-object check is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": ["not an object"],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "checks[0] must be an object" in error

    def test_missing_summary(self) -> None:
        """Test missing summary field is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "x", "result": "pass", "detail": "y"}],
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "summary" in error

    def test_empty_summary(self) -> None:
        """Test empty summary string is rejected."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "x", "result": "pass", "detail": "y"}],
            "summary": "   ",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "non-empty string" in error

    def test_non_json_input(self) -> None:
        """Test non-JSON input is rejected."""
        valid, error = validate_control_artifact("This is not JSON at all")
        assert valid is False
        assert "valid JSON" in error

    def test_json_array_rejected(self) -> None:
        """Test JSON array (not object) is rejected."""
        valid, error = validate_control_artifact('[{"verdict": "pass"}]')
        assert valid is False
        assert "JSON object" in error

    def test_multiple_checks(self) -> None:
        """Test artifact with multiple checks validates each."""
        import json

        artifact = json.dumps({
            "verdict": "fail",
            "checks": [
                {"name": "check1", "result": "pass", "detail": "ok"},
                {"name": "check2", "result": "fail", "detail": "issue found"},
                {"name": "check3", "result": "pass", "detail": "fine"},
            ],
            "summary": "One check failed",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is True
        assert error == ""

    def test_second_check_invalid(self) -> None:
        """Test that validation catches error in second check."""
        import json

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [
                {"name": "check1", "result": "pass", "detail": "ok"},
                {"name": "check2", "result": "invalid", "detail": "oops"},
            ],
            "summary": "ok",
        })
        valid, error = validate_control_artifact(artifact)
        assert valid is False
        assert "checks[1]" in error


# ============================================================
# Set Artifact Control Validation Tests
# ============================================================


class TestSetArtifactControlValidation:
    """Tests for control artifact validation in set_artifact tool."""

    def test_rejects_invalid_json_control_artifact(self) -> None:
        """Test that tool rejects non-JSON control artifact."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Audit check", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        result = set_artifact_tool(item_id=item_id, artifact="not json", role="control")
        assert "Error" in result
        assert "Invalid control artifact format" in result
        assert "Required format" in result
        # Artifact should NOT be stored
        assert task_list.ledger.items[item_id].artifact == ""

    def test_accepts_valid_pass_control_artifact(self) -> None:
        """Test that tool accepts valid pass control artifact."""
        import json

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Quality gate", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "completeness", "result": "pass", "detail": "all good"}],
            "summary": "All checks passed",
        })
        result = set_artifact_tool(item_id=item_id, artifact=artifact, role="control")
        assert "Artifact stored" in result
        assert task_list.ledger.items[item_id].artifact == artifact

    def test_accepts_valid_fail_control_artifact(self) -> None:
        """Test that tool accepts valid fail control artifact."""
        import json

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Accuracy audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = json.dumps({
            "verdict": "fail",
            "checks": [{"name": "dates", "result": "fail", "detail": "Wrong year"}],
            "summary": "Date accuracy check failed",
        })
        result = set_artifact_tool(item_id=item_id, artifact=artifact, role="control")
        assert "Artifact stored" in result
        assert task_list.ledger.items[item_id].artifact == artifact

    def test_rejects_missing_verdict_control_artifact(self) -> None:
        """Test that tool rejects control artifact missing verdict."""
        import json

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Check", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = json.dumps({
            "checks": [{"name": "x", "result": "pass", "detail": "y"}],
            "summary": "ok",
        })
        result = set_artifact_tool(item_id=item_id, artifact=artifact, role="control")
        assert "Error" in result
        assert "verdict" in result

    def test_rejects_bad_check_format(self) -> None:
        """Test that tool rejects control artifact with malformed checks."""
        import json

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "x"}],  # missing result and detail
            "summary": "ok",
        })
        result = set_artifact_tool(item_id=item_id, artifact=artifact, role="control")
        assert "Error" in result
        assert "checks[0]" in result

    def test_non_control_role_skips_control_validation(self) -> None:
        """Test that non-control roles bypass control artifact validation."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]
        set_artifact_tool = tools[3]

        add_tool(title="Notes", role="working")
        item_id = list(task_list.ledger.items.keys())[0]

        # Plain text that would fail control validation
        result = set_artifact_tool(item_id=item_id, artifact="just plain text", role="working")
        assert "Artifact stored" in result
        assert task_list.ledger.items[item_id].artifact == "just plain text"


# ============================================================
# Control Invariant Check Tests
# ============================================================


class TestControlInvariantCheck:
    """Tests for _check_control_invariants method."""

    def test_no_control_items_returns_none(self) -> None:
        """Test that no control items returns None."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        tools[0](title="Some work", role="working")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        result = executor._check_control_invariants(task_list.ledger)
        assert result is None

    def test_pass_verdict_returns_none(self) -> None:
        """Test that pass verdict returns None."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        tools[0](title="Audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]

        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "check1", "result": "pass", "detail": "ok"}],
            "summary": "All passed",
        })
        tools[3](item_id=item_id, artifact=artifact, role="control")
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        result = executor._check_control_invariants(task_list.ledger)
        assert result is None

    def test_fail_with_revisions_returns_none(self) -> None:
        """Test that fail verdict with existing revisions returns None."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        # Add a work item and mark done
        tools[0](title="Write report", role="deliverable")
        work_item_id = list(task_list.ledger.items.keys())[0]
        tools[1](item_id=work_item_id, status="done")

        # Add a control item with fail verdict
        tools[0](title="Audit", role="control")
        control_id = [k for k in task_list.ledger.items if k != work_item_id][0]
        artifact = json.dumps({
            "verdict": "fail",
            "checks": [{"name": "accuracy", "result": "fail", "detail": "wrong"}],
            "summary": "Failed",
        })
        tools[3](item_id=control_id, artifact=artifact, role="control")
        tools[1](item_id=control_id, status="done")

        # Flag revision on the work item (creates a revision child)
        tools[4](item_id=work_item_id, reason="Accuracy issue")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        result = executor._check_control_invariants(task_list.ledger)
        assert result is None

    def test_fail_without_revisions_returns_prompt(self) -> None:
        """Test that fail verdict without revisions returns continuation prompt."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        # Add a control item with fail verdict but no revisions
        tools[0](title="Audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]
        artifact = json.dumps({
            "verdict": "fail",
            "checks": [{"name": "date accuracy", "result": "fail", "detail": "wrong year"}],
            "summary": "Date check failed",
        })
        tools[3](item_id=item_id, artifact=artifact, role="control")
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        result = executor._check_control_invariants(task_list.ledger)
        assert result is not None
        assert "date accuracy" in result
        assert "flag_revision" in result

    def test_malformed_json_artifact_skipped(self) -> None:
        """Test that malformed JSON in control artifact is skipped gracefully."""
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        # Manually construct a ledger with a malformed artifact
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="abc",
                title="Broken audit",
                status=WorkItemStatus.DONE,
                artifact="not valid json {{{",
                artifact_role=ArtifactRole.CONTROL,
            ),
        )

        executor = AgentTurnExecutor(agent, task_list=WorkItemTaskList())
        result = executor._check_control_invariants(ledger)
        assert result is None

    def test_multiple_failed_checks_all_reported(self) -> None:
        """Test that multiple failed checks are all listed in the prompt."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        tools[0](title="Audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]
        artifact = json.dumps({
            "verdict": "fail",
            "checks": [
                {"name": "accuracy", "result": "fail", "detail": "wrong"},
                {"name": "completeness", "result": "fail", "detail": "missing sections"},
                {"name": "formatting", "result": "pass", "detail": "ok"},
            ],
            "summary": "Multiple failures",
        })
        tools[3](item_id=item_id, artifact=artifact, role="control")
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        result = executor._check_control_invariants(task_list.ledger)
        assert result is not None
        assert "accuracy" in result
        assert "completeness" in result
        # Pass check should not be in the list
        assert "formatting" not in result

    def test_pending_control_item_not_checked(self) -> None:
        """Test that pending control items are not checked for invariants."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        # Manually create a pending control item with fail verdict
        ledger = WorkItemLedger()
        ledger.add_item(
            WorkItem(
                id="abc",
                title="Pending audit",
                status=WorkItemStatus.PENDING,
                artifact=json.dumps({
                    "verdict": "fail",
                    "checks": [{"name": "x", "result": "fail", "detail": "y"}],
                    "summary": "failed",
                }),
                artifact_role=ArtifactRole.CONTROL,
            ),
        )

        executor = AgentTurnExecutor(agent, task_list=WorkItemTaskList())
        result = executor._check_control_invariants(ledger)
        assert result is None


# ============================================================
# Control Invariant Injection Tests
# ============================================================


class TestControlInvariantInjection:
    """Integration tests for control invariant enforcement in sync flow."""

    @pytest.fixture
    def mock_ctx(self):
        """Create a mock workflow context."""

        class MockContext:
            def __init__(self):
                self._state: dict[str, object] = {
                    HARNESS_TURN_COUNT_KEY: 1,
                    HARNESS_MAX_TURNS_KEY: 20,
                    HARNESS_TRANSCRIPT_KEY: [],
                }
                self._messages: list = []
                self._events: list = []
                self._outputs: list = []

            async def get_shared_state(self, key: str):
                if key not in self._state:
                    raise KeyError(key)
                return self._state[key]

            async def set_shared_state(self, key: str, value: object):
                self._state[key] = value

            async def send_message(self, msg):
                self._messages.append(msg)

            async def add_event(self, event):
                self._events.append(event)

            async def yield_output(self, output):
                self._outputs.append(output)

            def is_streaming(self):
                return False

        return MockContext()

    @pytest.mark.asyncio
    async def test_invariant_injects_continuation_message(self, mock_ctx) -> None:
        """Test that sync_work_item_ledger injects continuation when invariant violated."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        # Add and complete a control item with fail verdict
        tools[0](title="Quality audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]
        artifact = json.dumps({
            "verdict": "fail",
            "checks": [{"name": "timeline accuracy", "result": "fail", "detail": "dates wrong"}],
            "summary": "Failed audit",
        })
        tools[3](item_id=item_id, artifact=artifact, role="control")
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)

        await executor._sync_work_item_ledger(mock_ctx)

        # Check that a continuation message was injected into the cache
        user_messages = [m for m in executor._cache if getattr(getattr(m, "role", None), "value", None) == "user"]
        assert len(user_messages) == 1
        assert "timeline accuracy" in user_messages[0].text
        assert "flag_revision" in user_messages[0].text

    @pytest.mark.asyncio
    async def test_invariant_emits_event(self, mock_ctx) -> None:
        """Test that control_invariant_violation event is recorded in transcript."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        tools[0](title="Audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]
        artifact = json.dumps({
            "verdict": "fail",
            "checks": [{"name": "check1", "result": "fail", "detail": "bad"}],
            "summary": "Failed",
        })
        tools[3](item_id=item_id, artifact=artifact, role="control")
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        await executor._sync_work_item_ledger(mock_ctx)

        # Check transcript for control_invariant_violation event
        transcript = mock_ctx._state[HARNESS_TRANSCRIPT_KEY]
        invariant_events = [e for e in transcript if e["event_type"] == "control_invariant_violation"]
        assert len(invariant_events) == 1
        assert "prompt" in invariant_events[0]["data"]
        assert "flag_revision" in invariant_events[0]["data"]["prompt"]

    @pytest.mark.asyncio
    async def test_no_injection_when_revisions_exist(self, mock_ctx) -> None:
        """Test that no injection occurs when revisions already exist."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        # Add a work item, complete it, then flag for revision
        tools[0](title="Report", role="deliverable")
        work_id = list(task_list.ledger.items.keys())[0]
        tools[3](item_id=work_id, artifact="content", role="deliverable")
        tools[1](item_id=work_id, status="done")
        tools[4](item_id=work_id, reason="Issues found")

        # Add a control item with fail verdict
        tools[0](title="Audit", role="control")
        control_id = [k for k in task_list.ledger.items if k != work_id and not task_list.ledger.items[k].revision_of][
            0
        ]
        artifact = json.dumps({
            "verdict": "fail",
            "checks": [{"name": "issue", "result": "fail", "detail": "problem"}],
            "summary": "Failed",
        })
        tools[3](item_id=control_id, artifact=artifact, role="control")
        tools[1](item_id=control_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        await executor._sync_work_item_ledger(mock_ctx)

        # No continuation message should be injected
        user_messages = [m for m in executor._cache if getattr(getattr(m, "role", None), "value", None) == "user"]
        assert len(user_messages) == 0

    @pytest.mark.asyncio
    async def test_no_injection_for_pass_verdict(self, mock_ctx) -> None:
        """Test that no injection occurs for pass verdicts."""
        import json
        from unittest.mock import MagicMock

        from agent_framework._harness._agent_turn_executor import AgentTurnExecutor

        agent = MagicMock()
        agent.get_new_thread.return_value = MagicMock()

        task_list = WorkItemTaskList()
        tools = task_list.get_tools()

        tools[0](title="Audit", role="control")
        item_id = list(task_list.ledger.items.keys())[0]
        artifact = json.dumps({
            "verdict": "pass",
            "checks": [{"name": "all good", "result": "pass", "detail": "fine"}],
            "summary": "Passed",
        })
        tools[3](item_id=item_id, artifact=artifact, role="control")
        tools[1](item_id=item_id, status="done")

        executor = AgentTurnExecutor(agent, task_list=task_list)
        await executor._sync_work_item_ledger(mock_ctx)

        # No continuation message
        user_messages = [m for m in executor._cache if getattr(getattr(m, "role", None), "value", None) == "user"]
        assert len(user_messages) == 0

        # No invariant violation events
        transcript = mock_ctx._state[HARNESS_TRANSCRIPT_KEY]
        invariant_events = [e for e in transcript if e["event_type"] == "control_invariant_violation"]
        assert len(invariant_events) == 0
