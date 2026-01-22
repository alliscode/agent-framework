# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness Work Item tracking module."""


import pytest

from agent_framework._harness import (
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
)
from agent_framework._harness._constants import (
    HARNESS_MAX_TURNS_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
    HARNESS_WORK_ITEM_LEDGER_KEY,
)

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

    def test_get_tools_returns_three_tools(self) -> None:
        """Test that get_tools returns three tool functions."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        assert len(tools) == 3

    def test_add_tool(self) -> None:
        """Test the work_item_add tool closure."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        result = add_tool(title="Write tests", priority="high", notes="Important")
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

        add_tool(title="Simple task")
        item = list(task_list.ledger.items.values())[0]
        assert item.priority == WorkItemPriority.MEDIUM

    def test_add_tool_invalid_priority(self) -> None:
        """Test that invalid priority falls back to medium."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool = tools[0]

        add_tool(title="Task", priority="invalid")
        item = list(task_list.ledger.items.values())[0]
        assert item.priority == WorkItemPriority.MEDIUM

    def test_update_tool(self) -> None:
        """Test the work_item_update tool closure."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool, update_tool = tools[0], tools[1]

        # Add an item first
        add_tool(title="Build feature")
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

        add_tool(title="Task A", priority="high")
        add_tool(title="Task B")
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

        add_tool(title="Done task")
        add_tool(title="Pending task")
        item_ids = list(task_list.ledger.items.keys())
        update_tool(item_id=item_ids[0], status="done")

        result = list_tool(filter_status="done")
        assert "Done task" in result
        assert "Pending task" not in result

    def test_tools_share_same_ledger(self) -> None:
        """Test that all tools operate on the same ledger."""
        task_list = WorkItemTaskList()
        tools = task_list.get_tools()
        add_tool, update_tool, list_tool = tools[0], tools[1], tools[2]

        # Add via tool
        add_tool(title="Shared task")
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
        assert "incomplete work items" in result
        assert "[abc123]" in result
        assert "Implement handler" in result
        assert "(high)" in result
        assert "[def456]" in result
        assert "Add tests" in result
        assert "[~]" in result  # in_progress icon
        assert "ghi789" not in result  # done item should not appear
        assert "signal done again" in result


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
