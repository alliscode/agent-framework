# Copyright (c) Microsoft. All rights reserved.

"""State types for Agent Harness."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4


class HarnessStatus(Enum):
    """Status of the harness execution."""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STALLED = "stalled"


@dataclass
class StopReason:
    """Reason why the harness stopped execution.

    Attributes:
        kind: The type of stop reason.
        message: A human-readable message explaining the stop.
        details: Optional additional details about the stop.
    """

    kind: Literal["max_turns", "agent_done", "hard_stop", "stalled", "failed"]
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "kind": self.kind,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StopReason":
        """Deserialize from dictionary."""
        return cls(
            kind=data["kind"],
            message=data["message"],
            details=data.get("details"),
        )


@dataclass
class HarnessEvent:
    """Single event in the harness transcript.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: The type of event.
        timestamp: ISO 8601 timestamp when the event occurred.
        data: Event-specific data payload.
    """

    event_type: Literal[
        "turn_start",
        "agent_response",
        "tool_call",
        "tool_result",
        "repair",
        "stop_decision",
        "continuation_prompt",
        "stall_detected",
        "context_pressure",
        "work_item_reminder",
    ]
    data: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessEvent":
        """Deserialize from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            timestamp=data["timestamp"],
            data=data["data"],
        )


@dataclass
class PendingToolCall:
    """Tool call awaiting result (for repair detection).

    Attributes:
        call_id: Unique identifier for the tool call.
        tool_name: Name of the tool being called.
        args: Arguments passed to the tool.
        turn_number: The turn number when this tool call was made.
    """

    call_id: str
    tool_name: str
    args: dict[str, Any]
    turn_number: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "turn_number": self.turn_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingToolCall":
        """Deserialize from dictionary."""
        return cls(
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            args=data["args"],
            turn_number=data["turn_number"],
        )


@dataclass
class HarnessResult:
    """Final result from harness execution.

    Attributes:
        status: The final status of the harness.
        reason: The reason for stopping, if applicable.
        transcript: The complete event transcript.
        turn_count: The total number of turns executed.
    """

    status: HarnessStatus
    reason: StopReason | None = None
    transcript: list[HarnessEvent] = field(default_factory=lambda: [])
    turn_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "reason": self.reason.to_dict() if self.reason else None,
            "transcript": [e.to_dict() for e in self.transcript],
            "turn_count": self.turn_count,
        }


# Workflow event for streaming harness lifecycle to DevUI

from .._workflows._events import WorkflowEvent


class HarnessLifecycleEvent(WorkflowEvent):
    """Workflow event for streaming harness lifecycle updates.

    This event is emitted via ctx.add_event() so DevUI can display
    real-time harness state alongside traces and agent responses.

    Attributes:
        event_type: The type of lifecycle event.
        turn_number: Current turn number (if applicable).
        max_turns: Maximum allowed turns.
        harness_data: Event-specific payload.
        timestamp: ISO 8601 timestamp.
    """

    def __init__(
        self,
        event_type: Literal[
            "harness_started",
            "turn_started",
            "turn_completed",
            "continuation_prompt",
            "stall_detected",
            "context_pressure",
            "harness_completed",
        ],
        *,
        turn_number: int = 0,
        max_turns: int = 0,
        data: dict[str, Any] | None = None,
    ):
        """Initialize the harness lifecycle event.

        Args:
            event_type: The type of lifecycle event.
            turn_number: Current turn number (if applicable).
            max_turns: Maximum allowed turns.
            data: Event-specific payload.
        """
        super().__init__(data=data)
        self.event_type = event_type
        self.turn_number = turn_number
        self.max_turns = max_turns
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON transmission."""
        return {
            "event_type": self.event_type,
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"HarnessLifecycleEvent(event_type={self.event_type}, "
            f"turn={self.turn_number}/{self.max_turns}, origin={self.origin})"
        )


# Message types for inter-executor communication within the harness


@dataclass
class RepairTrigger:
    """Message to trigger the repair phase.

    Sent at workflow start and after each continue decision.
    """

    pass


@dataclass
class RepairComplete:
    """Message indicating repair phase is complete.

    Sent from RepairExecutor to AgentTurnExecutor.
    """

    repairs_made: int = 0


@dataclass
class TurnComplete:
    """Message indicating an agent turn is complete.

    Sent from AgentTurnExecutor to StopDecisionExecutor.

    Attributes:
        agent_done: Whether the agent signaled it has completed its task.
        error: Optional error message if the turn failed.
    """

    agent_done: bool = False
    error: str | None = None
