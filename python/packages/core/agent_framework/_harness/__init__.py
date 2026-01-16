# Copyright (c) Microsoft. All rights reserved.

"""Agent Harness - Runtime infrastructure for long-running agent execution.

The Agent Harness provides a workflow-based runtime for durable, recoverable
agent execution with turn-based control, transcript tracking, and layered
stop conditions.

Example:
    .. code-block:: python

        from agent_framework import ChatAgent
        from agent_framework._harness import AgentHarness, HarnessWorkflowBuilder

        # Simple API using AgentHarness wrapper
        agent = ChatAgent(chat_client=my_client, tools=[...])
        harness = AgentHarness(agent, max_turns=20)
        result = await harness.run("Solve this complex task...")

        # Or use the builder for more control
        builder = HarnessWorkflowBuilder(agent, max_turns=20)
        workflow = builder.build()
        result = await workflow.run(
            RepairTrigger(),
            **builder.get_harness_kwargs(),
        )

        # With context pressure management (Phase 2)
        harness = AgentHarness(
            agent,
            max_turns=50,
            enable_context_pressure=True,
            max_input_tokens=100000,
        )
"""

from ._agent_turn_executor import AgentTurnExecutor
from ._done_tool import TASK_COMPLETE_TOOL_NAME, get_task_complete_tool, task_complete
from ._constants import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SOFT_THRESHOLD_PERCENT,
    DEFAULT_STALL_THRESHOLD,
    HARNESS_COMPLETION_REPORT_KEY,
    HARNESS_CONTEXT_EDIT_HISTORY_KEY,
    HARNESS_COVERAGE_LEDGER_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_PENDING_TOOL_CALLS_KEY,
    HARNESS_PROGRESS_TRACKER_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_STOP_REASON_KEY,
    HARNESS_TASK_CONTRACT_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from ._context_pressure import (
    ClearEdit,
    ClearToolResultsStrategy,
    CompactConversationStrategy,
    CompactEdit,
    ContextEdit,
    ContextEditKind,
    ContextEditPlan,
    ContextPressureStrategy,
    DropEdit,
    DropOldestStrategy,
    ExternalizeEdit,
    TokenBudget,
    TranscriptRange,
    estimate_tokens,
    estimate_transcript_tokens,
    get_default_strategies,
)
from ._context_pressure_executor import ContextPressureComplete, ContextPressureExecutor
from ._contract_verifier import ContractVerificationResult, ContractVerifier, VerificationResult
from ._harness_builder import AgentHarness, HarnessWorkflowBuilder
from ._repair_executor import RepairExecutor
from ._state import (
    HarnessEvent,
    HarnessLifecycleEvent,
    HarnessResult,
    HarnessStatus,
    PendingToolCall,
    RepairComplete,
    RepairTrigger,
    StopReason,
    TurnComplete,
)
from ._stop_decision_executor import StopDecisionExecutor
from ._task_contract import (
    AcceptabilityCriteria,
    CompletionReport,
    CoverageLedger,
    Evidence,
    GapReport,
    Predicate,
    PredicateType,
    ProgressFingerprint,
    ProgressTracker,
    RequiredOutput,
    RequirementCoverage,
    RequirementStatus,
    TaskContract,
    UserQuestion,
)

__all__ = [
    # Constants
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SOFT_THRESHOLD_PERCENT",
    "DEFAULT_STALL_THRESHOLD",
    "HARNESS_COMPLETION_REPORT_KEY",
    "HARNESS_CONTEXT_EDIT_HISTORY_KEY",
    "HARNESS_COVERAGE_LEDGER_KEY",
    "HARNESS_MAX_TURNS_KEY",
    "HARNESS_PENDING_TOOL_CALLS_KEY",
    "HARNESS_PROGRESS_TRACKER_KEY",
    "HARNESS_STATUS_KEY",
    "HARNESS_STOP_REASON_KEY",
    "HARNESS_TASK_CONTRACT_KEY",
    "HARNESS_TOKEN_BUDGET_KEY",
    "HARNESS_TRANSCRIPT_KEY",
    "HARNESS_TURN_COUNT_KEY",
    # Main API
    "AgentHarness",
    "HarnessWorkflowBuilder",
    # Executors
    "AgentTurnExecutor",
    "ContextPressureExecutor",
    "RepairExecutor",
    "StopDecisionExecutor",
    # State types
    "ContextPressureComplete",
    "HarnessEvent",
    "HarnessLifecycleEvent",
    "HarnessResult",
    "HarnessStatus",
    "PendingToolCall",
    "RepairComplete",
    "RepairTrigger",
    "StopReason",
    "TurnComplete",
    # Context pressure types (Phase 2)
    "ClearEdit",
    "CompactEdit",
    "ContextEdit",
    "ContextEditKind",
    "ContextEditPlan",
    "ContextPressureStrategy",
    "DropEdit",
    "ExternalizeEdit",
    "TokenBudget",
    "TranscriptRange",
    # Context pressure strategies
    "ClearToolResultsStrategy",
    "CompactConversationStrategy",
    "DropOldestStrategy",
    "get_default_strategies",
    # Task contract types (Phase 3)
    "AcceptabilityCriteria",
    "CompletionReport",
    "ContractVerificationResult",
    "ContractVerifier",
    "CoverageLedger",
    "Evidence",
    "GapReport",
    "Predicate",
    "PredicateType",
    "ProgressFingerprint",
    "ProgressTracker",
    "RequiredOutput",
    "RequirementCoverage",
    "RequirementStatus",
    "TaskContract",
    "UserQuestion",
    "VerificationResult",
    # Utilities
    "estimate_tokens",
    "estimate_transcript_tokens",
    # Done signaling tool
    "TASK_COMPLETE_TOOL_NAME",
    "get_task_complete_tool",
    "task_complete",
]
