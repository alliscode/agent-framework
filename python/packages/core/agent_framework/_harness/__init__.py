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

# Production context compaction types (v2)
from ._compaction import (
    COMPACTION_PRECEDENCE,
    COMPACTION_RENDER_FORMAT_VERSION,
    # Durability types
    DEFAULT_DURABILITY_POLICIES,
    STRUCTURED_SUMMARY_SCHEMA_VERSION,
    SUMMARY_RENDER_VERSION,
    TIKTOKEN_AVAILABLE,
    # Renderer types
    ArtifactMetadata,
    # Summary types
    ArtifactReference,
    ArtifactStore,
    ArtifactStoreEntry,
    CacheEntry,
    ClearRecord,
    # Strategy types
    ClearStrategy,
    # Core types
    CompactionAction,
    # Event types
    CompactionCheckStartedEvent,
    CompactionCompletedEvent,
    CompactionCoordinator,
    CompactionErrorEvent,
    CompactionEvent,
    CompactionEventEmitter,
    CompactionEventType,
    CompactionMetrics,
    CompactionPlan,
    CompactionProposal,
    CompactionResult,
    # Storage and concurrency types
    CompactionStore,
    CompactionStrategy,
    CompactionTransaction,
    CompositeEventEmitter,
    ContentClearedEvent,
    ContentDroppedEvent,
    ContentExternalizedEvent,
    ContentRehydratedEvent,
    ContentSummarizedEvent,
    Decision,
    DeterminismMetadata,
    DropRecord,
    DropStrategy,
    ExternalizationRecord,
    ExternalizeStrategy,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    LoggingEventEmitter,
    MetricsCollector,
    # Tokenizer types
    ModelProvider,
    OpenItem,
    PromptRenderer,
    ProposalGeneratedEvent,
    ProposalRejectedEvent,
    ProviderAwareTokenizer,
    # Rehydration types
    RehydrationBlockedEvent,
    RehydrationConfig,
    RehydrationEvent,
    RehydrationInterceptor,
    RehydrationResult,
    RehydrationState,
    RenderedPrompt,
    SecurityContext,
    SimpleTokenizer,
    SpanReference,
    StructuredSummary,
    SummarizationRecord,
    Summarizer,
    SummarizeStrategy,
    SummaryCache,
    SummaryCacheKey,
    TiktokenTokenizer,
    ToolCall,
    ToolDurability,
    ToolDurabilityPolicy,
    ToolOutcome,
    ToolResultEnvelope,
    TurnContext,
    VersionConflictEvent,
    compute_content_hash,
    create_rehydration_interceptor,
    create_summary_cache_key,
    get_tokenizer,
    render_externalization_text,
    render_summary_text,
)
from ._compaction import (
    TokenBudget as TokenBudgetV2,  # Alias to avoid conflict with existing TokenBudget
)
from ._compaction_executor import CompactionComplete, CompactionExecutor
from ._constants import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SOFT_THRESHOLD_PERCENT,
    DEFAULT_STALL_THRESHOLD,
    HARNESS_COMPACTION_METRICS_KEY,
    HARNESS_COMPACTION_PLAN_KEY,
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
    HARNESS_WORK_ITEM_LEDGER_KEY,
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
from ._done_tool import TASK_COMPLETE_TOOL_NAME, get_task_complete_tool, task_complete
from ._harness_builder import AgentHarness, HarnessWorkflowBuilder
from ._renderers import (
    ACTIVITY_VERBS,
    HarnessRenderer,
    MarkdownRenderer,
    PassthroughRenderer,
    render_stream,
)
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
from ._work_items import (
    ArtifactContaminationLevel,
    ArtifactRole,
    ArtifactValidationResult,
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

__all__ = [
    # Renderer types
    "ACTIVITY_VERBS",
    # Production context compaction (v2)
    # Core types
    "COMPACTION_PRECEDENCE",
    "COMPACTION_RENDER_FORMAT_VERSION",
    # Durability types
    "DEFAULT_DURABILITY_POLICIES",
    # Constants
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SOFT_THRESHOLD_PERCENT",
    "DEFAULT_STALL_THRESHOLD",
    "HARNESS_COMPACTION_METRICS_KEY",
    "HARNESS_COMPACTION_PLAN_KEY",
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
    "HARNESS_WORK_ITEM_LEDGER_KEY",
    "STRUCTURED_SUMMARY_SCHEMA_VERSION",
    "SUMMARY_RENDER_VERSION",
    # Done signaling tool
    "TASK_COMPLETE_TOOL_NAME",
    "TIKTOKEN_AVAILABLE",
    # Task contract types (Phase 3)
    "AcceptabilityCriteria",
    # Main API
    "AgentHarness",
    # Executors
    "AgentTurnExecutor",
    # Artifact types
    "ArtifactContaminationLevel",
    "ArtifactMetadata",
    "ArtifactReference",
    "ArtifactRole",
    "ArtifactStore",
    "ArtifactStoreEntry",
    "ArtifactValidationResult",
    "CacheEntry",
    # Context pressure types (Phase 2)
    "ClearEdit",
    "ClearRecord",
    # Strategy types
    "ClearStrategy",
    # Context pressure strategies
    "ClearToolResultsStrategy",
    "CompactConversationStrategy",
    "CompactEdit",
    "CompactionAction",
    # Event types
    "CompactionCheckStartedEvent",
    # Compaction executor
    "CompactionComplete",
    "CompactionCompletedEvent",
    "CompactionCoordinator",
    "CompactionErrorEvent",
    "CompactionEvent",
    "CompactionEventEmitter",
    "CompactionEventType",
    "CompactionExecutor",
    "CompactionMetrics",
    "CompactionPlan",
    "CompactionProposal",
    "CompactionResult",
    # Storage and concurrency types
    "CompactionStore",
    "CompactionStrategy",
    "CompactionTransaction",
    "CompletionReport",
    "CompositeEventEmitter",
    "ContentClearedEvent",
    "ContentDroppedEvent",
    "ContentExternalizedEvent",
    "ContentRehydratedEvent",
    "ContentSummarizedEvent",
    "ContextEdit",
    "ContextEditKind",
    "ContextEditPlan",
    # State types
    "ContextPressureComplete",
    "ContextPressureExecutor",
    "ContextPressureStrategy",
    "ContractVerificationResult",
    "ContractVerifier",
    "CoverageLedger",
    "Decision",
    "DeterminismMetadata",
    "DropEdit",
    "DropOldestStrategy",
    "DropRecord",
    "DropStrategy",
    "Evidence",
    "ExternalizationRecord",
    "ExternalizeEdit",
    "ExternalizeStrategy",
    "GapReport",
    "HarnessEvent",
    "HarnessLifecycleEvent",
    "HarnessRenderer",
    "HarnessResult",
    "HarnessStatus",
    "HarnessWorkflowBuilder",
    "InMemoryArtifactStore",
    "InMemoryCompactionStore",
    "InMemorySummaryCache",
    "LoggingEventEmitter",
    "MarkdownRenderer",
    "MetricsCollector",
    # Tokenizer types (v2)
    "ModelProvider",
    "OpenItem",
    "PassthroughRenderer",
    "PendingToolCall",
    "Predicate",
    "PredicateType",
    "ProgressFingerprint",
    "ProgressTracker",
    "PromptRenderer",
    "ProposalGeneratedEvent",
    "ProposalRejectedEvent",
    "ProviderAwareTokenizer",
    # Rehydration types
    "RehydrationBlockedEvent",
    "RehydrationConfig",
    "RehydrationEvent",
    "RehydrationInterceptor",
    "RehydrationResult",
    "RehydrationState",
    "RenderedPrompt",
    "RepairComplete",
    "RepairExecutor",
    "RepairTrigger",
    "RequiredOutput",
    "RequirementCoverage",
    "RequirementStatus",
    "SecurityContext",
    "SimpleTokenizer",
    "SpanReference",
    "StopDecisionExecutor",
    "StopReason",
    "StructuredSummary",
    "SummarizationRecord",
    "SummarizeStrategy",
    "Summarizer",
    "SummaryCache",
    "SummaryCacheKey",
    "TaskContract",
    "TiktokenTokenizer",
    "TokenBudget",
    "TokenBudgetV2",
    "ToolCall",
    "ToolDurability",
    "ToolDurabilityPolicy",
    "ToolOutcome",
    "ToolResultEnvelope",
    "TranscriptRange",
    "TurnComplete",
    "TurnContext",
    "UserQuestion",
    "VerificationResult",
    "VersionConflictEvent",
    # Work item tracking types
    "WorkItem",
    "WorkItemLedger",
    "WorkItemPriority",
    "WorkItemStatus",
    "WorkItemTaskList",
    "WorkItemTaskListProtocol",
    # Utilities
    "compute_content_hash",
    "create_rehydration_interceptor",
    "create_summary_cache_key",
    "estimate_tokens",
    "estimate_transcript_tokens",
    "format_work_item_reminder",
    "get_default_strategies",
    "get_task_complete_tool",
    "get_tokenizer",
    "render_externalization_text",
    "render_stream",
    "render_summary_text",
    "task_complete",
    "validate_artifact_content",
    "validate_control_artifact",
]
