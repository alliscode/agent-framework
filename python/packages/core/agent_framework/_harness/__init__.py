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

        # With production compaction
        harness = AgentHarness(
            agent,
            max_turns=50,
            enable_compaction=True,
            max_input_tokens=100000,
        )
"""

from ._agent_turn_executor import AgentTurnExecutor

# Production context compaction types (v2)
from ._compaction import (
    COMPACTION_PRECEDENCE,
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
    # Adapter types
    CacheMessageStore,
    CacheThreadAdapter,
    ChatClientSummarizer,
    ClearRecord,
    # Strategy types
    ClearStrategy,
    # Core types
    CompactionAction,
    # Event types
    CompactionCoordinator,
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
    Decision,
    DeterminismMetadata,
    DropRecord,
    DropStrategy,
    ExternalizationRecord,
    ExternalizeStrategy,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    # Tokenizer types
    ModelProvider,
    OpenItem,
    ProviderAwareTokenizer,
    SimpleTokenizer,
    SpanReference,
    StructuredSummary,
    SummarizationRecord,
    Summarizer,
    SummarizeStrategy,
    SummaryCache,
    SummaryCacheKey,
    TiktokenTokenizer,
    TokenBudget,
    ToolDurability,
    ToolDurabilityPolicy,
    ToolOutcome,
    ToolResultEnvelope,
    TurnContext,
    compute_content_hash,
    create_summary_cache_key,
    get_tokenizer,
)
from ._compaction_executor import CompactionComplete, CompactionExecutor
from ._compaction_owner import OwnerCompactionResult, OwnerFallbackReason
from ._constants import (
    DEFAULT_BLOCKING_THRESHOLD_PERCENT,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SOFT_THRESHOLD_PERCENT,
    DEFAULT_STOP_POLICY_PROFILE,
    DEFAULT_STALL_THRESHOLD,
    HARNESS_COMPACTION_METRICS_KEY,
    HARNESS_COMPACTION_PLAN_KEY,
    HARNESS_COMPLETION_REPORT_KEY,
    HARNESS_COVERAGE_LEDGER_KEY,
    HARNESS_MAX_TURNS_KEY,
    HARNESS_PENDING_TOOL_CALLS_KEY,
    HARNESS_PROGRESS_TRACKER_KEY,
    HARNESS_STATUS_KEY,
    HARNESS_STOP_REASON_KEY,
    HARNESS_STOP_POLICY_PROFILE_KEY,
    HARNESS_TASK_CONTRACT_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
    HARNESS_WORK_ITEM_LEDGER_KEY,
)
from ._context_pressure import (
    estimate_tokens,
    estimate_transcript_tokens,
)
from ._context_providers import EnvironmentContextProvider, HarnessGuidanceProvider
from ._contract_verifier import ContractVerificationResult, ContractVerifier, VerificationResult
from ._done_tool import (
    TASK_COMPLETE_TOOL_NAME,
    WORK_COMPLETE_TOOL_NAME,
    get_task_complete_tool,
    get_work_complete_tool,
    task_complete,
    work_complete,
)
from ._harness_builder import AgentHarness, HarnessWorkflowBuilder
from ._hooks import (
    AgentStopEvent,
    AgentStopHook,
    AgentStopResult,
    HarnessHooks,
    HarnessToolMiddleware,
    PostToolHook,
    PreToolHook,
    ToolHookResult,
)
from ._jit_instructions import (
    DEFAULT_JIT_INSTRUCTIONS,
    JitContext,
    JitInstruction,
    JitInstructionProcessor,
)
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
from ._state_store import HarnessStateStore
from ._stop_decision_executor import StopDecisionExecutor
from ._stop_policy import StopPolicyProfile
from ._sub_agents import create_document_tool, create_explore_tool, create_task_tool
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
from ._turn_buffer import ExecutorLocalTurnBuffer, SharedStateTurnBuffer, TurnBuffer
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
    "DEFAULT_BLOCKING_THRESHOLD_PERCENT",
    # Durability types
    "DEFAULT_DURABILITY_POLICIES",
    "DEFAULT_JIT_INSTRUCTIONS",
    # Constants
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SOFT_THRESHOLD_PERCENT",
    "DEFAULT_STOP_POLICY_PROFILE",
    "DEFAULT_STALL_THRESHOLD",
    "HARNESS_COMPACTION_METRICS_KEY",
    "HARNESS_COMPACTION_PLAN_KEY",
    "HARNESS_COMPLETION_REPORT_KEY",
    "HARNESS_COVERAGE_LEDGER_KEY",
    "HARNESS_MAX_TURNS_KEY",
    "HARNESS_PENDING_TOOL_CALLS_KEY",
    "HARNESS_PROGRESS_TRACKER_KEY",
    "HARNESS_STATUS_KEY",
    "HARNESS_STOP_REASON_KEY",
    "HARNESS_STOP_POLICY_PROFILE_KEY",
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
    "WORK_COMPLETE_TOOL_NAME",
    # Task contract types (Phase 3)
    "AcceptabilityCriteria",
    # Main API
    "AgentHarness",
    # Hooks system (Phase 3)
    "AgentStopEvent",
    "AgentStopHook",
    "AgentStopResult",
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
    "CacheMessageStore",
    "CacheThreadAdapter",
    "ChatClientSummarizer",
    "ClearRecord",
    # Strategy types
    "ClearStrategy",
    "CompactionAction",
    # Event types
    # Compaction executor
    "CompactionComplete",
    "CompactionCoordinator",
    "CompactionEvent",
    "CompactionEventEmitter",
    "CompactionEventType",
    "CompactionExecutor",
    "CompactionMetrics",
    "CompactionPlan",
    "CompactionProposal",
    "CompactionResult",
    "OwnerFallbackReason",
    "OwnerCompactionResult",
    # Storage and concurrency types
    "CompactionStore",
    "CompactionStrategy",
    "CompletionReport",
    "ContractVerificationResult",
    "ContractVerifier",
    "CoverageLedger",
    "Decision",
    "DeterminismMetadata",
    "DropRecord",
    "DropStrategy",
    "EnvironmentContextProvider",
    "Evidence",
    "ExternalizationRecord",
    "ExternalizeStrategy",
    "GapReport",
    "HarnessEvent",
    "HarnessGuidanceProvider",
    "HarnessHooks",
    "HarnessLifecycleEvent",
    "HarnessRenderer",
    "HarnessResult",
    "HarnessStateStore",
    "HarnessStatus",
    "HarnessToolMiddleware",
    "HarnessWorkflowBuilder",
    "InMemoryArtifactStore",
    "InMemoryCompactionStore",
    "InMemorySummaryCache",
    # JIT instructions (Phase 7)
    "JitContext",
    "JitInstruction",
    "JitInstructionProcessor",
    "MarkdownRenderer",
    # Tokenizer types (v2)
    "ModelProvider",
    "OpenItem",
    "PassthroughRenderer",
    "PendingToolCall",
    "PostToolHook",
    "PreToolHook",
    "Predicate",
    "PredicateType",
    "ProgressFingerprint",
    "ProgressTracker",
    "ProviderAwareTokenizer",
    "RepairComplete",
    "RepairExecutor",
    "RepairTrigger",
    "RequiredOutput",
    "RequirementCoverage",
    "RequirementStatus",
    "SimpleTokenizer",
    "SpanReference",
    "StopDecisionExecutor",
    "StopPolicyProfile",
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
    "ToolDurability",
    "ToolDurabilityPolicy",
    "ToolHookResult",
    "ToolOutcome",
    "ToolResultEnvelope",
    "TurnBuffer",
    "TurnComplete",
    "TurnContext",
    "ExecutorLocalTurnBuffer",
    "SharedStateTurnBuffer",
    "UserQuestion",
    "VerificationResult",
    # Work item tracking types
    "WorkItem",
    "WorkItemLedger",
    "WorkItemPriority",
    "WorkItemStatus",
    "WorkItemTaskList",
    "WorkItemTaskListProtocol",
    # Utilities
    "compute_content_hash",
    "create_document_tool",
    "create_explore_tool",
    "create_summary_cache_key",
    "create_task_tool",
    "estimate_tokens",
    "estimate_transcript_tokens",
    "format_work_item_reminder",
    "get_task_complete_tool",
    "get_tokenizer",
    "get_work_complete_tool",
    "render_stream",
    "task_complete",
    "validate_artifact_content",
    "validate_control_artifact",
    "work_complete",
]
