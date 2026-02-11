# Copyright (c) Microsoft. All rights reserved.

"""Context compaction for Agent Harness.

This package provides production-quality context compaction based on an
immutable-log + compaction-plan + renderer architecture.

Key principles:
1. AgentThread is never mutated - it's the append-only source of truth
2. CompactionPlan is pure data describing what spans are compacted

## Core Types

- `SpanReference`: Self-sufficient reference to a span of messages
- `CompactionPlan`: Plan describing how to compact a thread
- `StructuredSummary`: Drift-resistant structured summary
- `ToolDurability`: Classification for tool result durability
- `ToolResultEnvelope`: Structured envelope for tool results
- `TurnContext`: Turn-level state for rehydration loop breaking

## Strategy Ladder

Strategies are applied in order from least to most aggressive:
1. Clear - Replace tool results with placeholders (respecting durability)
2. Summarize - LLM-compress older spans into structured summaries
3. Externalize - Write large content to storage, keep pointer + summary
4. Drop - Remove from prompt entirely (last resort)

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

# Adapter types
from ._adapters import (
    CacheMessageStore,
    CacheThreadAdapter,
)

# Durability types
from ._durability import (
    DEFAULT_DURABILITY_POLICIES,
    DeterminismMetadata,
    ToolDurability,
    ToolDurabilityPolicy,
    ToolResultEnvelope,
)

# Event types
from ._events import (
    CompactionEvent,
    CompactionEventEmitter,
    CompactionEventType,
    CompactionMetrics,
)

# Storage types
from ._store import (
    ArtifactMetadata,
    ArtifactStore,
    ArtifactStoreEntry,
    CacheEntry,
    CompactionStore,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    SummaryCache,
    compute_content_hash,
    create_summary_cache_key,
)

# Strategy types
from ._strategies import (
    ClearStrategy,
    CompactionCoordinator,
    CompactionProposal,
    CompactionResult,
    CompactionStrategy,
    DropStrategy,
    ExternalizeStrategy,
    Summarizer,
    SummarizeStrategy,
)

# Summarizer implementation
from ._summarizer import ChatClientSummarizer

# Summary types
from ._summary import (
    STRUCTURED_SUMMARY_SCHEMA_VERSION,
    SUMMARY_RENDER_VERSION,
    ArtifactReference,
    Decision,
    OpenItem,
    StructuredSummary,
    SummaryCacheKey,
    ToolOutcome,
)

# Tokenizer types
from ._tokenizer import (
    DEFAULT_ENCODING,
    MODEL_ENCODING_MAP,
    OPENAI_MESSAGE_OVERHEAD,
    OPENAI_REPLY_OVERHEAD,
    TIKTOKEN_AVAILABLE,
    ModelProvider,
    ProviderAwareTokenizer,
    SimpleTokenizer,
    TiktokenTokenizer,
    TokenBudget,
    get_tokenizer,
)

# Turn context types
from ._turn_context import (
    TurnContext,
)
from ._types import (
    COMPACTION_PRECEDENCE,
    ClearRecord,
    CompactionAction,
    CompactionPlan,
    DropRecord,
    ExternalizationRecord,
    SpanReference,
    SummarizationRecord,
)

__all__ = [
    # Constants
    "COMPACTION_PRECEDENCE",
    "DEFAULT_DURABILITY_POLICIES",
    "DEFAULT_ENCODING",
    "MODEL_ENCODING_MAP",
    "OPENAI_MESSAGE_OVERHEAD",
    "OPENAI_REPLY_OVERHEAD",
    "STRUCTURED_SUMMARY_SCHEMA_VERSION",
    "SUMMARY_RENDER_VERSION",
    "TIKTOKEN_AVAILABLE",
    # Core types
    "ArtifactMetadata",
    "ArtifactReference",
    "ArtifactStore",
    "ArtifactStoreEntry",
    "CacheEntry",
    # Adapter types
    "CacheMessageStore",
    "CacheThreadAdapter",
    "ChatClientSummarizer",
    "ClearRecord",
    "ClearStrategy",
    "CompactionAction",
    "CompactionCoordinator",
    "CompactionEvent",
    "CompactionEventEmitter",
    "CompactionEventType",
    "CompactionMetrics",
    "CompactionPlan",
    "CompactionProposal",
    "CompactionResult",
    "CompactionStore",
    "CompactionStrategy",
    "Decision",
    "DeterminismMetadata",
    "DropRecord",
    "DropStrategy",
    "ExternalizationRecord",
    "ExternalizeStrategy",
    "InMemoryArtifactStore",
    "InMemoryCompactionStore",
    "InMemorySummaryCache",
    "ModelProvider",
    "OpenItem",
    "ProviderAwareTokenizer",
    "SimpleTokenizer",
    "SpanReference",
    "StructuredSummary",
    "SummarizationRecord",
    "SummarizeStrategy",
    "Summarizer",
    "SummaryCache",
    "SummaryCacheKey",
    "TiktokenTokenizer",
    "TokenBudget",
    "ToolDurability",
    "ToolDurabilityPolicy",
    "ToolOutcome",
    "ToolResultEnvelope",
    "TurnContext",
    # Functions
    "compute_content_hash",
    "create_summary_cache_key",
    "get_tokenizer",
]
