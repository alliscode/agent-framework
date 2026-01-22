# Context Compaction: Production Design (v2)

## Overview

Context compaction manages the context window when conversations approach token limits.
The key architectural principle is **immutability**: the canonical conversation log is never
mutated. Instead, compaction produces a **plan** that a **renderer** applies to generate
the actual prompt sent to the model.

## Core Architectural Principles

### 1. Immutable Canonical Log
`AgentThread` is the append-only source of truth. We never delete or modify messages.
This preserves:
- Auditability ("what did the user actually say?")
- Reproducibility (can replay any run)
- Debugging ("why did it do that?")
- Concurrent safety (no identity-based bugs)

### 2. Compaction Plan as Pure Data
Compaction produces a `CompactionPlan` - a declarative description of what spans
are summarized, externalized, cleared, or dropped. The plan references message IDs,
not message content.

### 3. Prompt Renderer
A `PromptRenderer` takes `AgentThread + CompactionPlan` and produces the actual
request payload sent to the model. This separation enables:
- Testing ("golden prompt" snapshots)
- Policy iteration without data loss
- Multiple rendering strategies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                         │
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐                │
│  │ AgentThread │───▶│ CompactionPlan  │───▶│ PromptRenderer│───▶ Model     │
│  │  (immutable)│    │  (pure data)    │    │              │                │
│  └─────────────┘    └─────────────────┘    └──────────────┘                │
│         │                   │                                               │
│         │                   │                                               │
│         ▼                   ▼                                               │
│  ┌─────────────┐    ┌─────────────────┐                                    │
│  │ Audit Log   │    │ CompactionStore │                                    │
│  │ (full hist) │    │ (plans, summaries)│                                  │
│  └─────────────┘    └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Strategy Ladder

Strategies are applied in order from least to most aggressive:

1. **Externalize** - Write large content to storage, keep pointer + summary
2. **Summarize** - LLM-compress older spans into structured summaries
3. **Clear** - Replace tool results with minimal placeholders (respecting durability rules)
4. **Drop** - Remove from prompt view entirely (last resort)

> **Future: Index + Recall**
> When vector store integration is added, we'll support:
> - **Index**: Embed spans as they arrive or when compacted (background)
> - **Recall**: Inject relevant chunks into prompt based on current context
> This is *not* a ladder rung but a separate memory substrate that works alongside compaction.

## Core Data Structures

### CompactionPlan

```python
@dataclass
class SpanReference:
    """Reference to a contiguous span of messages.

    IMPORTANT: SpanReference is self-sufficient. It stores the explicit list
    of message IDs, not just start/end. This ensures:
    - Plans are truly "pure data" resolvable without thread access
    - No ambiguity if messages are appended later
    - No off-by-one bugs from start/end resolution

    Invariants:
    - message_ids is non-empty
    - message_ids preserves the original thread ordering
    - All IDs must exist in the thread at plan creation time
    """
    message_ids: list[str]  # Explicit list, not start/end
    first_turn: int  # Turn number of first message (for display)
    last_turn: int  # Turn number of last message (for display)

    @property
    def message_count(self) -> int:
        return len(self.message_ids)

    @property
    def start_message_id(self) -> str:
        return self.message_ids[0]

    @property
    def end_message_id(self) -> str:
        return self.message_ids[-1]

    def contains(self, message_id: str) -> bool:
        """Check if span contains a message ID. O(n) but spans are small."""
        return message_id in self.message_ids

@dataclass
class ExternalizationRecord:
    """Record of externalized content."""
    span: SpanReference
    artifact_id: str
    summary: "StructuredSummary"
    rehydrate_hint: str  # When agent should read this back

@dataclass
class SummarizationRecord:
    """Record of summarized content."""
    span: SpanReference
    summary: "StructuredSummary"
    summary_token_count: int

@dataclass
class ClearRecord:
    """Record of cleared content."""
    span: SpanReference
    preserved_fields: dict[str, Any]  # Tool name, outcome, key IDs

@dataclass
class DropRecord:
    """Record of dropped content."""
    span: SpanReference
    reason: str

class CompactionAction(Enum):
    """Action to take for a message during rendering."""
    INCLUDE = "include"        # Include as-is
    CLEAR = "clear"            # Replace with placeholder
    SUMMARIZE = "summarize"    # Replace with summary
    EXTERNALIZE = "externalize"  # Replace with pointer + summary
    DROP = "drop"              # Omit entirely

# Precedence order (higher = takes precedence)
COMPACTION_PRECEDENCE = {
    CompactionAction.DROP: 4,        # Highest - overrides everything
    CompactionAction.EXTERNALIZE: 3,
    CompactionAction.SUMMARIZE: 2,
    CompactionAction.CLEAR: 1,
    CompactionAction.INCLUDE: 0,     # Lowest - default
}

@dataclass
class CompactionPlan:
    """Complete compaction plan for a thread.

    IMPORTANT: Plans are normalized. A message ID appears in at most one
    record list. Precedence (Drop > Externalize > Summarize > Clear > Include)
    is enforced at plan-build time.
    """
    thread_id: str
    thread_version: int  # For optimistic concurrency
    created_at: datetime

    # What's been compacted (mutually exclusive by message_id)
    externalizations: list[ExternalizationRecord]
    summarizations: list[SummarizationRecord]
    clearings: list[ClearRecord]
    drops: list[DropRecord]

    # Budget tracking
    original_token_count: int
    compacted_token_count: int

    # Normalized action map (computed from records)
    _action_map: dict[str, tuple[CompactionAction, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Build normalized action map from records."""
        self._action_map = self._build_action_map()

    def _build_action_map(self) -> dict[str, tuple[CompactionAction, Any]]:
        """Build message_id -> (action, record) map with precedence.

        Also validates that no message ID appears in multiple record lists
        (except where precedence resolves the conflict).
        """
        action_map: dict[str, tuple[CompactionAction, Any]] = {}
        seen_ids: dict[str, tuple[CompactionAction, str]] = {}  # msg_id -> (action, record_type)

        def process_records(
            records: list,
            action: CompactionAction,
            record_type: str,
        ) -> list[str]:
            """Process records, track overlaps, return warnings."""
            warnings = []
            for record in records:
                for msg_id in record.span.message_ids:
                    if msg_id in seen_ids:
                        prev_action, prev_type = seen_ids[msg_id]
                        if COMPACTION_PRECEDENCE[action] > COMPACTION_PRECEDENCE[prev_action]:
                            warnings.append(
                                f"Message {msg_id} in both {prev_type} and {record_type}; "
                                f"{record_type} takes precedence"
                            )
                        else:
                            # Lower precedence, skip
                            continue
                    action_map[msg_id] = (action, record)
                    seen_ids[msg_id] = (action, record_type)
            return warnings

        # Process in precedence order (lowest first, higher overwrites)
        all_warnings = []
        all_warnings.extend(process_records(self.clearings, CompactionAction.CLEAR, "clearings"))
        all_warnings.extend(process_records(self.summarizations, CompactionAction.SUMMARIZE, "summarizations"))
        all_warnings.extend(process_records(self.externalizations, CompactionAction.EXTERNALIZE, "externalizations"))
        all_warnings.extend(process_records(self.drops, CompactionAction.DROP, "drops"))

        # Store warnings for observability
        self._normalization_warnings = all_warnings

        return action_map

    def get_action(self, message_id: str) -> tuple[CompactionAction, Any | None]:
        """Get the action for a message ID.

        Returns:
            (action, record) tuple. Record is None for INCLUDE.
        """
        return self._action_map.get(message_id, (CompactionAction.INCLUDE, None))

    @property
    def has_overlaps(self) -> bool:
        """True if plan had overlapping spans that were normalized."""
        return bool(getattr(self, '_normalization_warnings', []))

    @property
    def normalization_warnings(self) -> list[str]:
        """Warnings from plan normalization (overlapping spans)."""
        return getattr(self, '_normalization_warnings', [])
```

### StructuredSummary

Plain-text summaries drift over time. After 3-5 summarize cycles you get missing
constraints, lost decisions, merged entities. Structured summaries prevent this.

```python
@dataclass
class Decision:
    """A decision made during the conversation."""
    decision: str
    rationale: str
    turn_number: int
    timestamp: datetime

@dataclass
class OpenItem:
    """An unresolved item or TODO."""
    description: str
    context: str
    priority: Literal["high", "medium", "low"]

@dataclass
class ArtifactReference:
    """Reference to an externalized artifact."""
    artifact_id: str
    description: str
    rehydrate_hint: str  # "read if you need file contents"

@dataclass
class ToolOutcome:
    """Summary of a tool result."""
    tool_name: str
    outcome: Literal["success", "failure", "partial"]
    key_fields: dict[str, Any]  # IDs, important values
    error_message: str | None

@dataclass
class StructuredSummary:
    """Structured summary that resists drift."""
    span: SpanReference

    # Core sections
    facts: list[str]  # Stable info, user prefs, constraints
    decisions: list[Decision]
    open_items: list[OpenItem]
    artifacts: list[ArtifactReference]
    tool_outcomes: list[ToolOutcome]

    # Current state
    current_task: str | None
    current_plan: list[str] | None

    # Rendering
    def render_as_message(self) -> str:
        """Render as a message for inclusion in prompt."""
        ...
```

## Tool Result Durability Rules

Not all tool results are safe to clear. We classify by durability.

### ToolResultEnvelope (Required for Durability)

For durability rules to work, tool results must be structured, not opaque strings.
Every tool result in the thread is wrapped in an envelope:

```python
@dataclass
class ToolResultEnvelope:
    """Structured envelope for tool results in the thread.

    This is the canonical representation of a tool result. The envelope
    provides consistent structure for compaction to work with.
    """
    tool_name: str
    tool_call_id: str
    outcome: Literal["success", "failure", "partial"]

    # Content can be string or structured
    content: str | dict[str, Any]

    # Key fields that MUST survive compaction (IDs, decisions, errors)
    key_fields: dict[str, Any] = field(default_factory=dict)

    # Durability classification
    durability: "ToolDurability" = field(default=None)  # Set by tool or policy

    # For REPLAYABLE tools: metadata to detect if replay is still valid
    determinism: "DeterminismMetadata | None" = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def get_token_count(self, tokenizer: "ProviderAwareTokenizer") -> int:
        """Get token count of content."""
        if isinstance(self.content, str):
            return tokenizer.count_tokens(self.content)
        return tokenizer.count_tokens(json.dumps(self.content))

    def to_cleared_placeholder(self) -> str:
        """Render as minimal placeholder preserving key_fields."""
        parts = [f"[{self.tool_name}: {self.outcome}]"]
        if self.key_fields:
            parts.append(f"Key data: {json.dumps(self.key_fields)}")
        return "\n".join(parts)
```

**Why envelopes matter**: Without structured envelopes, `ClearRecord.preserved_fields`
has nothing to work with. The envelope guarantees we can always extract:
- Tool name and outcome
- Key fields (IDs, important values)
- Determinism metadata for REPLAYABLE tools

```python
class ToolDurability(Enum):
    """Durability classification for tool results."""

    EPHEMERAL = "ephemeral"
    # Safe to clear entirely (verbose logs, long file listings)
    # Example: directory listing, debug output

    ANCHORING = "anchoring"
    # Must keep summary + key fields (IDs, decisions)
    # Example: API response with order ID, database query results

    REPLAYABLE = "replayable"
    # Safe to clear because tool can be re-run cheaply/deterministically
    # BUT: requires determinism_metadata to verify result is still valid
    # Example: read_file (file hasn't changed), math calculation

    NON_REPLAYABLE = "non_replayable"
    # Must externalize or keep (rate-limited, paid, time-sensitive)
    # Example: web search results, paid API calls, stock prices

@dataclass
class DeterminismMetadata:
    """Metadata to verify if a REPLAYABLE tool result is still valid.

    REPLAYABLE tools must capture this at call time. Before clearing a
    REPLAYABLE result, we check if the source has changed. If drift is
    detected, the result becomes NON_REPLAYABLE for that span.
    """
    content_hash: str | None = None  # Hash of file/data at read time
    etag: str | None = None  # HTTP ETag for remote resources
    mtime: float | None = None  # Modification time (Unix timestamp)
    version: str | None = None  # API version or schema version

    def has_changed(self, current: "DeterminismMetadata") -> bool:
        """Check if source has drifted since this metadata was captured."""
        if self.content_hash and current.content_hash:
            return self.content_hash != current.content_hash
        if self.etag and current.etag:
            return self.etag != current.etag
        if self.mtime and current.mtime:
            return self.mtime != current.mtime
        # Can't verify - assume changed (fail safe)
        return True

@dataclass
class ToolDurabilityPolicy:
    """Policy for how to handle tool results during compaction."""

    durability: ToolDurability
    must_preserve_fields: list[str]  # Field names that must be kept
    externalize_threshold_tokens: int  # When to externalize vs summarize
    replay_cost: Literal["free", "cheap", "expensive"]

    # For REPLAYABLE tools: function to capture current determinism metadata
    capture_determinism: Callable[..., DeterminismMetadata] | None = None
```

Tools should declare their durability:

```python
@ai_function(
    durability=ToolDurability.ANCHORING,
    preserve_fields=["order_id", "status", "total"],
)
def create_order(...) -> str:
    ...

# REPLAYABLE tools should capture determinism metadata
@ai_function(
    durability=ToolDurability.REPLAYABLE,
)
def read_file(path: str) -> str:
    """Read file, capturing mtime for drift detection."""
    content = Path(path).read_text()
    # Framework captures determinism metadata automatically
    return ToolResult(
        content=content,
        determinism=DeterminismMetadata(
            mtime=Path(path).stat().st_mtime,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        )
    )
```

## Rehydration Contract

When content is externalized, the agent needs a reliable way to access it.
Pure "explicit tool call" is unreliable - models hallucinate or forget.

### Hybrid Rehydration Strategy

1. **Primary path**: Agent calls `read_artifact(artifact_id)` explicitly
2. **Safety net**: Automatic rehydration when:
   - Agent's message references an artifact ID
   - A tool schema requires data from an artifact
   - Agent asks a question that matches an artifact's rehydrate_hint

```python
@dataclass
class RehydrationConfig:
    """Configuration for automatic rehydration."""
    enabled: bool = True
    max_artifacts_per_turn: int = 3  # Prevent runaway injection
    max_tokens_per_artifact: int = 4000  # Truncate large artifacts
    total_budget_tokens: int = 8000  # Total rehydration budget
    cooldown_turns: int = 2  # Don't re-inject same artifact for N turns

    # Sensitivity gating: which sensitivity levels can be auto-rehydrated
    # Higher sensitivity requires explicit tool call (read_artifact)
    auto_rehydrate_sensitivities: set[str] = field(
        default_factory=lambda: {"public", "internal"}
    )
    # confidential/restricted require explicit tool call by default

class RehydrationInterceptor:
    """Intercepts agent requests and auto-rehydrates when needed.

    Security considerations:
    - Uses artifact_store access control (requester must have permission)
    - Limits total injected tokens to prevent context flooding
    - Tracks recently-injected artifacts to prevent loops
    - Validates artifact IDs match expected format
    - Sensitivity gating: only auto-rehydrates public/internal by default
      (confidential/restricted require explicit read_artifact call)
    """

    def __init__(
        self,
        config: RehydrationConfig,
        tokenizer: ProviderAwareTokenizer,
    ):
        self._config = config
        self._tokenizer = tokenizer
        self._recent_injections: dict[str, int] = {}  # artifact_id -> turn injected

    async def maybe_rehydrate(
        self,
        agent_message: str,
        pending_tool_calls: list[ToolCall],
        compaction_plan: CompactionPlan,
        artifact_store: ArtifactStore,
        current_turn: int,
        security_context: SecurityContext | None = None,
    ) -> list[RehydrationResult]:
        """Check if rehydration is needed and return content to inject.

        Returns:
            List of RehydrationResult with content and metadata.

        Raises:
            RehydrationBudgetExceeded: If injection would exceed token budget.
        """
        if not self._config.enabled:
            return []

        needed_artifacts: list[str] = []

        # Check for artifact ID references in message (validate format)
        for ext in compaction_plan.externalizations:
            if self._is_valid_artifact_ref(ext.artifact_id, agent_message):
                if not self._in_cooldown(ext.artifact_id, current_turn):
                    needed_artifacts.append(ext.artifact_id)

        # Check tool call arguments for artifact references
        for tool_call in pending_tool_calls:
            for arg_value in tool_call.arguments.values():
                if isinstance(arg_value, str):
                    for ext in compaction_plan.externalizations:
                        if ext.artifact_id in arg_value:
                            if not self._in_cooldown(ext.artifact_id, current_turn):
                                needed_artifacts.append(ext.artifact_id)

        # Deduplicate and limit
        needed_artifacts = list(dict.fromkeys(needed_artifacts))  # Preserve order
        needed_artifacts = needed_artifacts[:self._config.max_artifacts_per_turn]

        # Retrieve with budget enforcement and sensitivity gating
        results: list[RehydrationResult] = []
        skipped_sensitive: list[str] = []
        tokens_used = 0

        for artifact_id in needed_artifacts:
            # Check sensitivity before retrieving content
            metadata = await artifact_store.get_metadata(artifact_id)
            if metadata is None:
                continue  # Not found

            # Sensitivity gating: block auto-rehydration of sensitive artifacts
            if metadata.sensitivity not in self._config.auto_rehydrate_sensitivities:
                skipped_sensitive.append(artifact_id)
                # Emit event for observability
                await self._emit_event(
                    "rehydration_blocked_sensitivity",
                    artifact_id=artifact_id,
                    sensitivity=metadata.sensitivity,
                    reason="Auto-rehydration blocked; use read_artifact() explicitly",
                )
                continue

            content = await artifact_store.retrieve(
                artifact_id,
                requester_context=security_context,
            )
            if content is None:
                continue  # Access denied

            # Truncate if needed
            token_count = self._tokenizer.count_tokens(content)
            if token_count > self._config.max_tokens_per_artifact:
                content = self._truncate_content(content, self._config.max_tokens_per_artifact)
                token_count = self._config.max_tokens_per_artifact

            # Check total budget
            if tokens_used + token_count > self._config.total_budget_tokens:
                break  # Stop, don't exceed budget

            results.append(RehydrationResult(
                artifact_id=artifact_id,
                content=content,
                token_count=token_count,
                truncated=token_count >= self._config.max_tokens_per_artifact,
            ))
            tokens_used += token_count
            self._recent_injections[artifact_id] = current_turn

        return results

    def _in_cooldown(self, artifact_id: str, current_turn: int) -> bool:
        """Check if artifact was recently injected."""
        last_turn = self._recent_injections.get(artifact_id)
        if last_turn is None:
            return False
        return current_turn - last_turn < self._config.cooldown_turns

    def _is_valid_artifact_ref(self, artifact_id: str, text: str) -> bool:
        """Validate artifact reference to prevent injection attacks."""
        # Must be exact match, not substring of larger token
        import re
        pattern = rf'\b{re.escape(artifact_id)}\b'
        return bool(re.search(pattern, text))

@dataclass
class RehydrationResult:
    """Result of rehydrating an artifact."""
    artifact_id: str
    content: str
    token_count: int
    truncated: bool
```

### Rehydration Loop Breaking (Critical)

A common failure mode: agent references artifact → rehydration injects content →
token pressure increases → compaction externalizes more → agent references again → loop.

**Rules to prevent oscillation:**

1. **Rehydrated content is ephemeral**: It is NOT appended to the canonical log.
   It's injected at render time only for the current turn.

2. **Rehydration uses reserved budget**: Rehydrated tokens consume from
   `rehydration_reserve_tokens`, which is **excluded** from the compaction
   pressure calculation for that turn.

3. **Turn-level flag**: The harness tracks `rehydration_happened_this_turn`.
   If true, the compaction executor skips aggressive strategies (externalize,
   drop) for that turn to let the agent complete its work.

4. **Cooldown applies**: Same artifact won't be re-injected for N turns
   (default 2), giving compaction time to stabilize.

```python
@dataclass
class TurnContext:
    """Context passed through a single turn execution."""
    turn_number: int
    rehydration_happened: bool = False
    rehydrated_tokens: int = 0
    rehydrated_artifact_ids: list[str] = field(default_factory=list)

    def should_skip_aggressive_compaction(self) -> bool:
        """True if we should avoid externalize/drop this turn."""
        return self.rehydration_happened
```

**Why ephemeral matters**: If rehydrated content became part of the log,
it would be compacted again, creating a cycle. Ephemeral injection means
the content exists only in the rendered prompt, not the source of truth.

## Provider-Aware Token Budgeting

Token counting must include everything sent to the model:

```python
@dataclass
class TokenBudget:
    """Token budget that accounts for all overhead."""

    max_input_tokens: int
    soft_threshold_percent: float = 0.85

    # Overhead tracking
    system_prompt_tokens: int = 0
    tool_schema_tokens: int = 0
    formatting_overhead_tokens: int = 0  # Provider-specific wrapping
    safety_buffer_tokens: int = 500  # For response + safety margin

    # Reservation for rehydration (auto-injected artifact content)
    rehydration_reserve_tokens: int = 2000  # Space for ~1-2 artifacts
    max_rehydration_artifacts: int = 3  # Limit concurrent rehydrations

    @property
    def available_for_messages(self) -> int:
        """Tokens available for conversation messages."""
        overhead = (
            self.system_prompt_tokens +
            self.tool_schema_tokens +
            self.formatting_overhead_tokens +
            self.safety_buffer_tokens +
            self.rehydration_reserve_tokens  # Reserve space for rehydration
        )
        return int(self.max_input_tokens * self.soft_threshold_percent) - overhead

    @property
    def available_for_rehydration(self) -> int:
        """Tokens available for auto-rehydrated content."""
        return self.rehydration_reserve_tokens

class ProviderAwareTokenizer(Protocol):
    """Tokenizer that understands provider-specific formatting."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in raw text."""
        ...

    def count_message(self, message: ChatMessage) -> int:
        """Count tokens for a message including role overhead."""
        ...

    def count_tool_schemas(self, tools: list[Tool]) -> int:
        """Count tokens for tool schema injection."""
        ...

    def count_request(self, request: ChatRequest) -> int:
        """Count total tokens for a complete request."""
        ...
```

## Concurrency and Idempotency

Compaction must be safe under concurrent access:

```python
@dataclass
class CompactionTransaction:
    """Transactional compaction with optimistic concurrency."""

    thread_id: str
    expected_version: int
    plan: CompactionPlan

    async def commit(self, store: CompactionStore) -> bool:
        """Attempt to commit the compaction plan.

        Returns:
            True if committed, False if version conflict.
        """
        ...

class CompactionStore(Protocol):
    """Storage for compaction plans with concurrency control."""

    async def get_current_plan(self, thread_id: str) -> tuple[CompactionPlan | None, int]:
        """Get current plan and version number."""
        ...

    async def commit_plan(
        self,
        thread_id: str,
        plan: CompactionPlan,
        expected_version: int,
    ) -> bool:
        """Commit plan if version matches. Returns False on conflict."""
        ...
```

**Idempotency**: Same input span + policy → same compaction result.

### Summary Caching

Summaries are expensive (LLM calls) and must be cached. The cache key must include
all factors that affect the summary:

```python
@dataclass
class SummaryCacheKey:
    """Key for cached summaries."""
    content_hash: str  # SHA256 of span content
    schema_version: str  # StructuredSummary schema version (e.g., "v1.2")
    policy_version: str  # Compaction policy version
    model_id: str  # Summarization model ID
    prompt_version: str  # Summarization prompt hash

    def to_string(self) -> str:
        return f"{self.content_hash}:{self.schema_version}:{self.policy_version}:{self.model_id}:{self.prompt_version}"

# Schema versioning is critical to avoid cache poisoning
STRUCTURED_SUMMARY_SCHEMA_VERSION = "v1.0"  # Bump when StructuredSummary changes
```

**Why schema versioning matters**: If we add a field to `StructuredSummary` (e.g., `confidence_scores`),
cached summaries won't have it. The cache key must include schema version to invalidate stale summaries.

## Prompt Renderer

The renderer applies a compaction plan to produce the actual prompt.

### Replacement Semantics (Critical)

When a span is summarized or externalized, the renderer must decide:
- What **role** does the synthetic message have?
- Where in the **sequence** does it appear?
- What **format** is used?

**Rules:**

1. **Role**: Summaries and externalization pointers render as `role="assistant"`.
   - Rationale: Using `system` is dangerous (models treat it as privileged).
   - Using a custom role breaks provider compatibility.
   - `assistant` signals "this is context I provided earlier."

2. **Position**: The synthetic message is inserted at the position of the
   **first message in the span**. All other messages in the span are omitted.

3. **Format**: Stable, versioned prefix for reproducibility:
   ```
   [Context Summary - Turns {first_turn}-{last_turn}]
   {structured_summary.render_as_message()}
   ```

4. **One message per span**: Never merge adjacent summaries. Each span produces
   exactly one synthetic message.

5. **Externalization format**:
   ```
   [Externalized Content - artifact:{artifact_id}]
   Summary: {summary.render_as_message()}
   To retrieve full content, call: read_artifact("{artifact_id}")
   ```

```python
# Rendering format version - bump when format changes
COMPACTION_RENDER_FORMAT_VERSION = "v1.0"

class PromptRenderer:
    """Renders AgentThread + CompactionPlan into model request."""

    def __init__(
        self,
        tokenizer: ProviderAwareTokenizer,
        artifact_store: ArtifactStore,
    ):
        ...

    async def render(
        self,
        thread: AgentThread,
        plan: CompactionPlan,
        system_prompt: str,
        tools: list[Tool],
        rehydrated_content: list[RehydrationResult] | None = None,
    ) -> ChatRequest:
        """Render thread with compaction applied.

        For each message in thread:
        - If in summarization span → inject summary at first message position
        - If in externalization span → inject pointer + summary at first message position
        - If in clear span → inject placeholder with preserved fields
        - If in drop span → skip entirely
        - Otherwise → include as-is

        Args:
            rehydrated_content: Ephemeral content to inject (not part of thread).
        """
        messages = []
        processed_spans: set[str] = set()  # Track which spans we've rendered

        for msg in thread.messages:
            action, record = plan.get_action(msg.id)

            if action == CompactionAction.DROP:
                continue

            if action == CompactionAction.INCLUDE:
                messages.append(msg)
                continue

            if action == CompactionAction.CLEAR:
                messages.append(self._render_cleared(msg, record))
                continue

            # For SUMMARIZE and EXTERNALIZE, only render once per span
            span_key = record.span.start_message_id
            if span_key in processed_spans:
                continue  # Already rendered this span's summary
            processed_spans.add(span_key)

            if action == CompactionAction.SUMMARIZE:
                messages.append(self._render_summary(record))
            elif action == CompactionAction.EXTERNALIZE:
                messages.append(self._render_externalization(record))

        # Inject rehydrated content (ephemeral, at end before current turn)
        if rehydrated_content:
            for result in rehydrated_content:
                messages.append(self._render_rehydration(result))

        return self._build_request(messages, system_prompt, tools)

    def _render_summary(self, record: SummarizationRecord) -> ChatMessage:
        """Render a summary as an assistant message."""
        return ChatMessage(
            role="assistant",
            content=(
                f"[Context Summary - Turns {record.span.first_turn}-{record.span.last_turn}]\n"
                f"{record.summary.render_as_message()}"
            ),
        )

    def _render_externalization(self, record: ExternalizationRecord) -> ChatMessage:
        """Render an externalization pointer as an assistant message."""
        return ChatMessage(
            role="assistant",
            content=(
                f"[Externalized Content - artifact:{record.artifact_id}]\n"
                f"Summary: {record.summary.render_as_message()}\n"
                f'To retrieve full content, call: read_artifact("{record.artifact_id}")'
            ),
        )
```

## Updated Strategy Interfaces

```python
class CompactionStrategy(Protocol):
    """Protocol for compaction strategies."""

    @property
    def name(self) -> str:
        """Strategy name for logging/observability."""
        ...

    @property
    def aggressiveness(self) -> int:
        """Lower = less aggressive, applied first."""
        ...

    async def analyze(
        self,
        thread: AgentThread,
        current_plan: CompactionPlan | None,
        budget: TokenBudget,
        tokenizer: ProviderAwareTokenizer,
    ) -> list[CompactionProposal]:
        """Analyze thread and propose compactions.

        Returns:
            List of proposed compactions (spans to compact).
        """
        ...

    async def execute(
        self,
        proposal: CompactionProposal,
        thread: AgentThread,
        summarizer: Summarizer,
        artifact_store: ArtifactStore | None,
    ) -> CompactionRecord:
        """Execute a proposed compaction.

        Returns:
            Record of what was compacted (for inclusion in plan).
        """
        ...

@dataclass
class CompactionProposal:
    """A proposed compaction action."""
    strategy: str
    span: SpanReference
    estimated_tokens_freed: int
    reason: str
```

## Lifecycle Events for Observability

```python
class ContextCompactionEvent(HarnessLifecycleEvent):
    """Events emitted during context compaction."""

    event_type: Literal[
        "compaction_check_started",
        "compaction_proposal_generated",
        "compaction_externalized",
        "compaction_summarized",
        "compaction_cleared",
        "compaction_retrieved",  # Moved to vector store
        "compaction_dropped",
        "compaction_rehydrated",  # Auto-rehydration triggered
        "compaction_completed",
    ]

    # Budget info
    tokens_before: int
    tokens_after: int
    budget_limit: int

    # What happened
    strategy: str | None
    span: SpanReference | None
    tokens_freed: int

    # Details
    details: dict[str, Any]
```

## Configuration

```python
@dataclass
class ContextCompactionConfig:
    """Configuration for context compaction."""

    # Token budget
    max_input_tokens: int = 100_000
    soft_threshold_percent: float = 0.85

    # Components (all have sensible defaults)
    tokenizer: ProviderAwareTokenizer | None = None
    summarizer: Summarizer | None = None
    artifact_store: ArtifactStore | None = None
    vector_store: VectorStore | None = None  # For retrieve strategy
    compaction_store: CompactionStore | None = None

    # Strategy enablement
    enable_externalization: bool = True
    enable_summarization: bool = True
    enable_clearing: bool = True
    enable_retrieval: bool = False  # Requires vector_store
    enable_dropping: bool = True

    # Thresholds
    externalize_threshold_tokens: int = 1000
    summarize_target_ratio: float = 0.25
    preserve_recent_turns: int = 3

    # Summarization
    summarization_model: str | None = None  # Default: cheaper model
    summarization_cache_enabled: bool = True

    # Rehydration (auto-injection of externalized content)
    rehydration_enabled: bool = True
    rehydration_max_artifacts_per_turn: int = 3
    rehydration_max_tokens_per_artifact: int = 4000
    rehydration_total_budget_tokens: int = 8000
    rehydration_cooldown_turns: int = 2

    # Tool durability (can override per-tool settings)
    default_tool_durability: ToolDurability = ToolDurability.ANCHORING
    tool_durability_overrides: dict[str, ToolDurabilityPolicy] = field(default_factory=dict)
```

## Security Considerations for ArtifactStore

Externalization is a data exfil risk. The protocol should support:

```python
class ArtifactStore(Protocol):
    """Protocol for storing externalized content."""

    async def store(
        self,
        content: str,
        metadata: ArtifactMetadata,
    ) -> str:
        """Store content securely."""
        ...

    async def get_metadata(
        self,
        artifact_id: str,
    ) -> ArtifactMetadata | None:
        """Get metadata without retrieving content.

        Used for sensitivity gating in auto-rehydration.
        """
        ...

    async def retrieve(
        self,
        artifact_id: str,
        requester_context: SecurityContext | None = None,
    ) -> str | None:
        """Retrieve with access control."""
        ...

@dataclass
class ArtifactMetadata:
    """Metadata for stored artifacts."""
    thread_id: str
    tenant_id: str | None  # For multi-tenant isolation
    created_at: datetime
    ttl_seconds: int | None  # Auto-expire
    encryption_key_id: str | None
    sensitivity: Literal["public", "internal", "confidential", "restricted"]

@dataclass
class SecurityContext:
    """Context for access control decisions."""
    requester_id: str
    tenant_id: str | None
    permissions: set[str]
```

## Testing Strategy

### Invariant Tests
```python
def test_facts_preserved_after_compaction():
    """Critical facts must survive N compaction cycles."""
    thread = create_thread_with_facts(["user prefers Python", "budget is $1000"])
    for _ in range(5):
        plan = compactor.compact(thread)
        rendered = renderer.render(thread, plan)
    assert "Python" in rendered
    assert "1000" in rendered
```

### Task Continuation Tests
```python
def test_agent_completes_task_after_compaction():
    """Agent can complete multi-step task after compaction."""
    # Run agent until context pressure triggers compaction
    # Verify agent still completes the task correctly
```

### Summary Drift Tests
```python
def test_no_summary_drift():
    """Repeated summarization doesn't lose pinned constraints."""
    constraint = "Never delete production data"
    # Summarize 10 times
    # Verify constraint still present
```

### Budget Tests
```python
def test_stays_under_budget_with_real_overhead():
    """Total request size never exceeds budget."""
    # Include system prompt, tools, formatting
    # Verify actual token count < limit
```

## Future Improvements (Non-blocking for v1)

These are worth being aware of, but can be deferred:

### 1. Dropped Span Index for Future Recall
When `DropStrategy` is used, preserve a structured index of what was dropped
so that future recall (RAG) can be added without rework:

```python
@dataclass
class DroppedSpanIndex:
    """Index of dropped content for future retrieval."""
    span: SpanReference
    keywords: list[str]  # Extracted for search
    summary: str  # Brief description
    timestamp: datetime
    # Future: embedding vector for semantic search
```

### 2. Deterministic Summary Rendering
`StructuredSummary.render_as_message()` should be:
- Deterministic (same input → same output)
- Versioned (format version in output)
- Stable (field order doesn't change between runs)

This enables reproducibility testing and "golden prompt" snapshots.

```python
SUMMARY_RENDER_VERSION = "v1.0"

def render_as_message(self) -> str:
    """Render deterministically with version prefix."""
    # Use sorted keys, consistent formatting
    return f"[Summary {SUMMARY_RENDER_VERSION}]\n{self._render_body()}"
```

### 3. Quality Regression Metrics
Track "agent asked for missing info" after compaction—this is the canary
for quality regressions:

```python
@dataclass
class CompactionQualityMetrics:
    """Metrics for detecting compaction quality issues."""
    turns_since_compaction: int
    agent_asked_for_context: bool  # "What was the user's budget again?"
    rehydration_requests: int  # Explicit read_artifact calls
    task_completion_success: bool
    # Alert if agent frequently asks for compacted info
```

## Implementation Plan

### Phase 1: Core Data Structures
- [ ] `CompactionPlan`, `SpanReference`, records
- [ ] `StructuredSummary` with render method
- [ ] `ToolDurability` enum and policy
- [ ] `ToolResultEnvelope` for structured tool outputs
- [ ] `DeterminismMetadata` for REPLAYABLE tools
- [ ] `TurnContext` for turn-level state

### Phase 2: Tokenizer
- [ ] `ProviderAwareTokenizer` protocol
- [ ] `TiktokenTokenizer` implementation
- [ ] Request-level token counting

### Phase 3: Prompt Renderer
- [ ] `PromptRenderer` that applies plans
- [ ] Integration with `AgentThread`
- [ ] Tests for correct rendering

### Phase 4: Strategies
- [ ] `ExternalizeStrategy` with `ArtifactStore`
- [ ] `SummarizeStrategy` with `LLMSummarizer`
- [ ] `ClearStrategy` with durability rules
- [ ] `RetrieveStrategy` (optional, needs vector store)
- [ ] `DropStrategy` (last resort)

### Phase 5: Rehydration
- [ ] `RehydrationInterceptor`
- [ ] Auto-detection of artifact references
- [ ] Integration with agent turn executor

### Phase 6: Concurrency
- [ ] `CompactionStore` with versioning
- [ ] `CompactionTransaction` for safe commits
- [ ] Summary caching with content hash

### Phase 7: Observability
- [ ] Lifecycle events for DevUI
- [ ] Metrics (tokens freed, strategies applied, etc.)

### Phase 8: Testing
- [ ] Invariant test suite
- [ ] Task continuation tests
- [ ] Drift tests
- [ ] Budget tests
