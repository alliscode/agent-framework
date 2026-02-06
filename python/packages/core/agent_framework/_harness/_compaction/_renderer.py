# Copyright (c) Microsoft. All rights reserved.

"""Prompt renderer for context compaction.

The PromptRenderer applies a CompactionPlan to an AgentThread to produce
the actual prompt sent to the model. This separation enables:
- Testing ("golden prompt" snapshots)
- Policy iteration without data loss
- Multiple rendering strategies

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from ..._types import ChatMessage
from ._summary import StructuredSummary
from ._turn_context import RehydrationResult
from ._types import (
    ClearRecord,
    CompactionAction,
    CompactionPlan,
    ExternalizationRecord,
    SummarizationRecord,
)

if TYPE_CHECKING:
    from ..._threads import AgentThread
    from ._tokenizer import ProviderAwareTokenizer

logger = logging.getLogger(__name__)

# Rendering format version - bump when format changes
COMPACTION_RENDER_FORMAT_VERSION = "v1.0"


class ArtifactStore(Protocol):
    """Protocol for storing and retrieving externalized content.

    Security considerations:
    - Implementations should support tenant isolation
    - Access control should be enforced on retrieval
    - Content should be encrypted at rest
    """

    async def store(
        self,
        content: str,
        metadata: ArtifactMetadata,
    ) -> str:
        """Store content securely.

        Args:
            content: The content to store.
            metadata: Metadata about the artifact.

        Returns:
            Unique artifact ID for retrieval.
        """
        ...

    async def get_metadata(
        self,
        artifact_id: str,
    ) -> ArtifactMetadata | None:
        """Get metadata without retrieving content.

        Used for sensitivity gating in auto-rehydration.

        Args:
            artifact_id: The artifact ID.

        Returns:
            Metadata if found, None otherwise.
        """
        ...

    async def retrieve(
        self,
        artifact_id: str,
        requester_context: SecurityContext | None = None,
    ) -> str | None:
        """Retrieve content with access control.

        Args:
            artifact_id: The artifact ID.
            requester_context: Security context for access control.

        Returns:
            Content if found and authorized, None otherwise.
        """
        ...


@dataclass
class ArtifactMetadata:
    """Metadata for stored artifacts.

    Attributes:
        thread_id: ID of the thread this artifact belongs to.
        tenant_id: Tenant ID for multi-tenant isolation.
        created_at: When the artifact was created (ISO 8601).
        ttl_seconds: Time-to-live for auto-expiry.
        encryption_key_id: ID of the encryption key used.
        sensitivity: Sensitivity level for access control.
    """

    thread_id: str
    tenant_id: str | None = None
    created_at: str | None = None
    ttl_seconds: int | None = None
    encryption_key_id: str | None = None
    sensitivity: str = "internal"  # public, internal, confidential, restricted

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "thread_id": self.thread_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "encryption_key_id": self.encryption_key_id,
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        """Deserialize from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            tenant_id=data.get("tenant_id"),
            created_at=data.get("created_at"),
            ttl_seconds=data.get("ttl_seconds"),
            encryption_key_id=data.get("encryption_key_id"),
            sensitivity=data.get("sensitivity", "internal"),
        )


@dataclass
class SecurityContext:
    """Context for access control decisions.

    Attributes:
        requester_id: ID of the requester.
        tenant_id: Tenant ID for isolation.
        permissions: Set of permissions the requester has.
    """

    requester_id: str
    tenant_id: str | None = None
    permissions: set[str] = field(default_factory=lambda: set())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "requester_id": self.requester_id,
            "tenant_id": self.tenant_id,
            "permissions": list(self.permissions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityContext:
        """Deserialize from dictionary."""
        return cls(
            requester_id=data["requester_id"],
            tenant_id=data.get("tenant_id"),
            permissions=set(data.get("permissions", [])),
        )


@dataclass
class RenderedPrompt:
    """Result of rendering a thread with compaction.

    Attributes:
        messages: The rendered messages for the model.
        system_prompt: The system prompt (if any).
        tools: Tool definitions (if any).
        token_count: Estimated token count of the rendered prompt.
        compaction_applied: Whether any compaction was applied.
        spans_summarized: Number of spans that were summarized.
        spans_externalized: Number of spans that were externalized.
        spans_cleared: Number of spans that were cleared.
        spans_dropped: Number of spans that were dropped.
    """

    messages: list[ChatMessage]
    system_prompt: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    token_count: int = 0
    compaction_applied: bool = False
    spans_summarized: int = 0
    spans_externalized: int = 0
    spans_cleared: int = 0
    spans_dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "token_count": self.token_count,
            "compaction_applied": self.compaction_applied,
            "spans_summarized": self.spans_summarized,
            "spans_externalized": self.spans_externalized,
            "spans_cleared": self.spans_cleared,
            "spans_dropped": self.spans_dropped,
        }


class PromptRenderer:
    """Renders AgentThread + CompactionPlan into model request.

    The renderer applies compaction transformations to produce the actual
    prompt sent to the model. Key responsibilities:

    1. Apply compaction actions (summarize, externalize, clear, drop)
    2. Maintain message order and role semantics
    3. Inject rehydrated content (ephemeral)
    4. Track rendering statistics

    Replacement Semantics:
    - Role: Summaries and externalization pointers render as `role="assistant"`
    - Position: Synthetic message at position of first message in span
    - Format: Versioned prefix for reproducibility
    - One message per span (never merge adjacent summaries)

    Attributes:
        tokenizer: Tokenizer for counting tokens.
    """

    def __init__(
        self,
        tokenizer: ProviderAwareTokenizer | None = None,
    ):
        """Initialize the renderer.

        Args:
            tokenizer: Optional tokenizer for counting. If not provided,
                token_count in RenderedPrompt will be 0.
        """
        self._tokenizer = tokenizer

    async def render(
        self,
        thread: AgentThread,
        plan: CompactionPlan | None = None,
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        rehydrated_content: list[RehydrationResult] | None = None,
    ) -> RenderedPrompt:
        """Render thread with compaction applied.

        For each message in thread:
        - If in drop span → skip entirely
        - If in summarization span → inject summary at first message position
        - If in externalization span → inject pointer + summary at first
        - If in clear span → inject placeholder with preserved fields
        - Otherwise → include as-is

        Args:
            thread: The agent thread to render.
            plan: Optional compaction plan to apply. If None, renders as-is.
            system_prompt: Optional system prompt.
            tools: Optional tool definitions.
            rehydrated_content: Ephemeral content to inject (not part of thread).

        Returns:
            RenderedPrompt with messages and statistics.
        """
        # Get messages from thread
        if thread.message_store is None:
            messages_list: list[ChatMessage] = []
        else:
            messages_list = await thread.message_store.list_messages()

        # Track which spans we've rendered (to avoid duplicates)
        processed_spans: set[str] = set()

        # Statistics
        spans_summarized = 0
        spans_externalized = 0
        spans_cleared = 0
        spans_dropped = 0

        rendered_messages: list[ChatMessage] = []

        for msg in messages_list:
            # Get message ID - generate one if not present
            msg_id = msg.message_id
            if msg_id is None:
                # No ID means no compaction action for this message
                rendered_messages.append(msg)
                continue

            # Look up action in plan
            if plan is None:
                action = CompactionAction.INCLUDE
                record = None
            else:
                action, record = plan.get_action(msg_id)

            # Apply action
            if action == CompactionAction.DROP:
                spans_dropped += 1
                continue

            if action == CompactionAction.INCLUDE:
                rendered_messages.append(msg)
                continue

            if action == CompactionAction.CLEAR:
                rendered_messages.append(self._render_cleared(msg, record))
                spans_cleared += 1
                continue

            # For SUMMARIZE and EXTERNALIZE, only render once per span
            # record should never be None for these actions, but check for safety
            if record is None:
                logger.warning(
                    "Record is None for %s action on message %s",
                    action.value,
                    msg_id,
                )
                continue

            span_key = record.span.start_message_id
            if span_key in processed_spans:
                continue  # Already rendered this span's summary
            processed_spans.add(span_key)

            if action == CompactionAction.SUMMARIZE:
                rendered_messages.append(self._render_summary(record))
                spans_summarized += 1

            elif action == CompactionAction.EXTERNALIZE:
                rendered_messages.append(self._render_externalization(record))
                spans_externalized += 1

        # Inject rehydrated content (ephemeral, at end before current turn)
        if rehydrated_content:
            for result in rehydrated_content:
                rendered_messages.append(self._render_rehydration(result))

        # Calculate token count if tokenizer available
        token_count = 0
        if self._tokenizer is not None:
            token_count = self._count_tokens(
                rendered_messages,
                system_prompt,
                tools or [],
            )

        return RenderedPrompt(
            messages=rendered_messages,
            system_prompt=system_prompt,
            tools=tools or [],
            token_count=token_count,
            compaction_applied=plan is not None and not plan.is_empty,
            spans_summarized=spans_summarized,
            spans_externalized=spans_externalized,
            spans_cleared=spans_cleared,
            spans_dropped=spans_dropped,
        )

    def _render_summary(self, record: SummarizationRecord) -> ChatMessage:
        """Render a summary as an assistant message.

        Args:
            record: The summarization record.

        Returns:
            A ChatMessage with the summary content.
        """
        span = record.span
        summary_text = record.summary.render_as_message()

        content = f"[Context Summary - Turns {span.first_turn}-{span.last_turn}]\n{summary_text}"

        return ChatMessage(
            role="assistant",
            text=content,
            message_id=f"summary-{span.start_message_id}",
        )

    def _render_externalization(self, record: ExternalizationRecord) -> ChatMessage:
        """Render an externalization pointer as an assistant message.

        Args:
            record: The externalization record.

        Returns:
            A ChatMessage with the pointer and summary.
        """
        span = record.span
        summary_text = record.summary.render_as_message()

        content = (
            f"[Externalized Content - artifact:{record.artifact_id}]\n"
            f"Summary: {summary_text}\n"
            f'To retrieve full content, call: read_artifact("{record.artifact_id}")'
        )

        return ChatMessage(
            role="assistant",
            text=content,
            message_id=f"external-{span.start_message_id}",
        )

    def _render_cleared(
        self,
        original_msg: ChatMessage,
        record: ClearRecord | None,
    ) -> ChatMessage:
        """Render a cleared message as a placeholder.

        Args:
            original_msg: The original message being cleared.
            record: The clear record (may be None).

        Returns:
            A ChatMessage with placeholder content.
        """
        # Preserve role and basic structure
        role_value = original_msg.role.value if hasattr(original_msg.role, "value") else str(original_msg.role)

        # Build placeholder content
        parts = [f"[Cleared: {role_value} message]"]

        if record is not None and record.preserved_fields:
            # Sort keys for determinism
            fields = ", ".join(f"{k}={v}" for k, v in sorted(record.preserved_fields.items()))
            parts.append(f"Key data: {fields}")

        # Cast role to expected type (validated at runtime by ChatMessage)
        role = cast("Literal['system', 'user', 'assistant', 'tool']", role_value)

        return ChatMessage(
            role=role,
            text="\n".join(parts),
            message_id=original_msg.message_id,
        )

    def _render_rehydration(self, result: RehydrationResult) -> ChatMessage:
        """Render rehydrated content as an assistant message.

        Rehydrated content is ephemeral - it's injected for the current turn
        only and is not part of the canonical thread.

        Args:
            result: The rehydration result.

        Returns:
            A ChatMessage with the rehydrated content.
        """
        truncation_note = " (truncated)" if result.truncated else ""

        content = f"[Rehydrated Content - artifact:{result.artifact_id}{truncation_note}]\n{result.content}"

        return ChatMessage(
            role="assistant",
            text=content,
            message_id=f"rehydrated-{result.artifact_id}",
        )

    def _count_tokens(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None,
        tools: list[dict[str, Any]],
    ) -> int:
        """Count total tokens for the rendered prompt.

        Args:
            messages: The rendered messages.
            system_prompt: The system prompt.
            tools: Tool definitions.

        Returns:
            Total token count.
        """
        if self._tokenizer is None:
            return 0

        total = 0

        # System prompt
        if system_prompt:
            total += self._tokenizer.count_tokens(system_prompt)

        # Messages
        for msg in messages:
            msg_dict = self._message_to_dict(msg)
            total += self._tokenizer.count_message(msg_dict)

        # Tools
        if tools:
            total += self._tokenizer.count_tool_schemas(tools)

        return total

    def _message_to_dict(self, msg: ChatMessage) -> dict[str, Any]:
        """Convert ChatMessage to dict format for tokenizer.

        Args:
            msg: The chat message.

        Returns:
            Dict with role and content fields.
        """
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        return {
            "role": role,
            "content": msg.text,
        }


def render_summary_text(summary: StructuredSummary, span_first_turn: int, span_last_turn: int) -> str:
    """Render a summary as text for inclusion in a message.

    This is a standalone function for use when not using the full renderer.

    Args:
        summary: The structured summary.
        span_first_turn: First turn number of the span.
        span_last_turn: Last turn number of the span.

    Returns:
        Formatted summary text.
    """
    return f"[Context Summary - Turns {span_first_turn}-{span_last_turn}]\n{summary.render_as_message()}"


def render_externalization_text(
    artifact_id: str,
    summary: StructuredSummary,
    span_first_turn: int,
    span_last_turn: int,
) -> str:
    """Render an externalization pointer as text.

    This is a standalone function for use when not using the full renderer.

    Args:
        artifact_id: The artifact ID.
        summary: The structured summary.
        span_first_turn: First turn number of the span.
        span_last_turn: Last turn number of the span.

    Returns:
        Formatted externalization text.
    """
    summary_text = summary.render_as_message()
    return (
        f"[Externalized Content - artifact:{artifact_id}]\n"
        f"Summary (turns {span_first_turn}-{span_last_turn}): {summary_text}\n"
        f'To retrieve full content, call: read_artifact("{artifact_id}")'
    )
