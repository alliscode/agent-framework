# Copyright (c) Microsoft. All rights reserved.

"""Storage for context compaction.

This module provides:
- CompactionStore protocol and in-memory implementation for persisting compaction plans
- SummaryCache protocol and in-memory implementation for caching LLM summaries
- ArtifactStore protocol and in-memory implementation for storing externalized content
- ArtifactMetadata for artifact access control

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from ._summary import (
    STRUCTURED_SUMMARY_SCHEMA_VERSION,
    StructuredSummary,
    SummaryCacheKey,
)
from ._types import CompactionPlan

if TYPE_CHECKING:
    from ._types import SpanReference

logger = logging.getLogger(__name__)


class CompactionStore(Protocol):
    """Protocol for storing compaction plans.

    Implementations provide simple get/save/delete semantics for
    compaction plans keyed by thread ID.
    """

    async def get_plan(
        self,
        thread_id: str,
    ) -> CompactionPlan | None:
        """Get the current compaction plan.

        Args:
            thread_id: ID of the thread.

        Returns:
            The plan, or None if no plan exists.
        """
        ...

    async def save_plan(
        self,
        thread_id: str,
        plan: CompactionPlan,
    ) -> None:
        """Save a compaction plan.

        Args:
            thread_id: ID of the thread.
            plan: The compaction plan to save.
        """
        ...

    async def delete_plan(
        self,
        thread_id: str,
    ) -> bool:
        """Delete the compaction plan for a thread.

        Args:
            thread_id: ID of the thread.

        Returns:
            True if plan was deleted, False if no plan existed.
        """
        ...


class InMemoryCompactionStore:
    """In-memory implementation of CompactionStore.

    Suitable for development, testing, and single-process deployments.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._plans: dict[str, CompactionPlan] = {}

    async def get_plan(
        self,
        thread_id: str,
    ) -> CompactionPlan | None:
        """Get the current plan.

        Args:
            thread_id: ID of the thread.

        Returns:
            The plan, or None if not found.
        """
        return self._plans.get(thread_id)

    async def save_plan(
        self,
        thread_id: str,
        plan: CompactionPlan,
    ) -> None:
        """Save a plan.

        Args:
            thread_id: ID of the thread.
            plan: The plan to save.
        """
        self._plans[thread_id] = plan

    async def delete_plan(
        self,
        thread_id: str,
    ) -> bool:
        """Delete plan for a thread.

        Args:
            thread_id: ID of the thread.

        Returns:
            True if deleted, False if not found.
        """
        if thread_id not in self._plans:
            return False
        del self._plans[thread_id]
        return True

    def clear(self) -> None:
        """Clear all stored plans."""
        self._plans.clear()

    @property
    def plan_count(self) -> int:
        """Get the number of stored plans."""
        return len(self._plans)


class SummaryCache(Protocol):
    """Protocol for caching LLM-generated summaries.

    Summaries are expensive (LLM calls) and should be cached.
    The cache key includes all factors that affect the summary
    to prevent cache poisoning.

    Key factors:
    - Content hash (what was summarized)
    - Schema version (StructuredSummary format)
    - Policy version (summarization rules)
    - Model ID (which model generated it)
    - Prompt version (summarization prompt)
    """

    async def get(
        self,
        key: SummaryCacheKey,
    ) -> StructuredSummary | None:
        """Get a cached summary.

        Args:
            key: The cache key.

        Returns:
            Cached summary or None if not found.
        """
        ...

    async def put(
        self,
        key: SummaryCacheKey,
        summary: StructuredSummary,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store a summary in the cache.

        Args:
            key: The cache key.
            summary: The summary to cache.
            ttl_seconds: Optional time-to-live in seconds.
        """
        ...

    async def invalidate(
        self,
        key: SummaryCacheKey,
    ) -> bool:
        """Invalidate a cached summary.

        Args:
            key: The cache key.

        Returns:
            True if entry was removed, False if not found.
        """
        ...


@dataclass
class CacheEntry:
    """Entry in the summary cache.

    Attributes:
        summary: The cached summary.
        created_at: When the entry was created.
        expires_at: When the entry expires (None for no expiry).
    """

    summary: StructuredSummary
    created_at: datetime
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class InMemorySummaryCache:
    """In-memory implementation of SummaryCache.

    Suitable for development, testing, and single-process deployments.
    Supports TTL-based expiration and LRU eviction.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        default_ttl_seconds: int | None = None,
    ):
        """Initialize the cache.

        Args:
            max_entries: Maximum entries before LRU eviction.
            default_ttl_seconds: Default TTL (None for no expiry).
        """
        self._entries: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU
        self._max_entries = max_entries
        self._default_ttl_seconds = default_ttl_seconds

    async def get(
        self,
        key: SummaryCacheKey,
    ) -> StructuredSummary | None:
        """Get a cached summary.

        Args:
            key: The cache key.

        Returns:
            Cached summary or None if not found/expired.
        """
        key_str = key.to_string()
        entry = self._entries.get(key_str)
        if entry is None:
            return None

        if entry.is_expired():
            del self._entries[key_str]
            if key_str in self._access_order:
                self._access_order.remove(key_str)
            return None

        # Update access order (LRU)
        if key_str in self._access_order:
            self._access_order.remove(key_str)
        self._access_order.append(key_str)

        return entry.summary

    async def put(
        self,
        key: SummaryCacheKey,
        summary: StructuredSummary,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store a summary in the cache.

        Args:
            key: The cache key.
            summary: The summary to cache.
            ttl_seconds: TTL in seconds (uses default if None).
        """
        key_str = key.to_string()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds

        now = datetime.now(timezone.utc)
        expires_at = None
        if ttl is not None:
            from datetime import timedelta

            expires_at = now + timedelta(seconds=ttl)

        entry = CacheEntry(
            summary=summary,
            created_at=now,
            expires_at=expires_at,
        )

        # Evict if at capacity
        while len(self._entries) >= self._max_entries and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._entries:
                del self._entries[oldest_key]

        self._entries[key_str] = entry

        # Update access order
        if key_str in self._access_order:
            self._access_order.remove(key_str)
        self._access_order.append(key_str)

    async def invalidate(
        self,
        key: SummaryCacheKey,
    ) -> bool:
        """Invalidate a cached summary.

        Args:
            key: The cache key.

        Returns:
            True if removed, False if not found.
        """
        key_str = key.to_string()
        if key_str not in self._entries:
            return False
        del self._entries[key_str]
        if key_str in self._access_order:
            self._access_order.remove(key_str)
        return True

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._access_order.clear()

    @property
    def entry_count(self) -> int:
        """Get the number of cached entries."""
        return len(self._entries)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired_keys = [key for key, entry in self._entries.items() if entry.is_expired()]
        for key in expired_keys:
            del self._entries[key]
            if key in self._access_order:
                self._access_order.remove(key)
        return len(expired_keys)


def compute_content_hash(content: str) -> str:
    """Compute a content hash for cache keys.

    Uses SHA256 truncated to 16 characters for reasonable collision
    resistance while keeping keys short.

    Args:
        content: The content to hash.

    Returns:
        16-character hex hash.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_summary_cache_key(
    span: SpanReference,
    messages_content: str,
    *,
    policy_version: str = "v1.0",
    model_id: str = "default",
    prompt_version: str = "v1.0",
) -> SummaryCacheKey:
    """Create a cache key for a summary.

    Args:
        span: The span being summarized.
        messages_content: Concatenated content of messages in span.
        policy_version: Version of summarization policy.
        model_id: ID of the summarization model.
        prompt_version: Version of the summarization prompt.

    Returns:
        A SummaryCacheKey for caching.
    """
    content_hash = compute_content_hash(messages_content)

    return SummaryCacheKey(
        content_hash=content_hash,
        schema_version=STRUCTURED_SUMMARY_SCHEMA_VERSION,
        policy_version=policy_version,
        model_id=model_id,
        prompt_version=prompt_version,
    )


@dataclass
class ArtifactStoreEntry:
    """Entry in the in-memory artifact store.

    Attributes:
        content: The stored content.
        metadata: Artifact metadata.
        created_at: When the entry was created.
    """

    content: str
    metadata: Any  # ArtifactMetadata
    created_at: datetime


class InMemoryArtifactStore:
    """In-memory implementation of ArtifactStore protocol.

    Suitable for development and testing. Does not persist
    across restarts.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        self._artifacts: dict[str, ArtifactStoreEntry] = {}
        self._counter = 0

    async def store(
        self,
        content: str,
        metadata: Any,
    ) -> str:
        """Store content and return artifact ID.

        Args:
            content: The content to store.
            metadata: Artifact metadata.

        Returns:
            Generated artifact ID.
        """
        self._counter += 1
        artifact_id = f"artifact-{self._counter:08d}"
        self._artifacts[artifact_id] = ArtifactStoreEntry(
            content=content,
            metadata=metadata,
            created_at=datetime.now(timezone.utc),
        )
        return artifact_id

    async def get_metadata(
        self,
        artifact_id: str,
    ) -> Any | None:
        """Get artifact metadata.

        Args:
            artifact_id: The artifact ID.

        Returns:
            Metadata or None if not found.
        """
        entry = self._artifacts.get(artifact_id)
        if entry is None:
            return None
        return entry.metadata

    async def retrieve(
        self,
        artifact_id: str,
        requester_context: Any | None = None,
    ) -> str | None:
        """Retrieve artifact content.

        Args:
            artifact_id: The artifact ID.
            requester_context: Optional context (unused in this impl).

        Returns:
            Content or None if not found.
        """
        entry = self._artifacts.get(artifact_id)
        if entry is None:
            return None
        return entry.content

    async def delete(
        self,
        artifact_id: str,
    ) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            True if deleted, False if not found.
        """
        if artifact_id not in self._artifacts:
            return False
        del self._artifacts[artifact_id]
        return True

    def clear(self) -> None:
        """Clear all artifacts."""
        self._artifacts.clear()
        self._counter = 0

    @property
    def artifact_count(self) -> int:
        """Get the number of stored artifacts."""
        return len(self._artifacts)


# ---------------------------------------------------------------------------
# Artifact protocol and metadata types (moved from removed _renderer.py)
# ---------------------------------------------------------------------------


class ArtifactStore(Protocol):
    """Protocol for storing and retrieving externalized content."""

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

        Args:
            artifact_id: The artifact ID.

        Returns:
            Metadata if found, None otherwise.
        """
        ...

    async def retrieve(
        self,
        artifact_id: str,
        requester_context: Any | None = None,
    ) -> str | None:
        """Retrieve artifact content.

        Args:
            artifact_id: The artifact ID.
            requester_context: Optional context for access control.

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
