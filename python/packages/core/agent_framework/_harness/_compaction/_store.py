# Copyright (c) Microsoft. All rights reserved.

"""Storage and concurrency for context compaction.

This module provides:
- CompactionStore protocol for persisting compaction plans
- In-memory implementation for development/testing
- CompactionTransaction for optimistic concurrency control
- SummaryCache for caching expensive LLM summaries

Concurrency Model:
- Plans use optimistic locking via version numbers
- Commits fail if version has changed (conflict detection)
- Summaries are cached by content hash + schema version

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
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
    """Protocol for storing compaction plans with concurrency control.

    Implementations must support:
    - Get current plan with version number
    - Commit plan with optimistic locking (version check)
    - Delete/cleanup old plans

    Thread-safety requirements:
    - get_current_plan must be thread-safe
    - commit_plan must be atomic (check version + update)
    """

    async def get_current_plan(
        self,
        thread_id: str,
    ) -> tuple[CompactionPlan | None, int]:
        """Get the current compaction plan and version number.

        Args:
            thread_id: ID of the thread.

        Returns:
            Tuple of (plan, version). Plan is None if no plan exists.
            Version starts at 0 and increments with each commit.
        """
        ...

    async def commit_plan(
        self,
        thread_id: str,
        plan: CompactionPlan,
        expected_version: int,
    ) -> bool:
        """Commit a compaction plan with optimistic locking.

        The commit succeeds only if the current version matches
        expected_version. This prevents lost updates when multiple
        processes are compacting the same thread.

        Args:
            thread_id: ID of the thread.
            plan: The compaction plan to commit.
            expected_version: Expected current version (from get_current_plan).

        Returns:
            True if commit succeeded, False if version conflict.
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
    Uses a lock for thread-safety.

    Attributes:
        plans: Map of thread_id to (plan, version).
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._plans: dict[str, tuple[CompactionPlan, int]] = {}
        self._lock = threading.Lock()

    async def get_current_plan(
        self,
        thread_id: str,
    ) -> tuple[CompactionPlan | None, int]:
        """Get the current plan and version.

        Args:
            thread_id: ID of the thread.

        Returns:
            Tuple of (plan, version). Plan is None if not found.
        """
        with self._lock:
            if thread_id not in self._plans:
                return (None, 0)
            plan, version = self._plans[thread_id]
            return (plan, version)

    async def commit_plan(
        self,
        thread_id: str,
        plan: CompactionPlan,
        expected_version: int,
    ) -> bool:
        """Commit plan with optimistic locking.

        Args:
            thread_id: ID of the thread.
            plan: The plan to commit.
            expected_version: Expected version.

        Returns:
            True if committed, False on version conflict.
        """
        with self._lock:
            current_version = 0
            if thread_id in self._plans:
                _, current_version = self._plans[thread_id]

            if current_version != expected_version:
                logger.debug(
                    "Version conflict for thread %s: expected %d, got %d",
                    thread_id,
                    expected_version,
                    current_version,
                )
                return False

            new_version = current_version + 1
            plan.thread_version = new_version
            self._plans[thread_id] = (plan, new_version)

            logger.debug(
                "Committed plan for thread %s, version %d",
                thread_id,
                new_version,
            )
            return True

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
        with self._lock:
            if thread_id not in self._plans:
                return False
            del self._plans[thread_id]
            logger.debug("Deleted plan for thread %s", thread_id)
            return True

    def clear(self) -> None:
        """Clear all stored plans."""
        with self._lock:
            self._plans.clear()
            logger.debug("Cleared all plans")

    @property
    def plan_count(self) -> int:
        """Get the number of stored plans."""
        with self._lock:
            return len(self._plans)


@dataclass
class CompactionTransaction:
    """Transactional wrapper for compaction operations.

    Provides a clean API for read-modify-commit cycles with
    automatic version tracking.

    Usage:
        transaction = CompactionTransaction(store, thread_id)
        await transaction.begin()

        # Modify plan
        transaction.plan.clearings.append(new_record)

        # Commit with conflict detection
        success = await transaction.commit()
        if not success:
            # Handle conflict (retry, etc.)
            pass

    Attributes:
        store: The compaction store.
        thread_id: ID of the thread.
        plan: The current plan (None until begin() called).
        version: Current version (for optimistic locking).
    """

    store: CompactionStore
    thread_id: str
    plan: CompactionPlan | None = None
    version: int = 0
    _started: bool = field(default=False, repr=False)

    async def begin(self) -> CompactionPlan:
        """Begin the transaction by loading the current plan.

        Creates an empty plan if none exists.

        Returns:
            The current (or new) compaction plan.
        """
        self.plan, self.version = await self.store.get_current_plan(self.thread_id)

        if self.plan is None:
            self.plan = CompactionPlan.create_empty(self.thread_id, self.version)

        self._started = True
        return self.plan

    async def commit(self) -> bool:
        """Commit the modified plan.

        Returns:
            True if commit succeeded, False on version conflict.

        Raises:
            RuntimeError: If begin() was not called.
        """
        if not self._started or self.plan is None:
            raise RuntimeError("Transaction not started. Call begin() first.")

        return await self.store.commit_plan(
            self.thread_id,
            self.plan,
            self.version,
        )

    async def rollback(self) -> None:
        """Rollback the transaction by reloading the plan.

        Useful after a failed commit to get the latest version.
        """
        self.plan, self.version = await self.store.get_current_plan(self.thread_id)
        if self.plan is None:
            self.plan = CompactionPlan.create_empty(self.thread_id, self.version)


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
    Supports TTL-based expiration.

    Attributes:
        max_entries: Maximum number of entries (LRU eviction).
        default_ttl_seconds: Default TTL for entries.
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
        self._lock = threading.Lock()
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

        with self._lock:
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

        with self._lock:
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

        with self._lock:
            if key_str not in self._entries:
                return False
            del self._entries[key_str]
            if key_str in self._access_order:
                self._access_order.remove(key_str)
            return True

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()

    @property
    def entry_count(self) -> int:
        """Get the number of cached entries."""
        with self._lock:
            return len(self._entries)

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        removed = 0

        with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items() if entry.is_expired()
            ]

            for key in expired_keys:
                del self._entries[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                removed += 1

        return removed


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
    span: "SpanReference",
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
        self._lock = threading.Lock()
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
        with self._lock:
            self._counter += 1
            artifact_id = f"artifact-{self._counter:08d}"

            self._artifacts[artifact_id] = ArtifactStoreEntry(
                content=content,
                metadata=metadata,
                created_at=datetime.now(timezone.utc),
            )

            logger.debug("Stored artifact %s (%d bytes)", artifact_id, len(content))
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
        with self._lock:
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
            requester_context: Security context (ignored in this impl).

        Returns:
            Content or None if not found.
        """
        with self._lock:
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
        with self._lock:
            if artifact_id not in self._artifacts:
                return False
            del self._artifacts[artifact_id]
            logger.debug("Deleted artifact %s", artifact_id)
            return True

    def clear(self) -> None:
        """Clear all artifacts."""
        with self._lock:
            self._artifacts.clear()
            self._counter = 0
            logger.debug("Cleared all artifacts")

    @property
    def artifact_count(self) -> int:
        """Get the number of stored artifacts."""
        with self._lock:
            return len(self._artifacts)
