# Copyright (c) Microsoft. All rights reserved.

"""Adapters for bridging AgentTurnExecutor cache with compaction strategies.

The compaction strategies (ClearStrategy, DropStrategy, etc.) operate on
AgentThread objects, accessing messages via ``thread.message_store.list_messages()``.
However, AgentTurnExecutor maintains a flat ``self._cache: list[Any]`` of
ChatMessage objects — not an AgentThread.

This module provides lightweight adapters that wrap the cache as an
AgentThread-like object, allowing the CompactionCoordinator to operate
on the executor's cache without modifying the strategy interfaces.

After each compaction cycle, the cache is **flattened** — the compaction plan
is applied and the cache is replaced with the compacted view. This means
strategies always see the actual messages (including rendered summaries from
previous cycles), enabling re-summarization of accumulated summaries.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._types import ChatMessage


class CacheMessageStore:
    """Read-only message store backed by the executor cache.

    Implements the subset of ChatMessageStoreProtocol needed by
    compaction strategies (``list_messages()``).

    After flattening, the cache contains rendered summaries, cleared
    placeholders, and original messages — strategies see the real
    context the LLM would receive.
    """

    def __init__(self, cache: list[Any]):
        """Initialize with a reference to the executor's cache.

        Args:
            cache: The executor's message cache list.
        """
        self._cache = cache

    async def list_messages(self) -> list[ChatMessage]:
        """Return a copy of the cache as a message list.

        Assigns stable synthetic ``message_id`` values to any messages
        that lack one, so that compaction strategies can reference them.
        The IDs are written back to the original cache objects to remain
        stable across repeated calls.

        Returns:
            Copy of the cache contents.
        """
        for msg in self._cache:
            if getattr(msg, "message_id", None) is None:
                msg.message_id = f"cache-{uuid.uuid4().hex[:12]}"
        return list(self._cache)

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """No-op — the cache is managed by AgentTurnExecutor.

        Args:
            messages: Messages to add (ignored).
        """


class CacheThreadAdapter:
    """Wraps AgentTurnExecutor._cache as an AgentThread-like for compaction.

    Provides the ``message_store`` attribute that compaction strategies
    access to read messages. This is a lightweight adapter — it does not
    implement the full AgentThread interface.
    """

    def __init__(self, cache: list[Any]):
        """Initialize with a reference to the executor's cache.

        Args:
            cache: The executor's message cache list.
        """
        self.message_store = CacheMessageStore(cache)
