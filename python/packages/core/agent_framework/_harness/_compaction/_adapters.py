# Copyright (c) Microsoft. All rights reserved.

"""Adapters for bridging AgentTurnExecutor cache with compaction strategies.

The compaction strategies (ClearStrategy, DropStrategy, etc.) operate on
AgentThread objects, accessing messages via ``thread.message_store.list_messages()``.
However, AgentTurnExecutor maintains a flat ``self._cache: list[Any]`` of
ChatMessage objects — not an AgentThread.

This module provides lightweight adapters that wrap the cache as an
AgentThread-like object, allowing the CompactionCoordinator to operate
on the executor's cache without modifying the strategy interfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._types import ChatMessage


class CacheMessageStore:
    """Read-only message store backed by the executor cache.

    Implements the subset of ChatMessageStoreProtocol needed by
    compaction strategies (``list_messages()``).
    """

    def __init__(self, cache: list[Any]):
        """Initialize with a reference to the executor's cache.

        Args:
            cache: The executor's message cache list.
        """
        self._cache = cache

    async def list_messages(self) -> list[ChatMessage]:
        """Return a copy of the cache as a message list.

        Returns:
            Copy of the cache contents.
        """
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
