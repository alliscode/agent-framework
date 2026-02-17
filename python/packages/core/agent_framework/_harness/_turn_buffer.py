# Copyright (c) Microsoft. All rights reserved.

"""Turn buffer abstractions for harness turn state."""

from __future__ import annotations

from typing import Any, Protocol

from .._workflows._conversation_state import decode_chat_messages, encode_chat_messages


class TurnBuffer(Protocol):
    """Abstraction for managing turn message buffers."""

    def load_messages(self) -> list[Any]:
        """Return the current message list view."""

    def replace_messages(self, messages: list[Any]) -> None:
        """Replace the full message list."""

    def append_message(self, message: Any) -> None:
        """Append one message."""

    def append_messages(self, messages: list[Any]) -> None:
        """Append multiple messages."""

    def snapshot_version(self) -> int:
        """Get current mutation version."""


class ExecutorLocalTurnBuffer:
    """In-memory turn buffer implementation owned by a single executor."""

    def __init__(self, initial_messages: list[Any] | None = None) -> None:
        self._messages: list[Any] = list(initial_messages or [])
        self._version = 0

    def load_messages(self) -> list[Any]:
        return self._messages

    def replace_messages(self, messages: list[Any]) -> None:
        self._messages = list(messages)
        self._version += 1

    def append_message(self, message: Any) -> None:
        self._messages.append(message)
        self._version += 1

    def append_messages(self, messages: list[Any]) -> None:
        if messages:
            self._messages.extend(messages)
            self._version += 1

    def snapshot_version(self) -> int:
        return self._version


class SharedStateTurnBuffer:
    """Shared-state backed turn buffer snapshot adapter."""

    def __init__(self, *, key: str) -> None:
        self._key = key

    async def write_snapshot(self, ctx: Any, *, messages: list[Any], version: int) -> None:
        payload = {
            "version": version,
            "message_count": len(messages),
            "messages": encode_chat_messages(messages),
        }
        await ctx.set_shared_state(self._key, payload)

    async def read_snapshot(self, ctx: Any) -> tuple[list[Any], int]:
        try:
            payload = await ctx.get_shared_state(self._key)
        except KeyError:
            return [], 0
        if not isinstance(payload, dict):
            return [], 0
        version = int(payload.get("version", 0))
        messages_payload = payload.get("messages", [])
        if not isinstance(messages_payload, list):
            return [], version
        messages = decode_chat_messages(messages_payload)
        return messages, version

    async def read_snapshot_info(self, ctx: Any) -> dict[str, Any]:
        try:
            payload = await ctx.get_shared_state(self._key)
        except KeyError:
            return {"present": False, "version": 0, "message_count": 0}
        if not isinstance(payload, dict):
            return {"present": False, "version": 0, "message_count": 0}
        return {
            "present": True,
            "version": int(payload.get("version", 0)),
            "message_count": int(payload.get("message_count", 0)),
        }
