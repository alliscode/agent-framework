# Copyright (c) Microsoft. All rights reserved.

"""Tool durability and result envelope types for context compaction.

Not all tool results are safe to clear. This module defines:
- ToolDurability: Classification of how tool results can be compacted
- DeterminismMetadata: Metadata to verify if REPLAYABLE tools are still valid
- ToolResultEnvelope: Structured envelope for tool results in the thread

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ._tokenizer import ProviderAwareTokenizer


class ToolDurability(Enum):
    """Durability classification for tool results.

    Durability determines how a tool result can be compacted during
    context pressure management.
    """

    EPHEMERAL = "ephemeral"
    """Safe to clear entirely.

    Examples: directory listing, debug output, verbose logs.
    These provide transient information that doesn't need to be preserved.
    """

    ANCHORING = "anchoring"
    """Must keep summary + key fields.

    Examples: API response with order ID, database query results.
    These contain IDs or decisions that other operations depend on.
    """

    REPLAYABLE = "replayable"
    """Safe to clear because tool can be re-run cheaply/deterministically.

    Examples: read_file (if file hasn't changed), math calculation.
    IMPORTANT: Requires determinism_metadata to verify result is still valid.
    If drift is detected, the result becomes NON_REPLAYABLE for that span.
    """

    NON_REPLAYABLE = "non_replayable"
    """Must externalize or keep - cannot be safely cleared.

    Examples: web search results, paid API calls, stock prices.
    These are time-sensitive, rate-limited, or expensive to reproduce.
    """


@dataclass
class DeterminismMetadata:
    """Metadata to verify if a REPLAYABLE tool result is still valid.

    REPLAYABLE tools must capture this at call time. Before clearing a
    REPLAYABLE result, we check if the source has changed. If drift is
    detected, the result becomes NON_REPLAYABLE for that span.

    Attributes:
        content_hash: Hash of file/data at read time (first 16 chars of SHA256).
        etag: HTTP ETag for remote resources.
        mtime: Modification time (Unix timestamp).
        version: API version or schema version.
    """

    content_hash: str | None = None
    etag: str | None = None
    mtime: float | None = None
    version: str | None = None

    def has_changed(self, current: DeterminismMetadata) -> bool:
        """Check if source has drifted since this metadata was captured.

        Args:
            current: Current determinism metadata from the source.

        Returns:
            True if the source has changed (drift detected).
            Returns True if we can't verify (fail-safe).
        """
        # Try content hash first (most reliable)
        if self.content_hash and current.content_hash:
            return self.content_hash != current.content_hash

        # Try ETag
        if self.etag and current.etag:
            return self.etag != current.etag

        # Try mtime
        if self.mtime and current.mtime:
            return self.mtime != current.mtime

        # Can't verify - assume changed (fail safe)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content_hash": self.content_hash,
            "etag": self.etag,
            "mtime": self.mtime,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeterminismMetadata:
        """Deserialize from dictionary."""
        return cls(
            content_hash=data.get("content_hash"),
            etag=data.get("etag"),
            mtime=data.get("mtime"),
            version=data.get("version"),
        )

    @classmethod
    def from_file(cls, path: str) -> DeterminismMetadata:
        """Create metadata from a file path.

        Args:
            path: Path to the file.

        Returns:
            DeterminismMetadata with mtime and content_hash.
        """
        import hashlib
        from pathlib import Path

        file_path = Path(path)
        stat = file_path.stat()
        content = file_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()[:16]

        return cls(
            content_hash=content_hash,
            mtime=stat.st_mtime,
        )


@dataclass
class ToolDurabilityPolicy:
    """Policy for how to handle tool results during compaction.

    Tools can declare their durability policy via decorator or configuration.
    The policy determines what can be cleared and what must be preserved.

    Attributes:
        durability: The durability classification.
        must_preserve_fields: Field names that must be kept even when cleared.
        externalize_threshold_tokens: Token count above which to externalize.
        replay_cost: How expensive it is to replay this tool.
        capture_determinism: Optional function to capture current determinism state.
    """

    durability: ToolDurability
    must_preserve_fields: list[str] = field(default_factory=list)
    externalize_threshold_tokens: int = 1000
    replay_cost: Literal["free", "cheap", "expensive"] = "cheap"
    capture_determinism: Callable[..., DeterminismMetadata] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (without capture_determinism)."""
        return {
            "durability": self.durability.value,
            "must_preserve_fields": self.must_preserve_fields,
            "externalize_threshold_tokens": self.externalize_threshold_tokens,
            "replay_cost": self.replay_cost,
        }


# Default policies for common tool patterns
DEFAULT_DURABILITY_POLICIES: dict[str, ToolDurabilityPolicy] = {
    # File operations
    "read_file": ToolDurabilityPolicy(
        durability=ToolDurability.REPLAYABLE,
        replay_cost="free",
    ),
    "list_directory": ToolDurabilityPolicy(
        durability=ToolDurability.EPHEMERAL,
    ),
    "write_file": ToolDurabilityPolicy(
        durability=ToolDurability.ANCHORING,
        must_preserve_fields=["path", "success"],
    ),
    # Web/API operations
    "web_search": ToolDurabilityPolicy(
        durability=ToolDurability.NON_REPLAYABLE,
        externalize_threshold_tokens=500,
    ),
    "http_request": ToolDurabilityPolicy(
        durability=ToolDurability.NON_REPLAYABLE,
        must_preserve_fields=["status_code", "url"],
    ),
    # Compute operations
    "run_command": ToolDurabilityPolicy(
        durability=ToolDurability.ANCHORING,
        must_preserve_fields=["exit_code", "command"],
    ),
    "calculate": ToolDurabilityPolicy(
        durability=ToolDurability.REPLAYABLE,
        replay_cost="free",
    ),
}


@dataclass
class ToolResultEnvelope:
    """Structured envelope for tool results in the thread.

    For durability rules to work, tool results must be structured, not opaque
    strings. This envelope provides consistent structure for compaction.

    The envelope guarantees we can always extract:
    - Tool name and outcome
    - Key fields (IDs, important values)
    - Determinism metadata for REPLAYABLE tools

    Attributes:
        tool_name: Name of the tool that was called.
        tool_call_id: Unique identifier for this tool call.
        outcome: Whether the call succeeded, failed, or partially succeeded.
        content: The actual result content (string or structured).
        key_fields: Fields that MUST survive compaction.
        durability: Durability classification (set by tool or policy).
        determinism: Metadata for REPLAYABLE tools.
        started_at: When the tool call started.
        completed_at: When the tool call completed.
    """

    tool_name: str
    tool_call_id: str
    outcome: Literal["success", "failure", "partial"]
    content: str | dict[str, Any]
    key_fields: dict[str, Any] = field(default_factory=dict)
    durability: ToolDurability | None = None
    determinism: DeterminismMetadata | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def get_token_count(self, tokenizer: ProviderAwareTokenizer) -> int:
        """Get token count of content.

        Args:
            tokenizer: Tokenizer to use for counting.

        Returns:
            Token count of the content.
        """
        if isinstance(self.content, str):
            return tokenizer.count_tokens(self.content)
        return tokenizer.count_tokens(json.dumps(self.content))

    def to_cleared_placeholder(self) -> str:
        """Render as minimal placeholder preserving key_fields.

        Returns:
            A minimal string representation suitable for cleared content.
        """
        parts = [f"[{self.tool_name}: {self.outcome}]"]
        if self.key_fields:
            # Sort keys for determinism
            fields = ", ".join(f"{k}={v}" for k, v in sorted(self.key_fields.items()))
            parts.append(f"Key data: {fields}")
        return "\n".join(parts)

    def get_effective_durability(
        self,
        policy_overrides: dict[str, ToolDurabilityPolicy] | None = None,
        default: ToolDurability = ToolDurability.ANCHORING,
    ) -> ToolDurability:
        """Get the effective durability for this result.

        Priority:
        1. Explicit durability set on envelope
        2. Policy override for this tool name
        3. Default policy for this tool name
        4. Provided default

        Args:
            policy_overrides: Tool-specific policy overrides.
            default: Default durability if none specified.

        Returns:
            The effective ToolDurability.
        """
        # 1. Explicit durability on envelope
        if self.durability is not None:
            return self.durability

        # 2. Policy override
        if policy_overrides and self.tool_name in policy_overrides:
            return policy_overrides[self.tool_name].durability

        # 3. Default policy
        if self.tool_name in DEFAULT_DURABILITY_POLICIES:
            return DEFAULT_DURABILITY_POLICIES[self.tool_name].durability

        # 4. Provided default
        return default

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "outcome": self.outcome,
            "content": self.content,
            "key_fields": self.key_fields,
            "durability": self.durability.value if self.durability else None,
            "determinism": self.determinism.to_dict() if self.determinism else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResultEnvelope:
        """Deserialize from dictionary."""
        durability = None
        if data.get("durability"):
            durability = ToolDurability(data["durability"])

        determinism = None
        if data.get("determinism"):
            determinism = DeterminismMetadata.from_dict(data["determinism"])

        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])

        completed_at = None
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        return cls(
            tool_name=data["tool_name"],
            tool_call_id=data["tool_call_id"],
            outcome=data["outcome"],
            content=data["content"],
            key_fields=data.get("key_fields", {}),
            durability=durability,
            determinism=determinism,
            started_at=started_at,
            completed_at=completed_at,
        )

    @classmethod
    def from_tool_result(
        cls,
        tool_name: str,
        tool_call_id: str,
        result: str | dict[str, Any],
        *,
        outcome: Literal["success", "failure", "partial"] = "success",
        key_fields: dict[str, Any] | None = None,
        durability: ToolDurability | None = None,
        determinism: DeterminismMetadata | None = None,
    ) -> ToolResultEnvelope:
        """Create envelope from a tool result.

        Args:
            tool_name: Name of the tool.
            tool_call_id: ID of the tool call.
            result: The tool result content.
            outcome: Success/failure status.
            key_fields: Important fields to preserve.
            durability: Durability classification.
            determinism: Determinism metadata for REPLAYABLE tools.

        Returns:
            A ToolResultEnvelope wrapping the result.
        """
        return cls(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            outcome=outcome,
            content=result,
            key_fields=key_fields or {},
            durability=durability,
            determinism=determinism,
            completed_at=datetime.now(timezone.utc),
        )
