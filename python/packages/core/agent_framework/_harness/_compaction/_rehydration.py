# Copyright (c) Microsoft. All rights reserved.

"""Rehydration interceptor for automatic content injection.

When content is externalized, the agent needs a reliable way to access it.
Pure "explicit tool call" is unreliable - models hallucinate or forget.
This module implements a hybrid rehydration strategy:

1. Primary path: Agent calls `read_artifact(artifact_id)` explicitly
2. Safety net: Automatic rehydration when artifact IDs are referenced

Rehydration Loop Breaking:
- Rehydrated content is EPHEMERAL (not appended to canonical log)
- Uses reserved budget separate from compaction pressure calculation
- Turn-level tracking prevents aggressive compaction after rehydration
- Cooldown prevents re-injection of same artifact

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from ._turn_context import RehydrationConfig, RehydrationResult

if TYPE_CHECKING:
    from ._renderer import ArtifactMetadata, ArtifactStore, SecurityContext
    from ._tokenizer import ProviderAwareTokenizer
    from ._types import CompactionPlan

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Representation of a pending tool call.

    Attributes:
        id: Unique identifier for the tool call.
        name: Name of the tool being called.
        arguments: Arguments passed to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


@dataclass
class RehydrationEvent:
    """Event emitted when rehydration occurs.

    Attributes:
        artifact_id: ID of the rehydrated artifact.
        token_count: Tokens injected.
        truncated: Whether content was truncated.
        trigger: What triggered the rehydration.
        turn_number: Turn when rehydration occurred.
    """

    artifact_id: str
    token_count: int
    truncated: bool
    trigger: str  # "message_reference", "tool_argument", "explicit"
    turn_number: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "token_count": self.token_count,
            "truncated": self.truncated,
            "trigger": self.trigger,
            "turn_number": self.turn_number,
        }


@dataclass
class RehydrationBlockedEvent:
    """Event emitted when rehydration is blocked.

    Attributes:
        artifact_id: ID of the artifact that was blocked.
        reason: Why rehydration was blocked.
        turn_number: Turn when this occurred.
    """

    artifact_id: str
    reason: str
    turn_number: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "reason": self.reason,
            "turn_number": self.turn_number,
        }


class RehydrationInterceptor:
    """Intercepts agent requests and auto-rehydrates when needed.

    The interceptor detects when the agent references externalized content
    and automatically injects the content back into the context. This
    provides a safety net for models that don't reliably use explicit
    read_artifact() calls.

    Security Considerations:
    - Uses artifact_store access control (requester must have permission)
    - Limits total injected tokens to prevent context flooding
    - Tracks recently-injected artifacts to prevent loops
    - Validates artifact IDs match expected format
    - Sensitivity gating: only auto-rehydrates public/internal by default

    Attributes:
        config: Configuration for rehydration behavior.
        tokenizer: Tokenizer for counting tokens.
    """

    def __init__(
        self,
        config: RehydrationConfig,
        tokenizer: ProviderAwareTokenizer,
    ):
        """Initialize the rehydration interceptor.

        Args:
            config: Configuration for rehydration.
            tokenizer: Tokenizer for token counting.
        """
        self._config = config
        self._tokenizer = tokenizer
        # artifact_id -> turn when last injected
        self._recent_injections: dict[str, int] = {}
        # Events emitted during rehydration
        self._events: list[RehydrationEvent | RehydrationBlockedEvent] = []

    @property
    def config(self) -> RehydrationConfig:
        """Get the rehydration configuration."""
        return self._config

    @property
    def recent_injections(self) -> dict[str, int]:
        """Get the recent injections map."""
        return self._recent_injections.copy()

    @property
    def events(self) -> list[RehydrationEvent | RehydrationBlockedEvent]:
        """Get events emitted during rehydration."""
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear the events list."""
        self._events.clear()

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

        This method scans the agent's message and tool call arguments for
        references to externalized artifacts. If found and conditions are
        met (not in cooldown, within budget, proper sensitivity), the
        content is retrieved and returned for injection.

        Args:
            agent_message: The agent's message text.
            pending_tool_calls: Tool calls the agent is making.
            compaction_plan: Current compaction plan with externalizations.
            artifact_store: Store for retrieving artifact content.
            current_turn: Current turn number.
            security_context: Security context for access control.

        Returns:
            List of RehydrationResult with content to inject.
        """
        if not self._config.enabled:
            return []

        # Find artifacts that need rehydration
        needed_artifacts = self._find_needed_artifacts(
            agent_message,
            pending_tool_calls,
            compaction_plan,
            current_turn,
        )

        if not needed_artifacts:
            return []

        # Retrieve artifacts with budget enforcement
        return await self._retrieve_artifacts(
            needed_artifacts,
            artifact_store,
            current_turn,
            security_context,
        )

    def _find_needed_artifacts(
        self,
        agent_message: str,
        pending_tool_calls: list[ToolCall],
        compaction_plan: CompactionPlan,
        current_turn: int,
    ) -> list[tuple[str, str]]:
        """Find artifacts referenced in message or tool calls.

        Args:
            agent_message: The agent's message text.
            pending_tool_calls: Tool calls the agent is making.
            compaction_plan: Current compaction plan.
            current_turn: Current turn number.

        Returns:
            List of (artifact_id, trigger) tuples.
        """
        needed: list[tuple[str, str]] = []
        seen_ids: set[str] = set()

        # Check for artifact ID references in message
        for ext in compaction_plan.externalizations:
            artifact_id = ext.artifact_id

            # Skip if already seen
            if artifact_id in seen_ids:
                continue

            # Check if in cooldown
            if self._in_cooldown(artifact_id, current_turn):
                logger.debug(
                    "Artifact %s in cooldown (last injected turn %d, current %d)",
                    artifact_id,
                    self._recent_injections.get(artifact_id, -1),
                    current_turn,
                )
                continue

            # Check if referenced in message
            if self._is_valid_artifact_ref(artifact_id, agent_message):
                needed.append((artifact_id, "message_reference"))
                seen_ids.add(artifact_id)
                continue

            # Check tool call arguments
            for tool_call in pending_tool_calls:
                if self._is_artifact_in_tool_call(artifact_id, tool_call):
                    needed.append((artifact_id, "tool_argument"))
                    seen_ids.add(artifact_id)
                    break

        # Limit to max per turn
        return needed[: self._config.max_artifacts_per_turn]

    def _is_valid_artifact_ref(self, artifact_id: str, text: str) -> bool:
        """Validate artifact reference to prevent injection attacks.

        Must be exact match, not substring of larger token.

        Args:
            artifact_id: The artifact ID to look for.
            text: Text to search in.

        Returns:
            True if artifact ID is validly referenced.
        """
        if not text or not artifact_id:
            return False

        # Word boundary match to prevent partial matches
        pattern = rf"\b{re.escape(artifact_id)}\b"
        return bool(re.search(pattern, text))

    def _is_artifact_in_tool_call(
        self,
        artifact_id: str,
        tool_call: ToolCall,
    ) -> bool:
        """Check if artifact ID appears in tool call arguments.

        Args:
            artifact_id: The artifact ID to look for.
            tool_call: The tool call to check.

        Returns:
            True if artifact ID is in the arguments.
        """
        for arg_value in tool_call.arguments.values():
            if isinstance(arg_value, str):
                if self._is_valid_artifact_ref(artifact_id, arg_value):
                    return True
            elif isinstance(arg_value, dict):
                # Check nested dict values (cast to suppress Unknown type warning)
                nested_dict = cast("dict[str, Any]", arg_value)
                for nested_value in nested_dict.values():
                    if isinstance(nested_value, str) and self._is_valid_artifact_ref(artifact_id, nested_value):
                        return True
        return False

    def _in_cooldown(self, artifact_id: str, current_turn: int) -> bool:
        """Check if artifact was recently injected.

        Args:
            artifact_id: The artifact ID.
            current_turn: Current turn number.

        Returns:
            True if artifact is in cooldown period.
        """
        last_turn = self._recent_injections.get(artifact_id)
        if last_turn is None:
            return False
        return current_turn - last_turn < self._config.cooldown_turns

    async def _retrieve_artifacts(
        self,
        needed_artifacts: list[tuple[str, str]],
        artifact_store: ArtifactStore,
        current_turn: int,
        security_context: SecurityContext | None,
    ) -> list[RehydrationResult]:
        """Retrieve artifact content with budget enforcement.

        Args:
            needed_artifacts: List of (artifact_id, trigger) tuples.
            artifact_store: Store for retrieving content.
            current_turn: Current turn number.
            security_context: Security context for access control.

        Returns:
            List of RehydrationResult with content.
        """
        results: list[RehydrationResult] = []
        tokens_used = 0

        for artifact_id, trigger in needed_artifacts:
            # Check sensitivity before retrieving content
            metadata = await artifact_store.get_metadata(artifact_id)
            if metadata is None:
                logger.debug("Artifact %s not found", artifact_id)
                continue

            # Sensitivity gating
            if not self._check_sensitivity(artifact_id, metadata, current_turn):
                continue

            # Retrieve content
            content = await artifact_store.retrieve(
                artifact_id,
                requester_context=security_context,
            )
            if content is None:
                logger.debug(
                    "Access denied or not found for artifact %s",
                    artifact_id,
                )
                continue

            # Count and truncate if needed
            token_count = self._tokenizer.count_tokens(content)
            truncated = False

            if token_count > self._config.max_tokens_per_artifact:
                content = self._truncate_content(
                    content,
                    self._config.max_tokens_per_artifact,
                )
                token_count = self._config.max_tokens_per_artifact
                truncated = True

            # Check total budget
            if tokens_used + token_count > self._config.total_budget_tokens:
                logger.debug(
                    "Budget exceeded, stopping rehydration (used=%d, need=%d, limit=%d)",
                    tokens_used,
                    token_count,
                    self._config.total_budget_tokens,
                )
                break

            # Add result
            results.append(
                RehydrationResult(
                    artifact_id=artifact_id,
                    content=content,
                    token_count=token_count,
                    truncated=truncated,
                ),
            )

            # Track injection
            tokens_used += token_count
            self._recent_injections[artifact_id] = current_turn

            # Emit event
            self._events.append(
                RehydrationEvent(
                    artifact_id=artifact_id,
                    token_count=token_count,
                    truncated=truncated,
                    trigger=trigger,
                    turn_number=current_turn,
                ),
            )

            logger.debug(
                "Rehydrated artifact %s (%d tokens, truncated=%s, trigger=%s)",
                artifact_id,
                token_count,
                truncated,
                trigger,
            )

        return results

    def _check_sensitivity(
        self,
        artifact_id: str,
        metadata: ArtifactMetadata,
        current_turn: int,
    ) -> bool:
        """Check if artifact can be auto-rehydrated based on sensitivity.

        Higher sensitivity levels (confidential, restricted) require explicit
        read_artifact() calls and cannot be auto-rehydrated.

        Args:
            artifact_id: The artifact ID.
            metadata: Artifact metadata with sensitivity.
            current_turn: Current turn number.

        Returns:
            True if auto-rehydration is allowed.
        """
        sensitivity = metadata.sensitivity

        if sensitivity not in self._config.auto_rehydrate_sensitivities:
            logger.debug(
                "Auto-rehydration blocked for artifact %s (sensitivity=%s)",
                artifact_id,
                sensitivity,
            )
            self._events.append(
                RehydrationBlockedEvent(
                    artifact_id=artifact_id,
                    reason=f"Sensitivity '{sensitivity}' requires explicit read_artifact() call",
                    turn_number=current_turn,
                ),
            )
            return False

        return True

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token budget.

        Uses a simple approach: estimate characters from tokens and truncate.
        Adds a truncation marker at the end.

        Args:
            content: The content to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated content with marker.
        """
        # Rough estimate: 4 characters per token (for English)
        # Reserve some tokens for the truncation marker
        marker = "\n\n[... content truncated ...]"
        marker_tokens = self._tokenizer.count_tokens(marker)
        available_tokens = max_tokens - marker_tokens

        if available_tokens <= 0:
            return marker

        # Binary search for right truncation point
        # Start with character estimate
        estimated_chars = available_tokens * 4
        if estimated_chars >= len(content):
            return content

        # Truncate and verify
        truncated = content[:estimated_chars]
        actual_tokens = self._tokenizer.count_tokens(truncated)

        # Adjust if needed (simple linear adjustment)
        while actual_tokens > available_tokens and len(truncated) > 100:
            # Remove ~10% of content
            truncated = truncated[: int(len(truncated) * 0.9)]
            actual_tokens = self._tokenizer.count_tokens(truncated)

        return truncated + marker

    def reset_cooldowns(self) -> None:
        """Reset all cooldown tracking.

        Call this when starting a new conversation or after significant
        context changes.
        """
        self._recent_injections.clear()
        logger.debug("Rehydration cooldowns reset")

    def get_injection_history(self) -> dict[str, int]:
        """Get the injection history.

        Returns:
            Dict mapping artifact_id to last injection turn.
        """
        return self._recent_injections.copy()


@dataclass
class RehydrationState:
    """Serializable state for RehydrationInterceptor.

    Used for checkpointing and resuming rehydration state.

    Attributes:
        recent_injections: Map of artifact_id to last injection turn.
        config: Rehydration configuration.
    """

    recent_injections: dict[str, int] = field(default_factory=dict)
    config: RehydrationConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "recent_injections": self.recent_injections,
            "config": self.config.to_dict() if self.config else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RehydrationState:
        """Deserialize from dictionary."""
        config = None
        if data.get("config"):
            config = RehydrationConfig.from_dict(data["config"])

        return cls(
            recent_injections=data.get("recent_injections", {}),
            config=config,
        )


def create_rehydration_interceptor(
    tokenizer: ProviderAwareTokenizer,
    *,
    enabled: bool = True,
    max_artifacts_per_turn: int = 3,
    max_tokens_per_artifact: int = 4000,
    total_budget_tokens: int = 8000,
    cooldown_turns: int = 2,
    auto_rehydrate_sensitivities: set[str] | None = None,
) -> RehydrationInterceptor:
    """Create a RehydrationInterceptor with specified configuration.

    Factory function for creating interceptors with common configurations.

    Args:
        tokenizer: Tokenizer for token counting.
        enabled: Whether rehydration is enabled.
        max_artifacts_per_turn: Maximum artifacts to inject per turn.
        max_tokens_per_artifact: Maximum tokens per artifact.
        total_budget_tokens: Total token budget for rehydration.
        cooldown_turns: Turns before re-injecting same artifact.
        auto_rehydrate_sensitivities: Sensitivity levels that can be auto-rehydrated.

    Returns:
        Configured RehydrationInterceptor.
    """
    if auto_rehydrate_sensitivities is None:
        auto_rehydrate_sensitivities = {"public", "internal"}

    config = RehydrationConfig(
        enabled=enabled,
        max_artifacts_per_turn=max_artifacts_per_turn,
        max_tokens_per_artifact=max_tokens_per_artifact,
        total_budget_tokens=total_budget_tokens,
        cooldown_turns=cooldown_turns,
        auto_rehydrate_sensitivities=auto_rehydrate_sensitivities,
    )

    return RehydrationInterceptor(config, tokenizer)
