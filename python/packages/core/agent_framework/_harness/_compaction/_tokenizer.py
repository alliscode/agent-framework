# Copyright (c) Microsoft. All rights reserved.

"""Provider-aware tokenization for context compaction.

This module provides accurate token counting that accounts for:
- Provider-specific message formatting overhead
- Tool schema serialization
- System prompt tokens
- Safety buffers and rehydration reserves

Tiktoken is used when available (recommended for production).
Falls back to character-based estimation when tiktoken is not installed.

Install tiktoken for accurate counting:
    pip install tiktoken

See CONTEXT_COMPACTION_DESIGN.md for full architecture details.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)

# Try to import tiktoken, fall back gracefully
_tiktoken_available = False
try:
    import tiktoken

    _tiktoken_available = True
except ImportError:
    tiktoken = None  # type: ignore
    logger.debug("tiktoken not available, using character-based estimation")

TIKTOKEN_AVAILABLE: bool = _tiktoken_available


class ModelProvider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    UNKNOWN = "unknown"


# Model to encoding mapping for tiktoken
# Based on OpenAI's model-to-encoding mappings
MODEL_ENCODING_MAP: dict[str, str] = {
    # GPT-4 models (cl100k_base)
    "gpt-4": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    "gpt-4-vision-preview": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-mini-2024-07-18": "o200k_base",
    # GPT-3.5 models (cl100k_base)
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",
    # O1 models
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    # Embedding models
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

# Default encoding for unknown models
DEFAULT_ENCODING = "cl100k_base"

# Provider-specific overhead constants
# These are approximate and based on empirical observations

# OpenAI message overhead (tokens per message)
OPENAI_MESSAGE_OVERHEAD = 4  # <|im_start|>role\n...<|im_end|>
OPENAI_REPLY_OVERHEAD = 3  # assistant response priming

# Anthropic message overhead (approximate)
ANTHROPIC_MESSAGE_OVERHEAD = 3
ANTHROPIC_REPLY_OVERHEAD = 2

# Tool schema overhead (approximate tokens per tool)
TOOL_SCHEMA_BASE_OVERHEAD = 10  # Function definition wrapper
TOOL_PARAM_OVERHEAD = 5  # Per parameter


@runtime_checkable
class ProviderAwareTokenizer(Protocol):
    """Tokenizer that understands provider-specific formatting.

    This protocol defines the interface for tokenizers that can accurately
    count tokens including all provider-specific overhead.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens in raw text.

        Args:
            text: The text to tokenize.

        Returns:
            Token count.
        """
        ...

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens for a message including role overhead.

        Args:
            message: A chat message dict with 'role' and 'content'.

        Returns:
            Token count including role/formatting overhead.
        """
        ...

    def count_tool_schemas(self, tools: list[dict[str, Any]]) -> int:
        """Count tokens for tool schema injection.

        Args:
            tools: List of tool definitions (OpenAI function format).

        Returns:
            Token count for all tool schemas.
        """
        ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a list of messages.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count for all messages.
        """
        ...


class TiktokenTokenizer:
    """Tokenizer using tiktoken for accurate OpenAI-compatible token counting.

    This tokenizer provides accurate counts for OpenAI and Azure OpenAI models.
    For other providers, it provides reasonable approximations.

    Attributes:
        model: The model name for encoding selection.
        provider: The model provider.
        encoding: The tiktoken encoding being used.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: ModelProvider = ModelProvider.OPENAI,
    ):
        """Initialize the tokenizer.

        Args:
            model: Model name for encoding selection.
            provider: The model provider.

        Raises:
            ImportError: If tiktoken is not installed.
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for TiktokenTokenizer. Install it with: pip install tiktoken")

        self.model = model
        self.provider = provider
        self._encoding = self._get_encoding(model)

    def _get_encoding(self, model: str) -> Any:
        """Get the tiktoken encoding for a model.

        Args:
            model: The model name.

        Returns:
            A tiktoken Encoding object.
        """
        # tiktoken is guaranteed to be non-None here because __init__ raises
        # ImportError if tiktoken is not available
        if tiktoken is None:
            raise ImportError("tiktoken is required but not available")
        encoding_name = MODEL_ENCODING_MAP.get(model, DEFAULT_ENCODING)
        return tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in raw text.

        Args:
            text: The text to tokenize.

        Returns:
            Token count.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens for a message including role overhead.

        Follows OpenAI's token counting methodology for chat messages.

        Args:
            message: A chat message dict with 'role' and 'content'.

        Returns:
            Token count including role/formatting overhead.
        """
        tokens = 0

        # Role token
        role = message.get("role", "")
        tokens += self.count_tokens(role)

        # Content tokens
        content: str | list[Any] | None = message.get("content", "")
        if isinstance(content, str):
            tokens += self.count_tokens(content)
        elif isinstance(content, list):
            # Multi-part content (e.g., with images)
            for part in content:
                if isinstance(part, dict):
                    part_dict = cast("dict[str, Any]", part)
                    part_type = part_dict.get("type")
                    if part_type == "text":
                        text_content = part_dict.get("text")
                        if isinstance(text_content, str):
                            tokens += self.count_tokens(text_content)
                    elif part_type == "image_url":
                        # Images have variable token cost based on resolution
                        # Use a reasonable default
                        tokens += 85  # Low-res image default
                elif isinstance(part, str):
                    tokens += self.count_tokens(part)

        # Name field if present
        if "name" in message:
            tokens += self.count_tokens(message["name"])
            tokens += 1  # Separator

        # Tool calls if present
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tokens += self.count_tokens(tool_call.get("id", ""))
                func = tool_call.get("function", {})
                tokens += self.count_tokens(func.get("name", ""))
                tokens += self.count_tokens(func.get("arguments", ""))
                tokens += 5  # Tool call structure overhead

        # Tool call ID if present (for tool results)
        if "tool_call_id" in message:
            tokens += self.count_tokens(message["tool_call_id"])

        # Message overhead based on provider
        if self.provider in (ModelProvider.OPENAI, ModelProvider.AZURE_OPENAI):
            tokens += OPENAI_MESSAGE_OVERHEAD
        elif self.provider == ModelProvider.ANTHROPIC:
            tokens += ANTHROPIC_MESSAGE_OVERHEAD
        else:
            tokens += OPENAI_MESSAGE_OVERHEAD  # Default

        return tokens

    def count_tool_schemas(self, tools: list[dict[str, Any]]) -> int:
        """Count tokens for tool schema injection.

        Args:
            tools: List of tool definitions (OpenAI function format).

        Returns:
            Token count for all tool schemas.
        """
        if not tools:
            return 0

        total = 0
        for tool in tools:
            # Serialize the tool definition
            tool_str = json.dumps(tool, separators=(",", ":"))
            total += self.count_tokens(tool_str)
            total += TOOL_SCHEMA_BASE_OVERHEAD

            # Count parameters if present
            if "function" in tool:
                params = tool["function"].get("parameters", {})
                if "properties" in params:
                    total += len(params["properties"]) * TOOL_PARAM_OVERHEAD

        return total

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a list of messages.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count for all messages.
        """
        total = sum(self.count_message(msg) for msg in messages)

        # Add reply priming overhead
        if self.provider in (ModelProvider.OPENAI, ModelProvider.AZURE_OPENAI):
            total += OPENAI_REPLY_OVERHEAD
        elif self.provider == ModelProvider.ANTHROPIC:
            total += ANTHROPIC_REPLY_OVERHEAD

        return total


class SimpleTokenizer:
    """Simple character-based tokenizer for testing and fallback.

    This provides a rough approximation (1 token ≈ 4 characters for English).
    Use TiktokenTokenizer for production accuracy.

    The approximation is intentionally conservative (may overcount) to avoid
    accidentally exceeding token limits.
    """

    def __init__(
        self,
        chars_per_token: float = 4.0,
        provider: ModelProvider = ModelProvider.UNKNOWN,
    ):
        """Initialize the tokenizer.

        Args:
            chars_per_token: Average characters per token (default 4.0).
            provider: The model provider (for overhead calculations).
        """
        self.chars_per_token = chars_per_token
        self.provider = provider

    def count_tokens(self, text: str) -> int:
        """Count tokens using character approximation.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # Round up to be conservative
        return max(1, int(len(text) / self.chars_per_token + 0.5))

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens for a message including role overhead.

        Args:
            message: A chat message dict.

        Returns:
            Estimated token count.
        """
        tokens = 0

        # Role
        role = message.get("role", "")
        if isinstance(role, str):
            tokens += self.count_tokens(role)

        # Content
        content: str | list[Any] | None = message.get("content", "")
        if isinstance(content, str):
            tokens += self.count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    tokens += self.count_tokens(json.dumps(part))
                elif isinstance(part, str):
                    tokens += self.count_tokens(part)

        # Tool calls
        if "tool_calls" in message:
            tokens += self.count_tokens(json.dumps(message["tool_calls"]))

        # Overhead
        tokens += OPENAI_MESSAGE_OVERHEAD

        return tokens

    def count_tool_schemas(self, tools: list[dict[str, Any]]) -> int:
        """Count tokens for tool schemas.

        Args:
            tools: List of tool definitions.

        Returns:
            Estimated token count.
        """
        if not tools:
            return 0
        return self.count_tokens(json.dumps(tools))

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a list of messages.

        Args:
            messages: List of chat messages.

        Returns:
            Estimated total token count.
        """
        return sum(self.count_message(msg) for msg in messages) + OPENAI_REPLY_OVERHEAD


def get_tokenizer(
    model: str = "gpt-4o",
    provider: ModelProvider = ModelProvider.OPENAI,
    fallback_to_simple: bool = True,
) -> ProviderAwareTokenizer:
    """Get a tokenizer for the specified model.

    Args:
        model: The model name.
        provider: The model provider.
        fallback_to_simple: If True, fall back to SimpleTokenizer when
            tiktoken is not available. If False, raise ImportError.

    Returns:
        A ProviderAwareTokenizer instance.

    Raises:
        ImportError: If tiktoken is not available and fallback_to_simple is False.
    """
    if TIKTOKEN_AVAILABLE:
        return TiktokenTokenizer(model=model, provider=provider)
    if fallback_to_simple:
        logger.warning(
            "tiktoken not available, using SimpleTokenizer. "
            "Install tiktoken for accurate token counting: pip install tiktoken",
        )
        return SimpleTokenizer(provider=provider)
    raise ImportError("tiktoken is required for accurate token counting. Install it with: pip install tiktoken")


@dataclass
class TokenBudget:
    """Token budget that accounts for all overhead.

    This is the v2 budget for compaction internals. It tracks not just message
    tokens but all the overhead that goes into a request: system prompt, tool
    schemas, formatting, and reserves for response and rehydration. For the
    simpler SharedState pressure signaling budget, see
    ``_context_pressure.TokenBudget``.

    Attributes:
        max_input_tokens: Maximum tokens allowed in input.
        soft_threshold_percent: Percentage at which to trigger compaction.
        system_prompt_tokens: Tokens used by system prompt.
        tool_schema_tokens: Tokens used by tool schemas.
        formatting_overhead_tokens: Provider-specific formatting overhead.
        safety_buffer_tokens: Buffer for response and safety margin.
        rehydration_reserve_tokens: Reserved space for rehydration (not yet implemented).
        max_rehydration_artifacts: Maximum artifacts to rehydrate (not yet implemented).
    """

    max_input_tokens: int = 128_000
    soft_threshold_percent: float = 0.85

    # Overhead tracking
    system_prompt_tokens: int = 0
    tool_schema_tokens: int = 0
    formatting_overhead_tokens: int = 0
    safety_buffer_tokens: int = 500

    # Reservation for rehydration (not yet implemented — kept because it affects budget math)
    rehydration_reserve_tokens: int = 2000
    max_rehydration_artifacts: int = 3

    @property
    def total_overhead(self) -> int:
        """Calculate total overhead tokens."""
        return (
            self.system_prompt_tokens
            + self.tool_schema_tokens
            + self.formatting_overhead_tokens
            + self.safety_buffer_tokens
            + self.rehydration_reserve_tokens
        )

    @property
    def soft_threshold(self) -> int:
        """Calculate the soft threshold in tokens."""
        return int(self.max_input_tokens * self.soft_threshold_percent)

    @property
    def available_for_messages(self) -> int:
        """Tokens available for conversation messages."""
        return max(0, self.soft_threshold - self.total_overhead)

    @property
    def available_for_rehydration(self) -> int:
        """Tokens available for auto-rehydrated content."""
        return self.rehydration_reserve_tokens

    def is_under_pressure(self, current_tokens: int) -> bool:
        """Check if context is under pressure.

        Args:
            current_tokens: Current token count of messages.

        Returns:
            True if above soft threshold.
        """
        return current_tokens >= self.available_for_messages

    def tokens_over_threshold(self, current_tokens: int) -> int:
        """Calculate how many tokens over the threshold.

        Args:
            current_tokens: Current token count of messages.

        Returns:
            Number of tokens over threshold (0 if under).
        """
        return max(0, current_tokens - self.available_for_messages)

    def update_from_request(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tokenizer: ProviderAwareTokenizer,
    ) -> None:
        """Update overhead tracking from a request.

        Args:
            system_prompt: The system prompt text.
            tools: List of tool definitions.
            tokenizer: Tokenizer for counting.
        """
        self.system_prompt_tokens = tokenizer.count_tokens(system_prompt)
        self.tool_schema_tokens = tokenizer.count_tool_schemas(tools)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_input_tokens": self.max_input_tokens,
            "soft_threshold_percent": self.soft_threshold_percent,
            "system_prompt_tokens": self.system_prompt_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "formatting_overhead_tokens": self.formatting_overhead_tokens,
            "safety_buffer_tokens": self.safety_buffer_tokens,
            "rehydration_reserve_tokens": self.rehydration_reserve_tokens,
            "max_rehydration_artifacts": self.max_rehydration_artifacts,
            # Computed values
            "total_overhead": self.total_overhead,
            "soft_threshold": self.soft_threshold,
            "available_for_messages": self.available_for_messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenBudget:
        """Deserialize from dictionary."""
        return cls(
            max_input_tokens=data.get("max_input_tokens", 128_000),
            soft_threshold_percent=data.get("soft_threshold_percent", 0.85),
            system_prompt_tokens=data.get("system_prompt_tokens", 0),
            tool_schema_tokens=data.get("tool_schema_tokens", 0),
            formatting_overhead_tokens=data.get("formatting_overhead_tokens", 0),
            safety_buffer_tokens=data.get("safety_buffer_tokens", 500),
            rehydration_reserve_tokens=data.get("rehydration_reserve_tokens", 2000),
            max_rehydration_artifacts=data.get("max_rehydration_artifacts", 3),
        )

    @classmethod
    def for_model(
        cls,
        model: str,
        *,
        soft_threshold_percent: float = 0.85,
        safety_buffer_tokens: int = 500,
        rehydration_reserve_tokens: int = 2000,
    ) -> TokenBudget:
        """Create a TokenBudget with appropriate limits for a model.

        Args:
            model: The model name.
            soft_threshold_percent: When to trigger compaction.
            safety_buffer_tokens: Safety buffer for response.
            rehydration_reserve_tokens: Reserve for rehydration.

        Returns:
            A TokenBudget configured for the model.
        """
        # Model context window sizes (approximate)
        # Order matters: more specific patterns should come first
        model_limits: list[tuple[str, int]] = [
            # GPT-4o variants (must come before gpt-4)
            ("gpt-4o-mini", 128_000),
            ("gpt-4o", 128_000),
            # GPT-4 turbo variants (must come before gpt-4)
            ("gpt-4-turbo", 128_000),
            ("gpt-4-32k", 32_768),
            ("gpt-4", 8_192),
            # GPT-3.5 variants
            ("gpt-3.5-turbo-16k", 16_385),
            ("gpt-3.5-turbo", 16_385),
            # O1 variants
            ("o1-preview", 128_000),
            ("o1-mini", 128_000),
            ("o1", 200_000),
            # Claude variants (approximate)
            ("claude-3-5-sonnet", 200_000),
            ("claude-3-opus", 200_000),
            ("claude-3-sonnet", 200_000),
            ("claude-3-haiku", 200_000),
            ("claude-2", 100_000),
        ]

        # Find matching model or use default
        max_tokens = 128_000  # Default
        model_lower = model.lower()
        for pattern, limit in model_limits:
            if model_lower.startswith(pattern) or pattern in model_lower:
                max_tokens = limit
                break

        return cls(
            max_input_tokens=max_tokens,
            soft_threshold_percent=soft_threshold_percent,
            safety_buffer_tokens=safety_buffer_tokens,
            rehydration_reserve_tokens=rehydration_reserve_tokens,
        )
