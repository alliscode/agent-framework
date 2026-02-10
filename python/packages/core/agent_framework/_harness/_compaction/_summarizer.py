# Copyright (c) Microsoft. All rights reserved.

"""ChatClient-based Summarizer implementation for context compaction.

Provides a concrete Summarizer that uses a ChatClientProtocol to produce
StructuredSummary objects from conversation messages via LLM calls.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ._summary import (
    ArtifactReference,
    Decision,
    OpenItem,
    StructuredSummary,
    ToolOutcome,
)
from ._types import SpanReference

if TYPE_CHECKING:
    from ..._clients import ChatClientProtocol
    from ..._types import ChatMessage

logger = logging.getLogger(__name__)

# Prompt for structured summarization
_SUMMARIZE_SYSTEM_PROMPT = """\
You are a context compaction assistant. Your job is to summarize a block of \
conversation messages into a structured JSON summary that preserves the most \
important information.

Produce a JSON object with these fields (omit empty arrays/nulls):
{{
  "facts": ["stable facts, user preferences, constraints discovered"],
  "decisions": [{{"decision": "...", "rationale": "...", "turn_number": 0}}],
  "open_items": [{{"description": "...", "context": "...", "priority": "high|medium|low"}}],
  "tool_outcomes": [{{"tool_name": "...", "outcome": "success|failure|partial", \
"key_fields": {{}}, "error_message": null}}],
  "current_task": "what the agent is currently working on (string or null)",
  "current_plan": ["step 1", "step 2"]
}}

Rules:
- Preserve ALL decisions and their rationale.
- Preserve ALL unresolved/open items.
- For tool outcomes, keep the tool name, success/failure, and any key identifiers \
(file paths, IDs, counts) — drop verbose output.
- Facts should be concise one-liners.
- Target roughly {target_pct}% of the original token count.
- Respond with ONLY valid JSON, no markdown fencing.
"""


class ChatClientSummarizer:
    """Summarizer that uses a ChatClientProtocol to produce structured summaries.

    This calls the LLM with a summarization prompt and parses the response
    into a StructuredSummary. It is designed for use with cheaper/faster models
    to avoid impacting the main conversation budget.

    Attributes:
        chat_client: The chat client used for summarization calls.
    """

    def __init__(
        self,
        chat_client: "ChatClientProtocol",
        *,
        model_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the ChatClientSummarizer.

        Args:
            chat_client: Chat client to use for summarization LLM calls.
            model_id: Optional model override for summarization (e.g. a cheaper model).
            temperature: Temperature for summarization calls. Low for determinism.
            max_tokens: Max tokens for the summary response.
        """
        self._chat_client = chat_client
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def summarize(
        self,
        messages: list[ChatMessage],
        *,
        target_token_ratio: float = 0.25,
        preserve_facts: list[str] | None = None,
    ) -> StructuredSummary:
        """Summarize messages into a StructuredSummary via LLM call.

        Args:
            messages: The messages to summarize.
            target_token_ratio: Target ratio of summary to original tokens.
            preserve_facts: Facts that must be preserved in the summary.

        Returns:
            A StructuredSummary of the messages.
        """
        from ..._types import ChatMessage as CM

        # Build the conversation transcript for the summarizer
        transcript_lines: list[str] = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            text = msg.text or ""
            transcript_lines.append(f"[{role}] {text[:2000]}")

        transcript = "\n".join(transcript_lines)

        target_pct = int(target_token_ratio * 100)
        system_prompt = _SUMMARIZE_SYSTEM_PROMPT.format(target_pct=target_pct)

        if preserve_facts:
            system_prompt += "\n\nFacts that MUST appear in the summary:\n"
            for fact in preserve_facts:
                system_prompt += f"- {fact}\n"

        user_prompt = f"Summarize these conversation messages:\n\n{transcript}"

        call_kwargs: dict[str, Any] = {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._model_id:
            call_kwargs["model_id"] = self._model_id

        try:
            response = await self._chat_client.get_response(
                [
                    CM(role="system", text=system_prompt),
                    CM(role="user", text=user_prompt),
                ],
                **call_kwargs,
            )
            response_text = ""
            for resp_msg in response.messages:
                if resp_msg.text:
                    response_text += resp_msg.text
        except Exception:
            logger.warning("ChatClientSummarizer: LLM call failed, returning empty summary", exc_info=True)
            return self._empty_summary(messages)

        return self._parse_summary(response_text, messages, preserve_facts)

    def _parse_summary(
        self,
        response_text: str,
        messages: list[ChatMessage],
        preserve_facts: list[str] | None,
    ) -> StructuredSummary:
        """Parse LLM response JSON into a StructuredSummary.

        Falls back to an empty summary if parsing fails.
        """
        # Strip markdown fencing if present
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fencing)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("ChatClientSummarizer: Failed to parse JSON response, returning empty summary")
            return self._empty_summary(messages)

        # Build span from message IDs
        message_ids = [msg.message_id for msg in messages if msg.message_id]
        span = SpanReference(
            message_ids=message_ids,
            first_turn=0,
            last_turn=len(messages) - 1,
        )

        # Parse fields
        facts = data.get("facts", [])
        if preserve_facts:
            for fact in preserve_facts:
                if fact not in facts:
                    facts.append(fact)

        decisions = []
        for d in data.get("decisions", []):
            if isinstance(d, dict) and "decision" in d:
                decisions.append(
                    Decision(
                        decision=d["decision"],
                        rationale=d.get("rationale", ""),
                        turn_number=d.get("turn_number", 0),
                    )
                )

        open_items = []
        for item in data.get("open_items", []):
            if isinstance(item, dict) and "description" in item:
                open_items.append(
                    OpenItem(
                        description=item["description"],
                        context=item.get("context", ""),
                        priority=item.get("priority", "medium"),
                    )
                )

        tool_outcomes = []
        for t in data.get("tool_outcomes", []):
            if isinstance(t, dict) and "tool_name" in t:
                tool_outcomes.append(
                    ToolOutcome(
                        tool_name=t["tool_name"],
                        outcome=t.get("outcome", "success"),
                        key_fields=t.get("key_fields", {}),
                        error_message=t.get("error_message"),
                    )
                )

        # Artifact references (usually empty from summarization)
        artifacts = []
        for a in data.get("artifacts", []):
            if isinstance(a, dict) and "artifact_id" in a:
                artifacts.append(
                    ArtifactReference(
                        artifact_id=a["artifact_id"],
                        description=a.get("description", ""),
                        rehydrate_hint=a.get("rehydrate_hint", ""),
                    )
                )

        return StructuredSummary(
            span=span,
            facts=facts,
            decisions=decisions,
            open_items=open_items,
            artifacts=artifacts,
            tool_outcomes=tool_outcomes,
            current_task=data.get("current_task"),
            current_plan=data.get("current_plan"),
        )

    def _empty_summary(self, messages: list[ChatMessage]) -> StructuredSummary:
        """Create an empty summary as fallback."""
        message_ids = [msg.message_id for msg in messages if msg.message_id]
        span = SpanReference(
            message_ids=message_ids,
            first_turn=0,
            last_turn=len(messages) - 1,
        )
        return StructuredSummary.create_empty(span)
