# Copyright (c) Microsoft. All rights reserved.

"""Shared compaction view utilities.

These helpers apply compaction records to message sequences while preserving
tool call/result pairing invariants required by model APIs.
"""

from __future__ import annotations

import uuid
from typing import Any

from ._compaction import CompactionPlan


def ensure_message_ids(messages: list[Any]) -> None:
    """Assign deterministic harness message ids where missing."""
    for msg in messages:
        if getattr(msg, "message_id", None) is None:
            msg.message_id = f"msg-{uuid.uuid4().hex[:12]}"


def apply_compaction_plan_to_messages(cache: list[Any], plan: CompactionPlan) -> list[Any]:
    """Apply a compaction plan and return a compacted message view."""
    from ._compaction import CompactionAction

    compacted: list[Any] = []
    processed_spans: set[str] = set()

    for msg in cache:
        msg_id = getattr(msg, "message_id", None)
        if msg_id is None:
            compacted.append(msg)
            continue

        action, record = plan.get_action(msg_id)
        if action == CompactionAction.DROP:
            continue
        if action == CompactionAction.INCLUDE:
            compacted.append(msg)
            continue
        if action == CompactionAction.CLEAR:
            compacted.append(_render_cleared_message(msg, record))
            continue

        if record is None:
            continue
        span_key = record.span.start_message_id
        if span_key in processed_spans:
            continue
        processed_spans.add(span_key)

        if action == CompactionAction.SUMMARIZE:
            compacted.append(_render_summary_message(record))
        elif action == CompactionAction.EXTERNALIZE:
            compacted.append(_render_externalization_message(record))

    return fix_dangling_tool_calls(compacted)


def apply_direct_clear_to_messages(
    messages: list[Any],
    *,
    preserve_recent_messages: int = 2,
    target_tokens_to_free: int = 0,
) -> tuple[list[Any], int, int]:
    """Clear older large tool results directly and return updated view + metrics."""
    from .._types import ChatMessage, FunctionResultContent

    working = list(messages)
    if len(working) <= preserve_recent_messages:
        return fix_dangling_tool_calls(working), 0, 0

    cleared_count = 0
    tokens_freed_estimate = 0
    clearable_end = max(0, len(working) - preserve_recent_messages)

    for i in range(clearable_end):
        if target_tokens_to_free > 0 and tokens_freed_estimate >= target_tokens_to_free:
            break
        msg = working[i]
        contents = getattr(msg, "contents", None) or []
        if not contents:
            continue
        new_contents: list[Any] = []
        modified = False
        for content in contents:
            if isinstance(content, FunctionResultContent):
                result_str = str(content.result or "")
                if len(result_str) > 100:
                    if target_tokens_to_free > 0 and tokens_freed_estimate >= target_tokens_to_free:
                        new_contents.append(content)
                        continue
                    placeholder = "[Tool result cleared to save context]"
                    tokens_freed_estimate += max(0, len(result_str) - len(placeholder)) // 4
                    content = FunctionResultContent(call_id=content.call_id, result=placeholder)
                    modified = True
                    cleared_count += 1
            new_contents.append(content)
        if modified:
            role = getattr(msg, "role", "tool")
            role_value: str = str(getattr(role, "value", role))
            working[i] = ChatMessage(
                role=role_value,  # type: ignore[arg-type]
                contents=new_contents,
                message_id=getattr(msg, "message_id", None),
            )

    return fix_dangling_tool_calls(working), cleared_count, tokens_freed_estimate


def fix_dangling_tool_calls(messages: list[Any]) -> list[Any]:
    """Repair assistant tool-calls/tool-results pairing after compaction."""
    from .._types import ChatMessage, FunctionCallContent, FunctionResultContent

    assistant_call_ids: set[str] = set()
    answered_call_ids: set[str] = set()
    for msg in messages:
        contents = getattr(msg, "contents", None) or []
        for content in contents:
            if isinstance(content, FunctionCallContent) and content.call_id:
                assistant_call_ids.add(content.call_id)
            elif isinstance(content, FunctionResultContent) and content.call_id:
                answered_call_ids.add(content.call_id)

    result: list[Any] = []
    for msg in messages:
        role_value = str(getattr(getattr(msg, "role", ""), "value", getattr(msg, "role", "")))
        if role_value == "tool":
            contents = getattr(msg, "contents", None) or []
            tool_call_ids = [c.call_id for c in contents if isinstance(c, FunctionResultContent) and c.call_id]
            if tool_call_ids and all(cid not in assistant_call_ids for cid in tool_call_ids):
                continue

        result.append(msg)
        contents = getattr(msg, "contents", None) or []
        orphaned_ids = [
            c.call_id
            for c in contents
            if isinstance(c, FunctionCallContent) and c.call_id and c.call_id not in answered_call_ids
        ]
        for call_id in orphaned_ids:
            result.append(
                ChatMessage(
                    role="tool",
                    contents=[
                        FunctionResultContent(
                            call_id=call_id,
                            result="[Tool result cleared during context compaction]",
                        )
                    ],
                    message_id=f"placeholder-{call_id[:12]}",
                )
            )
    return result


def _render_cleared_message(original_msg: Any, record: Any) -> Any:
    from .._types import ChatMessage, FunctionResultContent

    role_attr = getattr(original_msg, "role", "assistant")
    role_value: str = str(getattr(role_attr, "value", role_attr))
    parts = [f"[Cleared: {role_value} message]"]
    if record is not None and hasattr(record, "preserved_fields") and record.preserved_fields:
        fields = ", ".join(f"{k}={v}" for k, v in sorted(record.preserved_fields.items()))
        parts.append(f"Key data: {fields}")
    placeholder_text = "\n".join(parts)

    if role_value == "tool":
        contents = getattr(original_msg, "contents", None) or []
        for content in contents:
            if isinstance(content, FunctionResultContent):
                return ChatMessage(
                    role="tool",
                    contents=[
                        FunctionResultContent(
                            call_id=content.call_id,
                            result=placeholder_text,
                        )
                    ],
                    message_id=getattr(original_msg, "message_id", None),
                )

    return ChatMessage(  # type: ignore[call-overload]
        role=role_value,  # type: ignore[arg-type]
        text=placeholder_text,
        message_id=getattr(original_msg, "message_id", None),
    )


def _render_summary_message(record: Any) -> Any:
    from .._types import ChatMessage

    span = record.span
    summary_text = record.summary.render_as_message()
    content = f"[Context Summary - Turns {span.first_turn}-{span.last_turn}]\n{summary_text}"
    return ChatMessage(
        role="assistant",
        text=content,
        message_id=f"summary-{span.start_message_id}",
    )


def _render_externalization_message(record: Any) -> Any:
    from .._types import ChatMessage

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
