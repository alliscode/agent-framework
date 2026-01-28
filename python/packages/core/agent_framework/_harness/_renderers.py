# Copyright (c) Microsoft. All rights reserved.

"""Pluggable renderers for Agent Harness event streams.

Renderers wrap the harness event stream and transform lifecycle events
into formatted text that can be displayed in different environments
(terminals, DevUI, markdown viewers, etc.).

Usage:
    from agent_framework._harness import AgentHarness, MarkdownRenderer, render_stream

    harness = AgentHarness(agent, ...)
    renderer = MarkdownRenderer()

    async for event in render_stream(harness, "Do the task", renderer):
        # Events now include injected formatted text for progress/activity
        ...
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .._types import AgentRunResponseUpdate
from .._workflows._events import AgentRunUpdateEvent, WorkflowOutputEvent
from ._state import HarnessLifecycleEvent, HarnessResult

# Activity verbs cycled through for turn indicators
ACTIVITY_VERBS = [
    "reasoning",
    "analyzing",
    "synthesizing",
    "drafting",
    "evaluating",
    "connecting",
    "composing",
    "reviewing",
    "considering",
    "structuring",
]


def _progress_bar_text(done: int, total: int, width: int = 20) -> str:
    """Generate a text-based progress bar."""
    if total == 0:
        return ""
    progress = done / total
    filled = int(width * progress)
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    pct = int(progress * 100)
    return f"[{bar}] {done}/{total} complete ({pct}%)"


# ── Renderer Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class HarnessRenderer(Protocol):
    """Protocol for harness event stream renderers.

    Each method receives event data and returns formatted text to inject
    into the stream, or None to suppress output for that event.
    """

    def on_turn_started(self, turn_number: int) -> str | None:
        """Called when a new agent turn begins.

        Args:
            turn_number: The 1-based turn number.

        Returns:
            Formatted text to inject, or None.
        """
        ...

    def on_deliverables_updated(self, data: dict[str, Any]) -> str | None:
        """Called when work item progress changes.

        Args:
            data: Event data with keys: count, total_items, done_items, items.

        Returns:
            Formatted text to inject, or None.
        """
        ...

    def on_result(self, result: HarnessResult) -> str | None:
        """Called when the harness produces a final result.

        Args:
            result: The final HarnessResult.

        Returns:
            Formatted text to inject, or None.
        """
        ...

    def on_text(self, text: str) -> str | None:
        """Called for each streaming text chunk from the agent.

        Args:
            text: The text chunk.

        Returns:
            Transformed text, or None to suppress.
        """
        ...


# ── Passthrough Renderer ───────────────────────────────────────────────────


class PassthroughRenderer:
    """Renderer that only passes through agent text, suppressing all chrome.

    Use this when you want raw agent output without any progress indicators,
    activity verbs, or deliverable formatting.
    """

    def on_turn_started(self, turn_number: int) -> str | None:
        """Suppress turn indicators."""
        return None

    def on_deliverables_updated(self, data: dict[str, Any]) -> str | None:
        """Suppress progress updates."""
        return None

    def on_result(self, result: HarnessResult) -> str | None:
        """Suppress result summary."""
        return None

    def on_text(self, text: str) -> str | None:
        """Pass through text unchanged."""
        return text


# ── Markdown Renderer ──────────────────────────────────────────────────────


class MarkdownRenderer:
    """Renderer that formats harness events as Markdown.

    Suitable for DevUI or any markdown-capable display. Produces:
    - Activity verb indicators as bold italic text
    - Progress bars as inline monospace text
    - Deliverable blocks as quoted markdown sections
    - Final result summary with pass/fail status
    """

    def __init__(self, show_deliverables_inline: bool = True) -> None:
        """Initialize the markdown renderer.

        Args:
            show_deliverables_inline: If True, render new deliverables inline
                as they appear. If False, only show them in the final summary.
        """
        self._show_deliverables_inline = show_deliverables_inline
        self._seen_deliverable_ids: set[str] = set()
        self._last_done: int = -1

    def on_turn_started(self, turn_number: int) -> str | None:
        """Render an activity verb indicator."""
        verb = ACTIVITY_VERBS[(turn_number - 1) % len(ACTIVITY_VERBS)]
        return f"\n\n***● {verb}...***\n\n"

    def on_deliverables_updated(self, data: dict[str, Any]) -> str | None:
        """Render progress bar and any new deliverables."""
        done = data.get("done_items", 0)
        total = data.get("total_items", 0)

        parts: list[str] = []

        # Only show progress bar if it changed
        if total > 0 and done != self._last_done:
            self._last_done = done
            bar = _progress_bar_text(done, total)
            parts.append(f"\n`{bar}`\n")

        # Render new deliverables inline
        if self._show_deliverables_inline:
            for item in data.get("items", []):
                item_id = item.get("item_id", "")
                if item_id and item_id not in self._seen_deliverable_ids:
                    self._seen_deliverable_ids.add(item_id)
                    title = item.get("title", "Untitled")
                    content = item.get("content", "")
                    parts.append(self._format_deliverable(title, content))

        return "".join(parts) if parts else None

    def on_result(self, result: HarnessResult) -> str | None:
        """Render final result summary."""
        status_icon = "✓" if result.status.value == "done" else "✗"
        parts = [f"\n\n---\n\n**{status_icon} Complete** ({result.turn_count} turns)\n"]

        # Show any deliverables not already rendered inline
        unseen = [
            d for d in result.deliverables
            if d.get("item_id") not in self._seen_deliverable_ids
        ]
        if unseen:
            parts.append(f"\n**Deliverables ({len(unseen)}):**\n")
            for d in unseen:
                parts.append(self._format_deliverable(
                    d.get("title", "Untitled"),
                    d.get("content", ""),
                ))

        return "".join(parts)

    def on_text(self, text: str) -> str | None:
        """Pass through text unchanged."""
        return text

    def _format_deliverable(self, title: str, content: str) -> str:
        """Format a single deliverable as a markdown block."""
        # Indent content as a blockquote
        quoted = "\n".join(f"> {line}" for line in content.split("\n"))
        return f"\n#### {title}\n\n{quoted}\n"


# ── Stream Wrapper ─────────────────────────────────────────────────────────


def _make_text_event(text: str) -> AgentRunUpdateEvent:
    """Create a synthetic AgentRunUpdateEvent with text content."""
    update = AgentRunResponseUpdate(text=text)
    return AgentRunUpdateEvent(executor_id="harness_renderer", data=update)


async def render_stream(
    harness: Any,
    message: str,
    renderer: HarnessRenderer,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Wrap a harness run_stream with a renderer.

    This async generator consumes events from harness.run_stream(),
    passes them through the renderer for formatting, and yields:
    - Original events (for consumers that need raw lifecycle data)
    - Injected text events for rendered output

    Args:
        harness: An AgentHarness instance.
        message: The initial message to send.
        renderer: The renderer to apply.
        **kwargs: Additional kwargs passed to harness.run_stream().

    Yields:
        Workflow events, with injected text events from the renderer.
    """
    async for event in harness.run_stream(message, **kwargs):
        # Handle lifecycle events
        if isinstance(event, HarnessLifecycleEvent):
            if event.event_type == "turn_started":
                turn_number = (
                    event.data.get("turn_number", 1) if event.data else 1
                )
                text = renderer.on_turn_started(turn_number)
                if text:
                    yield _make_text_event(text)

            elif event.event_type == "deliverables_updated" and event.data:
                text = renderer.on_deliverables_updated(event.data)
                if text:
                    yield _make_text_event(text)

            # Always yield the original lifecycle event too
            yield event

        # Handle streaming text
        elif isinstance(event, AgentRunUpdateEvent) and event.data:
            update: AgentRunResponseUpdate = event.data
            if hasattr(update, "text") and update.text:
                text = renderer.on_text(update.text)
                if text:
                    if text == update.text:
                        yield event  # unchanged, pass original
                    else:
                        yield _make_text_event(text)
                # If None, suppress the event entirely
            else:
                yield event  # non-text update, pass through

        # Handle final result
        elif isinstance(event, WorkflowOutputEvent) and isinstance(event.data, HarnessResult):
            text = renderer.on_result(event.data)
            if text:
                yield _make_text_event(text)
            yield event  # always yield the original result event

        # Everything else passes through unchanged
        else:
            yield event
