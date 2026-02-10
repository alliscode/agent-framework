# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Agent Harness pluggable renderers."""

import pytest

from agent_framework._harness import (
    ACTIVITY_VERBS,
    HarnessRenderer,
    HarnessResult,
    HarnessStatus,
    MarkdownRenderer,
    PassthroughRenderer,
    render_stream,
)
from agent_framework._harness._renderers import _progress_bar_text
from agent_framework._harness._state import HarnessLifecycleEvent  # noqa: F401 (used in tests)

# ============================================================
# Protocol Tests
# ============================================================


class TestHarnessRendererProtocol:
    """Tests for the HarnessRenderer protocol."""

    def test_passthrough_satisfies_protocol(self) -> None:
        """Test that PassthroughRenderer satisfies HarnessRenderer."""
        renderer = PassthroughRenderer()
        assert isinstance(renderer, HarnessRenderer)

    def test_markdown_satisfies_protocol(self) -> None:
        """Test that MarkdownRenderer satisfies HarnessRenderer."""
        renderer = MarkdownRenderer()
        assert isinstance(renderer, HarnessRenderer)

    def test_custom_renderer_satisfies_protocol(self) -> None:
        """Test that a custom implementation satisfies HarnessRenderer."""

        class CustomRenderer:
            def on_turn_started(self, turn_number: int) -> str | None:
                return f"Turn {turn_number}"

            def on_deliverables_updated(self, data: dict) -> str | None:
                return None

            def on_result(self, result: HarnessResult) -> str | None:
                return "Done"

            def on_text(self, text: str) -> str | None:
                return text.upper()

        renderer = CustomRenderer()
        assert isinstance(renderer, HarnessRenderer)


# ============================================================
# Progress Bar Helper Tests
# ============================================================


class TestProgressBarText:
    """Tests for _progress_bar_text helper."""

    def test_zero_total(self) -> None:
        """Test that zero total returns empty string."""
        assert _progress_bar_text(0, 0) == ""

    def test_empty_progress(self) -> None:
        """Test 0% progress."""
        result = _progress_bar_text(0, 5)
        assert "0/5" in result
        assert "0%" in result
        assert "░" in result

    def test_full_progress(self) -> None:
        """Test 100% progress."""
        result = _progress_bar_text(5, 5)
        assert "5/5" in result
        assert "100%" in result
        assert "█" in result

    def test_partial_progress(self) -> None:
        """Test 50% progress."""
        result = _progress_bar_text(2, 4)
        assert "2/4" in result
        assert "50%" in result

    def test_custom_width(self) -> None:
        """Test custom bar width."""
        result = _progress_bar_text(5, 10, width=10)
        # 50% of 10 = 5 filled chars
        assert "█████░░░░░" in result


# ============================================================
# PassthroughRenderer Tests
# ============================================================


class TestPassthroughRenderer:
    """Tests for PassthroughRenderer."""

    def test_suppresses_turn_started(self) -> None:
        """Test that turn_started returns None."""
        renderer = PassthroughRenderer()
        assert renderer.on_turn_started(1) is None
        assert renderer.on_turn_started(5) is None

    def test_suppresses_deliverables_updated(self) -> None:
        """Test that deliverables_updated returns None."""
        renderer = PassthroughRenderer()
        data = {"done_items": 2, "total_items": 5, "items": []}
        assert renderer.on_deliverables_updated(data) is None

    def test_suppresses_result(self) -> None:
        """Test that on_result returns None."""
        renderer = PassthroughRenderer()
        result = HarnessResult(status=HarnessStatus.DONE, turn_count=3)
        assert renderer.on_result(result) is None

    def test_passes_text_unchanged(self) -> None:
        """Test that on_text passes through unchanged."""
        renderer = PassthroughRenderer()
        assert renderer.on_text("Hello world") == "Hello world"
        assert renderer.on_text("") == ""
        assert renderer.on_text("  spaces  ") == "  spaces  "


# ============================================================
# MarkdownRenderer Tests
# ============================================================


class TestMarkdownRenderer:
    """Tests for MarkdownRenderer."""

    def test_turn_started_uses_activity_verbs(self) -> None:
        """Test that turn_started cycles through activity verbs."""
        renderer = MarkdownRenderer()
        result1 = renderer.on_turn_started(1)
        assert ACTIVITY_VERBS[0] in result1
        assert "●" in result1

        result2 = renderer.on_turn_started(2)
        assert ACTIVITY_VERBS[1] in result2

    def test_turn_started_cycles_verbs(self) -> None:
        """Test that verbs cycle after exhausting the list."""
        renderer = MarkdownRenderer()
        n = len(ACTIVITY_VERBS)
        result_first = renderer.on_turn_started(1)
        result_cycle = renderer.on_turn_started(n + 1)
        # Turn n+1 should use the same verb as turn 1
        assert result_first == result_cycle

    def test_turn_started_format(self) -> None:
        """Test the markdown format of turn indicator."""
        renderer = MarkdownRenderer()
        result = renderer.on_turn_started(1)
        # Should be bold with bullet
        assert result.strip().startswith("**")
        assert result.strip().endswith("**")
        assert "..." in result

    def test_deliverables_updated_shows_progress_bar(self) -> None:
        """Test that progress bar is shown when items change."""
        renderer = MarkdownRenderer()
        data = {"done_items": 2, "total_items": 5, "items": []}
        result = renderer.on_deliverables_updated(data)
        assert result is not None
        assert "2/5" in result
        assert "█" in result
        assert "`" in result  # monospace wrapping

    def test_deliverables_updated_no_duplicate_bar(self) -> None:
        """Test that same progress doesn't produce duplicate bar."""
        renderer = MarkdownRenderer()
        data = {"done_items": 2, "total_items": 5, "items": []}
        renderer.on_deliverables_updated(data)
        # Same data again
        result = renderer.on_deliverables_updated(data)
        assert result is None  # no change, suppress

    def test_deliverables_updated_renders_new_deliverables(self) -> None:
        """Test that new deliverables are rendered inline."""
        renderer = MarkdownRenderer(show_deliverables_inline=True)
        data = {
            "done_items": 1,
            "total_items": 2,
            "items": [
                {"item_id": "abc", "title": "Report", "content": "## Findings\nData here"},
            ],
        }
        result = renderer.on_deliverables_updated(data)
        assert "Report" in result
        assert "Findings" in result
        assert "---" in result  # horizontal rule separator

    def test_deliverables_updated_skips_seen_items(self) -> None:
        """Test that already-seen deliverables are not re-rendered."""
        renderer = MarkdownRenderer(show_deliverables_inline=True)
        data = {
            "done_items": 1,
            "total_items": 2,
            "items": [
                {"item_id": "abc", "title": "Report", "content": "content"},
            ],
        }
        renderer.on_deliverables_updated(data)

        # Same item again
        result = renderer.on_deliverables_updated({
            "done_items": 2,
            "total_items": 2,
            "items": [{"item_id": "abc", "title": "Report", "content": "content"}],
        })
        # Should still have progress bar (done changed) but no deliverable block
        assert "Report" not in (result or "")  # no duplicate deliverable heading

    def test_deliverables_disabled_inline(self) -> None:
        """Test that inline deliverables can be disabled."""
        renderer = MarkdownRenderer(show_deliverables_inline=False)
        data = {
            "done_items": 1,
            "total_items": 2,
            "items": [{"item_id": "abc", "title": "Report", "content": "content"}],
        }
        result = renderer.on_deliverables_updated(data)
        # Should have progress bar but no deliverable content
        assert "1/2" in (result or "")
        assert "#### Report" not in (result or "")

    def test_deliverables_updated_zero_total(self) -> None:
        """Test that zero total items suppresses output."""
        renderer = MarkdownRenderer()
        data = {"done_items": 0, "total_items": 0, "items": []}
        result = renderer.on_deliverables_updated(data)
        assert result is None

    def test_on_result_shows_status(self) -> None:
        """Test that on_result shows completion status."""
        renderer = MarkdownRenderer()
        result = HarnessResult(status=HarnessStatus.DONE, turn_count=5)
        output = renderer.on_result(result)
        assert "Complete" in output
        assert "5 turns" in output

    def test_on_result_shows_unseen_deliverables(self) -> None:
        """Test that on_result shows deliverables not seen inline."""
        renderer = MarkdownRenderer()
        result = HarnessResult(
            status=HarnessStatus.DONE,
            turn_count=3,
            deliverables=[
                {"item_id": "abc", "title": "Final Report", "content": "## Report\nDone"},
            ],
        )
        output = renderer.on_result(result)
        assert "Final Report" in output
        assert "Report" in output

    def test_on_result_skips_seen_deliverables(self) -> None:
        """Test that on_result skips deliverables already shown inline."""
        renderer = MarkdownRenderer(show_deliverables_inline=True)
        # Simulate seeing a deliverable inline
        renderer.on_deliverables_updated({
            "done_items": 1,
            "total_items": 1,
            "items": [{"item_id": "abc", "title": "Report", "content": "data"}],
        })

        result = HarnessResult(
            status=HarnessStatus.DONE,
            turn_count=2,
            deliverables=[{"item_id": "abc", "title": "Report", "content": "data"}],
        )
        output = renderer.on_result(result)
        assert "Deliverables" not in output

    def test_on_text_passes_through(self) -> None:
        """Test that on_text passes through unchanged."""
        renderer = MarkdownRenderer()
        assert renderer.on_text("Hello") == "Hello"
        assert renderer.on_text("") == ""

    def test_format_deliverable_block(self) -> None:
        """Test deliverable formatting with horizontal rules."""
        renderer = MarkdownRenderer()
        output = renderer._format_deliverable("My Title", "Line 1\nLine 2\nLine 3")
        assert "**📄 My Title**" in output
        assert "---" in output
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output


# ============================================================
# render_stream Tests
# ============================================================


class TestRenderStream:
    """Tests for the render_stream async generator."""

    @pytest.fixture
    def mock_harness(self):
        """Create a mock harness that yields controlled events."""

        class MockHarness:
            def __init__(self, events: list):
                self._events = events

            async def run_stream(self, message, **kwargs):
                for event in self._events:
                    yield event

        return MockHarness

    @pytest.mark.asyncio
    async def test_passthrough_only_yields_text(self, mock_harness) -> None:
        """Test that PassthroughRenderer only yields text events."""
        from agent_framework import AgentRunResponseUpdate
        from agent_framework._workflows._events import AgentRunUpdateEvent

        events = [
            HarnessLifecycleEvent(event_type="turn_started", data={"turn_number": 1}),
            AgentRunUpdateEvent(executor_id="agent", data=AgentRunResponseUpdate(text="Hello ")),
            HarnessLifecycleEvent(
                event_type="deliverables_updated",
                data={
                    "done_items": 1,
                    "total_items": 2,
                    "count": 0,
                    "items": [],
                },
            ),
            AgentRunUpdateEvent(executor_id="agent", data=AgentRunResponseUpdate(text="world")),
        ]
        harness = mock_harness(events)
        renderer = PassthroughRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        # Should get: lifecycle (passed through) + text + lifecycle + text
        # Passthrough suppresses turn_started and deliverables_updated text injection
        # but lifecycle events are always yielded
        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 2
        assert text_events[0].data.text == "Hello "
        assert text_events[1].data.text == "world"

    @pytest.mark.asyncio
    async def test_markdown_injects_turn_indicator(self, mock_harness) -> None:
        """Test that MarkdownRenderer injects turn indicator text."""
        from agent_framework import AgentRunResponseUpdate
        from agent_framework._workflows._events import AgentRunUpdateEvent

        events = [
            HarnessLifecycleEvent(event_type="turn_started", data={"turn_number": 1}),
            AgentRunUpdateEvent(executor_id="agent", data=AgentRunResponseUpdate(text="Working...")),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]

        # First text event should be the injected activity indicator
        assert len(text_events) >= 2
        assert ACTIVITY_VERBS[0] in text_events[0].data.text
        assert text_events[1].data.text == "Working..."

    @pytest.mark.asyncio
    async def test_markdown_injects_progress_bar(self, mock_harness) -> None:
        """Test that MarkdownRenderer injects progress bar."""
        events = [
            HarnessLifecycleEvent(
                event_type="deliverables_updated",
                data={
                    "done_items": 2,
                    "total_items": 4,
                    "count": 0,
                    "items": [],
                },
            ),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        from agent_framework._workflows._events import AgentRunUpdateEvent

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]

        assert len(text_events) == 1
        assert "2/4" in text_events[0].data.text

    @pytest.mark.asyncio
    async def test_lifecycle_events_always_passed_through(self, mock_harness) -> None:
        """Test that lifecycle events are always yielded regardless of renderer."""
        events = [
            HarnessLifecycleEvent(event_type="turn_started", data={"turn_number": 1}),
            HarnessLifecycleEvent(
                event_type="deliverables_updated",
                data={
                    "done_items": 0,
                    "total_items": 1,
                    "count": 0,
                    "items": [],
                },
            ),
        ]
        harness = mock_harness(events)

        for renderer in [PassthroughRenderer(), MarkdownRenderer()]:
            collected = []
            async for event in render_stream(harness, "test", renderer):
                collected.append(event)

            lifecycle = [e for e in collected if isinstance(e, HarnessLifecycleEvent)]
            assert len(lifecycle) == 2

    @pytest.mark.asyncio
    async def test_result_event_always_passed_through(self, mock_harness) -> None:
        """Test that WorkflowOutputEvent with HarnessResult is always yielded."""
        from agent_framework._workflows._events import WorkflowOutputEvent

        result = HarnessResult(status=HarnessStatus.DONE, turn_count=3)
        events = [
            WorkflowOutputEvent(data=result, source_executor_id="stop"),
        ]
        harness = mock_harness(events)
        renderer = PassthroughRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        output_events = [e for e in collected if isinstance(e, WorkflowOutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].data is result

    @pytest.mark.asyncio
    async def test_markdown_result_injects_summary(self, mock_harness) -> None:
        """Test that MarkdownRenderer injects result summary text."""
        from agent_framework._workflows._events import AgentRunUpdateEvent, WorkflowOutputEvent

        result = HarnessResult(
            status=HarnessStatus.DONE,
            turn_count=4,
            deliverables=[{"item_id": "x", "title": "Report", "content": "data"}],
        )
        events = [
            WorkflowOutputEvent(data=result, source_executor_id="stop"),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 1
        assert "Complete" in text_events[0].data.text
        assert "4 turns" in text_events[0].data.text

    @pytest.mark.asyncio
    async def test_custom_renderer(self, mock_harness) -> None:
        """Test render_stream with a custom renderer that transforms text."""
        from agent_framework import AgentRunResponseUpdate
        from agent_framework._workflows._events import AgentRunUpdateEvent

        class UpperRenderer:
            def on_turn_started(self, turn_number):
                return None

            def on_deliverables_updated(self, data):
                return None

            def on_result(self, result):
                return None

            def on_text(self, text):
                return text.upper()

        events = [
            AgentRunUpdateEvent(executor_id="agent", data=AgentRunResponseUpdate(text="hello")),
        ]
        harness = mock_harness(events)
        renderer = UpperRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 1
        assert text_events[0].data.text == "HELLO"

    @pytest.mark.asyncio
    async def test_text_suppression(self, mock_harness) -> None:
        """Test that returning None from on_text suppresses the event."""
        from agent_framework import AgentRunResponseUpdate
        from agent_framework._workflows._events import AgentRunUpdateEvent

        class SuppressRenderer:
            def on_turn_started(self, turn_number):
                return None

            def on_deliverables_updated(self, data):
                return None

            def on_result(self, result):
                return None

            def on_text(self, text):
                return None  # suppress all text

        events = [
            AgentRunUpdateEvent(executor_id="agent", data=AgentRunResponseUpdate(text="hidden")),
        ]
        harness = mock_harness(events)
        renderer = SuppressRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 0

    @pytest.mark.asyncio
    async def test_injected_events_have_renderer_executor_id(self, mock_harness) -> None:
        """Test that injected text events have 'harness_renderer' executor_id."""
        from agent_framework._workflows._events import AgentRunUpdateEvent

        events = [
            HarnessLifecycleEvent(event_type="turn_started", data={"turn_number": 1}),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        injected = [e for e in collected if isinstance(e, AgentRunUpdateEvent) and e.executor_id == "harness_renderer"]
        assert len(injected) == 1  # the turn indicator


# ============================================================
# Activity Verbs Tests
# ============================================================


class TestActivityVerbs:
    """Tests for the ACTIVITY_VERBS constant."""

    def test_has_verbs(self) -> None:
        """Test that ACTIVITY_VERBS is non-empty."""
        assert len(ACTIVITY_VERBS) >= 5

    def test_all_lowercase(self) -> None:
        """Test that all verbs are lowercase."""
        for verb in ACTIVITY_VERBS:
            assert verb == verb.lower()

    def test_no_duplicates(self) -> None:
        """Test that there are no duplicate verbs."""
        assert len(ACTIVITY_VERBS) == len(set(ACTIVITY_VERBS))


# ============================================================
# Compaction Visibility Tests (Phase 12)
# ============================================================


class TestMarkdownRendererCompaction:
    """Tests for compaction event rendering in MarkdownRenderer."""

    def test_compaction_started_format(self) -> None:
        """Test that compaction_started renders with token counts."""
        renderer = MarkdownRenderer()
        data = {"current_tokens": 85000, "soft_threshold": 100000, "tokens_over_threshold": 5000}
        result = renderer.on_compaction_started(data)
        assert result is not None
        assert "⟳" in result
        assert "Compacting context" in result
        assert "85,000" in result
        assert "100,000" in result

    def test_compaction_started_with_strategies(self) -> None:
        """Test that compaction_started shows available strategies."""
        renderer = MarkdownRenderer()
        data = {
            "current_tokens": 157684,
            "soft_threshold": 85000,
            "strategies_available": ["clear", "drop"],
        }
        result = renderer.on_compaction_started(data)
        assert result is not None
        assert "via clear+drop" in result
        assert "157,684" in result

    def test_compaction_started_with_summarize_strategy(self) -> None:
        """Test that compaction_started shows summarize in strategy list."""
        renderer = MarkdownRenderer()
        data = {
            "current_tokens": 100000,
            "soft_threshold": 85000,
            "strategies_available": ["clear", "summarize", "drop"],
        }
        result = renderer.on_compaction_started(data)
        assert result is not None
        assert "via clear+summarize+drop" in result

    def test_compaction_started_no_strategies(self) -> None:
        """Test that compaction_started works without strategies_available."""
        renderer = MarkdownRenderer()
        data = {"current_tokens": 85000, "soft_threshold": 100000}
        result = renderer.on_compaction_started(data)
        assert result is not None
        assert "via" not in result
        assert "Compacting context..." in result

    def test_compaction_started_zero_values(self) -> None:
        """Test compaction_started with zero values."""
        renderer = MarkdownRenderer()
        data = {"current_tokens": 0, "soft_threshold": 0}
        result = renderer.on_compaction_started(data)
        assert result is not None
        assert "0" in result

    def test_compaction_completed_format(self) -> None:
        """Test that compaction_completed renders with before/after sizes."""
        renderer = MarkdownRenderer()
        data = {
            "tokens_before": 85000,
            "tokens_after": 52000,
            "tokens_freed": 33000,
            "duration_ms": 2300,
        }
        result = renderer.on_compaction_completed(data)
        assert result is not None
        assert "✓" in result
        assert "85,000" in result
        assert "52,000" in result
        assert "33,000" in result
        assert "2.3s" in result

    def test_compaction_completed_with_level(self) -> None:
        """Test that compaction_completed shows compaction level."""
        renderer = MarkdownRenderer()
        data = {
            "tokens_before": 157684,
            "tokens_after": 5609,
            "tokens_freed": 152075,
            "duration_ms": 0,
            "strategies_applied": ["clear", "drop"],
            "compaction_level": "destructive",
        }
        result = renderer.on_compaction_completed(data)
        assert result is not None
        assert "[destructive]" in result
        assert "157,684" in result
        assert "5,609" in result

    def test_compaction_completed_compressed_level(self) -> None:
        """Test that compaction_completed shows compressed level."""
        renderer = MarkdownRenderer()
        data = {
            "tokens_before": 157684,
            "tokens_after": 40000,
            "tokens_freed": 117684,
            "duration_ms": 3200,
            "strategies_applied": ["clear", "summarize", "drop"],
            "compaction_level": "compressed",
        }
        result = renderer.on_compaction_completed(data)
        assert result is not None
        assert "[compressed]" in result
        assert "3.2s" in result

    def test_compaction_completed_optimized_level(self) -> None:
        """Test that compaction_completed shows optimized level."""
        renderer = MarkdownRenderer()
        data = {
            "tokens_before": 85000,
            "tokens_after": 70000,
            "tokens_freed": 15000,
            "duration_ms": 50,
            "strategies_applied": ["clear"],
            "compaction_level": "optimized",
        }
        result = renderer.on_compaction_completed(data)
        assert result is not None
        assert "[optimized]" in result

    def test_compaction_completed_no_level(self) -> None:
        """Test that compaction_completed works without compaction_level."""
        renderer = MarkdownRenderer()
        data = {"tokens_before": 1000, "tokens_after": 800, "tokens_freed": 200, "duration_ms": 0}
        result = renderer.on_compaction_completed(data)
        assert result is not None
        assert "0.0s" in result
        # No level tag when not provided
        assert "[" not in result


class TestPassthroughRendererCompaction:
    """Tests for compaction event suppression in PassthroughRenderer."""

    def test_compaction_started_suppressed(self) -> None:
        """Test that compaction_started returns None."""
        renderer = PassthroughRenderer()
        data = {"current_tokens": 85000, "soft_threshold": 100000}
        assert renderer.on_compaction_started(data) is None

    def test_compaction_completed_suppressed(self) -> None:
        """Test that compaction_completed returns None."""
        renderer = PassthroughRenderer()
        data = {"tokens_before": 85000, "tokens_after": 52000, "tokens_freed": 33000, "duration_ms": 2300}
        assert renderer.on_compaction_completed(data) is None


class TestRenderStreamCompaction:
    """Tests for compaction event routing in render_stream."""

    @pytest.fixture
    def mock_harness(self):
        """Create a mock harness that yields controlled events."""

        class MockHarness:
            def __init__(self, events: list):
                self._events = events

            async def run_stream(self, message, **kwargs):
                for event in self._events:
                    yield event

        return MockHarness

    @pytest.mark.asyncio
    async def test_compaction_started_injected_for_markdown(self, mock_harness) -> None:
        """Test that compaction_started injects text with MarkdownRenderer."""
        from agent_framework._workflows._events import AgentRunUpdateEvent

        events = [
            HarnessLifecycleEvent(
                event_type="compaction_started",
                data={
                    "current_tokens": 85000,
                    "soft_threshold": 100000,
                    "tokens_over_threshold": 5000,
                    "strategies_available": ["clear", "drop"],
                },
            ),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 1
        assert "85,000" in text_events[0].data.text
        assert "Compacting" in text_events[0].data.text
        assert "via clear+drop" in text_events[0].data.text

    @pytest.mark.asyncio
    async def test_compaction_completed_injected_for_markdown(self, mock_harness) -> None:
        """Test that compaction_completed injects text with MarkdownRenderer."""
        from agent_framework._workflows._events import AgentRunUpdateEvent

        events = [
            HarnessLifecycleEvent(
                event_type="compaction_completed",
                data={
                    "tokens_before": 85000,
                    "tokens_after": 52000,
                    "tokens_freed": 33000,
                    "duration_ms": 2300,
                    "strategies_applied": ["clear", "drop"],
                    "compaction_level": "destructive",
                },
            ),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 1
        assert "52,000" in text_events[0].data.text
        assert "33,000" in text_events[0].data.text
        assert "[destructive]" in text_events[0].data.text

    @pytest.mark.asyncio
    async def test_compaction_events_suppressed_for_passthrough(self, mock_harness) -> None:
        """Test that compaction events don't inject text for PassthroughRenderer."""
        from agent_framework._workflows._events import AgentRunUpdateEvent

        events = [
            HarnessLifecycleEvent(
                event_type="compaction_started",
                data={"current_tokens": 85000, "soft_threshold": 100000},
            ),
            HarnessLifecycleEvent(
                event_type="compaction_completed",
                data={"tokens_before": 85000, "tokens_after": 52000, "tokens_freed": 33000, "duration_ms": 2300},
            ),
        ]
        harness = mock_harness(events)
        renderer = PassthroughRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        text_events = [
            e
            for e in collected
            if isinstance(e, AgentRunUpdateEvent) and e.data and hasattr(e.data, "text") and e.data.text
        ]
        assert len(text_events) == 0

        # But lifecycle events should still pass through
        lifecycle = [e for e in collected if isinstance(e, HarnessLifecycleEvent)]
        assert len(lifecycle) == 2

    @pytest.mark.asyncio
    async def test_compaction_lifecycle_events_always_passed_through(self, mock_harness) -> None:
        """Test that compaction lifecycle events are always yielded."""
        events = [
            HarnessLifecycleEvent(
                event_type="compaction_started",
                data={"current_tokens": 85000, "soft_threshold": 100000},
            ),
            HarnessLifecycleEvent(
                event_type="compaction_completed",
                data={"tokens_before": 85000, "tokens_after": 52000, "tokens_freed": 33000, "duration_ms": 2300},
            ),
        ]
        harness = mock_harness(events)
        renderer = MarkdownRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        lifecycle = [e for e in collected if isinstance(e, HarnessLifecycleEvent)]
        assert len(lifecycle) == 2
        assert lifecycle[0].event_type == "compaction_started"
        assert lifecycle[1].event_type == "compaction_completed"

    @pytest.mark.asyncio
    async def test_compaction_works_with_renderer_lacking_methods(self, mock_harness) -> None:
        """Test that renderers without compaction methods don't cause errors."""

        class MinimalRenderer:
            def on_turn_started(self, turn_number):
                return None

            def on_deliverables_updated(self, data):
                return None

            def on_result(self, result):
                return None

            def on_text(self, text):
                return text

        events = [
            HarnessLifecycleEvent(
                event_type="compaction_started",
                data={"current_tokens": 85000, "soft_threshold": 100000},
            ),
            HarnessLifecycleEvent(
                event_type="compaction_completed",
                data={"tokens_before": 85000, "tokens_after": 52000, "tokens_freed": 33000, "duration_ms": 2300},
            ),
        ]
        harness = mock_harness(events)
        renderer = MinimalRenderer()

        collected = []
        async for event in render_stream(harness, "test", renderer):
            collected.append(event)

        # Should not crash; lifecycle events still passed through
        lifecycle = [e for e in collected if isinstance(e, HarnessLifecycleEvent)]
        assert len(lifecycle) == 2


class TestClassifyCompactionLevel:
    """Tests for compaction level classification."""

    def test_destructive_clear_and_drop(self) -> None:
        """Test that clear+drop is classified as destructive."""
        from agent_framework._harness._agent_turn_executor import _classify_compaction_level

        assert _classify_compaction_level(["clear", "drop"]) == "destructive"

    def test_compressed_with_summarize(self) -> None:
        """Test that summarize presence is classified as compressed."""
        from agent_framework._harness._agent_turn_executor import _classify_compaction_level

        assert _classify_compaction_level(["clear", "summarize", "drop"]) == "compressed"

    def test_compressed_summarize_only(self) -> None:
        """Test that summarize alone is classified as compressed."""
        from agent_framework._harness._agent_turn_executor import _classify_compaction_level

        assert _classify_compaction_level(["summarize"]) == "compressed"

    def test_optimized_clear_only(self) -> None:
        """Test that clear only is classified as optimized."""
        from agent_framework._harness._agent_turn_executor import _classify_compaction_level

        assert _classify_compaction_level(["clear"]) == "optimized"

    def test_optimized_empty(self) -> None:
        """Test that no strategies is classified as optimized."""
        from agent_framework._harness._agent_turn_executor import _classify_compaction_level

        assert _classify_compaction_level([]) == "optimized"
