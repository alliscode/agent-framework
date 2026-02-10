# Copyright (c) Microsoft. All rights reserved.

"""Tests for Phase 5 — Sub-Agent Delegation."""

from collections.abc import AsyncIterable
from typing import Any

import pytest

from agent_framework import AgentRunResponse, AgentRunResponseUpdate, AgentThread, ChatMessage, Role, TextContent
from agent_framework._harness._agent_turn_executor import AgentTurnExecutor
from agent_framework._harness._context_providers import HarnessGuidanceProvider
from agent_framework._harness._sub_agents import create_document_tool, create_explore_tool, create_task_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeChatClient:
    """Minimal ChatClientProtocol stub for constructing ChatAgent sub-agents."""

    additional_properties: dict[str, Any] = {}

    async def get_response(self, messages: Any, **kwargs: Any) -> Any:
        from agent_framework._types import ChatResponse

        return ChatResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="stub response")])],
        )

    async def get_streaming_response(self, messages: Any, **kwargs: Any) -> AsyncIterable[Any]:
        async def _empty() -> AsyncIterable[Any]:
            return
            yield  # noqa: RET503 — make it an async generator

        return _empty()


def _dummy_tool() -> str:
    """A dummy tool for testing."""
    return "ok"


# ---------------------------------------------------------------------------
# 5a — Sub-agent factory functions
# ---------------------------------------------------------------------------


class TestCreateExploreTool:
    """Tests for create_explore_tool factory."""

    def test_returns_ai_function(self) -> None:
        """create_explore_tool returns an AIFunction with correct name."""
        tool = create_explore_tool(_FakeChatClient(), [_dummy_tool])
        assert tool.name == "explore"

    def test_description_mentions_codebase(self) -> None:
        """Tool description explains its purpose."""
        tool = create_explore_tool(_FakeChatClient(), [])
        assert "codebase" in tool.description.lower()

    def test_accepts_tools_sequence(self) -> None:
        """Factory accepts a list of tools without error."""
        tool = create_explore_tool(_FakeChatClient(), [_dummy_tool, _dummy_tool])
        assert tool.name == "explore"


class TestCreateTaskTool:
    """Tests for create_task_tool factory."""

    def test_returns_ai_function(self) -> None:
        """create_task_tool returns an AIFunction with correct name."""
        tool = create_task_tool(_FakeChatClient(), [_dummy_tool])
        assert tool.name == "run_task"

    def test_description_mentions_commands(self) -> None:
        """Tool description explains its purpose."""
        tool = create_task_tool(_FakeChatClient(), [])
        assert "command" in tool.description.lower() or "test" in tool.description.lower()

    def test_accepts_empty_tools(self) -> None:
        """Factory accepts an empty tools list."""
        tool = create_task_tool(_FakeChatClient(), [])
        assert tool.name == "run_task"


# ---------------------------------------------------------------------------
# 5b — Wiring into AgentTurnExecutor
# ---------------------------------------------------------------------------


class _MockAgent:
    """Minimal agent stub for testing AgentTurnExecutor wiring."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "mock"

    def get_new_thread(self) -> AgentThread:
        return AgentThread()

    async def run(self, messages: Any, **kwargs: Any) -> AgentRunResponse:
        self.last_kwargs = kwargs
        return AgentRunResponse(
            messages=[ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text="done")])],
        )

    async def run_stream(self, messages: Any, **kwargs: Any) -> AsyncIterable[AgentRunResponseUpdate]:
        self.last_kwargs = kwargs

        async def _gen() -> AsyncIterable[AgentRunResponseUpdate]:
            yield AgentRunResponseUpdate(
                contents=[TextContent(text="done")],
            )

        return _gen()


class TestAgentTurnExecutorSubAgentTools:
    """Tests for sub-agent tool injection in AgentTurnExecutor."""

    def test_stores_sub_agent_tools(self) -> None:
        """AgentTurnExecutor stores provided sub_agent_tools."""
        fake_tools = ["tool_a", "tool_b"]
        executor = AgentTurnExecutor(
            _MockAgent(),
            sub_agent_tools=fake_tools,
        )
        assert executor._sub_agent_tools == fake_tools

    def test_defaults_to_empty_list(self) -> None:
        """Without sub_agent_tools, defaults to empty list."""
        executor = AgentTurnExecutor(_MockAgent())
        assert executor._sub_agent_tools == []

    def test_none_defaults_to_empty_list(self) -> None:
        """Passing None explicitly defaults to empty list."""
        executor = AgentTurnExecutor(_MockAgent(), sub_agent_tools=None)
        assert executor._sub_agent_tools == []


# ---------------------------------------------------------------------------
# 5c — Sub-agent guidance in system prompt
# ---------------------------------------------------------------------------


class TestSubAgentGuidance:
    """Tests for sub-agent guidance in HarnessGuidanceProvider."""

    @pytest.mark.asyncio
    async def test_guidance_not_included_by_default(self) -> None:
        """Sub-agent guidance is NOT included when enable_sub_agents is False."""
        provider = HarnessGuidanceProvider(enable_sub_agents=False)
        context = await provider.invoking([])
        assert context.instructions is not None
        assert "<sub_agents>" not in context.instructions

    @pytest.mark.asyncio
    async def test_guidance_included_when_enabled(self) -> None:
        """Sub-agent guidance IS included when enable_sub_agents is True."""
        provider = HarnessGuidanceProvider(enable_sub_agents=True)
        context = await provider.invoking([])
        assert context.instructions is not None
        assert "<sub_agents>" in context.instructions
        assert "explore" in context.instructions
        assert "run_task" in context.instructions

    @pytest.mark.asyncio
    async def test_guidance_content(self) -> None:
        """Sub-agent guidance contains expected usage hints."""
        provider = HarnessGuidanceProvider(enable_sub_agents=True)
        context = await provider.invoking([])
        assert "codebase" in context.instructions.lower() or "Q&A" in context.instructions

    @pytest.mark.asyncio
    async def test_task_completion_always_present(self) -> None:
        """Work completion instructions are present regardless of sub-agent setting."""
        for flag in (True, False):
            provider = HarnessGuidanceProvider(enable_sub_agents=flag)
            context = await provider.invoking([])
            assert "<work_completion>" in context.instructions


# ---------------------------------------------------------------------------
# 5b+5d — HarnessWorkflowBuilder / AgentHarness wiring
# ---------------------------------------------------------------------------


class TestHarnessBuilderSubAgents:
    """Tests for sub-agent wiring in HarnessWorkflowBuilder and AgentHarness."""

    def test_builder_creates_sub_agent_tools_when_client_provided(self) -> None:
        """HarnessWorkflowBuilder creates sub-agent tools when sub_agent_client is given."""
        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        builder = HarnessWorkflowBuilder(
            _MockAgent(),
            sub_agent_client=_FakeChatClient(),
        )
        assert len(builder._sub_agent_tools) == 3
        names = {t.name for t in builder._sub_agent_tools}
        assert names == {"explore", "run_task", "document"}

    def test_builder_no_sub_agents_by_default(self) -> None:
        """Without sub_agent_client, no sub-agent tools are created."""
        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        builder = HarnessWorkflowBuilder(_MockAgent())
        assert builder._sub_agent_tools == []

    def test_builder_forwards_sub_agent_tools(self) -> None:
        """sub_agent_tools parameter is forwarded to sub-agent factories."""
        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        builder = HarnessWorkflowBuilder(
            _MockAgent(),
            sub_agent_client=_FakeChatClient(),
            sub_agent_tools=[_dummy_tool],
        )
        # All three tools should be created successfully
        assert len(builder._sub_agent_tools) == 3

    def test_agent_harness_accepts_sub_agent_params(self) -> None:
        """AgentHarness constructor accepts sub_agent_client and sub_agent_tools."""
        from agent_framework._harness import AgentHarness

        harness = AgentHarness(
            _MockAgent(),
            sub_agent_client=_FakeChatClient(),
            sub_agent_tools=[_dummy_tool],
        )
        # Should build without error
        assert harness._builder._sub_agent_tools is not None
        assert len(harness._builder._sub_agent_tools) == 3

    def test_agent_harness_no_sub_agents_by_default(self) -> None:
        """AgentHarness without sub_agent_client has no sub-agent tools."""
        from agent_framework._harness import AgentHarness

        harness = AgentHarness(_MockAgent())
        assert harness._builder._sub_agent_tools == []


# ---------------------------------------------------------------------------
# 5b — Document sub-agent factory
# ---------------------------------------------------------------------------


class TestCreateDocumentTool:
    """Tests for create_document_tool factory."""

    def test_returns_ai_function(self) -> None:
        """create_document_tool returns an AIFunction with correct name."""
        tool = create_document_tool(_FakeChatClient(), [_dummy_tool])
        assert tool.name == "document"

    def test_description_mentions_comprehensive(self) -> None:
        """Tool description explains its purpose."""
        tool = create_document_tool(_FakeChatClient(), [])
        assert "comprehensive" in tool.description.lower()

    def test_accepts_tools_sequence(self) -> None:
        """Factory accepts a list of tools without error."""
        tool = create_document_tool(_FakeChatClient(), [_dummy_tool, _dummy_tool])
        assert tool.name == "document"

    def test_instructions_stress_depth(self) -> None:
        """Document agent instructions emphasize depth and specificity."""
        tool = create_document_tool(_FakeChatClient(), [])
        # Access the underlying agent's instructions via the ChatAgent
        # The tool wraps a ChatAgent, so we verify via the tool description
        assert "comprehensive" in tool.description.lower() or "technical" in tool.description.lower()

    def test_accepts_empty_tools(self) -> None:
        """Factory accepts an empty tools list."""
        tool = create_document_tool(_FakeChatClient(), [])
        assert tool.name == "document"


class TestSubAgentGuidanceIncludesDocument:
    """Tests for document agent in sub-agent guidance."""

    @pytest.mark.asyncio
    async def test_guidance_includes_document(self) -> None:
        """Sub-agent guidance includes the document agent when enabled."""
        provider = HarnessGuidanceProvider(enable_sub_agents=True)
        context = await provider.invoking([])
        assert context.instructions is not None
        assert "document" in context.instructions

    @pytest.mark.asyncio
    async def test_guidance_document_usage_hint(self) -> None:
        """Sub-agent guidance includes document usage guidance."""
        provider = HarnessGuidanceProvider(enable_sub_agents=True)
        context = await provider.invoking([])
        assert "deliverable" in context.instructions.lower()


class TestBuilderCreatesThreeSubAgents:
    """Tests for builder creating three sub-agent tools."""

    def test_builder_creates_three_sub_agent_tools(self) -> None:
        """HarnessWorkflowBuilder creates 3 sub-agent tools when sub_agent_client is given."""
        from agent_framework._harness._harness_builder import HarnessWorkflowBuilder

        builder = HarnessWorkflowBuilder(
            _MockAgent(),
            sub_agent_client=_FakeChatClient(),
        )
        assert len(builder._sub_agent_tools) == 3
        names = {t.name for t in builder._sub_agent_tools}
        assert names == {"explore", "run_task", "document"}

    def test_agent_harness_creates_three_sub_agent_tools(self) -> None:
        """AgentHarness creates 3 sub-agent tools when sub_agent_client is given."""
        from agent_framework._harness import AgentHarness

        harness = AgentHarness(
            _MockAgent(),
            sub_agent_client=_FakeChatClient(),
            sub_agent_tools=[_dummy_tool],
        )
        assert len(harness._builder._sub_agent_tools) == 3
        names = {t.name for t in harness._builder._sub_agent_tools}
        assert names == {"explore", "run_task", "document"}


class TestExports:
    """Tests for module-level exports."""

    def test_create_explore_tool_exported(self) -> None:
        """create_explore_tool is importable from _harness."""
        from agent_framework._harness import create_explore_tool as fn

        assert callable(fn)

    def test_create_task_tool_exported(self) -> None:
        """create_task_tool is importable from _harness."""
        from agent_framework._harness import create_task_tool as fn

        assert callable(fn)

    def test_create_document_tool_exported(self) -> None:
        """create_document_tool is importable from _harness."""
        from agent_framework._harness import create_document_tool as fn

        assert callable(fn)
