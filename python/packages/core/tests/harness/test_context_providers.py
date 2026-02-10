# Copyright (c) Microsoft. All rights reserved.

"""Tests for harness context providers (Phase 2 — Rich System Prompt Construction)."""

import os
import platform

import pytest

from agent_framework._harness._context_providers import (
    EnvironmentContextProvider,
    HarnessGuidanceProvider,
)
from agent_framework._memory import AggregateContextProvider


class TestEnvironmentContextProvider:
    """Tests for EnvironmentContextProvider."""

    @pytest.mark.asyncio
    async def test_injects_cwd_and_os(self) -> None:
        """Provider includes working directory and OS in instructions."""
        provider = EnvironmentContextProvider()
        context = await provider.invoking([])

        assert context.instructions is not None
        assert os.getcwd() in context.instructions
        assert platform.system() in context.instructions

    @pytest.mark.asyncio
    async def test_sandbox_path_overrides_cwd(self, tmp_path: object) -> None:
        """When sandbox_path is set, it is used instead of os.getcwd()."""
        sandbox = str(tmp_path)
        provider = EnvironmentContextProvider(sandbox_path=sandbox)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert sandbox in context.instructions

    @pytest.mark.asyncio
    async def test_directory_listing_included(self, tmp_path: object) -> None:
        """Directory listing of the sandbox path is included."""
        sandbox = str(tmp_path)
        # Create test files
        for name in ["alpha.txt", "beta.py", "gamma.md"]:
            os.close(os.open(os.path.join(sandbox, name), os.O_CREAT | os.O_WRONLY))

        provider = EnvironmentContextProvider(sandbox_path=sandbox)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "alpha.txt" in context.instructions
        assert "beta.py" in context.instructions
        assert "gamma.md" in context.instructions

    @pytest.mark.asyncio
    async def test_max_entries_limits_listing(self, tmp_path: object) -> None:
        """Only max_entries directory entries are included."""
        sandbox = str(tmp_path)
        for i in range(10):
            os.close(os.open(os.path.join(sandbox, f"file_{i:02d}.txt"), os.O_CREAT | os.O_WRONLY))

        provider = EnvironmentContextProvider(sandbox_path=sandbox, max_entries=3)
        context = await provider.invoking([])

        assert context.instructions is not None
        # Sorted entries: file_00, file_01, file_02 should be present
        assert "file_00.txt" in context.instructions
        assert "file_02.txt" in context.instructions
        # file_03 and beyond should not be present
        assert "file_03.txt" not in context.instructions

    @pytest.mark.asyncio
    async def test_invalid_path_graceful_fallback(self) -> None:
        """Invalid sandbox path produces graceful fallback message."""
        provider = EnvironmentContextProvider(sandbox_path="/nonexistent/path/xyz")
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "unable to list directory" in context.instructions

    @pytest.mark.asyncio
    async def test_xml_tags_present(self) -> None:
        """Instructions are wrapped in <environment_context> tags."""
        provider = EnvironmentContextProvider()
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "<environment_context>" in context.instructions
        assert "</environment_context>" in context.instructions


class TestHarnessGuidanceProvider:
    """Tests for HarnessGuidanceProvider."""

    @pytest.mark.asyncio
    async def test_always_includes_task_completion(self) -> None:
        """Task completion instructions are always included."""
        provider = HarnessGuidanceProvider(enable_tool_guidance=False, enable_work_items=False)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "<work_completion>" in context.instructions
        assert "work_complete" in context.instructions

    @pytest.mark.asyncio
    async def test_always_includes_response_style(self) -> None:
        """Response style guidance is always included."""
        provider = HarnessGuidanceProvider(enable_tool_guidance=False, enable_work_items=False)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "<response_style>" in context.instructions
        assert "</response_style>" in context.instructions
        assert "brief acknowledgment" in context.instructions

    @pytest.mark.asyncio
    async def test_tool_guidance_included_by_default(self) -> None:
        """Tool strategy guidance is included when enable_tool_guidance=True (default)."""
        provider = HarnessGuidanceProvider()
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "TOOL STRATEGY GUIDE" in context.instructions

    @pytest.mark.asyncio
    async def test_tool_guidance_excluded(self) -> None:
        """Tool strategy guidance is excluded when enable_tool_guidance=False."""
        provider = HarnessGuidanceProvider(enable_tool_guidance=False)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "TOOL STRATEGY GUIDE" not in context.instructions

    @pytest.mark.asyncio
    async def test_work_item_guidance_included(self) -> None:
        """Work item and planning guidance are included when enable_work_items=True."""
        provider = HarnessGuidanceProvider(enable_work_items=True)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "work_item_add" in context.instructions
        assert "work_item_set_artifact" in context.instructions

    @pytest.mark.asyncio
    async def test_work_item_guidance_excluded_by_default(self) -> None:
        """Work item guidance is not included by default."""
        provider = HarnessGuidanceProvider()
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "work_item_add" not in context.instructions

    @pytest.mark.asyncio
    async def test_all_sections_combined(self) -> None:
        """All sections are combined when all flags are enabled."""
        provider = HarnessGuidanceProvider(enable_tool_guidance=True, enable_work_items=True)
        context = await provider.invoking([])

        assert context.instructions is not None
        assert "TOOL STRATEGY GUIDE" in context.instructions
        assert "work_item_add" in context.instructions
        assert "<work_completion>" in context.instructions


class TestContextProviderIntegration:
    """Integration tests for context providers with AggregateContextProvider."""

    @pytest.mark.asyncio
    async def test_providers_compose_in_aggregate(self, tmp_path: object) -> None:
        """Both providers can be composed in an AggregateContextProvider."""
        sandbox = str(tmp_path)
        env_provider = EnvironmentContextProvider(sandbox_path=sandbox)
        guidance_provider = HarnessGuidanceProvider()
        aggregate = AggregateContextProvider([env_provider, guidance_provider])

        context = await aggregate.invoking([])

        assert context.instructions is not None
        # Environment context
        assert sandbox in context.instructions
        # Guidance
        assert "TOOL STRATEGY GUIDE" in context.instructions
        assert "<work_completion>" in context.instructions
        assert "<response_style>" in context.instructions

    @pytest.mark.asyncio
    async def test_providers_return_no_messages_or_tools(self) -> None:
        """Context providers only return instructions, not messages or tools."""
        env_provider = EnvironmentContextProvider()
        guidance_provider = HarnessGuidanceProvider()

        env_ctx = await env_provider.invoking([])
        guide_ctx = await guidance_provider.invoking([])

        assert len(env_ctx.messages) == 0
        assert len(env_ctx.tools) == 0
        assert len(guide_ctx.messages) == 0
        assert len(guide_ctx.tools) == 0
