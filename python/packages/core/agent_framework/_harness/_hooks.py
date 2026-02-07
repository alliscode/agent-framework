# Copyright (c) Microsoft. All rights reserved.

"""Harness-level lifecycle hooks for external extensibility.

This module provides the hooks system (Phase 3) that allows external code to
intercept, modify, and control harness execution — equivalent to Copilot CLI's
``preToolUse``, ``postToolUse``, and ``agentStop`` hooks.

Three hook points:

1. **Pre-tool hooks** — called before each tool invocation. Can deny the call.
2. **Post-tool hooks** — called after each tool invocation. Observational.
3. **Agent-stop hooks** — called when the agent signals done but before the
   harness accepts the stop. Can block and force continuation.

Example:
    .. code-block:: python

        from agent_framework._harness import AgentHarness, HarnessHooks


        async def my_stop_gate(event):
            if event.turn_count < 5:
                return AgentStopResult(decision="block", reason="Too few turns")
            return None


        hooks = HarnessHooks(agent_stop=[my_stop_gate])
        harness = AgentHarness(agent, hooks=hooks)
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .._middleware import FunctionInvocationContext, FunctionMiddleware

logger = logging.getLogger(__name__)


@dataclass
class AgentStopEvent:
    """Event payload for agent-stop hooks.

    Attributes:
        turn_count: The current turn number.
        tool_usage: Mapping of tool name → call count for the session.
        called_task_complete: Whether the agent called task_complete.
    """

    turn_count: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)
    called_task_complete: bool = False


@dataclass
class AgentStopResult:
    """Result from an agent-stop hook.

    Attributes:
        decision: ``"block"`` to force continuation, ``"allow"`` to accept stop.
        reason: Human-readable reason (injected as continuation prompt on block).
    """

    decision: str = "allow"
    reason: str = ""


@dataclass
class ToolHookResult:
    """Result from a pre-tool hook.

    Attributes:
        decision: ``"deny"`` to prevent tool execution, ``"allow"`` to proceed.
        reason: Human-readable reason (returned as tool result on deny).
    """

    decision: str = "allow"
    reason: str = ""


# Type aliases for hook callbacks
PreToolHook = Callable[[str, dict[str, Any]], Awaitable[ToolHookResult | None]]
PostToolHook = Callable[[str, dict[str, Any], Any], Awaitable[None]]
AgentStopHook = Callable[[AgentStopEvent], Awaitable[AgentStopResult | None]]


@dataclass
class HarnessHooks:
    """Harness-level lifecycle hooks for external extensibility.

    Attributes:
        pre_tool: Callbacks invoked before each tool call. Return ``ToolHookResult``
            with ``decision="deny"`` to prevent execution.
        post_tool: Callbacks invoked after each tool call. Observational only.
        agent_stop: Callbacks invoked when the agent signals done. Return
            ``AgentStopResult`` with ``decision="block"`` to force continuation.
    """

    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    agent_stop: list[AgentStopHook] = field(default_factory=list)


class HarnessToolMiddleware(FunctionMiddleware):
    """FunctionMiddleware that delegates to registered pre/post tool callbacks.

    This is the ``preToolUse`` / ``postToolUse`` equivalent. It composes
    alongside the existing ``WorkItemEventMiddleware`` in the harness.

    Attributes:
        _hooks: The HarnessHooks instance containing callbacks.
    """

    def __init__(self, hooks: HarnessHooks) -> None:
        """Initialize with a HarnessHooks instance.

        Args:
            hooks: The hooks whose pre_tool/post_tool lists will be invoked.
        """
        self._hooks = hooks

    async def process(
        self,
        context: FunctionInvocationContext,
        next: Callable[[FunctionInvocationContext], Awaitable[None]],
    ) -> None:
        """Process a function invocation with pre/post hooks.

        Args:
            context: The function invocation context.
            next: The next handler in the middleware chain.
        """
        tool_name = context.function.name
        # Extract arguments as a dict for the hooks
        args = (
            context.arguments.model_dump()
            if hasattr(context.arguments, "model_dump")
            else dict(context.arguments)
            if context.arguments
            else {}
        )

        # Pre-tool hooks — any "deny" short-circuits execution
        for hook in self._hooks.pre_tool:
            try:
                result = await hook(tool_name, args)
                if result and result.decision == "deny":
                    logger.info("HarnessToolMiddleware: Tool '%s' denied: %s", tool_name, result.reason)
                    context.result = f"Denied: {result.reason}"
                    context.terminate = True
                    return
            except Exception:
                logger.exception("HarnessToolMiddleware: pre-tool hook error for '%s'", tool_name)

        # Execute the actual tool
        await next(context)

        # Post-tool hooks — observational, exceptions are logged but swallowed
        for hook in self._hooks.post_tool:
            try:
                await hook(tool_name, args, context.result)
            except Exception:
                logger.exception("HarnessToolMiddleware: post-tool hook error for '%s'", tool_name)
