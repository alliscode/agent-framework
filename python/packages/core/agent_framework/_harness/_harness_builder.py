# Copyright (c) Microsoft. All rights reserved.

"""HarnessWorkflowBuilder for creating agent harness workflows."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .._agents import AgentProtocol
from .._threads import AgentThread
from .._workflows._checkpoint import CheckpointStorage
from .._workflows._workflow import Workflow
from .._workflows._workflow_builder import WorkflowBuilder
from ._agent_turn_executor import AgentTurnExecutor
from ._constants import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_TURNS,
    DEFAULT_SOFT_THRESHOLD_PERCENT,
    DEFAULT_STALL_THRESHOLD,
)
from ._context_pressure_executor import ContextPressureExecutor
from ._repair_executor import RepairExecutor
from ._state import RepairTrigger
from ._stop_decision_executor import StopDecisionExecutor

if TYPE_CHECKING:
    from ._task_contract import TaskContract

# Key for harness configuration passed via kwargs
HARNESS_CONFIG_KEY = "harness.config"


class HarnessWorkflowBuilder:
    """Builder for creating agent harness workflows.

    The harness workflow wraps an agent with infrastructure for:
    - Durable execution with checkpointing
    - Turn-based execution with configurable limits
    - Transcript tracking for observability
    - Repair of execution invariants (dangling tool calls)
    - Layered stop conditions

    Example:
        .. code-block:: python

            from agent_framework import ChatAgent
            from agent_framework._harness import HarnessWorkflowBuilder

            # Create an agent
            agent = ChatAgent(chat_client=my_client, tools=[...])

            # Build the harness workflow
            builder = HarnessWorkflowBuilder(agent, max_turns=20)
            workflow = builder.build()

            # Run the harness
            result = await workflow.run(
                "Solve this complex task...",
                **builder.get_harness_kwargs(),
            )
    """

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        agent_thread: AgentThread | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        checkpoint_storage: CheckpointStorage | None = None,
        # Phase 2: Context pressure
        enable_context_pressure: bool = False,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        soft_threshold_percent: float = DEFAULT_SOFT_THRESHOLD_PERCENT,
        # Phase 3: Task contracts and stall detection
        task_contract: "TaskContract | None" = None,
        enable_stall_detection: bool = False,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        # Continuation prompts
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 2,
        continuation_prompt: str | None = None,
    ):
        """Initialize the HarnessWorkflowBuilder.

        Args:
            agent: The agent to wrap in the harness.
            agent_thread: Optional thread for the agent. If None, creates a new thread.
            max_turns: Maximum number of agent turns before stopping. Default is 50.
            checkpoint_storage: Optional checkpoint storage for durability.
            enable_context_pressure: Whether to enable context pressure management.
            max_input_tokens: Maximum input tokens for context pressure. Default is 100000.
            soft_threshold_percent: Percentage at which to trigger context pressure. Default is 0.85.
            task_contract: Optional task contract for automation mode. When provided,
                contract verification is automatically enabled.
            enable_stall_detection: Whether to detect stalled progress.
            stall_threshold: Number of unchanged turns before stall detection. Default is 3.
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done. Default is 2.
            continuation_prompt: Custom continuation prompt text.
        """
        self._agent = agent
        self._agent_thread = agent_thread
        self._max_turns = max_turns
        self._checkpoint_storage = checkpoint_storage
        self._enable_context_pressure = enable_context_pressure
        self._max_input_tokens = max_input_tokens
        self._soft_threshold_percent = soft_threshold_percent
        self._task_contract = task_contract
        self._enable_stall_detection = enable_stall_detection
        self._stall_threshold = stall_threshold
        self._enable_continuation_prompts = enable_continuation_prompts
        self._max_continuation_prompts = max_continuation_prompts
        self._continuation_prompt = continuation_prompt

    def get_harness_kwargs(self) -> dict[str, Any]:
        """Get the kwargs to pass to workflow.run() for harness configuration.

        Returns:
            Dictionary of kwargs including harness configuration.
        """
        config: dict[str, Any] = {
            "max_turns": self._max_turns,
        }

        if self._enable_context_pressure:
            config["context_pressure"] = {
                "enabled": True,
                "max_input_tokens": self._max_input_tokens,
                "soft_threshold_percent": self._soft_threshold_percent,
            }

        if self._task_contract is not None:
            config["task_contract"] = self._task_contract.to_dict()

        return {HARNESS_CONFIG_KEY: config}

    def build(self) -> Workflow:
        """Build the harness workflow.

        Returns:
            A Workflow instance configured as an agent harness.
        """
        # Create the workflow builder
        builder = WorkflowBuilder(
            name="AgentHarness",
            description="Agent harness workflow for durable, long-running agent execution",
        )

        # Register executors
        builder.register_executor(
            lambda: RepairExecutor(id="harness_repair"),
            name="repair",
        )

        # Optionally add context pressure executor
        if self._enable_context_pressure:
            builder.register_executor(
                lambda: ContextPressureExecutor(
                    max_input_tokens=self._max_input_tokens,
                    soft_threshold_percent=self._soft_threshold_percent,
                    id="harness_context_pressure",
                ),
                name="context_pressure",
            )

        # Capture continuation settings for lambda
        enable_cont = self._enable_continuation_prompts
        max_cont = self._max_continuation_prompts
        cont_prompt = self._continuation_prompt

        builder.register_executor(
            lambda: AgentTurnExecutor(
                self._agent,
                agent_thread=self._agent_thread,
                enable_continuation_prompts=enable_cont,
                max_continuation_prompts=max_cont,
                continuation_prompt=cont_prompt,
                id="harness_agent_turn",
            ),
            name="agent_turn",
        )

        # Configure stop decision with Phase 3 options
        # Contract verification is enabled when a contract is provided
        enable_contract_verification = self._task_contract is not None
        builder.register_executor(
            lambda: StopDecisionExecutor(
                enable_contract_verification=enable_contract_verification,
                enable_stall_detection=self._enable_stall_detection,
                stall_threshold=self._stall_threshold,
                id="harness_stop_decision",
            ),
            name="stop_decision",
        )

        # Wire the harness loop based on whether context pressure is enabled
        if self._enable_context_pressure:
            # repair -> context_pressure -> agent_turn -> stop_decision -> repair (loop)
            builder.add_edge("repair", "context_pressure")
            builder.add_edge("context_pressure", "agent_turn")
        else:
            # repair -> agent_turn -> stop_decision -> repair (loop)
            builder.add_edge("repair", "agent_turn")

        builder.add_edge("agent_turn", "stop_decision")
        builder.add_edge("stop_decision", "repair")

        # Set the start executor
        builder.set_start_executor("repair")

        # Configure checkpointing if provided
        if self._checkpoint_storage:
            builder.with_checkpointing(self._checkpoint_storage)

        # Set max iterations high enough for max_turns
        # Each turn is roughly 3-4 supersteps depending on context pressure
        supersteps_per_turn = 4 if self._enable_context_pressure else 3
        builder.set_max_iterations(self._max_turns * supersteps_per_turn * 2)

        return builder.build()


class AgentHarness:
    """Convenience wrapper for running an agent with harness infrastructure.

    Provides a simpler API than using HarnessWorkflowBuilder directly.

    Supports two modes:
    - **Interactive mode** (default): Model judgment determines completion.
      Use stall detection to catch spinning.
    - **Automation mode**: Provide a TaskContract for formal verification
      before accepting completion.

    Example:
        .. code-block:: python

            from agent_framework import ChatAgent
            from agent_framework._harness import AgentHarness, TaskContract

            agent = ChatAgent(chat_client=my_client, tools=[...])

            # Interactive mode - model judgment
            harness = AgentHarness(agent, max_turns=20)
            result = await harness.run("Solve this complex task...")

            # With stall detection
            harness = AgentHarness(
                agent,
                max_turns=50,
                enable_stall_detection=True,
                stall_threshold=3,
            )

            # Automation mode - formal contract verification
            harness = AgentHarness(
                agent,
                max_turns=50,
                task_contract=TaskContract.simple(
                    "Fix the bug",
                    "Identify root cause",
                    "Implement fix",
                    "Tests pass",
                ),
            )
    """

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        agent_thread: AgentThread | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        checkpoint_storage: CheckpointStorage | None = None,
        # Phase 2: Context pressure
        enable_context_pressure: bool = False,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        soft_threshold_percent: float = DEFAULT_SOFT_THRESHOLD_PERCENT,
        # Phase 3: Task contracts and stall detection
        task_contract: "TaskContract | None" = None,
        enable_stall_detection: bool = False,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        # Continuation prompts
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 2,
        continuation_prompt: str | None = None,
    ):
        """Initialize the AgentHarness.

        Args:
            agent: The agent to run with harness infrastructure.
            agent_thread: Optional thread for the agent. If None, creates a new thread.
            max_turns: Maximum number of agent turns before stopping. Default is 50.
            checkpoint_storage: Optional checkpoint storage for durability.
            enable_context_pressure: Whether to enable context pressure management.
            max_input_tokens: Maximum input tokens for context pressure. Default is 100000.
            soft_threshold_percent: Percentage at which to trigger context pressure. Default is 0.85.
            task_contract: Optional task contract for automation mode. When provided,
                contract verification is automatically enabled.
            enable_stall_detection: Whether to detect stalled progress.
            stall_threshold: Number of unchanged turns before stall detection. Default is 3.
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done. Default is 2.
            continuation_prompt: Custom continuation prompt text.
        """
        self._builder = HarnessWorkflowBuilder(
            agent,
            agent_thread=agent_thread,
            max_turns=max_turns,
            checkpoint_storage=checkpoint_storage,
            enable_context_pressure=enable_context_pressure,
            max_input_tokens=max_input_tokens,
            soft_threshold_percent=soft_threshold_percent,
            task_contract=task_contract,
            enable_stall_detection=enable_stall_detection,
            stall_threshold=stall_threshold,
            enable_continuation_prompts=enable_continuation_prompts,
            max_continuation_prompts=max_continuation_prompts,
            continuation_prompt=continuation_prompt,
        )
        self._workflow: Workflow | None = None
        self._harness_kwargs = self._builder.get_harness_kwargs()

    def _ensure_workflow(self) -> Workflow:
        """Ensure the workflow is built."""
        if self._workflow is None:
            self._workflow = self._builder.build()
        return self._workflow

    async def run(self, message: str | Any, **kwargs: Any) -> Any:
        """Run the harness to completion.

        Args:
            message: The initial message to send to the agent.
            **kwargs: Additional kwargs to pass to the workflow.

        Returns:
            The workflow run result containing all events and outputs.
        """
        workflow = self._ensure_workflow()

        # Store the initial message in harness config
        harness_config = dict(self._harness_kwargs.get(HARNESS_CONFIG_KEY, {}))
        harness_config["initial_message"] = message
        merged_kwargs = {**self._harness_kwargs, **kwargs, HARNESS_CONFIG_KEY: harness_config}

        # Create the initial trigger message
        trigger = RepairTrigger()

        return await workflow.run(trigger, **merged_kwargs)

    async def run_stream(self, message: str | Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Run the harness and stream events.

        Args:
            message: The initial message to send to the agent.
            **kwargs: Additional kwargs to pass to the workflow.

        Yields:
            Workflow events as they occur.
        """
        workflow = self._ensure_workflow()

        # Store the initial message in harness config
        harness_config = dict(self._harness_kwargs.get(HARNESS_CONFIG_KEY, {}))
        harness_config["initial_message"] = message
        merged_kwargs = {**self._harness_kwargs, **kwargs, HARNESS_CONFIG_KEY: harness_config}

        # Create the initial trigger message
        trigger = RepairTrigger()

        async for event in workflow.run_stream(trigger, **merged_kwargs):
            yield event
