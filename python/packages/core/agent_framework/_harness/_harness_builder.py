# Copyright (c) Microsoft. All rights reserved.

"""HarnessWorkflowBuilder for creating agent harness workflows."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

from .._agents import AgentProtocol
from .._memory import ContextProvider
from .._threads import AgentThread
from .._workflows._checkpoint import CheckpointStorage
from .._workflows._workflow import Workflow
from .._workflows._workflow_builder import WorkflowBuilder
from ._agent_turn_executor import AgentTurnExecutor
from ._compaction_executor import CompactionExecutor
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
    from .._clients import ChatClientProtocol
    from .._tools import ToolProtocol
    from ._compaction import (
        ArtifactStore,
        CompactionStore,
        CompactionStrategy,
        ProviderAwareTokenizer,
        Summarizer,
        SummaryCache,
    )
    from ._hooks import HarnessHooks
    from ._task_contract import TaskContract
    from ._work_items import WorkItemTaskListProtocol

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
        # Phase 2: Context pressure (legacy)
        enable_context_pressure: bool = False,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        soft_threshold_percent: float = DEFAULT_SOFT_THRESHOLD_PERCENT,
        # Phase 9: Production compaction (preferred)
        enable_compaction: bool = False,
        compaction_store: "CompactionStore | None" = None,
        artifact_store: "ArtifactStore | None" = None,
        summary_cache: "SummaryCache | None" = None,
        compaction_strategies: "list[CompactionStrategy] | None" = None,
        summarizer: "Summarizer | None" = None,
        tokenizer: "ProviderAwareTokenizer | None" = None,
        model_name: str = "gpt-4o",
        # Phase 3: Task contracts and stall detection
        task_contract: "TaskContract | None" = None,
        enable_stall_detection: bool = False,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        # Continuation prompts
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 5,
        continuation_prompt: str | None = None,
        # Work item tracking
        enable_work_items: bool = False,
        task_list: "WorkItemTaskListProtocol | None" = None,
        # Phase 2: Rich system prompt construction
        sandbox_path: str | None = None,
        # Phase 3: Hooks system
        hooks: "HarnessHooks | None" = None,
        # Phase 5: Sub-agent delegation
        sub_agent_client: "ChatClientProtocol | None" = None,
        sub_agent_tools: "Sequence[ToolProtocol | Callable[..., Any]] | None" = None,
    ):
        """Initialize the HarnessWorkflowBuilder.

        Args:
            agent: The agent to wrap in the harness.
            agent_thread: Optional thread for the agent. If None, creates a new thread.
            max_turns: Maximum number of agent turns before stopping. Default is 50.
            checkpoint_storage: Optional checkpoint storage for durability.
            enable_context_pressure: Whether to enable legacy context pressure management.
            max_input_tokens: Maximum input tokens for context pressure/compaction. Default is 100000.
            soft_threshold_percent: Percentage at which to trigger compaction. Default is 0.85.
            enable_compaction: Whether to enable production compaction (preferred over context_pressure).
            compaction_store: Injectable store for compaction plans. Defaults to in-memory.
            artifact_store: Injectable store for externalized content. None disables externalization.
            summary_cache: Injectable cache for LLM summaries. None disables caching.
            compaction_strategies: Custom list of compaction strategies.
            summarizer: LLM summarizer for summarize/externalize strategies.
            tokenizer: Tokenizer for token counting. Defaults to model-appropriate tokenizer.
            model_name: Model name for default tokenizer selection.
            task_contract: Optional task contract for automation mode. When provided,
                contract verification is automatically enabled.
            enable_stall_detection: Whether to detect stalled progress.
            stall_threshold: Number of unchanged turns before stall detection. Default is 3.
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done. Default is 5.
            continuation_prompt: Custom continuation prompt text.
            enable_work_items: Whether to enable work item tracking. When True, a default
                WorkItemTaskList is created and its tools are injected at runtime.
            task_list: Optional custom task list implementation. When provided,
                work item tracking is automatically enabled.
            sandbox_path: Optional path to the sandbox/working directory. When provided,
                environment context is injected into the system prompt.
            hooks: Optional harness hooks for lifecycle interception (pre/post tool,
                agent stop).
            sub_agent_client: Optional chat client for sub-agents (ideally a fast/cheap model).
                When provided, explore and run_task sub-agent tools are created and injected.
            sub_agent_tools: Optional tools to give sub-agents. When None, sub-agents get no tools.
        """
        self._agent = agent
        self._agent_thread = agent_thread
        self._max_turns = max_turns
        self._checkpoint_storage = checkpoint_storage
        self._enable_context_pressure = enable_context_pressure
        self._max_input_tokens = max_input_tokens
        self._soft_threshold_percent = soft_threshold_percent
        # Compaction settings
        self._enable_compaction = enable_compaction
        self._compaction_store = compaction_store
        self._artifact_store = artifact_store
        self._summary_cache = summary_cache
        self._compaction_strategies = compaction_strategies
        self._summarizer = summarizer
        self._tokenizer = tokenizer
        self._model_name = model_name
        # Task contract settings
        self._task_contract = task_contract
        self._enable_stall_detection = enable_stall_detection
        self._stall_threshold = stall_threshold
        self._enable_continuation_prompts = enable_continuation_prompts
        self._max_continuation_prompts = max_continuation_prompts
        self._continuation_prompt = continuation_prompt
        # Work item tracking
        self._task_list = self._resolve_task_list(enable_work_items, task_list)

        # Hooks
        self._hooks = hooks

        # Phase 5: Sub-agent delegation
        self._sub_agent_tools: list[Any] = []
        if sub_agent_client is not None:
            from ._sub_agents import create_document_tool, create_explore_tool, create_task_tool

            agent_tools = list(sub_agent_tools or [])
            self._sub_agent_tools = [
                create_explore_tool(sub_agent_client, agent_tools),
                create_task_tool(sub_agent_client, agent_tools),
                create_document_tool(sub_agent_client, agent_tools),
            ]

        # Phase 2: Wire context providers for rich system prompt
        self._wire_context_providers(
            sandbox_path,
            enable_work_items or task_list is not None,
            enable_sub_agents=len(self._sub_agent_tools) > 0,
        )

    @staticmethod
    def _resolve_task_list(
        enable_work_items: bool,
        task_list: "WorkItemTaskListProtocol | None",
    ) -> "WorkItemTaskListProtocol | None":
        """Resolve the task list from flags.

        If task_list is provided, use it directly.
        If enable_work_items is True, create a default WorkItemTaskList.
        """
        if task_list is not None:
            return task_list
        if enable_work_items:
            from ._work_items import WorkItemTaskList

            return WorkItemTaskList()
        return None

    def _wire_context_providers(
        self,
        sandbox_path: str | None,
        enable_work_items: bool,
        *,
        enable_sub_agents: bool = False,
    ) -> None:
        """Wire EnvironmentContextProvider and HarnessGuidanceProvider to the agent.

        Appends harness-level context providers to the agent's existing context_provider,
        preserving any user-configured providers.

        Args:
            sandbox_path: Optional sandbox directory path for environment context.
            enable_work_items: Whether work item guidance should be included.
            enable_sub_agents: Whether sub-agent guidance should be included.
        """
        from .._memory import AggregateContextProvider
        from ._context_providers import EnvironmentContextProvider, HarnessGuidanceProvider

        providers: list[ContextProvider] = []

        if sandbox_path:
            providers.append(EnvironmentContextProvider(sandbox_path=sandbox_path))

        providers.append(
            HarnessGuidanceProvider(
                enable_work_items=enable_work_items,
                enable_tool_guidance=True,
                enable_sub_agents=enable_sub_agents,
            )
        )

        if not providers:
            return

        agent = self._agent
        if hasattr(agent, "context_provider"):
            existing = agent.context_provider
            if isinstance(existing, AggregateContextProvider):
                for p in providers:
                    existing.add(p)
            elif existing is not None:
                agent.context_provider = AggregateContextProvider([existing, *providers])
            else:
                agent.context_provider = AggregateContextProvider(providers)

    def get_harness_kwargs(self) -> dict[str, Any]:
        """Get the kwargs to pass to workflow.run() for harness configuration.

        Returns:
            Dictionary of kwargs including harness configuration.
        """
        config: dict[str, Any] = {
            "max_turns": self._max_turns,
        }

        if self._enable_compaction:
            config["compaction"] = {
                "enabled": True,
                "max_input_tokens": self._max_input_tokens,
                "soft_threshold_percent": self._soft_threshold_percent,
            }
        elif self._enable_context_pressure:
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

        # Optionally add compaction or context pressure executor
        # Compaction (Phase 9) is preferred over context pressure (Phase 2)
        if self._enable_compaction:
            # Capture settings for lambda
            compaction_store = self._compaction_store
            artifact_store = self._artifact_store
            summary_cache = self._summary_cache
            strategies = self._compaction_strategies
            summarizer = self._summarizer
            tokenizer = self._tokenizer
            model_name = self._model_name
            max_tokens = self._max_input_tokens
            threshold = self._soft_threshold_percent

            builder.register_executor(
                lambda: CompactionExecutor(
                    compaction_store=compaction_store,
                    artifact_store=artifact_store,
                    summary_cache=summary_cache,
                    strategies=strategies,
                    summarizer=summarizer,
                    max_input_tokens=max_tokens,
                    soft_threshold_percent=threshold,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    id="harness_compaction",
                ),
                name="compaction",
            )
        elif self._enable_context_pressure:
            builder.register_executor(
                lambda: ContextPressureExecutor(
                    max_input_tokens=self._max_input_tokens,
                    soft_threshold_percent=self._soft_threshold_percent,
                    id="harness_context_pressure",
                ),
                name="context_pressure",
            )

        # Capture continuation and work item settings for lambda
        enable_cont = self._enable_continuation_prompts
        max_cont = self._max_continuation_prompts
        cont_prompt = self._continuation_prompt
        task_list = self._task_list
        enable_compact = self._enable_compaction
        hooks = self._hooks
        sub_agent_tools = self._sub_agent_tools

        builder.register_executor(
            lambda: AgentTurnExecutor(
                self._agent,
                agent_thread=self._agent_thread,
                enable_continuation_prompts=enable_cont,
                max_continuation_prompts=max_cont,
                continuation_prompt=cont_prompt,
                enable_compaction=enable_compact,
                task_list=task_list,
                hooks=hooks,
                sub_agent_tools=sub_agent_tools,
                id="harness_agent_turn",
            ),
            name="agent_turn",
        )

        # Configure stop decision with Phase 3 options
        # Contract verification is enabled when a contract is provided
        enable_contract_verification = self._task_contract is not None
        enable_work_item_verification = self._task_list is not None
        builder.register_executor(
            lambda: StopDecisionExecutor(
                enable_contract_verification=enable_contract_verification,
                enable_stall_detection=self._enable_stall_detection,
                enable_work_item_verification=enable_work_item_verification,
                require_task_complete=True,
                stall_threshold=self._stall_threshold,
                hooks=hooks,
                id="harness_stop_decision",
            ),
            name="stop_decision",
        )

        # Wire the harness loop based on whether compaction/context pressure is enabled
        if self._enable_compaction:
            # repair -> compaction -> agent_turn -> stop_decision -> repair (loop)
            builder.add_edge("repair", "compaction")
            builder.add_edge("compaction", "agent_turn")
        elif self._enable_context_pressure:
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
        # Each turn is roughly 3-4 supersteps depending on compaction/context pressure
        has_pressure_executor = self._enable_compaction or self._enable_context_pressure
        supersteps_per_turn = 4 if has_pressure_executor else 3
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

            # With work item tracking (self-critique loop)
            harness = AgentHarness(agent, enable_work_items=True, max_turns=20)

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
        # Phase 2: Context pressure (legacy)
        enable_context_pressure: bool = False,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        soft_threshold_percent: float = DEFAULT_SOFT_THRESHOLD_PERCENT,
        # Phase 9: Production compaction (preferred)
        enable_compaction: bool = False,
        compaction_store: "CompactionStore | None" = None,
        artifact_store: "ArtifactStore | None" = None,
        summary_cache: "SummaryCache | None" = None,
        compaction_strategies: "list[CompactionStrategy] | None" = None,
        summarizer: "Summarizer | None" = None,
        tokenizer: "ProviderAwareTokenizer | None" = None,
        model_name: str = "gpt-4o",
        # Phase 3: Task contracts and stall detection
        task_contract: "TaskContract | None" = None,
        enable_stall_detection: bool = False,
        stall_threshold: int = DEFAULT_STALL_THRESHOLD,
        # Continuation prompts
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 5,
        continuation_prompt: str | None = None,
        # Work item tracking
        enable_work_items: bool = False,
        task_list: "WorkItemTaskListProtocol | None" = None,
        # Phase 2: Rich system prompt construction
        sandbox_path: str | None = None,
        # Phase 3: Hooks system
        hooks: "HarnessHooks | None" = None,
        # Phase 5: Sub-agent delegation
        sub_agent_client: "ChatClientProtocol | None" = None,
        sub_agent_tools: "Sequence[ToolProtocol | Callable[..., Any]] | None" = None,
    ):
        """Initialize the AgentHarness.

        Args:
            agent: The agent to run with harness infrastructure.
            agent_thread: Optional thread for the agent. If None, creates a new thread.
            max_turns: Maximum number of agent turns before stopping. Default is 50.
            checkpoint_storage: Optional checkpoint storage for durability.
            enable_context_pressure: Whether to enable legacy context pressure management.
            max_input_tokens: Maximum input tokens for context pressure/compaction. Default is 100000.
            soft_threshold_percent: Percentage at which to trigger compaction. Default is 0.85.
            enable_compaction: Whether to enable production compaction (preferred over context_pressure).
            compaction_store: Injectable store for compaction plans. Defaults to in-memory.
            artifact_store: Injectable store for externalized content. None disables externalization.
            summary_cache: Injectable cache for LLM summaries. None disables caching.
            compaction_strategies: Custom list of compaction strategies.
            summarizer: LLM summarizer for summarize/externalize strategies.
            tokenizer: Tokenizer for token counting. Defaults to model-appropriate tokenizer.
            model_name: Model name for default tokenizer selection.
            task_contract: Optional task contract for automation mode. When provided,
                contract verification is automatically enabled.
            enable_stall_detection: Whether to detect stalled progress.
            stall_threshold: Number of unchanged turns before stall detection. Default is 3.
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done. Default is 5.
            continuation_prompt: Custom continuation prompt text.
            enable_work_items: Whether to enable work item tracking. When True, a default
                WorkItemTaskList is created and its tools are injected at runtime.
            task_list: Optional custom task list implementation. When provided,
                work item tracking is automatically enabled.
            sandbox_path: Optional path to the sandbox/working directory. When provided,
                environment context is injected into the system prompt.
            hooks: Optional harness hooks for lifecycle interception (pre/post tool,
                agent stop).
            sub_agent_client: Optional chat client for sub-agents (ideally a fast/cheap model).
                When provided, explore and run_task sub-agent tools are created and injected.
            sub_agent_tools: Optional tools to give sub-agents. When None, sub-agents get no tools.
        """
        self._builder = HarnessWorkflowBuilder(
            agent,
            agent_thread=agent_thread,
            max_turns=max_turns,
            checkpoint_storage=checkpoint_storage,
            enable_context_pressure=enable_context_pressure,
            max_input_tokens=max_input_tokens,
            soft_threshold_percent=soft_threshold_percent,
            enable_compaction=enable_compaction,
            compaction_store=compaction_store,
            artifact_store=artifact_store,
            summary_cache=summary_cache,
            compaction_strategies=compaction_strategies,
            summarizer=summarizer,
            tokenizer=tokenizer,
            model_name=model_name,
            task_contract=task_contract,
            enable_stall_detection=enable_stall_detection,
            stall_threshold=stall_threshold,
            enable_continuation_prompts=enable_continuation_prompts,
            max_continuation_prompts=max_continuation_prompts,
            continuation_prompt=continuation_prompt,
            enable_work_items=enable_work_items,
            task_list=task_list,
            sandbox_path=sandbox_path,
            hooks=hooks,
            sub_agent_client=sub_agent_client,
            sub_agent_tools=sub_agent_tools,
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
