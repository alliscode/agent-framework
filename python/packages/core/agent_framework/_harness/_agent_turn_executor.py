# Copyright (c) Microsoft. All rights reserved.

"""AgentTurnExecutor for running agent turns within the harness."""

import logging
from typing import TYPE_CHECKING, Any

from .._agents import AgentProtocol
from .._threads import AgentThread  # kept for checkpoint backward-compat deserialization
from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._constants import (
    HARNESS_SHARED_TURN_BUFFER_KEY,
)
from ._state import RepairComplete, TurnComplete
from ._turn_buffer import ExecutorLocalTurnBuffer, SharedStateTurnBuffer
from ._turn_services import (
    TurnBudgetSyncService,
    TurnBufferSyncService,
    TurnCheckpointService,
    TurnCompletionSignalService,
    TurnCompactionEngine,
    TurnCompactionService,
    TurnCompactionViewService,
    TurnContinuationService,
    TurnErrorHandlingService,
    TurnExecutionCoordinator,
    TurnEventWriter,
    TurnGuidanceService,
    TurnInitialMessageService,
    TurnInvocationService,
    TurnJitInstructionService,
    TurnLoggingService,
    TurnOutcomeEvaluator,
    TurnOwnerModeService,
    TurnPreambleService,
    TurnPostprocessService,
    TurnPromptAssembler,
    TurnTokenBudgetService,
    TurnToolingService,
    TurnSystemMessageService,
    TurnWorkItemPromptService,
    TurnWorkItemStateService,
)

if TYPE_CHECKING:
    from ._compaction._strategies import Summarizer
    from ._hooks import HarnessHooks
    from ._jit_instructions import JitInstruction
    from ._work_items import WorkItemEventMiddleware, WorkItemTaskListProtocol

logger = logging.getLogger(__name__)


def _classify_compaction_level(
    strategies_applied: list[str],
    tokens_before: int = 0,
    tokens_after: int = 0,
) -> str:
    """Classify the compaction level based on strategies and actual impact.

    Returns:
        - "compressed": LLM summarization was used (context preserved at reduced fidelity)
        - "destructive": 90%+ of context was destroyed (regardless of strategy)
        - "optimized": minor reduction, context mostly intact
    """
    if "summarize" in strategies_applied:
        return "compressed"
    # Classify by actual impact when no LLM summarization was used
    if tokens_before > 0:
        reduction = (tokens_before - tokens_after) / tokens_before
        if reduction >= 0.9:
            return "destructive"
    elif "drop" in strategies_applied:
        return "destructive"
    return "optimized"


class AgentTurnExecutor(Executor):
    """Executor that orchestrates a single harness turn.

    The executor coordinates internal turn services for:
    - prompt assembly and guidance injection,
    - compaction execution,
    - tool/middleware composition,
    - token budget persistence,
    - outcome evaluation and continuation prompting,
    - transcript/lifecycle event writing,
    - work-item state syncing and invariant enforcement.
    """

    # Continuation nudge — injected when the agent ends a turn without calling work_complete
    DEFAULT_CONTINUATION_PROMPT = (
        "You stopped without calling work_complete or making tool calls.\n\n"
        "IMPORTANT: Do NOT repeat, rephrase, or summarize anything you already said. "
        "The user has already seen your previous output.\n\n"
        "Choose exactly one action:\n"
        "1. If your task is complete → call work_complete now (no extra text needed).\n"
        "2. If work remains → make the next tool call immediately.\n"
        "3. If something failed → try a different approach.\n\n"
        "Do not narrate. Do not restate your findings. Just act."
    )

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        agent_thread: AgentThread | None = None,
        enable_continuation_prompts: bool = True,
        max_continuation_prompts: int = 5,
        continuation_prompt: str | None = None,
        enable_compaction: bool = False,
        summarizer: "Summarizer | None" = None,
        artifact_store: Any | None = None,
        task_list: "WorkItemTaskListProtocol | None" = None,
        hooks: "HarnessHooks | None" = None,
        sub_agent_tools: list[Any] | None = None,
        jit_instructions: "list[JitInstruction] | None" = None,
        initial_message_service: TurnInitialMessageService | None = None,
        guidance_service: TurnGuidanceService | None = None,
        system_message_service: TurnSystemMessageService | None = None,
        checkpoint_service: TurnCheckpointService | None = None,
        work_item_prompt_service: TurnWorkItemPromptService | None = None,
        budget_sync_service: TurnBudgetSyncService | None = None,
        jit_instruction_service: TurnJitInstructionService | None = None,
        continuation_service: TurnContinuationService | None = None,
        execution_coordinator: TurnExecutionCoordinator | None = None,
        id: str = "harness_agent_turn",
    ):
        """Initialize the AgentTurnExecutor.

        Args:
            agent: The agent to run for each turn.
            agent_thread: Deprecated — ignored. The harness manages its own message cache
                and passes the full history directly to the agent on each call, so an
                AgentThread is not needed (and would cause duplicate messages).
            enable_continuation_prompts: Whether to prompt agent to continue if it stops early.
            max_continuation_prompts: Maximum continuation prompts before accepting done signal.
            continuation_prompt: Custom continuation prompt text.
            enable_compaction: Whether to apply CompactionPlan from SharedState.
            summarizer: Optional Summarizer for LLM-based context summarization during compaction.
                When provided, SummarizeStrategy is added between ClearStrategy and DropStrategy.
            artifact_store: Optional artifact store for ExternalizeStrategy. When provided along
                with a summarizer, ExternalizeStrategy is added to the compaction ladder.
            task_list: Optional work item task list for self-critique tracking.
            hooks: Optional harness hooks for pre/post tool interception.
            sub_agent_tools: Optional list of sub-agent tools (explore, run_task) to inject.
            jit_instructions: Optional custom JIT instructions. Defaults to built-in set.
            initial_message_service: Optional injected initial-message service.
            guidance_service: Optional injected guidance prompt service.
            system_message_service: Optional injected system-message append service.
            checkpoint_service: Optional injected checkpoint save/restore service.
            work_item_prompt_service: Optional injected work-item prompt service.
            budget_sync_service: Optional injected budget sync service.
            jit_instruction_service: Optional injected JIT instruction service.
            continuation_service: Optional injected continuation service.
            execution_coordinator: Optional injected turn execution coordinator.
            id: Unique identifier for this executor.
        """
        super().__init__(id)
        self._agent = agent
        self._turn_buffer = ExecutorLocalTurnBuffer()
        self._shared_turn_buffer = SharedStateTurnBuffer(key=HARNESS_SHARED_TURN_BUFFER_KEY)
        self._enable_continuation_prompts = enable_continuation_prompts
        self._max_continuation_prompts = max_continuation_prompts
        self._continuation_prompt = continuation_prompt or self.DEFAULT_CONTINUATION_PROMPT
        self._enable_compaction = enable_compaction
        self._compaction_count = 0
        self._summarizer = summarizer
        self._artifact_store = artifact_store
        self._task_list = task_list

        # Initialize tokenizer for cache token counting when compaction is enabled
        self._tokenizer: "ProviderAwareTokenizer | None" = None
        if enable_compaction:
            from ._compaction._tokenizer import ProviderAwareTokenizer, get_tokenizer

            self._tokenizer = get_tokenizer()

        # Create work item event middleware if task_list is provided
        self._work_item_middleware: "WorkItemEventMiddleware | None" = None
        if task_list is not None:
            from ._work_items import WorkItemEventMiddleware

            self._work_item_middleware = WorkItemEventMiddleware(task_list.ledger)

        # Create harness tool middleware if hooks have pre/post tool callbacks
        from ._hooks import HarnessToolMiddleware

        self._harness_tool_middleware: HarnessToolMiddleware | None = None
        if hooks and (hooks.pre_tool or hooks.post_tool):
            self._harness_tool_middleware = HarnessToolMiddleware(hooks)

        # Phase 5: Sub-agent tools
        self._sub_agent_tools = sub_agent_tools or []

        self._tooling_service = TurnToolingService(
            task_list=self._task_list,
            sub_agent_tools=self._sub_agent_tools,
            work_item_middleware=self._work_item_middleware,
            harness_tool_middleware=self._harness_tool_middleware,
        )
        self._token_budget_service = TurnTokenBudgetService(tokenizer=self._tokenizer)
        self._budget_sync_service = budget_sync_service or TurnBudgetSyncService(
            token_budget_service=self._token_budget_service,
            turn_buffer=self._turn_buffer,
        )
        self._buffer_sync_service = TurnBufferSyncService(shared_turn_buffer=self._shared_turn_buffer)
        self._event_writer = TurnEventWriter()
        self._owner_mode_service = TurnOwnerModeService()
        self._initial_message_service = initial_message_service or TurnInitialMessageService()
        self._system_message_service = system_message_service or TurnSystemMessageService(turn_buffer=self._turn_buffer)
        self._checkpoint_service = checkpoint_service or TurnCheckpointService(turn_buffer=self._turn_buffer)
        self._logging_service = TurnLoggingService()
        self._preamble_service = TurnPreambleService(event_writer=self._event_writer)
        self._completion_signal_service = TurnCompletionSignalService()
        self._continuation_service = continuation_service or TurnContinuationService(
            continuation_prompt=self._continuation_prompt,
            max_continuation_prompts=self._max_continuation_prompts,
            task_list=self._task_list,
            event_writer=self._event_writer,
            append_system_message=self._system_message_service.append_system_message,
        )
        self._error_handling_service = TurnErrorHandlingService(
            event_writer=self._event_writer,
            completion_signal_service=self._completion_signal_service,
        )
        self._outcome_evaluator = TurnOutcomeEvaluator(
            enable_continuation_prompts=self._enable_continuation_prompts,
            max_continuation_prompts=self._max_continuation_prompts,
            get_continuation_count=self._continuation_service.get_continuation_count,
            inject_continuation_prompt=self._continuation_service.inject_continuation_prompt,
        )
        self._work_item_state_service = TurnWorkItemStateService(
            task_list=self._task_list,
            work_item_middleware=self._work_item_middleware,
            event_writer=self._event_writer,
            append_system_message=self._system_message_service.append_system_message,
        )
        self._work_item_prompt_service = work_item_prompt_service or TurnWorkItemPromptService(
            turn_buffer=self._turn_buffer,
            event_writer=self._event_writer,
        )
        self._guidance_service = guidance_service or TurnGuidanceService(turn_buffer=self._turn_buffer)
        self._prompt_assembler = TurnPromptAssembler(
            task_list_enabled=self._task_list is not None,
            get_initial_message=self._initial_message_service.get_initial_message,
            append_to_cache=self._turn_buffer.append_message,
            inject_work_item_guidance=self._guidance_service.inject_work_item_guidance,
            inject_tool_strategy_guidance=self._guidance_service.inject_tool_strategy_guidance,
            inject_planning_prompt=self._guidance_service.inject_planning_prompt,
            inject_work_item_state=self._work_item_prompt_service.inject_work_item_state,
            maybe_inject_work_item_reminder=self._work_item_prompt_service.maybe_inject_work_item_reminder,
            inject_jit_instructions=lambda ctx, turn_count: self._jit_instruction_service.inject_instructions(
                ctx,
                turn_count=turn_count,
                compaction_count=self._compaction_count,
                tool_usage=self._work_item_prompt_service.get_tool_usage(),
            ),
        )
        self._compaction_view_service = TurnCompactionViewService(
            enable_compaction=self._enable_compaction,
            turn_buffer=self._turn_buffer,
        )
        self._compaction_engine = TurnCompactionEngine(
            turn_buffer=self._turn_buffer,
            tokenizer=self._tokenizer,
            summarizer=self._summarizer,
            artifact_store=self._artifact_store,
            load_compaction_plan=lambda ctx: self._compaction_view_service.load_compaction_plan(ctx),
            apply_compaction_plan=self._compaction_view_service.apply_compaction_plan,
        )
        self._compaction_service = TurnCompactionService(
            enable_compaction=self._enable_compaction,
            owner_mode_service=self._owner_mode_service,
            is_compaction_needed=self._is_compaction_needed,
            ensure_message_ids=self._compaction_engine.ensure_message_ids,
            get_budget_estimate=self._token_budget_service.get_budget_estimate,
            run_full_compaction=lambda ctx, turn_count: self._compaction_engine.run_full_compaction(
                ctx, turn_count, self._token_budget_service.get_budget_estimate
            ),
            apply_direct_clear=self._compaction_engine.apply_direct_clear,
            update_token_budget=lambda ctx: self._budget_sync_service.sync(ctx),
            classify_compaction_level=_classify_compaction_level,
            increment_compaction_count=lambda: setattr(self, "_compaction_count", self._compaction_count + 1),
        )
        self._invocation_service = TurnInvocationService(
            agent=self._agent,
            executor_id=self.id,
            tooling_service=self._tooling_service,
            work_item_state_service=self._work_item_state_service,
            turn_buffer=self._turn_buffer,
            get_messages_for_agent=lambda ctx, owner_mode: self._compaction_view_service.get_messages_for_agent(
                ctx,
                owner_mode,
            ),
        )
        self._postprocess_service = TurnPostprocessService(
            event_writer=self._event_writer,
            outcome_evaluator=self._outcome_evaluator,
            work_item_state_service=self._work_item_state_service,
            task_list_enabled=self._task_list is not None,
            update_token_budget=self._budget_sync_service.sync,
            get_budget_estimate=self._token_budget_service.get_budget_estimate,
            buffer_sync_service=self._buffer_sync_service,
            turn_buffer=self._turn_buffer,
        )
        self._execution_coordinator = execution_coordinator or TurnExecutionCoordinator(
            preamble_service=self._preamble_service,
            owner_mode_service=self._owner_mode_service,
            buffer_sync_service=self._buffer_sync_service,
            prompt_assembler=self._prompt_assembler,
            compaction_service=self._compaction_service,
            invocation_service=self._invocation_service,
            postprocess_service=self._postprocess_service,
            completion_signal_service=self._completion_signal_service,
            error_handling_service=self._error_handling_service,
            logging_service=self._logging_service,
            turn_buffer=self._turn_buffer,
            get_compaction_count=lambda: self._compaction_count,
        )

        # Phase 7: JIT instruction processor
        from ._jit_instructions import JitInstructionProcessor

        self._jit_processor = (
            JitInstructionProcessor(
                instructions=list(jit_instructions) if jit_instructions is not None else None  # type: ignore[arg-type]
            )
            if jit_instructions is not None
            else JitInstructionProcessor()
        )
        self._jit_instruction_service = jit_instruction_service or TurnJitInstructionService(
            jit_processor=self._jit_processor,
            turn_buffer=self._turn_buffer,
        )

    @property
    def _cache(self) -> list[Any]:
        """Back-compat view over the turn buffer during Phase 3 migration."""
        return self._turn_buffer.load_messages()

    @_cache.setter
    def _cache(self, messages: list[Any]) -> None:
        self._turn_buffer.replace_messages(messages)

    @handler
    async def run_turn(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
        """Execute a single agent turn.

        Args:
            trigger: The repair complete message indicating we can proceed.
            ctx: Workflow context for state access and message sending.
        """
        await self._execution_coordinator.execute_turn(trigger, ctx)

    @staticmethod
    def _is_compaction_needed(trigger: RepairComplete) -> bool:
        """Check if the trigger signals that compaction is needed.

        Args:
            trigger: The trigger message (may be CompactionComplete with compaction_needed flag).

        Returns:
            True if compaction should be applied to the cache.
        """
        from ._compaction_executor import CompactionComplete

        return isinstance(trigger, CompactionComplete) and trigger.compaction_needed

    async def on_checkpoint_save(self) -> dict[str, Any]:
        """Save executor state for checkpointing.

        Returns:
            State dictionary for restoration.
        """
        return await self._checkpoint_service.save()

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        """Restore executor state from checkpoint.

        Args:
            state: The saved state dictionary.
        """
        await self._checkpoint_service.restore(state)

    def set_initial_messages(self, messages: list[Any]) -> None:
        """Set initial messages for the agent conversation.

        Args:
            messages: The initial messages to seed the conversation with.
        """
        self._checkpoint_service.set_initial_messages(messages)

    async def _sync_work_item_ledger(self, ctx: WorkflowContext[Any]) -> None:
        """Back-compat shim for legacy tests/callers using executor-private API."""
        await self._work_item_state_service.sync_ledger(ctx)

    def _check_control_invariants(self, ledger: Any) -> str | None:
        """Back-compat shim for legacy tests/callers using executor-private API."""
        return self._work_item_state_service.check_control_invariants(ledger)
