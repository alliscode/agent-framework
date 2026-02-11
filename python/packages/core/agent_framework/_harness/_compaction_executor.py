# Copyright (c) Microsoft. All rights reserved.

"""CompactionExecutor for production context compaction.

This executor replaces the placeholder ContextPressureExecutor with
production-quality context compaction using the compaction system.

Key features:
- Injectable storage protocols (supports different deployment environments)
- Uses CompactionCoordinator with strategy ladder
- Stores CompactionPlan in SharedState for AgentTurnExecutor to apply
- Emits observability events
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from .._workflows._executor import Executor, handler
from .._workflows._workflow_context import WorkflowContext
from ._compaction import (
    ClearStrategy,
    CompactionCoordinator,
    CompactionPlan,
    CompactionStore,
    InMemoryCompactionStore,
    ProviderAwareTokenizer,
    Summarizer,
    SummaryCache,
    TurnContext,
    get_tokenizer,
)
from ._compaction import TokenBudget as TokenBudgetV2
from ._constants import (
    HARNESS_COMPACTION_METRICS_KEY,
    HARNESS_COMPACTION_PLAN_KEY,
    HARNESS_TOKEN_BUDGET_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from ._context_pressure import TokenBudget
from ._state import HarnessLifecycleEvent, RepairComplete

if TYPE_CHECKING:
    from .._threads import AgentThread
    from ._compaction import ArtifactStore, CompactionStrategy

logger = logging.getLogger(__name__)


@dataclass
class CompactionComplete(RepairComplete):
    """Message indicating compaction check is complete.

    Extends RepairComplete to maintain compatibility with executor chain.

    Attributes:
        plan_updated: Whether a new compaction plan was created.
        tokens_freed: Estimated tokens freed by compaction.
        proposals_applied: Number of proposals applied.
        compaction_needed: Whether context pressure was detected and compaction should be applied.
    """

    plan_updated: bool = False
    tokens_freed: int = 0
    proposals_applied: int = 0
    compaction_needed: bool = False
    blocking: bool = False


class CompactionExecutor(Executor):
    """Executor that performs production context compaction.

    This executor runs before each agent turn to check if the conversation
    context is approaching token limits. If pressure is detected, it uses
    the CompactionCoordinator to apply strategies in order from least to
    most aggressive:

    1. Clear - Replace older tool results with placeholders
    2. Summarize - LLM-compress older spans (if summarizer provided)
    3. Externalize - Store large content to artifact store (if stores provided)
    4. Drop - Remove content entirely (last resort)

    The executor produces a CompactionPlan stored in SharedState, which
    AgentTurnExecutor uses to build the actual prompt.

    This design supports deployment flexibility:
    - InMemory stores for local/testing
    - Filesystem stores for Claude Code
    - Database stores for government cloud
    """

    def __init__(
        self,
        *,
        # Storage protocols (injectable for different environments)
        compaction_store: CompactionStore | None = None,
        artifact_store: "ArtifactStore | None" = None,
        summary_cache: SummaryCache | None = None,
        # Strategy configuration
        strategies: "list[CompactionStrategy] | None" = None,
        summarizer: Summarizer | None = None,
        # Budget configuration
        max_input_tokens: int = 128000,
        soft_threshold_percent: float = 0.80,
        # Tokenizer
        tokenizer: ProviderAwareTokenizer | None = None,
        model_name: str = "gpt-4o",
        # Executor ID
        id: str = "harness_compaction",
    ):
        """Initialize the CompactionExecutor.

        Args:
            compaction_store: Store for compaction plans. Defaults to in-memory.
            artifact_store: Store for externalized content. None disables externalization.
            summary_cache: Cache for LLM summaries. None disables caching.
            strategies: Custom list of strategies. Defaults to ClearStrategy only.
            summarizer: LLM summarizer for summarize/externalize strategies.
            max_input_tokens: Maximum allowed input tokens.
            soft_threshold_percent: Percentage at which to trigger compaction (0-1).
            tokenizer: Tokenizer for counting. Defaults to model-appropriate tokenizer.
            model_name: Model name for default tokenizer selection.
            id: Unique identifier for this executor.
        """
        super().__init__(id)

        # Storage
        self._compaction_store = compaction_store or InMemoryCompactionStore()
        self._artifact_store = artifact_store
        self._summary_cache = summary_cache
        self._summarizer = summarizer

        # Budget
        self._max_input_tokens = max_input_tokens
        self._soft_threshold_percent = soft_threshold_percent

        # Tokenizer
        self._tokenizer = tokenizer or get_tokenizer(model_name)

        # Build strategies
        self._strategies = strategies or self._build_default_strategies()

        # Coordinator
        self._coordinator = CompactionCoordinator(self._strategies)

    def _build_default_strategies(self) -> "list[CompactionStrategy]":
        """Build default strategy list based on available components."""
        from ._compaction import DropStrategy

        strategies: list[CompactionStrategy] = []

        # Always include ClearStrategy (no dependencies)
        strategies.append(ClearStrategy())

        # Include SummarizeStrategy if summarizer provided
        if self._summarizer is not None:
            from ._compaction import SummarizeStrategy

            strategies.append(SummarizeStrategy(self._summarizer))

        # Include ExternalizeStrategy if artifact_store and summarizer provided
        if self._artifact_store is not None and self._summarizer is not None:
            from ._compaction import ExternalizeStrategy

            strategies.append(
                ExternalizeStrategy(
                    self._artifact_store,
                    self._summarizer,
                ),
            )

        # Always include DropStrategy (last resort)
        strategies.append(DropStrategy())

        return strategies

    @handler
    async def check_compaction(self, trigger: RepairComplete, ctx: WorkflowContext[CompactionComplete]) -> None:
        """Check context pressure and apply compaction if needed.

        Args:
            trigger: The repair complete message indicating we can proceed.
            ctx: Workflow context for state access and message sending.
        """
        # 1. Get or create token budget (using existing TokenBudget from context_pressure)
        budget = await self._get_or_create_budget(ctx)

        # 2. Get current turn for context
        turn_number = await self._get_turn_count(ctx)

        # 3. Load current compaction plan
        plan, version = await self._load_plan(ctx)

        # 4. Read current token estimate from budget (updated by AgentTurnExecutor)
        current_tokens = budget.current_estimate

        # 5. Check if under pressure
        if not budget.is_under_pressure:
            logger.debug(
                "CompactionExecutor: Under budget (%d/%d tokens)",
                current_tokens,
                budget.soft_threshold,
            )
            await self._save_budget(ctx, budget)
            await ctx.send_message(CompactionComplete(repairs_made=trigger.repairs_made))
            return

        logger.info(
            "CompactionExecutor: Context pressure detected (%d/%d tokens, %d over threshold)",
            current_tokens,
            budget.soft_threshold,
            budget.tokens_over_threshold,
        )

        # 5b. Emit compaction_started event for UI feedback
        strategies_available = [s.name for s in self._strategies]
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="compaction_started",
                turn_number=turn_number,
                data={
                    "current_tokens": current_tokens,
                    "soft_threshold": budget.soft_threshold,
                    "tokens_over_threshold": budget.tokens_over_threshold,
                    "strategies_available": strategies_available,
                },
            ),
        )

        # 6. Create/update plan
        # Note: In Plan Pipeline architecture, CompactionExecutor doesn't have
        # direct access to the message cache. The actual compaction analysis
        # happens in AgentTurnExecutor which has thread access.
        # CompactionExecutor's role is to:
        # - Check token budget and detect pressure
        # - Signal that compaction is needed
        # - AgentTurnExecutor calls compact_thread() when it has access to messages
        if plan is None:
            plan = CompactionPlan.create_empty(thread_id="harness")

        # Mark plan as needing compaction (AgentTurnExecutor will do actual work)
        plan_updated = False
        tokens_freed = 0
        proposals_applied = 0

        # 7. Store the plan
        await self._save_plan(ctx, plan, version)
        await self._save_budget(ctx, budget)

        # 8. Store compaction metrics
        await self._save_metrics(
            ctx,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "turn_number": turn_number,
                "tokens_before": current_tokens,
                "tokens_freed": tokens_freed,
                "proposals_applied": proposals_applied,
                "under_pressure": budget.is_under_pressure,
            },
        )

        # 9. Emit lifecycle event (use "context_pressure" which is a valid event type)
        await ctx.add_event(
            HarnessLifecycleEvent(
                event_type="context_pressure",
                turn_number=turn_number,
                data={
                    "plan_updated": plan_updated,
                    "tokens_freed": tokens_freed,
                    "proposals_applied": proposals_applied,
                },
            ),
        )

        # 10. Signal completion — tell AgentTurnExecutor that compaction is needed
        # At 80%: compaction_needed=True (ClearStrategy)
        # At 95%: compaction_needed=True + blocking=True (must compact before proceeding)
        is_blocking = budget.is_blocking if hasattr(budget, "is_blocking") else False

        await ctx.send_message(
            CompactionComplete(
                repairs_made=trigger.repairs_made,
                plan_updated=plan_updated,
                tokens_freed=tokens_freed,
                proposals_applied=proposals_applied,
                compaction_needed=True,
                blocking=is_blocking,
            ),
        )

    async def compact_thread(
        self,
        thread: "AgentThread",
        current_plan: CompactionPlan | None,
        budget: TokenBudgetV2,
        turn_number: int,
    ) -> CompactionPlan:
        """Run compaction on a thread (called by AgentTurnExecutor).

        This is the main compaction entry point when thread access is available.

        Args:
            thread: The agent thread with messages.
            current_plan: Existing compaction plan to build on.
            budget: Current token budget (v2 from compaction package).
            turn_number: Current turn number.

        Returns:
            Updated CompactionPlan.
        """
        # Need current token count to check if over threshold
        # This should be provided by the caller or calculated from thread
        if thread.message_store is not None:
            messages = await thread.message_store.list_messages()
            current_tokens = sum(self._tokenizer.count_tokens(msg.text) for msg in messages)
        else:
            current_tokens = 0

        # Calculate tokens to free
        tokens_over = budget.tokens_over_threshold(current_tokens)
        tokens_to_free = max(0, tokens_over + int(budget.max_input_tokens * 0.1))

        if tokens_to_free == 0:
            return current_plan or CompactionPlan.create_empty(thread_id="harness")

        # Create turn context
        turn_context = TurnContext(turn_number=turn_number)

        # Run coordinator
        result = await self._coordinator.compact(
            thread,
            current_plan,
            budget,
            self._tokenizer,
            tokens_to_free=tokens_to_free,
            turn_context=turn_context,
        )

        logger.info(
            "CompactionExecutor: Compaction complete - freed %d tokens, applied %d/%d proposals",
            result.tokens_freed,
            result.proposals_applied,
            result.proposals_generated,
        )

        return result.plan

    async def _get_or_create_budget(self, ctx: WorkflowContext[Any]) -> TokenBudget:
        """Get or create the token budget from shared state."""
        try:
            budget_data = await ctx.get_shared_state(HARNESS_TOKEN_BUDGET_KEY)
            if budget_data and isinstance(budget_data, dict):
                budget_dict = cast("dict[str, Any]", budget_data)
                return TokenBudget.from_dict(budget_dict)
        except KeyError:
            pass

        # Create default budget
        return TokenBudget(
            max_input_tokens=self._max_input_tokens,
            soft_threshold_percent=self._soft_threshold_percent,
        )

    async def _save_budget(self, ctx: WorkflowContext[Any], budget: TokenBudget) -> None:
        """Save the token budget to shared state."""
        await ctx.set_shared_state(HARNESS_TOKEN_BUDGET_KEY, budget.to_dict())

    async def _get_turn_count(self, ctx: WorkflowContext[Any]) -> int:
        """Get the current turn count."""
        try:
            return int(await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY) or 0)
        except KeyError:
            return 0

    async def _load_plan(self, ctx: WorkflowContext[Any]) -> tuple[CompactionPlan | None, int]:
        """Load the current compaction plan from shared state.

        Args:
            ctx: The workflow context.

        Returns:
            Tuple of (plan, version). Plan is None if no plan exists.
        """
        try:
            plan_data = await ctx.get_shared_state(HARNESS_COMPACTION_PLAN_KEY)
            if plan_data and isinstance(plan_data, dict):
                plan_dict = cast("dict[str, Any]", plan_data)
                version = int(plan_dict.get("_version", plan_dict.get("thread_version", 0)))
                plan = CompactionPlan.from_dict(plan_dict)
                return plan, version
        except KeyError:
            pass
        return None, 0

    async def _save_plan(self, ctx: WorkflowContext[Any], plan: CompactionPlan, version: int) -> None:
        """Save the compaction plan to shared state."""
        plan_data = plan.to_dict()
        plan_data["_version"] = version + 1
        await ctx.set_shared_state(HARNESS_COMPACTION_PLAN_KEY, plan_data)

    async def _save_metrics(self, ctx: WorkflowContext[Any], metrics: dict[str, Any]) -> None:
        """Save compaction metrics to shared state."""
        try:
            existing: list[dict[str, Any]] = list(await ctx.get_shared_state(HARNESS_COMPACTION_METRICS_KEY) or [])
        except KeyError:
            existing = []

        existing.append(metrics)
        # Keep only last 100 metrics
        if len(existing) > 100:
            existing = existing[-100:]

        await ctx.set_shared_state(HARNESS_COMPACTION_METRICS_KEY, existing)
