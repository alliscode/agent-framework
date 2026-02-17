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
from typing import TYPE_CHECKING, Any

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
    TokenBudget,
    TurnContext,
    get_tokenizer,
)
from ._compaction_owner import CompactionOwnerService, OwnerCompactionResult, OwnerFallbackReason
from ._compaction_shadow import CompactionShadowService
from ._compaction_state import CompactionStateStore
from ._compaction_policy import CompactionPolicy, OwnerFallbackPolicy
from ._compaction_lifecycle import CompactionLifecycleEmitter
from ._compaction_thread import CompactionThreadService
from ._compaction_telemetry import (
    OwnerCompactionOutcome,
    pressure_metrics_payload,
)
from ._state import RepairComplete

if TYPE_CHECKING:
    from .._threads import AgentThread
    from ._compaction import ArtifactStore, CompactionStrategy
    from ._compaction_lifecycle import CompactionLifecycleEmitter
    from ._compaction_owner import CompactionOwnerService
    from ._compaction_policy import CompactionPolicy
    from ._compaction_shadow import CompactionShadowService
    from ._compaction_state import CompactionStateStore
    from ._compaction_thread import CompactionThreadService

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
        # Owner rollout policy
        owner_bootstrap_fallback_turn_limit: int = 2,
        enforce_owner_fallback_gate: bool = True,
        # Injectable runtime services
        state_store: "CompactionStateStore | None" = None,
        owner_service: "CompactionOwnerService | None" = None,
        shadow_service: "CompactionShadowService | None" = None,
        policy: "CompactionPolicy | None" = None,
        lifecycle: "CompactionLifecycleEmitter | None" = None,
        thread_service: "CompactionThreadService | None" = None,
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
            owner_bootstrap_fallback_turn_limit: Number of early turns where bootstrap
                fallback reasons (missing/invalid shared buffer snapshot) are allowed.
            enforce_owner_fallback_gate: Whether to mark non-allowed owner fallbacks as
                gate violations for rollout telemetry.
            state_store: Optional injected shared-state accessor.
            owner_service: Optional injected owner-mode compaction service.
            shadow_service: Optional injected shadow candidate service.
            policy: Optional injected compaction flow policy.
            lifecycle: Optional injected lifecycle emitter.
            thread_service: Optional injected thread compaction service.
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

        self._state_store = state_store or CompactionStateStore(
            max_input_tokens=self._max_input_tokens,
            soft_threshold_percent=self._soft_threshold_percent,
        )
        self._owner_service = owner_service or CompactionOwnerService(
            coordinator=self._coordinator,
            tokenizer=self._tokenizer,
            save_budget=lambda ctx, budget: self._state_store.save_budget(ctx, budget),
        )
        self._shadow_service = shadow_service or CompactionShadowService(
            coordinator=self._coordinator,
            tokenizer=self._tokenizer,
        )
        self._policy = policy or CompactionPolicy(
            owner_fallback_policy=OwnerFallbackPolicy(
                bootstrap_turn_limit=owner_bootstrap_fallback_turn_limit,
                enforce_gate=enforce_owner_fallback_gate,
            )
        )
        self._lifecycle = lifecycle or CompactionLifecycleEmitter()
        self._thread_service = thread_service or CompactionThreadService(
            coordinator=self._coordinator,
            tokenizer=self._tokenizer,
        )

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
        budget = await self._state_store.get_or_create_budget(ctx)

        # 2. Get current turn for context
        turn_number = await self._state_store.get_turn_count(ctx)
        owner_mode = await self._state_store.get_owner_mode(ctx)
        strategies_available = [s.name for s in self._strategies]
        await self._shadow_service.publish_candidate(
            ctx=ctx,
            owner_mode=owner_mode,
            turn_number=turn_number,
            current_tokens=budget.current_estimate,
            budget=budget,
            strategies_available=strategies_available,
        )

        # 3. Load current compaction plan
        plan, version = await self._state_store.load_plan(ctx)

        # 4. Read current token estimate from budget (updated by AgentTurnExecutor)
        current_tokens = budget.current_estimate

        # 5. Check if under pressure
        if not self._policy.is_under_pressure(budget):
            logger.debug(
                "CompactionExecutor: Under budget (%d/%d tokens)",
                current_tokens,
                budget.soft_threshold,
            )
            await self._state_store.save_budget(ctx, budget)
            await ctx.send_message(CompactionComplete(repairs_made=trigger.repairs_made))
            return

        logger.info(
            "CompactionExecutor: Context pressure detected (%d/%d tokens, %d over threshold)",
            current_tokens,
            budget.soft_threshold,
            budget.tokens_over,
        )
        owner_applied, owner_fallback_reason = await self._handle_owner_mode_attempt(
            trigger=trigger,
            ctx=ctx,
            owner_mode=owner_mode,
            budget=budget,
            current_tokens=current_tokens,
            turn_number=turn_number,
        )
        if owner_applied:
            return
        owner_fallback_reason_str = owner_fallback_reason.value if owner_fallback_reason else None
        is_owner_fallback_allowed = getattr(self._policy, "is_owner_fallback_allowed", None)
        if callable(is_owner_fallback_allowed):
            owner_fallback_allowed = is_owner_fallback_allowed(
                owner_mode=owner_mode,
                owner_fallback_reason=owner_fallback_reason_str,
                turn_number=turn_number,
            )
        else:
            owner_fallback_allowed = True

        is_owner_fallback_gate_violation = getattr(self._policy, "is_owner_fallback_gate_violation", None)
        if callable(is_owner_fallback_gate_violation):
            owner_fallback_gate_violation = is_owner_fallback_gate_violation(
                owner_mode=owner_mode,
                owner_fallback_reason=owner_fallback_reason_str,
                turn_number=turn_number,
            )
        else:
            owner_fallback_gate_violation = False
        if owner_fallback_gate_violation:
            logger.warning(
                "CompactionExecutor: owner fallback gate violation (mode=%s, reason=%s, turn=%d)",
                owner_mode,
                owner_fallback_reason_str,
                turn_number,
            )

        # 5b. Emit compaction_started event for UI feedback
        await self._lifecycle.emit_started(
            ctx,
            turn_number=turn_number,
            current_tokens=current_tokens,
            budget=budget,
            strategies_available=strategies_available,
            owner_mode=owner_mode,
            owner_fallback_reason=owner_fallback_reason_str,
            owner_fallback_allowed=owner_fallback_allowed,
            owner_fallback_gate_violation=owner_fallback_gate_violation,
        )

        # 6. Create/update plan
        # Note: In Plan Pipeline architecture, CompactionExecutor doesn't have
        # direct access to the message cache. The actual compaction analysis
        # happens in AgentTurnExecutor which has thread access.
        # CompactionExecutor's role is to:
        # - Check token budget and detect pressure
        # - Signal that compaction is needed
        # - AgentTurnExecutor calls compact_thread() when it has access to messages
        plan = self._policy.ensure_plan(plan)

        # Mark plan as needing compaction (AgentTurnExecutor will do actual work)
        plan_updated = False
        tokens_freed = 0
        proposals_applied = 0

        # 7. Store the plan
        await self._state_store.save_plan(ctx, plan, version)
        await self._state_store.save_budget(ctx, budget)

        # 8. Store compaction metrics
        await self._state_store.append_metrics(
            ctx,
            pressure_metrics_payload(
                turn_number=turn_number,
                tokens_before=current_tokens,
                tokens_freed=tokens_freed,
                proposals_applied=proposals_applied,
                under_pressure=self._policy.is_under_pressure(budget),
                owner_mode=owner_mode,
                owner_fallback_allowed=owner_fallback_allowed,
                owner_fallback_gate_violation=owner_fallback_gate_violation,
            ),
        )

        # 9. Emit lifecycle event (use "context_pressure" which is a valid event type)
        await self._lifecycle.emit_pressure(
            ctx,
            turn_number=turn_number,
            plan_updated=plan_updated,
            tokens_freed=tokens_freed,
            proposals_applied=proposals_applied,
            owner_mode=owner_mode,
            owner_fallback_reason=owner_fallback_reason_str,
            owner_fallback_allowed=owner_fallback_allowed,
            owner_fallback_gate_violation=owner_fallback_gate_violation,
        )

        # 10. Signal completion — tell AgentTurnExecutor that compaction is needed
        # At 80%: compaction_needed=True (ClearStrategy)
        # At 95%: compaction_needed=True + blocking=True (must compact before proceeding)
        is_blocking = self._policy.is_blocking(budget)

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

    async def _try_compaction_executor_ownership(
        self,
        *,
        ctx: WorkflowContext[Any],
        budget: TokenBudget,
        current_tokens: int,
        turn_number: int,
    ) -> OwnerCompactionResult:
        """Delegate owner-mode compaction to dedicated service."""
        return await self._owner_service.try_apply(
            ctx=ctx,
            budget=budget,
            current_tokens=current_tokens,
            turn_number=turn_number,
        )

    async def _handle_owner_mode_attempt(
        self,
        *,
        trigger: RepairComplete,
        ctx: WorkflowContext[Any],
        owner_mode: str,
        budget: TokenBudget,
        current_tokens: int,
        turn_number: int,
    ) -> tuple[bool, OwnerFallbackReason | None]:
        """Attempt owner-mode compaction and handle success side effects."""
        if not self._policy.should_attempt_owner_path(owner_mode):
            return False, None

        owner_result = await self._try_compaction_executor_ownership(
            ctx=ctx,
            budget=budget,
            current_tokens=current_tokens,
            turn_number=turn_number,
        )
        if owner_result.applied:
            tokens_freed = owner_result.tokens_freed
            proposals_applied = owner_result.proposals_applied
            strategies_applied = list(owner_result.strategies_applied or [])
            owner_outcome = OwnerCompactionOutcome(
                turn_number=turn_number,
                tokens_before=current_tokens,
                tokens_freed=tokens_freed,
                proposals_applied=proposals_applied,
                strategies_applied=strategies_applied,
                owner_mode=owner_mode,
                under_pressure=self._policy.is_under_pressure(budget),
            )
            await self._state_store.append_metrics(ctx, owner_outcome.metrics_payload())
            await self._lifecycle.emit_owner_completed(ctx, turn_number=turn_number, owner_outcome=owner_outcome)
            await ctx.send_message(
                CompactionComplete(
                    repairs_made=trigger.repairs_made,
                    plan_updated=True,
                    tokens_freed=tokens_freed,
                    proposals_applied=proposals_applied,
                    compaction_needed=False,  # owner path applied; agent_turn fallback not needed
                    blocking=budget.is_blocking,
                ),
            )
            return True, None

        logger.info(
            "CompactionExecutor: owner mode fallback to agent_turn (reason=%s)",
            owner_result.fallback_reason.value if owner_result.fallback_reason else "unknown",
        )
        return False, owner_result.fallback_reason

    async def compact_thread(
        self,
        thread: "AgentThread",
        current_plan: CompactionPlan | None,
        budget: TokenBudget,
        turn_number: int,
    ) -> CompactionPlan:
        """Run compaction on a thread (called by AgentTurnExecutor)."""
        return await self._thread_service.compact_thread(thread, current_plan, budget, turn_number)
