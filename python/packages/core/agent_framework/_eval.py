# Copyright (c) Microsoft. All rights reserved.

"""Provider-agnostic evaluation framework for Microsoft Agent Framework.

Defines the core evaluation types and orchestration functions that work with
any evaluation provider (Azure AI Foundry, local evaluators, third-party
libraries, etc.).

Typical usage::

    from agent_framework import evaluate_agent, EvalResults
    from agent_framework_azure_ai import FoundryEvals

    evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")
    results = await evaluate_agent(agent=agent, queries=["Hello"], evaluators=evals)
    results.assert_passed()
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, Sequence, Union, runtime_checkable

from ._tools import FunctionTool
from ._types import AgentResponse, Message

if TYPE_CHECKING:
    from ._workflows._agent_executor import AgentExecutorResponse
    from ._workflows._workflow import Workflow, WorkflowRunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


class ConversationSplit(str, Enum):
    """Built-in strategies for splitting a conversation into query/response halves.

    Different splits evaluate different aspects of agent behavior:

    - ``LAST_TURN``: Split at the last user message.  Everything up to and
      including that message is the query; everything after is the response.
      Evaluates whether the agent answered the *latest* question well.

    - ``FULL``: The first user message (and any preceding system messages) is
      the query; the entire remainder of the conversation is the response.
      Evaluates whether the *whole conversation trajectory* served the
      original request.

    For custom splits (e.g. split before a memory-retrieval tool call),
    pass a callable instead — see ``ConversationSplitter``.
    """

    LAST_TURN = "last_turn"
    FULL = "full"


ConversationSplitter = Union[
    ConversationSplit,
    Callable[[list[Message]], tuple[list[Message], list[Message]]],
]
"""Type accepted by ``EvalItem.to_dict(split=...)``.

Either a built-in ``ConversationSplit`` enum value **or** a callable with
signature::

    def my_splitter(conversation: list[Message]) -> tuple[list[Message], list[Message]]:
        '''Return (query_messages, response_messages).'''

Custom splitters let you evaluate domain-specific boundaries — for example,
splitting just before a memory-retrieval tool call to evaluate recall quality::

    def split_before_memory(conversation):
        for i, msg in enumerate(conversation):
            for c in msg.contents or []:
                if c.type == "function_call" and c.name == "retrieve_memory":
                    return conversation[:i], conversation[i:]
        # Fallback: split at last user message
        return EvalItem._split_last_turn_static(conversation)

    item.to_dict(split=split_before_memory)
"""


@dataclass
class EvalItem:
    """A single item to be evaluated.

    Represents one query/response interaction in a provider-agnostic format.
    ``conversation`` is the single source of truth — ``query`` and ``response``
    are derived from it via the split strategy.

    Attributes:
        conversation: Full conversation as ``Message`` objects.
        tools: Typed tool objects (e.g. ``FunctionTool``) for evaluator logic.
        context: Optional grounding context document.
        expected: Optional expected output for ground-truth comparison.
        response_id: Responses API response ID (for providers that support
            server-side retrieval).
        split_strategy: Split strategy controlling how ``query`` and
            ``response`` are derived from the conversation. Defaults to
            ``ConversationSplit.LAST_TURN``.
    """

    conversation: list[Message]
    tools: list[FunctionTool] | None = None
    context: str | None = None
    expected: str | None = None
    response_id: str | None = None
    split_strategy: ConversationSplitter | None = None

    @property
    def query(self) -> str:
        """User query text, derived from the query side of the conversation split."""
        query_msgs, _ = self._split_conversation(
            self.split_strategy or ConversationSplit.LAST_TURN
        )
        user_texts = [m.text for m in query_msgs if m.role == "user" and m.text]
        return " ".join(user_texts).strip()

    @property
    def response(self) -> str:
        """Agent response text, derived from the response side of the conversation split."""
        _, response_msgs = self._split_conversation(
            self.split_strategy or ConversationSplit.LAST_TURN
        )
        assistant_texts = [m.text for m in response_msgs if m.role == "assistant" and m.text]
        return " ".join(assistant_texts).strip()

    def to_dict(
        self,
        *,
        split: ConversationSplitter | None = None,
    ) -> dict[str, Any]:
        """Convert to a flat dict for serialization.

        Produces ``query``, ``response``, ``query_messages`` and
        ``response_messages`` by splitting the conversation according to
        *split*:

        - ``LAST_TURN`` (default): split at the last user message.
        - ``FULL``: split after the first user message.
        - A callable: your function receives the conversation list and
          returns ``(query_messages, response_messages)``.

        When *split* is ``None`` (the default), uses ``self.split_strategy``
        if set, otherwise ``ConversationSplit.LAST_TURN``.
        """
        effective_split = split or self.split_strategy or ConversationSplit.LAST_TURN
        query_msgs, response_msgs = self._split_conversation(effective_split)

        query_text = " ".join(
            m.text for m in query_msgs if m.role == "user" and m.text
        ).strip()
        response_text = " ".join(
            m.text for m in response_msgs if m.role == "assistant" and m.text
        ).strip()

        item: dict[str, Any] = {
            "query": query_text,
            "response": response_text,
            "query_messages": AgentEvalConverter.convert_messages(query_msgs),
            "response_messages": AgentEvalConverter.convert_messages(response_msgs),
        }
        if self.tools:
            item["tool_definitions"] = [
                {"name": t.name, "description": t.description, "parameters": t.parameters()}
                for t in self.tools
            ]
        if self.context:
            item["context"] = self.context
        return item

    def _split_conversation(
        self, split: ConversationSplitter
    ) -> tuple[list[Message], list[Message]]:
        """Split ``self.conversation`` into (query_messages, response_messages)."""
        if callable(split) and not isinstance(split, ConversationSplit):
            return split(self.conversation)
        if split == ConversationSplit.FULL:
            return self._split_full()
        return self._split_last_turn()

    def _split_last_turn(self) -> tuple[list[Message], list[Message]]:
        """Split at the last user message (default strategy)."""
        return self._split_last_turn_static(self.conversation)

    @staticmethod
    def _split_last_turn_static(
        conversation: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        """Split at the last user message.  Usable as a fallback in custom splitters."""
        last_user_idx = -1
        for i, msg in enumerate(conversation):
            if msg.role == "user":
                last_user_idx = i

        if last_user_idx >= 0:
            return (
                conversation[: last_user_idx + 1],
                conversation[last_user_idx + 1 :],
            )
        return [], list(conversation)

    def _split_full(self) -> tuple[list[Message], list[Message]]:
        """Split after the first user message (evaluates whole trajectory)."""
        first_user_idx = -1
        for i, msg in enumerate(self.conversation):
            if msg.role == "user":
                first_user_idx = i
                break

        if first_user_idx >= 0:
            return (
                self.conversation[: first_user_idx + 1],
                self.conversation[first_user_idx + 1 :],
            )
        return [], list(self.conversation)

    @classmethod
    def per_turn_items(
        cls,
        conversation: list[Message],
        *,
        tools: list[FunctionTool] | None = None,
        context: str | None = None,
    ) -> list[EvalItem]:
        """Split a multi-turn conversation into one ``EvalItem`` per turn.

        Each user message starts a new turn.  The resulting ``EvalItem``
        has cumulative context: ``query_messages`` contains the full
        conversation up to and including that user message, and
        ``response_messages`` contains the agent's actions up to the next
        user message.  This lets you evaluate each response independently
        with its full preceding context.

        Args:
            conversation: Full conversation as ``Message`` objects.
            tools: Tool objects shared across all items.
            context: Optional grounding context shared across all items.

        Returns:
            A list of ``EvalItem`` instances, one per user turn.
        """
        user_indices = [i for i, m in enumerate(conversation) if m.role == "user"]
        if not user_indices:
            return []

        items: list[EvalItem] = []
        for turn_idx, ui in enumerate(user_indices):
            # Response runs from after the user message to the next user
            # message (or end of conversation).
            next_ui = user_indices[turn_idx + 1] if turn_idx + 1 < len(user_indices) else len(conversation)

            items.append(
                cls(
                    conversation=conversation[: next_ui],
                    tools=tools,
                    context=context,
                )
            )

        return items

@dataclass
class EvalScoreResult:
    """Result from a single evaluator on a single item.

    Attributes:
        name: Evaluator name (e.g. ``"relevance"``).
        score: Numeric score from the evaluator.
        passed: Whether the item passed this evaluator's threshold.
        sample: Optional raw evaluator output (rationale, metadata).
    """

    name: str
    score: float
    passed: bool | None = None
    sample: dict[str, Any] | None = None


@dataclass
class EvalItemResult:
    """Per-item result from an evaluation run.

    Every competitor framework surfaces per-item detail. This type bridges
    the gap between run-level summary and the portal, giving programmatic
    access to individual pass/fail/error status, evaluator scores with
    rationale, token usage, and error categorization.

    Attributes:
        item_id: Provider-assigned item identifier.
        status: ``"pass"``, ``"fail"``, or ``"error"``.
        scores: Per-evaluator results for this item.
        error_code: Error category when ``status == "error"``
            (e.g. ``"QueryExtractionError"``).
        error_message: Human-readable error detail.
        response_id: Responses API response ID, if applicable.
        input_text: The query/input that was evaluated.
        output_text: The response/output that was evaluated.
        token_usage: Token counts (``prompt_tokens``,
            ``completion_tokens``, ``total_tokens``).
        metadata: Additional provider-specific data.
    """

    item_id: str
    status: str
    scores: list[EvalScoreResult] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    response_id: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        """Whether this item errored (infrastructure failure, not quality)."""
        return self.status in ("error", "errored")

    @property
    def is_passed(self) -> bool:
        """Whether this item passed all evaluators."""
        return self.status == "pass"

    @property
    def is_failed(self) -> bool:
        """Whether this item failed at least one evaluator."""
        return self.status == "fail"


@dataclass
class EvalResults:
    """Results from an evaluation run by a single provider.

    Attributes:
        provider: Name of the evaluation provider that produced these results.
        eval_id: The evaluation definition ID (provider-specific).
        run_id: The evaluation run ID (provider-specific).
        status: Run status - ``"completed"``, ``"failed"``, ``"canceled"``,
            or ``"timeout"`` if polling exceeded the deadline.
        result_counts: Pass/fail/error counts, populated when completed.
        report_url: URL to view results in the provider's portal.
        error: Error details when the run failed.
        per_evaluator: Per-evaluator result counts, keyed by evaluator name.
        items: Per-item results with individual pass/fail/error status,
            evaluator scores, error details, and token usage. Populated
            when the provider supports per-item retrieval (e.g. Foundry
            ``output_items`` API).
        sub_results: Per-agent breakdown for workflow evaluations, keyed by
            agent/executor name.

    Example::

        results = await evaluate_agent(agent=my_agent, queries=["Hello"], evaluators=evals)
        for r in results:
            print(f"{r.provider}: {r.passed}/{r.total}")

            # Per-item detail
            for item in r.items:
                print(f"  {item.item_id}: {item.status}")
                for score in item.scores:
                    print(f"    {score.name}: {score.score} ({'pass' if score.passed else 'fail'})")
                if item.is_error:
                    print(f"    Error: {item.error_code} - {item.error_message}")

        # Workflow eval - per-agent breakdown
        for r in results:
            for name, sub in r.sub_results.items():
                print(f"  {name}: {sub.passed}/{sub.total}")
    """

    provider: str
    eval_id: str
    run_id: str
    status: str
    result_counts: dict[str, int] | None = None
    report_url: str | None = None
    error: str | None = None
    per_evaluator: dict[str, dict[str, int]] = field(default_factory=dict)
    items: list[EvalItemResult] = field(default_factory=list)
    sub_results: dict[str, "EvalResults"] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        """Number of passing results."""
        return (self.result_counts or {}).get("passed", 0)

    @property
    def failed(self) -> int:
        """Number of failing results."""
        return (self.result_counts or {}).get("failed", 0)

    @property
    def errored(self) -> int:
        """Number of errored results."""
        return (self.result_counts or {}).get("errored", 0)

    @property
    def total(self) -> int:
        """Total number of results (passed + failed + errored)."""
        return self.passed + self.failed + self.errored

    @property
    def all_passed(self) -> bool:
        """Whether all results passed with no failures or errors.

        For workflow evals with sub-agents, checks that all sub-results passed.
        Returns ``False`` if the run did not complete successfully.
        """
        if self.status not in ("completed",):
            return False
        if self.sub_results:
            return all(sub.all_passed for sub in self.sub_results.values())
        # Leaf result - check own counts
        return self.failed == 0 and self.errored == 0 and self.total > 0

    def assert_passed(self, msg: str | None = None) -> None:
        """Assert all results passed. Raises ``AssertionError`` for CI use.

        Args:
            msg: Optional custom failure message.
        """
        if not self.all_passed:
            detail = msg or (
                f"Eval run {self.run_id} {self.status}: "
                f"{self.passed} passed, {self.failed} failed, {self.errored} errored."
            )
            if self.report_url:
                detail += f" See {self.report_url} for details."
            if self.error:
                detail += f" Error: {self.error}"
            errored = [i for i in self.items if i.is_error]
            if errored:
                errors = [f"{i.item_id}: {i.error_code or 'unknown'}" for i in errored[:3]]
                detail += f" Errored items: {'; '.join(errors)}."
            if self.sub_results:
                failed = [name for name, sub in self.sub_results.items() if not sub.all_passed]
                if failed:
                    detail += f" Failed: {', '.join(failed)}."
            raise AssertionError(detail)


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for evaluation providers.

    Any evaluation backend (Azure AI Foundry, local LLM-as-judge, custom
    scorers, etc.) implements this protocol. The provider encapsulates all
    connection details, evaluator selection, and execution logic.

    Example implementation::

        class MyEvaluator:
            @property
            def name(self) -> str:
                return "my-evaluator"

            async def evaluate(self, items: Sequence[EvalItem], *, eval_name: str = "Eval") -> EvalResults:
                # Score each item and return results
                ...
    """

    @property
    def name(self) -> str:
        """Human-readable name of this evaluator."""
        ...

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str = "Agent Framework Eval",
    ) -> EvalResults:
        """Evaluate a batch of items and return results.

        The evaluator determines which metrics to run. It may auto-detect
        capabilities from the items (e.g., run tool evaluators only when
        ``tools`` is present).

        Args:
            items: Eval data items to score.
            eval_name: Display name for the evaluation run.

        Returns:
            ``EvalResults`` with status, counts, and optional portal link.
        """
        ...


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class AgentEvalConverter:
    """Converts agent-framework types to evaluation format.

    Handles the type gap between agent-framework's ``Message`` / ``Content`` /
    ``FunctionTool`` types and the OpenAI-style agent message schema used by
    evaluation providers.

    Example::

        converter = AgentEvalConverter()
        response = await agent.run([Message("user", ["Hello"])])
        item = converter.to_eval_item(query="Hello", response=response, agent=agent)
    """

    @staticmethod
    def convert_message(message: Message) -> list[dict[str, Any]]:
        """Convert a single ``Message`` to Foundry agent evaluator format.

        Uses typed content lists as required by Foundry evaluators::

            {"role": "assistant", "content": [{"type": "tool_call", ...}]}

        A single agent-framework ``Message`` with multiple ``function_result``
        contents produces multiple output messages (one per tool result).

        Args:
            message: An agent-framework ``Message``.

        Returns:
            A list of Foundry-format message dicts.
        """
        role = message.role
        contents = message.contents or []

        content_items: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for c in contents:
            if c.type == "text" and c.text:
                content_items.append({"type": "text", "text": c.text})
            elif c.type == "function_call":
                args = c.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args}
                tc: dict[str, Any] = {
                    "type": "tool_call",
                    "tool_call_id": c.call_id or "",
                    "name": c.name or "",
                }
                if args:
                    tc["arguments"] = args
                content_items.append(tc)
            elif c.type == "function_result":
                result_val = c.result
                if isinstance(result_val, str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        result_val = json.loads(result_val)
                tool_results.append({
                    "call_id": c.call_id or "",
                    "result": result_val,
                })

        output: list[dict[str, Any]] = []

        if tool_results:
            for tr in tool_results:
                output.append({
                    "role": "tool",
                    "tool_call_id": tr["call_id"],
                    "content": [{"type": "tool_result", "tool_result": tr["result"]}],
                })
        elif content_items:
            output.append({"role": role, "content": content_items})
        else:
            output.append({
                "role": role,
                "content": [{"type": "text", "text": ""}],
            })

        return output

    @staticmethod
    def convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert a sequence of ``Message`` objects to Foundry evaluator format.

        Args:
            messages: Agent-framework messages.

        Returns:
            A list of Foundry-format message dicts with typed content lists.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            result.extend(AgentEvalConverter.convert_message(msg))
        return result

    @staticmethod
    def extract_tools(agent: Any) -> list[dict[str, Any]]:
        """Extract tool definitions from an agent instance.

        Reads ``agent.default_options["tools"]`` and ``agent.mcp_tools``
        and converts each ``FunctionTool`` to ``{name, description, parameters}``.

        Args:
            agent: An agent-framework agent instance.

        Returns:
            A list of tool definition dicts.
        """
        tools: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw_tools = getattr(agent, "default_options", {}).get("tools", [])
        for t in raw_tools:
            if isinstance(t, FunctionTool) and t.name not in seen:
                tools.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters(),
                })
                seen.add(t.name)
        # Include tools from connected MCP servers
        for mcp in getattr(agent, "mcp_tools", []):
            for t in getattr(mcp, "functions", []):
                if isinstance(t, FunctionTool) and t.name not in seen:
                    tools.append({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters(),
                    })
                    seen.add(t.name)
        return tools

    @staticmethod
    def to_eval_item(
        *,
        query: str | Sequence[Message],
        response: AgentResponse[Any],
        agent: Any | None = None,
        tools: Sequence[FunctionTool] | None = None,
        context: str | None = None,
    ) -> EvalItem:
        """Convert a complete agent interaction to an ``EvalItem``.

        Args:
            query: The user query string, or input messages.
            response: The agent's response.
            agent: Optional agent instance to auto-extract tool definitions.
            tools: Explicit tool list (takes precedence over *agent*).
            context: Optional context document for groundedness evaluation.

        Returns:
            An ``EvalItem`` suitable for passing to any ``Evaluator``.
        """
        if isinstance(query, str):
            input_msgs = [Message("user", [query])]
        else:
            input_msgs = list(query)

        all_msgs = list(input_msgs) + list(response.messages or [])

        typed_tools: list[FunctionTool] = []
        if tools:
            typed_tools = list(tools)
        elif agent:
            raw_tools = getattr(agent, "default_options", {}).get("tools", [])
            typed_tools = [t for t in raw_tools if isinstance(t, FunctionTool)]
            # Include tools from connected MCP servers
            seen = {t.name for t in typed_tools}
            for mcp in getattr(agent, "mcp_tools", []):
                for t in getattr(mcp, "functions", []):
                    if isinstance(t, FunctionTool) and t.name not in seen:
                        typed_tools.append(t)
                        seen.add(t.name)

        return EvalItem(
            conversation=all_msgs,
            tools=typed_tools or None,
            context=context,
            response_id=getattr(response, "response_id", None),
        )


# ---------------------------------------------------------------------------
# Workflow extraction helpers
# ---------------------------------------------------------------------------


@dataclass
class _AgentEvalData:
    """Per-agent data extracted from a workflow run for evaluation."""

    executor_id: str
    query: str | Sequence[Message]
    response: AgentResponse[Any]
    agent: Any | None = None


def _extract_agent_eval_data(
    workflow_result: WorkflowRunResult,
    workflow: Workflow | None = None,
) -> list[_AgentEvalData]:
    """Walk a WorkflowRunResult and extract per-agent query/response pairs.

    Pairs ``executor_invoked`` with ``executor_completed`` events for each
    ``AgentExecutor``. Skips internal framework executors.
    """
    from ._workflows._agent_executor import AgentExecutor as AE
    from ._workflows._agent_executor import AgentExecutorResponse

    invoked_data: dict[str, Any] = {}
    results: list[_AgentEvalData] = []

    for event in workflow_result:
        if event.type == "executor_invoked" and event.executor_id:
            invoked_data[event.executor_id] = event.data

        elif event.type == "executor_completed" and event.executor_id:
            executor_id = event.executor_id

            # Skip internal framework executors
            if executor_id.startswith("_") or any(
                kw in executor_id.lower() for kw in ("input-conversation", "end-conversation", "end")
            ):
                continue

            completion_data = event.data
            agent_exec_response: AgentExecutorResponse | None = None

            if isinstance(completion_data, list):
                for item in completion_data:
                    if isinstance(item, AgentExecutorResponse):
                        agent_exec_response = item
                        break
            elif isinstance(completion_data, AgentExecutorResponse):
                agent_exec_response = completion_data

            if agent_exec_response is None:
                continue

            query: str | list[Message]
            if agent_exec_response.full_conversation:
                user_msgs = [m for m in agent_exec_response.full_conversation if m.role == "user"]
                query = user_msgs or agent_exec_response.full_conversation
            elif executor_id in invoked_data:
                input_data = invoked_data[executor_id]
                query = input_data if isinstance(input_data, (str, list)) else str(input_data)
            else:
                continue

            agent_ref = None
            if workflow is not None:
                executor = workflow.executors.get(executor_id)
                if executor is not None and isinstance(executor, AE):
                    agent_ref = executor.agent

            results.append(
                _AgentEvalData(
                    executor_id=executor_id,
                    query=query,
                    response=agent_exec_response.agent_response,
                    agent=agent_ref,
                )
            )

    return results


def _extract_overall_query(workflow_result: WorkflowRunResult) -> str | None:
    """Extract the original user query from a workflow result."""
    for event in workflow_result:
        if event.type == "executor_invoked" and event.data is not None:
            data = event.data
            if isinstance(data, str):
                return data
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, Message):
                    return " ".join(m.text for m in data if hasattr(m, "text") and m.role == "user")
                if isinstance(first, str):
                    return " ".join(data)
            return str(data)
    return None


# ---------------------------------------------------------------------------
# Public orchestration functions
# ---------------------------------------------------------------------------


class _MinimalAgent:
    """Stub agent used when evaluate_response is called without an agent."""

    name = "unknown"


async def evaluate_agent(
    *,
    agent: Any = None,
    queries: Sequence[str] | None = None,
    responses: AgentResponse[Any] | Sequence[AgentResponse[Any]] | None = None,
    evaluators: Evaluator | Callable[..., Any] | Sequence[Evaluator | Callable[..., Any]],
    eval_name: str | None = None,
    context: str | None = None,
    conversation_split: ConversationSplitter | None = None,
) -> list[EvalResults]:
    """Run an agent against test queries and evaluate the results.

    The simplest path for evaluating an agent during development. For each
    query, runs the agent, converts the interaction to eval format, and
    submits to the evaluator(s).

    If ``responses`` is provided, skips running the agent and evaluates those
    responses directly — but still extracts tool definitions from the agent.
    In this mode ``queries`` is optional: if omitted, the evaluator will use
    the response's ``response_id`` (requires a provider that supports
    server-side retrieval).

    Args:
        agent: An agent-framework agent instance.
        queries: Test queries to run the agent against. Required when
            ``responses`` is not provided.
        responses: Pre-existing ``AgentResponse``(s) to evaluate without
            running the agent.  When provided, ``queries`` are used only to
            populate the eval item's ``query`` field (not to invoke the agent).
        evaluators: One or more ``Evaluator`` instances.
        eval_name: Display name (defaults to agent name).
        context: Optional context for groundedness evaluation.
        conversation_split: Split strategy applied to all items, overriding
            each evaluator's default.  See ``ConversationSplitter``.

    Returns:
        A list of ``EvalResults``, one per evaluator provider.

    Raises:
        ValueError: If neither ``queries`` nor ``responses`` is provided.

    Example — run and evaluate::

        results = await evaluate_agent(
            agent=my_agent,
            queries=["What's the weather?"],
            evaluators=evals,
        )

    Example — evaluate existing responses::

        response = await agent.run([Message("user", ["What's the weather?"])])
        results = await evaluate_agent(
            agent=agent,
            responses=response,
            queries=["What's the weather?"],
            evaluators=evals,
        )
    """
    converter = AgentEvalConverter()
    items: list[EvalItem] = []

    if responses is not None:
        # Evaluate pre-existing responses (don't run the agent)
        resp_list = [responses] if isinstance(responses, AgentResponse) else list(responses)

        if queries is not None:
            query_list = list(queries)
            if len(query_list) != len(resp_list):
                raise ValueError(
                    f"Got {len(query_list)} queries but {len(resp_list)} responses."
                )
            for q, r in zip(query_list, resp_list):
                items.append(
                    converter.to_eval_item(
                        query=q, response=r, agent=agent, context=context,
                    )
                )
        else:
            # No queries — build minimal items with response_id
            for r in resp_list:
                response_id = getattr(r, "response_id", None)
                if response_id is None:
                    raise ValueError(
                        "Response does not have a response_id. Provide "
                        "'queries' so the conversation can be reconstructed "
                        "for evaluation."
                    )
                items.append(
                    EvalItem(
                        conversation=[],
                        response_id=response_id,
                    )
                )
    elif queries is not None:
        # Run the agent against test queries
        for query in queries:
            response = await agent.run([Message("user", [query])])
            items.append(
                converter.to_eval_item(
                    query=query, response=response, agent=agent, context=context,
                )
            )
    else:
        raise ValueError("Provide either 'queries' or 'responses' (or both).")

    # Stamp split strategy on items so evaluators respect it
    if conversation_split is not None:
        for item in items:
            item.split_strategy = conversation_split

    name = eval_name or f"Eval: {getattr(agent, 'name', None) or getattr(agent, 'id', 'agent')}"
    return await _run_evaluators(evaluators, items, eval_name=name)


async def evaluate_response(
    *,
    response: AgentResponse[Any] | Sequence[AgentResponse[Any]],
    query: str | Sequence[Message] | Sequence[str | Sequence[Message]] | None = None,
    agent: Any | None = None,
    evaluators: Evaluator | Sequence[Evaluator],
    eval_name: str = "Agent Framework Response Eval",
) -> list[EvalResults]:
    """Deprecated: use ``evaluate_agent(responses=...)`` instead.

    Evaluate one or more agent responses that have already been produced.
    This is a thin wrapper that delegates to ``evaluate_agent``.
    """
    # Normalize queries for evaluate_agent (it expects Sequence[str] | None)
    queries_norm: Sequence[str] | None = None
    if query is not None:
        responses_list = [response] if isinstance(response, AgentResponse) else list(response)
        queries_norm = list(_normalize_queries(query, len(responses_list)))

    # Build a dummy agent if none provided (evaluate_agent requires agent=)
    if agent is None:
        agent = _MinimalAgent()

    return await evaluate_agent(
        agent=agent,
        responses=response,
        queries=queries_norm,
        evaluators=evaluators,
        eval_name=eval_name,
    )


async def evaluate_workflow(
    *,
    workflow: Workflow,
    workflow_result: WorkflowRunResult | None = None,
    queries: Sequence[str] | None = None,
    evaluators: Evaluator | Callable[..., Any] | Sequence[Evaluator | Callable[..., Any]],
    eval_name: str | None = None,
    include_overall: bool = True,
    include_per_agent: bool = True,
    conversation_split: ConversationSplitter | None = None,
) -> list[EvalResults]:
    """Evaluate a multi-agent workflow with per-agent breakdown.

    Evaluates each sub-agent individually and (optionally) the workflow's
    overall output. Returns one ``EvalResults`` per evaluator provider, each
    with per-agent breakdowns in ``sub_results``.

    **Two modes:**

    - **Post-hoc**: Pass ``workflow_result`` from a previous
      ``workflow.run()`` call.
    - **Run + evaluate**: Pass ``queries`` and the workflow will be run
      against each query, then evaluated.

    Args:
        workflow: The workflow instance.
        workflow_result: A completed ``WorkflowRunResult``.
        queries: Test queries to run through the workflow.
        evaluators: One or more ``Evaluator`` instances.
        eval_name: Display name for the evaluation.
        include_overall: Whether to evaluate the workflow's final output.
        include_per_agent: Whether to evaluate each sub-agent individually.
        conversation_split: Split strategy applied to all items, overriding
            each evaluator's default.  See ``ConversationSplitter``.

    Returns:
        A list of ``EvalResults``, one per evaluator provider, each with
        per-agent breakdown in ``sub_results``.

    Example::

        from agent_framework_azure_ai import FoundryEvals

        evals = FoundryEvals(project_client=client, model_deployment="gpt-4o")
        result = await workflow.run("Plan a trip to Paris")

        eval_results = await evaluate_workflow(
            workflow=workflow,
            workflow_result=result,
            evaluators=evals,
        )
        for r in eval_results:
            print(f"{r.provider}:")
            for name, sub in r.sub_results.items():
                print(f"  {name}: {sub.passed}/{sub.total}")
    """
    from ._workflows._workflow import WorkflowRunResult as WRR

    if workflow_result is None and queries is None:
        raise ValueError("Provide either 'workflow_result' or 'queries'.")

    wf_name = eval_name or f"Workflow Eval: {workflow.__class__.__name__}"
    converter = AgentEvalConverter()
    evaluator_list = [evaluators] if isinstance(evaluators, Evaluator) else list(evaluators)

    # Collect per-agent data and overall items
    all_agent_data: list[_AgentEvalData] = []
    overall_items: list[EvalItem] = []

    if queries is not None:
        results_list: list[WRR] = []
        for q in queries:
            result = await workflow.run(q)
            if not isinstance(result, WRR):
                raise TypeError(f"Expected WorkflowRunResult from workflow.run(), got {type(result).__name__}.")
            results_list.append(result)
            all_agent_data.extend(_extract_agent_eval_data(result, workflow))
            if include_overall:
                overall_item = _build_overall_item(q, result)
                if overall_item:
                    overall_items.append(overall_item)
    else:
        assert workflow_result is not None  # noqa: S101
        all_agent_data = _extract_agent_eval_data(workflow_result, workflow)
        if include_overall:
            original_query = _extract_overall_query(workflow_result)
            if original_query:
                overall_item = _build_overall_item(original_query, workflow_result)
                if overall_item:
                    overall_items.append(overall_item)

    # Group agent data by executor ID
    agents_by_id: dict[str, list[_AgentEvalData]] = {}
    if include_per_agent and all_agent_data:
        for ad in all_agent_data:
            agents_by_id.setdefault(ad.executor_id, []).append(ad)

    # Build per-agent items once (shared across providers).
    # Clear response_id so per-agent evals always use the dataset path.
    # The Responses API retrieval path doesn't work for agents whose input
    # is another agent's full conversation (the evaluator can't extract a
    # clean user query from the stored response).
    agent_items_by_id: dict[str, list[EvalItem]] = {}
    for executor_id, agent_data_list in agents_by_id.items():
        items = [
            converter.to_eval_item(query=ad.query, response=ad.response, agent=ad.agent) for ad in agent_data_list
        ]
        for item in items:
            item.response_id = None
        agent_items_by_id[executor_id] = items

    if not agent_items_by_id and not overall_items:
        raise ValueError(
            "No agent executor data found in the workflow result. Ensure the workflow uses AgentExecutor-based agents."
        )

    # Stamp split strategy on all items so evaluators respect it
    if conversation_split is not None:
        for items in agent_items_by_id.values():
            for item in items:
                item.split_strategy = conversation_split
        for item in overall_items:
            item.split_strategy = conversation_split

    # Run each provider, building per-agent sub_results for each
    all_results: list[EvalResults] = []
    for ev in evaluator_list:
        suffix = f" ({ev.name})" if len(evaluator_list) > 1 else ""
        sub_results: dict[str, EvalResults] = {}

        # Per-agent evals
        for executor_id, items in agent_items_by_id.items():
            agent_result = await ev.evaluate(items, eval_name=f"{wf_name} — {executor_id}{suffix}")
            sub_results[executor_id] = agent_result

        # Overall eval
        if include_overall and overall_items:
            overall_result = await ev.evaluate(overall_items, eval_name=f"{wf_name} — overall{suffix}")
        elif sub_results:
            # Aggregate from sub-results
            total_passed = sum(s.passed for s in sub_results.values())
            total_failed = sum(s.failed for s in sub_results.values())
            total_errored = sum(s.errored for s in sub_results.values())
            all_completed = all(s.status == "completed" for s in sub_results.values())
            overall_result = EvalResults(
                provider=ev.name,
                eval_id="aggregate",
                run_id="aggregate",
                status="completed" if all_completed else "partial",
                result_counts={
                    "passed": total_passed,
                    "failed": total_failed,
                    "errored": total_errored,
                },
            )
        else:
            raise ValueError(
                "No agent executor data found in the workflow result. "
                "Ensure the workflow uses AgentExecutor-based agents."
            )

        overall_result.sub_results = sub_results
        all_results.append(overall_result)

    return all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_queries(
    query: str | Sequence[Message] | Sequence[str | Sequence[Message]],
    expected_count: int,
) -> list[str | Sequence[Message]]:
    """Normalize query input to a list matching the expected count."""
    if isinstance(query, (str, Message)):
        queries: list[str | Sequence[Message]] = [query] * expected_count if expected_count == 1 else [query]  # type: ignore[list-item]
    elif isinstance(query, list) and len(query) > 0 and isinstance(query[0], Message):
        queries = [query] * expected_count if expected_count == 1 else [query]  # type: ignore[list-item]
    else:
        queries = list(query)  # type: ignore[arg-type]

    if len(queries) != expected_count:
        raise ValueError(f"Number of queries ({len(queries)}) does not match number of responses ({expected_count}).")
    return queries


def _build_overall_item(
    query: str,
    workflow_result: WorkflowRunResult,
) -> EvalItem | None:
    """Build an EvalItem for the overall workflow output."""
    outputs = workflow_result.get_outputs()
    if not outputs:
        return None

    final_output = outputs[-1]
    if isinstance(final_output, list) and final_output and isinstance(final_output[0], Message):
        response_text = " ".join(m.text for m in final_output if m.role == "assistant")
        overall_response = AgentResponse(messages=[Message("assistant", [response_text])])
    elif isinstance(final_output, AgentResponse):
        overall_response = final_output
    else:
        overall_response = AgentResponse(messages=[Message("assistant", [str(final_output)])])

    return AgentEvalConverter.to_eval_item(query=query, response=overall_response)


async def _run_evaluators(
    evaluators: Evaluator | Sequence[Evaluator | Callable[..., Any]],
    items: Sequence[EvalItem],
    *,
    eval_name: str,
) -> list[EvalResults]:
    """Run one or more evaluators and return a result per provider.

    Bare ``EvalCheck`` callables (including ``@function_evaluator`` decorated
    functions and helpers like ``keyword_check``) are auto-wrapped in a
    ``LocalEvaluator`` so they can be passed directly in the evaluators list.
    """
    from ._local_eval import LocalEvaluator

    raw_list: list[Any]
    if isinstance(evaluators, Evaluator):
        raw_list = [evaluators]
    elif callable(evaluators):
        raw_list = [evaluators]
    else:
        raw_list = list(evaluators)

    # Auto-wrap bare callables (EvalCheck / @function_evaluator) into LocalEvaluator
    evaluator_list: list[Evaluator] = []
    pending_checks: list[Callable[..., Any]] = []

    for item in raw_list:
        if isinstance(item, Evaluator):
            # Flush any pending checks as a single LocalEvaluator
            if pending_checks:
                evaluator_list.append(LocalEvaluator(*pending_checks))
                pending_checks = []
            evaluator_list.append(item)
        elif callable(item):
            pending_checks.append(item)
        else:
            raise TypeError(
                f"Expected an Evaluator or callable, got {type(item).__name__}"
            )

    if pending_checks:
        evaluator_list.append(LocalEvaluator(*pending_checks))

    results: list[EvalResults] = []
    for ev in evaluator_list:
        suffix = f" ({ev.name})" if len(evaluator_list) > 1 else ""
        result = await ev.evaluate(items, eval_name=f"{eval_name}{suffix}")
        results.append(result)

    return results
