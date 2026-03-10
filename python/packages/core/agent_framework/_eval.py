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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from ._tools import FunctionTool
from ._types import AgentResponse, Message

if TYPE_CHECKING:
    from ._workflows._agent_executor import AgentExecutorResponse
    from ._workflows._workflow import Workflow, WorkflowRunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class EvalItem:
    """A single item to be evaluated.

    Represents one query/response interaction in a provider-agnostic format.
    Evaluation providers convert this to their internal format.

    Attributes:
        query: The user's input query.
        response: The agent's text response.
        conversation: Full conversation as typed content messages.
        tool_definitions: Tool definitions if the agent has tools.
        context: Optional grounding context document.
        expected: Optional expected output for ground-truth comparison.
        response_id: Responses API response ID (for providers that support
            server-side retrieval).
    """

    query: str
    response: str
    conversation: list[dict[str, Any]]
    tool_definitions: list[dict[str, Any]] | None = None
    context: str | None = None
    expected: str | None = None
    response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for serialization.

        Produces ``query_messages`` (system + user) and ``response_messages``
        (assistant + tool) from the conversation, matching the Foundry agent
        evaluator schema where ``query`` and ``response`` each accept either
        a string or a conversation array.
        """
        query_msgs: list[dict[str, Any]] = []
        response_msgs: list[dict[str, Any]] = []
        for msg in self.conversation:
            if msg.get("role") in ("system", "user"):
                query_msgs.append(msg)
            else:
                response_msgs.append(msg)

        item: dict[str, Any] = {
            "query": self.query,
            "response": self.response,
            "query_messages": query_msgs,
            "response_messages": response_msgs,
        }
        if self.tool_definitions:
            item["tool_definitions"] = self.tool_definitions
        if self.context:
            item["context"] = self.context
        return item


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
    def passed_items(self) -> list[EvalItemResult]:
        """Items that passed all evaluators."""
        return [i for i in self.items if i.is_passed]

    @property
    def failed_items(self) -> list[EvalItemResult]:
        """Items that failed at least one evaluator."""
        return [i for i in self.items if i.is_failed]

    @property
    def errored_items(self) -> list[EvalItemResult]:
        """Items that errored (infrastructure failures, not quality)."""
        return [i for i in self.items if i.is_error]

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
            if self.errored_items:
                errors = [f"{i.item_id}: {i.error_code or 'unknown'}" for i in self.errored_items[:3]]
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
        ``tool_definitions`` is present).

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

        Reads ``agent.default_options["tools"]`` and converts each
        ``FunctionTool`` to ``{name, description, parameters}``.

        Args:
            agent: An agent-framework agent instance.

        Returns:
            A list of tool definition dicts.
        """
        tools: list[dict[str, Any]] = []
        raw_tools = getattr(agent, "default_options", {}).get("tools", [])
        for t in raw_tools:
            if isinstance(t, FunctionTool):
                tools.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters(),
                })
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
            query_str = query
            input_msgs = [Message("user", [query])]
        else:
            input_msgs = list(query)
            query_str = " ".join(m.text for m in input_msgs if m.role == "user")

        all_msgs = list(input_msgs) + list(response.messages or [])
        conversation = AgentEvalConverter.convert_messages(all_msgs)

        tool_defs: list[dict[str, Any]] = []
        if tools:
            for t in tools:
                tool_defs.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters(),
                })
        elif agent:
            tool_defs = AgentEvalConverter.extract_tools(agent)

        return EvalItem(
            query=query_str,
            response=response.text or "",
            conversation=conversation,
            tool_definitions=tool_defs or None,
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
    agent: Any,
    queries: Sequence[str] | None = None,
    responses: AgentResponse[Any] | Sequence[AgentResponse[Any]] | None = None,
    evaluators: Evaluator | Sequence[Evaluator],
    eval_name: str | None = None,
    context: str | None = None,
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
                        query="",
                        response=r.text or "",
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
    evaluators: Evaluator | Sequence[Evaluator],
    eval_name: str | None = None,
    include_overall: bool = True,
    include_per_agent: bool = True,
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
    evaluators: Evaluator | Sequence[Evaluator],
    items: Sequence[EvalItem],
    *,
    eval_name: str,
) -> list[EvalResults]:
    """Run one or more evaluators and return a result per provider."""
    evaluator_list = [evaluators] if isinstance(evaluators, Evaluator) else list(evaluators)

    results: list[EvalResults] = []
    for ev in evaluator_list:
        suffix = f" ({ev.name})" if len(evaluator_list) > 1 else ""
        result = await ev.evaluate(items, eval_name=f"{eval_name}{suffix}")
        results.append(result)

    return results
