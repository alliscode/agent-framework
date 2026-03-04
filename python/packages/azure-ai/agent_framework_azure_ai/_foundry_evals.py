# Copyright (c) Microsoft. All rights reserved.

"""Azure AI Foundry Evals integration for Microsoft Agent Framework.

Provides helpers to evaluate agent-framework agents using Foundry's built-in
evaluators. See docs/decisions/0018-foundry-evals-integration.md for the
design rationale.

Typical usage::

    from agent_framework_azure_ai import Evaluators, evaluate_agent

    results = await evaluate_agent(
        agent=my_agent,
        queries=["What's the weather in Seattle?"],
        evaluators=[Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY],
        project_client=project_client,
        model_deployment="gpt-4o",
    )
    assert results.all_passed
    print(results.report_url)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from agent_framework import AgentResponse, FunctionTool, Message

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

    from agent_framework import (
        AgentExecutor,
        AgentExecutorResponse,
        Workflow,
        WorkflowEvent,
        WorkflowRunResult,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in evaluator constants
# ---------------------------------------------------------------------------


class Evaluators:
    """Constants for Foundry built-in evaluator names.

    Use these instead of raw strings for IDE autocomplete and typo prevention::

        from agent_framework_azure_ai import Evaluators

        evaluators = [Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY]
    """

    # Agent behavior
    INTENT_RESOLUTION: str = "intent_resolution"
    TASK_ADHERENCE: str = "task_adherence"
    TASK_COMPLETION: str = "task_completion"
    TASK_NAVIGATION_EFFICIENCY: str = "task_navigation_efficiency"

    # Tool usage
    TOOL_CALL_ACCURACY: str = "tool_call_accuracy"
    TOOL_SELECTION: str = "tool_selection"
    TOOL_INPUT_ACCURACY: str = "tool_input_accuracy"
    TOOL_OUTPUT_UTILIZATION: str = "tool_output_utilization"
    TOOL_CALL_SUCCESS: str = "tool_call_success"

    # Quality
    COHERENCE: str = "coherence"
    FLUENCY: str = "fluency"
    RELEVANCE: str = "relevance"
    GROUNDEDNESS: str = "groundedness"
    RESPONSE_COMPLETENESS: str = "response_completeness"
    SIMILARITY: str = "similarity"

    # Safety
    VIOLENCE: str = "violence"
    SEXUAL: str = "sexual"
    SELF_HARM: str = "self_harm"
    HATE_UNFAIRNESS: str = "hate_unfairness"


# Evaluators that require tool_definitions in the data mapping.
_TOOL_EVALUATORS: set[str] = {
    "builtin.tool_call_accuracy",
    "builtin.tool_selection",
    "builtin.tool_input_accuracy",
    "builtin.tool_output_utilization",
    "builtin.tool_call_success",
}

_BUILTIN_EVALUATORS: dict[str, str] = {
    # Agent behavior
    "intent_resolution": "builtin.intent_resolution",
    "task_adherence": "builtin.task_adherence",
    "task_completion": "builtin.task_completion",
    "task_navigation_efficiency": "builtin.task_navigation_efficiency",
    # Tool usage
    "tool_call_accuracy": "builtin.tool_call_accuracy",
    "tool_selection": "builtin.tool_selection",
    "tool_input_accuracy": "builtin.tool_input_accuracy",
    "tool_output_utilization": "builtin.tool_output_utilization",
    "tool_call_success": "builtin.tool_call_success",
    # Quality
    "coherence": "builtin.coherence",
    "fluency": "builtin.fluency",
    "relevance": "builtin.relevance",
    "groundedness": "builtin.groundedness",
    "response_completeness": "builtin.response_completeness",
    "similarity": "builtin.similarity",
    # Safety
    "violence": "builtin.violence",
    "sexual": "builtin.sexual",
    "self_harm": "builtin.self_harm",
    "hate_unfairness": "builtin.hate_unfairness",
}

# Default evaluator sets used when evaluators=None
_DEFAULT_EVALUATORS: list[str] = [
    Evaluators.RELEVANCE,
    Evaluators.COHERENCE,
    Evaluators.TASK_ADHERENCE,
]

_DEFAULT_TOOL_EVALUATORS: list[str] = [
    Evaluators.TOOL_CALL_ACCURACY,
]


def _resolve_evaluator(name: str) -> str:
    """Resolve a short evaluator name to its fully-qualified ``builtin.*`` form.

    Args:
        name: Short name (e.g. ``"relevance"``) or fully-qualified name
            (e.g. ``"builtin.relevance"``).

    Returns:
        The fully-qualified evaluator name.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name.startswith("builtin."):
        return name
    resolved = _BUILTIN_EVALUATORS.get(name)
    if resolved is None:
        raise ValueError(f"Unknown evaluator '{name}'. Available: {sorted(_BUILTIN_EVALUATORS)}")
    return resolved


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EvalResults:
    """Results from a Foundry evaluation run.

    Attributes:
        eval_id: The evaluation definition ID.
        run_id: The evaluation run ID.
        status: Run status — ``"completed"``, ``"failed"``, ``"canceled"``,
            or ``"timeout"`` if polling exceeded the deadline.
        result_counts: Pass/fail/error counts, populated when completed.
        report_url: URL to view results in the Foundry portal.
        per_evaluator: Per-evaluator result counts, keyed by evaluator name.
        sub_results: Per-agent or per-component evaluation results, keyed by
            executor/agent name. Empty for single-agent evals, populated by
            ``evaluate_workflow()``. Supports nested workflows.

    Example::

        results = await evaluate_agent(...)
        assert results.all_passed
        print(f"{results.passed}/{results.total} passed")
        print(f"Portal: {results.report_url}")

        # Workflow eval — per-agent breakdown
        for name, sub in results.sub_results.items():
            print(f"{name}: {sub.passed}/{sub.total}")
    """

    eval_id: str
    run_id: str
    status: str
    result_counts: dict[str, int] | None = None
    report_url: str | None = None
    per_evaluator: dict[str, dict[str, int]] = field(default_factory=dict)
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

        For workflow evals, also checks that all sub-agent results passed.
        Returns ``False`` if the run did not complete successfully.
        """
        own_passed = self.status == "completed" and self.failed == 0 and self.errored == 0 and self.total > 0
        if not own_passed:
            return False
        return all(sub.all_passed for sub in self.sub_results.values())

    def assert_passed(self, msg: str | None = None) -> None:
        """Assert all results passed. Raises ``AssertionError`` for CI use.

        Args:
            msg: Optional custom failure message.
        """
        if not self.all_passed:
            detail = msg or (
                f"Eval run {self.run_id} {self.status}: "
                f"{self.passed} passed, {self.failed} failed, {self.errored} errored. "
                f"See {self.report_url or 'Foundry portal'} for details."
            )
            if self.sub_results:
                failed_agents = [
                    name for name, sub in self.sub_results.items() if not sub.all_passed
                ]
                if failed_agents:
                    detail += f" Failed agents: {', '.join(failed_agents)}."
            raise AssertionError(detail)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class AgentEvalConverter:
    """Converts agent-framework types to the format Foundry evaluators expect.

    Handles the type gap between agent-framework's ``Message`` / ``Content`` /
    ``FunctionTool`` types and the OpenAI-style agent message schema used by
    Foundry evaluators.

    Example::

        converter = AgentEvalConverter()
        response = await agent.run([Message("user", ["Hello"])])
        item = converter.to_eval_item(query="Hello", response=response, agent=agent)
    """

    @staticmethod
    def convert_message(message: Message) -> list[dict[str, Any]]:
        """Convert a single ``Message`` to one or more OpenAI chat format dicts.

        A single agent-framework ``Message`` with multiple ``function_result``
        contents produces multiple OpenAI-format messages (one per tool result).

        Args:
            message: An agent-framework ``Message``.

        Returns:
            A list of OpenAI chat format message dicts.
        """
        role = message.role
        contents = message.contents or []

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for c in contents:
            if c.type == "text" and c.text:
                text_parts.append(c.text)
            elif c.type == "function_call":
                args = c.arguments
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls.append({
                    "id": c.call_id or "",
                    "type": "function",
                    "function": {
                        "name": c.name or "",
                        "arguments": args or "{}",
                    },
                })
            elif c.type == "function_result":
                result_val = c.result
                if not isinstance(result_val, str):
                    result_val = json.dumps(result_val) if result_val is not None else ""
                tool_results.append({
                    "call_id": c.call_id or "",
                    "content": result_val,
                })

        output: list[dict[str, Any]] = []

        if role == "assistant" or tool_calls:
            msg: dict[str, Any] = {"role": "assistant"}
            msg["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            output.append(msg)
        elif tool_results:
            # Each function result becomes a separate "tool" message
            for tr in tool_results:
                output.append({
                    "role": "tool",
                    "tool_call_id": tr["call_id"],
                    "content": tr["content"],
                })
        else:
            output.append({
                "role": role,
                "content": "\n".join(text_parts) if text_parts else "",
            })

        return output

    @staticmethod
    def convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert a sequence of ``Message`` objects to OpenAI chat format.

        Args:
            messages: Agent-framework messages.

        Returns:
            A list of OpenAI chat format message dicts.
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
    ) -> dict[str, Any]:
        """Convert a complete agent interaction to a Foundry eval data item.

        Args:
            query: The user query string, or input messages.
            response: The agent's response.
            agent: Optional agent instance to auto-extract tool definitions.
            tools: Explicit tool list (takes precedence over *agent*).
            context: Optional context document for groundedness evaluation.

        Returns:
            A dict suitable for use as a JSONL eval data item, containing at
            least ``query``, ``response``, and ``conversation`` keys.
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

        item: dict[str, Any] = {
            "query": query_str,
            "response": response.text or "",
            "conversation": json.dumps(conversation),
        }
        if tool_defs:
            item["tool_definitions"] = json.dumps(tool_defs)
        if context:
            item["context"] = context

        return item


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_testing_criteria(
    evaluators: Sequence[str],
    model_deployment: str,
    *,
    include_data_mapping: bool = False,
) -> list[dict[str, Any]]:
    """Build ``testing_criteria`` for ``evals.create()``.

    Args:
        evaluators: Evaluator names.
        model_deployment: Model deployment for the LLM judge.
        include_data_mapping: Whether to include field-level data mapping
            (required for the JSONL data source, not needed for response-based).
    """
    criteria: list[dict[str, Any]] = []
    for name in evaluators:
        qualified = _resolve_evaluator(name)
        short = name if not name.startswith("builtin.") else name.split(".")[-1]

        entry: dict[str, Any] = {
            "type": "azure_ai_evaluator",
            "name": short,
            "evaluator_name": qualified,
            "initialization_parameters": {"deployment_name": model_deployment},
        }

        if include_data_mapping:
            mapping: dict[str, str] = {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            }
            if qualified == "builtin.groundedness":
                mapping["context"] = "{{item.context}}"
            if qualified in _TOOL_EVALUATORS:
                mapping["tool_definitions"] = "{{item.tool_definitions}}"
            entry["data_mapping"] = mapping

        criteria.append(entry)
    return criteria


def _build_item_schema(*, has_context: bool = False, has_tools: bool = False) -> dict[str, Any]:
    """Build the ``item_schema`` for custom JSONL eval definitions."""
    properties: dict[str, Any] = {
        "query": {"type": "string"},
        "response": {"type": "string"},
        "conversation": {"type": "string"},
    }
    if has_context:
        properties["context"] = {"type": "string"}
    if has_tools:
        properties["tool_definitions"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": ["query", "response"],
    }


def _resolve_default_evaluators(
    evaluators: Sequence[str] | None,
    agent: Any | None = None,
) -> list[str]:
    """Resolve evaluators, applying defaults when ``None``.

    Defaults to relevance + coherence + task_adherence. Automatically adds
    tool_call_accuracy when the agent has tools.
    """
    if evaluators is not None:
        return list(evaluators)

    result = list(_DEFAULT_EVALUATORS)
    if agent is not None and AgentEvalConverter.extract_tools(agent):
        result.extend(_DEFAULT_TOOL_EVALUATORS)
    return result


async def _call_client(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a sync or async client method transparently.

    If ``func`` returns a coroutine (async client), awaits it directly.
    Otherwise wraps the sync call in ``asyncio.to_thread()``.
    """
    import inspect

    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return await asyncio.to_thread(lambda: result)


async def _poll_eval_run(
    client: OpenAI | AsyncOpenAI,
    eval_id: str,
    run_id: str,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Poll an eval run until completion or timeout."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        run = await _call_client(client.evals.runs.retrieve, run_id=run_id, eval_id=eval_id)
        if run.status in ("completed", "failed", "canceled"):
            return EvalResults(
                eval_id=eval_id,
                run_id=run_id,
                status=run.status,
                result_counts=_extract_result_counts(run),
                report_url=getattr(run, "report_url", None),
                per_evaluator=_extract_per_evaluator(run),
            )
        remaining = deadline - loop.time()
        if remaining <= 0:
            return EvalResults(eval_id=eval_id, run_id=run_id, status="timeout")
        logger.debug("Eval run %s status: %s (%.0fs remaining)", run_id, run.status, remaining)
        await asyncio.sleep(min(poll_interval, remaining))


def _extract_result_counts(run: Any) -> dict[str, int] | None:
    """Safely extract result_counts from an eval run object."""
    counts = getattr(run, "result_counts", None)
    if counts is None:
        return None
    if isinstance(counts, dict):
        return counts
    # Handle ResultCounts object type from the OpenAI SDK
    try:
        return {k: v for k, v in vars(counts).items() if isinstance(v, int)}
    except TypeError:
        return None


def _extract_per_evaluator(run: Any) -> dict[str, dict[str, int]]:
    """Safely extract per-evaluator result breakdowns from an eval run."""
    per_eval: dict[str, dict[str, int]] = {}
    per_testing_criteria = getattr(run, "per_testing_criteria_results", None)
    if per_testing_criteria is None:
        return per_eval
    try:
        items = per_testing_criteria if isinstance(per_testing_criteria, list) else []
        for item in items:
            name = getattr(item, "name", None) or getattr(item, "testing_criteria", "unknown")
            counts = _extract_result_counts(item)
            if name and counts:
                per_eval[name] = counts
    except (TypeError, AttributeError):
        pass
    return per_eval


def _resolve_openai_client(
    openai_client: OpenAI | AsyncOpenAI | None = None,
    project_client: Any | None = None,
) -> OpenAI | AsyncOpenAI:
    """Resolve an OpenAI client from explicit client or project_client.

    Accepts either an ``openai_client`` directly (sync or async) or an
    ``AIProjectClient`` (sync or async) from which the OpenAI client is
    obtained via ``.get_openai_client()``.

    Both sync and async ``AIProjectClient`` variants are supported. The
    returned client type (sync ``OpenAI`` or ``AsyncOpenAI``) depends on
    which project client variant was provided.
    """
    if openai_client is not None:
        return openai_client
    if project_client is not None:
        return project_client.get_openai_client()
    raise ValueError("Provide either 'openai_client' or 'project_client'.")


# ---------------------------------------------------------------------------
# Public API — evaluation functions
# ---------------------------------------------------------------------------


async def evaluate_dataset(
    *,
    items: Sequence[dict[str, Any]],
    evaluators: Sequence[str] | None = None,
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    eval_name: str = "Agent Framework Eval",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Evaluate pre-collected data items using Foundry evaluators (Path 3).

    Items should be produced by ``AgentEvalConverter.to_eval_item()`` or
    structured as dicts with at least ``query`` and ``response`` keys.

    Args:
        items: Eval data items.
        evaluators: Evaluator names (e.g. ``[Evaluators.RELEVANCE]``).
            Defaults to relevance, coherence, and task_adherence.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance. The OpenAI client
            is obtained automatically via ``project_client.get_openai_client()``.
        model_deployment: Model deployment name for the evaluator LLM judge.
        eval_name: Display name for the evaluation.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.

    Returns:
        ``EvalResults`` with status, result counts, and portal link.

    Example::

        items = [converter.to_eval_item(query=q, response=r) for q, r in pairs]
        results = await evaluate_dataset(
            items=items,
            evaluators=[Evaluators.RELEVANCE, Evaluators.COHERENCE],
            project_client=project_client,
            model_deployment="gpt-4o",
        )
    """
    client = _resolve_openai_client(openai_client, project_client)
    resolved_evaluators = _resolve_default_evaluators(evaluators)
    has_context = any("context" in item for item in items)
    has_tools = any("tool_definitions" in item for item in items)

    eval_obj = await _call_client(
        client.evals.create,
        name=eval_name,
        data_source_config={
            "type": "custom",
            "item_schema": _build_item_schema(has_context=has_context, has_tools=has_tools),
            "include_sample_schema": True,
        },
        testing_criteria=_build_testing_criteria(resolved_evaluators, model_deployment, include_data_mapping=True),
    )

    data_source = {
        "type": "jsonl",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in items],
        },
    }

    run = await _call_client(
        client.evals.runs.create,
        eval_id=eval_obj.id,
        name=f"{eval_name} Run",
        data_source=data_source,
    )

    return await _poll_eval_run(client, eval_obj.id, run.id, poll_interval, timeout)


async def _evaluate_responses(
    *,
    response_ids: Sequence[str],
    evaluators: Sequence[str],
    openai_client: OpenAI | AsyncOpenAI,
    model_deployment: str,
    eval_name: str = "Agent Framework Response Eval",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Evaluate existing Responses API responses by ID.

    Internal implementation — use ``evaluate_traces(response_ids=...)`` instead.
    """
    eval_obj = await _call_client(
        openai_client.evals.create,
        name=eval_name,
        data_source_config={"type": "azure_ai_source", "scenario": "responses"},
        testing_criteria=_build_testing_criteria(evaluators, model_deployment),
    )

    data_source = {
        "type": "azure_ai_responses",
        "item_generation_params": {
            "type": "response_retrieval",
            "data_mapping": {"response_id": "{{item.resp_id}}"},
            "source": {
                "type": "file_content",
                "content": [{"item": {"resp_id": rid}} for rid in response_ids],
            },
        },
    }

    run = await _call_client(
        openai_client.evals.runs.create,
        eval_id=eval_obj.id,
        name=f"{eval_name} Run",
        data_source=data_source,
    )

    return await _poll_eval_run(openai_client, eval_obj.id, run.id, poll_interval, timeout)


async def evaluate_response(
    *,
    response: AgentResponse[Any] | Sequence[AgentResponse[Any]],
    query: str | Sequence[Message] | Sequence[str | Sequence[Message]] | None = None,
    agent: Any | None = None,
    evaluators: Sequence[str] | None = None,
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    eval_name: str = "Agent Framework Response Eval",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Evaluate one or more agent responses that have already been produced.

    The simplest post-hoc evaluation path — pass the response you got from
    ``agent.run()`` and get a full evaluation.

    **Two modes:**

    - **Responses API** (e.g. ``AzureOpenAIResponsesClient``): Only ``response``
      is needed for quality evaluators (relevance, coherence, etc.). Foundry
      retrieves the full conversation from the stored ``response_id``.
    - **Any other client** (or **tool evaluators**): Also provide ``query``
      (the input messages) and ``agent`` (for tool definitions). Tool evaluators
      like ``tool_call_accuracy`` always require ``query`` and ``agent`` because
      tool definitions are not available through Responses API retrieval.

    Args:
        response: One or more ``AgentResponse`` objects from ``agent.run()``.
        query: The user query or input messages. Required when the response
            does not have a ``response_id``, or when using tool evaluators.
            For multiple responses, pass a list of queries in the same order.
        agent: Optional agent instance. Used to auto-extract tool definitions
            and to resolve default evaluators (adds ``tool_call_accuracy``
            when the agent has tools).
        evaluators: Evaluator names (e.g. ``[Evaluators.RELEVANCE]``).
            Defaults to relevance, coherence, and task_adherence.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance.
        model_deployment: Model deployment name for the evaluator LLM judge.
        eval_name: Display name for the evaluation.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.

    Returns:
        ``EvalResults`` with status, result counts, and portal link.

    Raises:
        ValueError: If any response lacks a ``response_id`` and ``query``
            is not provided.

    Example::

        # Responses API — query not needed
        response = await agent.run([Message("user", ["What's the weather?"])])
        results = await evaluate_response(
            response=response,
            project_client=project_client,
            model_deployment="gpt-4o",
        )

        # Any other client — provide query and agent
        response = await agent.run([Message("user", ["What's the weather?"])])
        results = await evaluate_response(
            response=response,
            query="What's the weather?",
            agent=agent,
            project_client=project_client,
            model_deployment="gpt-4o",
        )
    """
    client = _resolve_openai_client(openai_client, project_client)
    resolved_evaluators = _resolve_default_evaluators(evaluators, agent=agent)

    responses = [response] if isinstance(response, AgentResponse) else list(response)

    # Check if any evaluator needs tool_definitions (not available via response retrieval)
    needs_tool_defs = any(
        _resolve_evaluator(e) in _TOOL_EVALUATORS for e in resolved_evaluators
    )

    # Fast path: all responses have response_ids → use Responses API retrieval
    # (only when no tool evaluators — tool_definitions aren't in stored responses)
    all_have_ids = all(getattr(r, "response_id", None) for r in responses)

    if all_have_ids and query is None and not needs_tool_defs:
        response_ids = [r.response_id for r in responses]  # type: ignore[union-attr]
        return await _evaluate_responses(
            response_ids=response_ids,
            evaluators=resolved_evaluators,
            openai_client=client,
            model_deployment=model_deployment,
            eval_name=eval_name,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    # Dataset path: build eval items from query + response
    if query is None:
        if needs_tool_defs:
            raise ValueError(
                "Tool evaluators (e.g., tool_call_accuracy) require tool definitions "
                "which are not available through the Responses API retrieval path. "
                "Provide 'query' (the input messages) and 'agent' (for tool "
                "definitions) to use the dataset evaluation path."
            )
        raise ValueError(
            "Response does not have a response_id. Provide 'query' (the input "
            "messages) so the conversation can be reconstructed for evaluation. "
            "Optionally pass 'agent' to include tool definitions."
        )

    # Normalize queries to a list matching responses
    if isinstance(query, str) or isinstance(query, Message):
        queries: list[str | Sequence[Message]] = [query] * len(responses) if len(responses) == 1 else [query]  # type: ignore[list-item]
    elif isinstance(query, list) and len(query) > 0 and isinstance(query[0], Message):
        # Single query as list of Messages
        queries = [query] * len(responses) if len(responses) == 1 else [query]  # type: ignore[list-item]
    else:
        queries = list(query)  # type: ignore[arg-type]

    if len(queries) != len(responses):
        raise ValueError(
            f"Number of queries ({len(queries)}) does not match "
            f"number of responses ({len(responses)}). Provide one query "
            f"per response."
        )

    items = []
    for q, r in zip(queries, responses):
        items.append(AgentEvalConverter.to_eval_item(query=q, response=r, agent=agent))

    return await evaluate_dataset(
        items=items,
        evaluators=resolved_evaluators,
        openai_client=client,
        model_deployment=model_deployment,
        eval_name=eval_name,
        poll_interval=poll_interval,
        timeout=timeout,
    )


async def evaluate_traces(
    *,
    evaluators: Sequence[str] | None = None,
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    response_ids: Sequence[str] | None = None,
    trace_ids: Sequence[str] | None = None,
    agent_id: str | None = None,
    lookback_hours: int = 24,
    eval_name: str = "Agent Framework Trace Eval",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Evaluate agent behavior from OTel traces or response IDs (Path 1).

    The highest-impact path — works with any agent that emits OTel traces to
    App Insights. Provide *response_ids* for specific responses, *trace_ids*
    for specific traces, or *agent_id* with *lookback_hours* to evaluate
    recent activity.

    Args:
        evaluators: Evaluator names (e.g. ``[Evaluators.RELEVANCE]``).
            Defaults to relevance, coherence, and task_adherence.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance.
        model_deployment: Model deployment name for the evaluator LLM judge.
        response_ids: Evaluate specific Responses API responses.
        trace_ids: Evaluate specific OTel trace IDs from App Insights.
        agent_id: Filter traces by agent ID (used with *lookback_hours*).
        lookback_hours: Hours of trace history to evaluate (default 24).
        eval_name: Display name for the evaluation.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.

    Returns:
        ``EvalResults`` with status, result counts, and portal link.

    Example::

        # Evaluate by response IDs
        results = await evaluate_traces(
            response_ids=[response.response_id],
            evaluators=[Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY],
            project_client=project_client,
            model_deployment="gpt-4o",
        )

        # Evaluate recent traces
        results = await evaluate_traces(
            agent_id="travel-bot",
            evaluators=[Evaluators.INTENT_RESOLUTION, Evaluators.TASK_ADHERENCE],
            project_client=project_client,
            model_deployment="gpt-4o",
            lookback_hours=24,
        )
    """
    client = _resolve_openai_client(openai_client, project_client)
    resolved_evaluators = _resolve_default_evaluators(evaluators)

    if response_ids:
        return await _evaluate_responses(
            response_ids=response_ids,
            evaluators=resolved_evaluators,
            openai_client=client,
            model_deployment=model_deployment,
            eval_name=eval_name,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    if not trace_ids and not agent_id:
        raise ValueError("Provide at least one of: response_ids, trace_ids, or agent_id")

    # Trace-based data source — Foundry queries App Insights directly.
    # NOTE: The exact data source shape for azure_ai_traces may need
    # adjustment based on the Foundry API version.
    trace_source: dict[str, Any] = {
        "type": "azure_ai_traces",
        "lookback_hours": lookback_hours,
    }
    if trace_ids:
        trace_source["trace_ids"] = list(trace_ids)
    if agent_id:
        trace_source["agent_id"] = agent_id

    eval_obj = await _call_client(
        client.evals.create,
        name=eval_name,
        data_source_config={"type": "azure_ai_source", "scenario": "traces"},
        testing_criteria=_build_testing_criteria(resolved_evaluators, model_deployment),
    )

    run = await _call_client(
        client.evals.runs.create,
        eval_id=eval_obj.id,
        name=f"{eval_name} Run",
        data_source=trace_source,
    )

    return await _poll_eval_run(client, eval_obj.id, run.id, poll_interval, timeout)


async def evaluate_foundry_target(
    *,
    target: dict[str, Any],
    test_queries: Sequence[str],
    evaluators: Sequence[str] | None = None,
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    eval_name: str = "Agent Framework Target Eval",
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Evaluate a Foundry-registered agent or model deployment (Path 2).

    Foundry invokes the target, captures the output, and evaluates it. Use
    this for scheduled evals, red teaming, and CI/CD quality gates.

    Args:
        target: Target configuration dict, e.g.
            ``{"type": "azure_ai_agent", "name": "my-agent"}`` or
            ``{"type": "azure_ai_model", "deployment_name": "gpt-4o"}``.
        test_queries: Queries for Foundry to send to the target.
        evaluators: Evaluator names (e.g. ``[Evaluators.TASK_COMPLETION]``).
            Defaults to relevance, coherence, and task_adherence.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance.
        model_deployment: Model deployment name for the evaluator LLM judge.
        eval_name: Display name for the evaluation.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.

    Returns:
        ``EvalResults`` with status, result counts, and portal link.

    Example::

        results = await evaluate_foundry_target(
            target={"type": "azure_ai_agent", "name": "my-agent"},
            test_queries=["Book a flight to Paris", "Cancel my subscription"],
            evaluators=[Evaluators.TASK_COMPLETION, Evaluators.TOOL_CALL_ACCURACY],
            project_client=project_client,
            model_deployment="gpt-4o",
        )
    """
    client = _resolve_openai_client(openai_client, project_client)
    resolved_evaluators = _resolve_default_evaluators(evaluators)

    eval_obj = await _call_client(
        client.evals.create,
        name=eval_name,
        data_source_config={
            "type": "azure_ai_source",
            "scenario": "target_completions",
        },
        testing_criteria=_build_testing_criteria(resolved_evaluators, model_deployment),
    )

    # NOTE: The exact data source shape for azure_ai_target_completions may
    # need adjustment based on the Foundry API version.
    data_source: dict[str, Any] = {
        "type": "azure_ai_target_completions",
        "target": target,
        "source": {
            "type": "file_content",
            "content": [{"item": {"query": q}} for q in test_queries],
        },
    }

    run = await _call_client(
        client.evals.runs.create,
        eval_id=eval_obj.id,
        name=f"{eval_name} Run",
        data_source=data_source,
    )

    return await _poll_eval_run(client, eval_obj.id, run.id, poll_interval, timeout)


def setup_continuous_eval(
    *,
    agent_name: str,
    evaluators: Sequence[str],
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    sampling_rate: float = 1.0,
) -> None:
    """Set up continuous evaluation for a Foundry-registered agent (Path 4).

    Creates an evaluation rule so every response from the agent is
    automatically evaluated.

    Args:
        agent_name: Name of the Foundry-registered agent.
        evaluators: Evaluator names to apply to each response.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance.
        model_deployment: Model deployment name for the evaluator LLM judge.
        sampling_rate: Fraction of responses to evaluate (0.0-1.0, default 1.0).

    Raises:
        NotImplementedError: The continuous evaluation rules API shape is not
            yet finalized. This function documents the intended interface.
    """
    # The Foundry evaluation rules API (schedules.create_or_update) requires
    # an agent registered in Foundry with responses using agent_reference.
    # The exact API shape needs to be confirmed with the Foundry team.
    raise NotImplementedError(
        "Continuous evaluation setup requires the Foundry evaluation rules API, "
        "which is not yet integrated. Use evaluate_responses() or evaluate_traces() "
        "to evaluate individual responses in the meantime."
    )


async def evaluate_agent(
    *,
    agent: Any,
    queries: Sequence[str],
    evaluators: Sequence[str] | None = None,
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    eval_name: str | None = None,
    context: str | None = None,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Run an agent against test queries and evaluate the results (Path 3).

    The simplest path for evaluating an agent during development. For each
    query, runs the agent, converts the interaction to Foundry format using
    ``AgentEvalConverter``, and submits for evaluation.

    Args:
        agent: An agent-framework agent instance.
        queries: Test queries to run the agent against.
        evaluators: Evaluator names (e.g. ``[Evaluators.RELEVANCE]``).
            Defaults to relevance, coherence, and task_adherence.
            Automatically adds tool_call_accuracy if the agent has tools.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance. The OpenAI client
            is obtained automatically via ``project_client.get_openai_client()``.
        model_deployment: Model deployment name for the evaluator LLM judge.
        eval_name: Display name (defaults to agent name).
        context: Optional context for groundedness evaluation.
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.

    Returns:
        ``EvalResults`` with status, result counts, and portal link.

    Example::

        results = await evaluate_agent(
            agent=my_agent,
            queries=["What's the weather?", "Book a flight to London"],
            evaluators=[Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY],
            project_client=project_client,
            model_deployment="gpt-4o",
        )
        results.assert_passed()  # raises AssertionError if any failures
    """
    resolved_evaluators = _resolve_default_evaluators(evaluators, agent=agent)
    converter = AgentEvalConverter()
    items: list[dict[str, Any]] = []

    for query in queries:
        response = await agent.run([Message("user", [query])])
        items.append(
            converter.to_eval_item(
                query=query,
                response=response,
                agent=agent,
                context=context,
            )
        )

    name = eval_name or f"Eval: {getattr(agent, 'name', None) or getattr(agent, 'id', 'agent')}"

    return await evaluate_dataset(
        items=items,
        evaluators=resolved_evaluators,
        openai_client=openai_client,
        project_client=project_client,
        model_deployment=model_deployment,
        eval_name=name,
        poll_interval=poll_interval,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Workflow evaluation helpers
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
    ``AgentExecutor``. Skips internal framework executors (IDs starting with
    ``_`` or containing "input" / "end" in lowercase).

    Args:
        workflow_result: The completed workflow result.
        workflow: Optional workflow instance for resolving agent references
            (used to extract tool definitions).

    Returns:
        List of ``_AgentEvalData`` with query, response, and agent for each
        agent executor that ran.
    """
    from agent_framework import AgentExecutor, AgentExecutorResponse

    # Collect executor_invoked data (input to each executor)
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

            # Extract the AgentExecutorResponse from completion data
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

            # Extract query from full_conversation (user messages) or invoked data
            query: str | list[Message]
            if agent_exec_response.full_conversation:
                user_msgs = [
                    m for m in agent_exec_response.full_conversation if m.role == "user"
                ]
                if user_msgs:
                    query = user_msgs
                else:
                    query = agent_exec_response.full_conversation
            elif executor_id in invoked_data:
                input_data = invoked_data[executor_id]
                if isinstance(input_data, str):
                    query = input_data
                elif isinstance(input_data, list):
                    query = input_data
                else:
                    query = str(input_data)
            else:
                continue

            # Resolve agent reference from workflow for tool extraction
            agent_ref = None
            if workflow is not None:
                executor = workflow.executors.get(executor_id)
                if executor is not None and isinstance(executor, AgentExecutor):
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
    """Extract the original user query from a workflow result.

    Looks at the first ``executor_invoked`` event to find the initial input.
    """
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


async def evaluate_workflow(
    *,
    workflow: Workflow,
    workflow_result: WorkflowRunResult | None = None,
    queries: Sequence[str] | None = None,
    evaluators: Sequence[str] | None = None,
    openai_client: OpenAI | None = None,
    project_client: Any | None = None,
    model_deployment: str,
    eval_name: str | None = None,
    include_overall: bool = True,
    include_per_agent: bool = True,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> EvalResults:
    """Evaluate a multi-agent workflow with per-agent breakdown.

    Evaluates each sub-agent individually and (optionally) the workflow's
    overall output, returning a single ``EvalResults`` with per-agent
    breakdowns in ``sub_results``.

    **Two modes:**

    - **Post-hoc**: Pass ``workflow_result`` from a previous
      ``workflow.run()`` call. Each agent executor's input/output is
      extracted from the event log.
    - **Run + evaluate**: Pass ``queries`` and the workflow will be run
      against each query, then evaluated.

    Args:
        workflow: The workflow instance (needed to resolve agent references
            for tool extraction).
        workflow_result: A completed ``WorkflowRunResult``. Provide this
            or ``queries``.
        queries: Test queries to run through the workflow. Each triggers a
            separate workflow run. Provide this or ``workflow_result``.
        evaluators: Evaluator names (e.g. ``[Evaluators.RELEVANCE]``).
            Defaults to relevance, coherence, and task_adherence.
        openai_client: OpenAI client. Provide this or *project_client*.
        project_client: An ``AIProjectClient`` instance.
        model_deployment: Model deployment name for the evaluator LLM judge.
        eval_name: Display name for the evaluation.
        include_overall: Whether to evaluate the workflow's final output
            as a whole (default ``True``).
        include_per_agent: Whether to evaluate each sub-agent individually
            (default ``True``).
        poll_interval: Seconds between status polls.
        timeout: Maximum seconds to wait for completion.

    Returns:
        ``EvalResults`` with per-agent breakdown in ``sub_results``.
        The top-level results represent the overall workflow eval
        (if ``include_overall`` is True), or an aggregate.

    Example::

        result = await workflow.run("Plan a trip to Paris")

        eval_results = await evaluate_workflow(
            workflow=workflow,
            workflow_result=result,
            project_client=project_client,
            model_deployment="gpt-4o",
        )

        print(f"Overall: {eval_results.passed}/{eval_results.total}")
        for name, sub in eval_results.sub_results.items():
            print(f"  {name}: {sub.passed}/{sub.total}")
    """
    from agent_framework import WorkflowRunResult as WRR

    client = _resolve_openai_client(openai_client, project_client)

    if workflow_result is None and queries is None:
        raise ValueError("Provide either 'workflow_result' or 'queries'.")

    # Run the workflow if queries are provided
    all_agent_data: list[_AgentEvalData] = []
    overall_items: list[dict[str, Any]] = []

    if queries is not None:
        results_list: list[WRR] = []
        for query in queries:
            result = await workflow.run(query)
            if not isinstance(result, WRR):
                raise TypeError(
                    f"Expected WorkflowRunResult from workflow.run(), got {type(result).__name__}. "
                    "Make sure to call workflow.run() without stream=True."
                )
            results_list.append(result)

            # Extract per-agent data from each run
            agent_data = _extract_agent_eval_data(result, workflow)
            all_agent_data.extend(agent_data)

            # Extract overall item (original query → final output)
            if include_overall:
                outputs = result.get_outputs()
                if outputs:
                    final_output = outputs[-1]
                    # Final output may be list[Message] or AgentResponse
                    if isinstance(final_output, list) and final_output and isinstance(final_output[0], Message):
                        response_text = " ".join(m.text for m in final_output if m.role == "assistant")
                        overall_response = AgentResponse(messages=[Message("assistant", [response_text])])
                    elif isinstance(final_output, AgentResponse):
                        overall_response = final_output
                    else:
                        overall_response = AgentResponse(messages=[Message("assistant", [str(final_output)])])

                    overall_items.append(
                        AgentEvalConverter.to_eval_item(query=query, response=overall_response)
                    )
    else:
        assert workflow_result is not None
        all_agent_data = _extract_agent_eval_data(workflow_result, workflow)

        if include_overall:
            original_query = _extract_overall_query(workflow_result)
            outputs = workflow_result.get_outputs()
            if original_query and outputs:
                final_output = outputs[-1]
                if isinstance(final_output, list) and final_output and isinstance(final_output[0], Message):
                    response_text = " ".join(m.text for m in final_output if m.role == "assistant")
                    overall_response = AgentResponse(messages=[Message("assistant", [response_text])])
                elif isinstance(final_output, AgentResponse):
                    overall_response = final_output
                else:
                    overall_response = AgentResponse(messages=[Message("assistant", [str(final_output)])])

                overall_items.append(
                    AgentEvalConverter.to_eval_item(query=original_query, response=overall_response)
                )

    wf_name = eval_name or f"Workflow Eval: {workflow.__class__.__name__}"
    converter = AgentEvalConverter()
    sub_results: dict[str, EvalResults] = {}

    # Evaluate each agent individually
    if include_per_agent and all_agent_data:
        # Group agent data by executor_id
        agents_by_id: dict[str, list[_AgentEvalData]] = {}
        for ad in all_agent_data:
            agents_by_id.setdefault(ad.executor_id, []).append(ad)

        for executor_id, agent_data_list in agents_by_id.items():
            agent_items: list[dict[str, Any]] = []
            agent_ref = agent_data_list[0].agent

            for ad in agent_data_list:
                agent_items.append(
                    converter.to_eval_item(
                        query=ad.query,
                        response=ad.response,
                        agent=ad.agent,
                    )
                )

            agent_evaluators = _resolve_default_evaluators(evaluators, agent=agent_ref)
            agent_eval = await evaluate_dataset(
                items=agent_items,
                evaluators=agent_evaluators,
                openai_client=client,
                model_deployment=model_deployment,
                eval_name=f"{wf_name} — {executor_id}",
                poll_interval=poll_interval,
                timeout=timeout,
            )
            sub_results[executor_id] = agent_eval

    # Evaluate overall workflow output
    overall_eval: EvalResults
    if include_overall and overall_items:
        resolved_evaluators = _resolve_default_evaluators(evaluators)
        overall_eval = await evaluate_dataset(
            items=overall_items,
            evaluators=resolved_evaluators,
            openai_client=client,
            model_deployment=model_deployment,
            eval_name=f"{wf_name} — overall",
            poll_interval=poll_interval,
            timeout=timeout,
        )
    elif sub_results:
        # Aggregate from sub-results
        total_passed = sum(s.passed for s in sub_results.values())
        total_failed = sum(s.failed for s in sub_results.values())
        total_errored = sum(s.errored for s in sub_results.values())
        all_completed = all(s.status == "completed" for s in sub_results.values())
        overall_eval = EvalResults(
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

    overall_eval.sub_results = sub_results
    return overall_eval
