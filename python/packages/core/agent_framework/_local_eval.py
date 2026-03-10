# Copyright (c) Microsoft. All rights reserved.

"""Local evaluation checks for fast, API-free agent evaluation.

Provides lightweight checks that run locally without any API calls.
Useful for inner-loop development, CI smoke tests, and combining with
cloud-based evaluators for comprehensive coverage.

Example — built-in checks::

    from agent_framework import LocalEvaluator, keyword_check, evaluate_agent

    local = LocalEvaluator(
        keyword_check("weather", "temperature"),
        tool_called_check("get_weather"),
    )
    results = await evaluate_agent(agent=agent, queries=queries, evaluators=local)

Example — custom function evaluators::

    from agent_framework import LocalEvaluator, function_evaluator, evaluate_agent

    @function_evaluator
    def mentions_weather(query: str, response: str) -> bool:
        return "weather" in response.lower()

    @function_evaluator
    def word_overlap(response: str, expected: str) -> float:
        r_words = set(response.lower().split())
        e_words = set(expected.lower().split())
        return len(r_words & e_words) / max(len(e_words), 1)

    local = LocalEvaluator(mentions_weather, word_overlap)
    results = await evaluate_agent(agent=agent, queries=queries, evaluators=local)
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ._eval import EvalItem, EvalResults

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single check on a single evaluation item.

    Attributes:
        passed: Whether the check passed.
        reason: Human-readable explanation.
        check_name: Name of the check that produced this result.
    """

    passed: bool
    reason: str
    check_name: str


EvalCheck = Callable[[EvalItem], CheckResult]
"""A check function that takes an ``EvalItem`` and returns a ``CheckResult``."""

# Async variant — returns a coroutine instead of a plain CheckResult
AsyncEvalCheck = Callable[[EvalItem], Any]  # Actually returns Awaitable[CheckResult]


def keyword_check(*keywords: str, case_sensitive: bool = False) -> EvalCheck:
    """Check that the response contains all specified keywords.

    Args:
        *keywords: Required keywords that must appear in the response.
        case_sensitive: Whether matching is case-sensitive (default ``False``).

    Returns:
        A check function for use with ``LocalEvaluator``.

    Example::

        check = keyword_check("weather", "temperature")
    """

    def _check(item: EvalItem) -> CheckResult:
        text = item.response if case_sensitive else item.response.lower()
        missing = [k for k in keywords if (k if case_sensitive else k.lower()) not in text]
        if missing:
            return CheckResult(False, f"Missing keywords: {missing}", "keyword_check")
        return CheckResult(True, "All keywords found", "keyword_check")

    return _check


def tool_called_check(*tool_names: str) -> EvalCheck:
    """Check that specific tools were called during the conversation.

    Inspects the conversation history for ``tool_calls`` entries matching
    the expected tool names.

    Args:
        *tool_names: Names of tools that should have been called.

    Returns:
        A check function for use with ``LocalEvaluator``.

    Example::

        check = tool_called_check("get_weather", "get_flight_price")
    """

    def _check(item: EvalItem) -> CheckResult:
        called: set[str] = set()
        for msg in item.conversation:
            # Foundry format: content is a list of typed items
            for ci in msg.get("content", []):
                if isinstance(ci, dict) and ci.get("type") == "tool_call":
                    name = ci.get("name", "")
                    if name:
                        called.add(name)
            # Also check OpenAI format for compatibility
            for tc in msg.get("tool_calls", []):
                name = tc.get("function", {}).get("name", "")
                if name:
                    called.add(name)
        missing = [t for t in tool_names if t not in called]
        if missing:
            return CheckResult(
                False,
                f"Expected tools not called: {missing} (called: {sorted(called)})",
                "tool_called",
            )
        return CheckResult(True, f"All expected tools called: {sorted(called)}", "tool_called")

    return _check


# ---------------------------------------------------------------------------
# Function evaluator — wrap plain functions as EvalChecks
# ---------------------------------------------------------------------------

# Parameters recognized by the function evaluator wrapper
_KNOWN_PARAMS = frozenset({
    "query", "response", "expected", "conversation", "tool_definitions", "context",
})


def _resolve_function_args(fn: Callable[..., Any], item: EvalItem) -> dict[str, Any]:
    """Build a kwargs dict for *fn* based on its signature and the EvalItem.

    Supported parameter names:

    ========== ====================================================
    Name       Value from EvalItem
    ========== ====================================================
    query      ``item.query``
    response   ``item.response``
    expected   ``item.expected``  (empty string if not set)
    conversation ``item.conversation``
    tool_definitions ``item.tool_definitions``
    context    ``item.context``
    ========== ====================================================

    Parameters with default values are only supplied when their name is
    recognised.  Unknown required parameters raise ``TypeError``.
    """
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}

    field_map: dict[str, Any] = {
        "query": item.query,
        "response": item.response,
        "expected": item.expected or "",
        "conversation": item.conversation,
        "tool_definitions": item.tool_definitions,
        "context": item.context,
    }

    for name, param in sig.parameters.items():
        if name in field_map:
            kwargs[name] = field_map[name]
        elif param.default is inspect.Parameter.empty:
            raise TypeError(
                f"Function evaluator '{fn.__name__}' has unknown required parameter "
                f"'{name}'.  Supported: {sorted(_KNOWN_PARAMS)}"
            )
        # else: has a default — leave it to Python

    return kwargs


def _coerce_result(value: Any, check_name: str) -> CheckResult:
    """Convert a function evaluator return value to a ``CheckResult``.

    Accepted return types:

    * ``bool`` — True/False maps directly to pass/fail.
    * ``int | float`` — ≥ 0.5 is pass (score is included in reason).
    * ``CheckResult`` — returned as-is.
    * ``dict`` with ``score`` or ``passed`` key — converted to CheckResult.
    """
    if isinstance(value, CheckResult):
        return value

    if isinstance(value, bool):
        return CheckResult(value, "passed" if value else "failed", check_name)

    if isinstance(value, (int, float)):
        passed = value >= 0.5
        return CheckResult(passed, f"score={value:.3f}", check_name)

    if isinstance(value, dict):
        if "score" in value:
            score = float(value["score"])
            passed = score >= value.get("threshold", 0.5)
            reason = value.get("reason", f"score={score:.3f}")
            return CheckResult(passed, reason, check_name)
        if "passed" in value:
            return CheckResult(
                bool(value["passed"]),
                value.get("reason", "passed" if value["passed"] else "failed"),
                check_name,
            )

    raise TypeError(
        f"Function evaluator '{check_name}' returned unsupported type "
        f"{type(value).__name__}. Expected bool, float, dict, or CheckResult."
    )


def function_evaluator(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> EvalCheck | Callable[[Callable[..., Any]], EvalCheck]:
    """Wrap a plain function as an ``EvalCheck`` for use with ``LocalEvaluator``.

    Works with both sync and async functions.  The function's parameter names
    determine what data it receives from the ``EvalItem``.  Any combination of
    the following parameter names is valid:

    * ``query`` — the user query (str)
    * ``response`` — the agent response (str)
    * ``expected`` — expected output for ground-truth comparison (str)
    * ``conversation`` — full conversation history (list[dict])
    * ``tool_definitions`` — tool schemas available to the agent (list[dict])
    * ``context`` — grounding context (str | None)

    Return ``bool``, ``float`` (≥0.5 = pass), ``dict`` with ``score`` or
    ``passed`` key, or ``CheckResult``.

    Can be used as a decorator (with or without arguments) or called directly::

        # Decorator — no args
        @function_evaluator
        def mentions_weather(query: str, response: str) -> bool:
            return "weather" in response.lower()

        # Decorator — with name
        @function_evaluator(name="length_check")
        def is_not_too_long(response: str) -> bool:
            return len(response) < 2000

        # Direct wrapping
        check = function_evaluator(my_scorer, name="my_scorer")

        # Async function — handled automatically
        @function_evaluator
        async def llm_judge(query: str, response: str) -> float:
            result = await my_llm_client.score(query, response)
            return result.score

        # Use with LocalEvaluator
        local = LocalEvaluator(mentions_weather, is_not_too_long, check, llm_judge)

    Args:
        fn: The function to wrap.  If omitted, returns a decorator.
        name: Display name for the check (defaults to ``fn.__name__``).
    """

    def _wrap(func: Callable[..., Any]) -> EvalCheck:
        check_name = name or getattr(func, "__name__", "function_evaluator")
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            async def _async_check(item: EvalItem) -> CheckResult:
                kwargs = _resolve_function_args(func, item)
                result = await func(**kwargs)
                return _coerce_result(result, check_name)

            _async_check.__name__ = check_name  # type: ignore[attr-defined]
            _async_check.__doc__ = func.__doc__
            _async_check._is_async_check = True  # type: ignore[attr-defined]
            return _async_check
        else:

            def _check(item: EvalItem) -> CheckResult:
                kwargs = _resolve_function_args(func, item)
                result = func(**kwargs)
                return _coerce_result(result, check_name)

            _check.__name__ = check_name  # type: ignore[attr-defined]
            _check.__doc__ = func.__doc__
            return _check

    # Support @function_evaluator (no parens) and @function_evaluator(name="x")
    if fn is not None:
        return _wrap(fn)
    return _wrap


def async_function_evaluator(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> EvalCheck | Callable[[Callable[..., Any]], EvalCheck]:
    """Deprecated: use :func:`function_evaluator` which handles async automatically."""
    return function_evaluator(fn, name=name)


class LocalEvaluator:
    """Evaluation provider that runs checks locally without API calls.

    Implements the ``Evaluator`` protocol. Each check function is applied
    to every item. An item passes only if all checks pass.

    Example::

        from agent_framework import LocalEvaluator, keyword_check, evaluate_agent

        local = LocalEvaluator(
            keyword_check("weather"),
            tool_called_check("get_weather"),
        )
        results = await evaluate_agent(agent=agent, queries=queries, evaluators=local)

    To mix with cloud evaluators::

        from agent_framework_azure_ai import FoundryEvals

        results = await evaluate_agent(
            agent=agent,
            queries=queries,
            evaluators=[local, FoundryEvals(project_client=client, model_deployment="gpt-4o")],
        )
    """

    def __init__(self, *checks: EvalCheck):
        self._checks = checks

    @property
    def name(self) -> str:
        """Human-readable name of this evaluator."""
        return "Local"

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str = "Local Eval",
    ) -> EvalResults:
        """Run all checks on each item and return aggregated results.

        An item passes only if every check passes for that item. Per-check
        breakdowns are available in ``per_evaluator``.

        Supports both sync and async check functions (from
        :func:`function_evaluator` and :func:`async_function_evaluator`).
        """
        passed = 0
        failed = 0
        per_check: dict[str, dict[str, int]] = {}
        failure_reasons: list[str] = []

        for item in items:
            item_passed = True
            for check_fn in self._checks:
                raw = check_fn(item)
                # Await if the check returned a coroutine (async_function_evaluator)
                if inspect.isawaitable(raw):
                    result = await raw
                else:
                    result = raw
                counts = per_check.setdefault(result.check_name, {"passed": 0, "failed": 0, "errored": 0})
                if result.passed:
                    counts["passed"] += 1
                else:
                    counts["failed"] += 1
                    item_passed = False
                    failure_reasons.append(f"{result.check_name}: {result.reason}")

            if item_passed:
                passed += 1
            else:
                failed += 1

        return EvalResults(
            provider=self.name,
            eval_id="local",
            run_id=eval_name,
            status="completed",
            result_counts={"passed": passed, "failed": failed, "errored": 0},
            per_evaluator=per_check,
            error="; ".join(failure_reasons) if failure_reasons else None,
        )
