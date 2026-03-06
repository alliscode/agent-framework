# Copyright (c) Microsoft. All rights reserved.

"""Local evaluation checks for fast, API-free agent evaluation.

Provides lightweight checks that run locally without any API calls.
Useful for inner-loop development, CI smoke tests, and combining with
cloud-based evaluators for comprehensive coverage.

Example::

    from agent_framework import LocalEvaluator, keyword_check, evaluate_agent

    local = LocalEvaluator(
        keyword_check("weather", "temperature"),
        tool_called_check("get_weather"),
    )
    results = await evaluate_agent(agent=agent, queries=queries, evaluators=local)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ._eval import EvalItem, EvalResults


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
        """
        passed = 0
        failed = 0
        per_check: dict[str, dict[str, int]] = {}
        failure_reasons: list[str] = []

        for item in items:
            item_passed = True
            for check_fn in self._checks:
                result = check_fn(item)
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
