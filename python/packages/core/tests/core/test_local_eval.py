# Copyright (c) Microsoft. All rights reserved.

"""Tests for evaluator and async_function_evaluator."""

from __future__ import annotations

import inspect

import pytest

from agent_framework._eval import EvalItem
from agent_framework._types import Message
from agent_framework._local_eval import (
    CheckResult,
    LocalEvaluator,
    async_function_evaluator,
    evaluator,
    keyword_check,
    tool_called_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    query: str = "What's the weather in Paris?",
    response: str = "It's sunny and 75°F",
    expected: str | None = None,
    conversation: list | None = None,
    tools: list | None = None,
    context: str | None = None,
) -> EvalItem:
    if conversation is None:
        conversation = [Message("user", [query]), Message("assistant", [response])]
    return EvalItem(
        conversation=conversation,
        expected=expected,
        tools=tools,
        context=context,
    )


# ---------------------------------------------------------------------------
# Tier 1: (query, response) -> result
# ---------------------------------------------------------------------------

class TestTier1SimpleChecks:
    def test_bool_return_true(self):
        @evaluator
        def has_temperature(query: str, response: str) -> bool:
            return "°F" in response

        result = has_temperature(_make_item())
        assert result.passed is True
        assert result.check_name == "has_temperature"

    def test_bool_return_false(self):
        @evaluator
        def has_celsius(query: str, response: str) -> bool:
            return "°C" in response

        result = has_celsius(_make_item())
        assert result.passed is False

    def test_float_return_passing(self):
        @evaluator
        def length_score(response: str) -> float:
            return min(len(response) / 10, 1.0)

        result = length_score(_make_item())
        assert result.passed is True
        assert "score=" in result.reason

    def test_float_return_failing(self):
        @evaluator
        def always_low(response: str) -> float:
            return 0.1

        result = always_low(_make_item())
        assert result.passed is False

    def test_response_only(self):
        """Function with only 'response' param should work."""
        @evaluator
        def is_short(response: str) -> bool:
            return len(response) < 1000

        result = is_short(_make_item())
        assert result.passed is True

    def test_query_only(self):
        """Function with only 'query' param should work."""
        @evaluator
        def is_question(query: str) -> bool:
            return "?" in query

        result = is_question(_make_item())
        assert result.passed is True


# ---------------------------------------------------------------------------
# Tier 2: (query, response, expected) -> result
# ---------------------------------------------------------------------------

class TestTier2GroundTruth:
    def test_exact_match(self):
        @evaluator
        def exact_match(response: str, expected: str) -> bool:
            return response.strip() == expected.strip()

        item = _make_item(response="42", expected="42")
        assert exact_match(item).passed is True

        item2 = _make_item(response="43", expected="42")
        assert exact_match(item2).passed is False

    def test_expected_defaults_to_empty(self):
        """When expected is None on the item, it should be passed as ''."""
        @evaluator
        def check_expected(expected: str) -> bool:
            return expected == ""

        result = check_expected(_make_item(expected=None))
        assert result.passed is True

    def test_similarity_score(self):
        @evaluator
        def word_overlap(response: str, expected: str) -> float:
            r_words = set(response.lower().split())
            e_words = set(expected.lower().split())
            if not e_words:
                return 1.0
            return len(r_words & e_words) / len(e_words)

        item = _make_item(response="sunny warm day", expected="warm sunny afternoon")
        result = word_overlap(item)
        assert result.passed is True  # 2/3 overlap ≥ 0.5


# ---------------------------------------------------------------------------
# Tier 3: full context (conversation, tools, context)
# ---------------------------------------------------------------------------

class TestTier3FullContext:
    def test_conversation_access(self):
        @evaluator
        def multi_turn(query: str, response: str, *, conversation: list) -> bool:
            return len(conversation) >= 2

        item = _make_item(conversation=[Message("user", []), Message("assistant", [])])
        assert multi_turn(item).passed is True

        item2 = _make_item(conversation=[Message("user", [])])
        assert multi_turn(item2).passed is False

    def test_tools_access(self):
        @evaluator
        def has_tools(tools: list) -> bool:
            return len(tools) > 0

        mock_tool = type('MockTool', (), {'name': 'get_weather', 'description': 'Get weather', 'parameters': lambda self: {}})()
        item = _make_item(tools=[mock_tool])
        assert has_tools(item).passed is True

    def test_context_access(self):
        @evaluator
        def grounded(response: str, context: str) -> bool:
            if not context:
                return True
            return any(word in response.lower() for word in context.lower().split())

        item = _make_item(response="It's sunny", context="sunny warm")
        assert grounded(item).passed is True

    def test_all_params(self):
        @evaluator
        def full_check(
            query: str,
            response: str,
            expected: str,
            conversation: list,
            tools: list,
            context: str,
        ) -> bool:
            return all([query, response, expected is not None, isinstance(conversation, list)])

        item = _make_item(expected="foo", context="bar")
        assert full_check(item).passed is True


# ---------------------------------------------------------------------------
# Return type coercion
# ---------------------------------------------------------------------------

class TestReturnTypeCoercion:
    def test_dict_with_score(self):
        @evaluator
        def scored(response: str) -> dict:
            return {"score": 0.9, "reason": "good answer"}

        result = scored(_make_item())
        assert result.passed is True
        assert result.reason == "good answer"

    def test_dict_with_score_below_threshold(self):
        @evaluator
        def low_scored(response: str) -> dict:
            return {"score": 0.3}

        result = low_scored(_make_item())
        assert result.passed is False

    def test_dict_with_custom_threshold(self):
        @evaluator
        def custom_threshold(response: str) -> dict:
            return {"score": 0.3, "threshold": 0.2}

        result = custom_threshold(_make_item())
        assert result.passed is True

    def test_dict_with_passed(self):
        @evaluator
        def explicit_pass(response: str) -> dict:
            return {"passed": True, "reason": "all good"}

        result = explicit_pass(_make_item())
        assert result.passed is True
        assert result.reason == "all good"

    def test_check_result_passthrough(self):
        @evaluator
        def returns_check_result(response: str) -> CheckResult:
            return CheckResult(True, "direct result", "custom")

        result = returns_check_result(_make_item())
        assert result.passed is True
        assert result.reason == "direct result"
        assert result.check_name == "custom"

    def test_unsupported_return_type(self):
        @evaluator
        def bad_return(response: str) -> str:
            return "oops"

        with pytest.raises(TypeError, match="unsupported type"):
            bad_return(_make_item())

    def test_int_return(self):
        @evaluator
        def int_score(response: str) -> int:
            return 1

        result = int_score(_make_item())
        assert result.passed is True


# ---------------------------------------------------------------------------
# Decorator variants
# ---------------------------------------------------------------------------

class TestDecoratorVariants:
    def test_decorator_no_parens(self):
        @evaluator
        def my_check(response: str) -> bool:
            return True

        assert my_check(_make_item()).passed is True

    def test_decorator_with_name(self):
        @evaluator(name="custom_name")
        def my_check(response: str) -> bool:
            return True

        assert my_check.__name__ == "custom_name"
        result = my_check(_make_item())
        assert result.check_name == "custom_name"

    def test_direct_call(self):
        def raw_fn(query: str, response: str) -> bool:
            return len(response) > 0

        check = evaluator(raw_fn, name="direct")
        result = check(_make_item())
        assert result.passed is True
        assert result.check_name == "direct"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_required_param_raises(self):
        @evaluator
        def bad_params(query: str, unknown_param: str) -> bool:
            return True

        with pytest.raises(TypeError, match="unknown required parameter"):
            bad_params(_make_item())

    def test_unknown_optional_param_ok(self):
        @evaluator
        def optional_unknown(query: str, foo: str = "default") -> bool:
            return foo == "default"

        result = optional_unknown(_make_item())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_async_function_works_with_evaluator(self):
        """Using an async function with @evaluator should work."""
        @evaluator
        async def async_fn(response: str) -> bool:
            return True

        result = async_fn(_make_item())
        # Should return an awaitable
        assert inspect.isawaitable(result)
        check_result = await result
        assert check_result.passed is True


# ---------------------------------------------------------------------------
# Integration with LocalEvaluator
# ---------------------------------------------------------------------------

class TestLocalEvaluatorIntegration:
    @pytest.mark.asyncio
    async def test_mixed_checks(self):
        """Function evaluators mix with built-in checks in LocalEvaluator."""
        @evaluator
        def length_ok(response: str) -> bool:
            return len(response) > 5

        local = LocalEvaluator(
            keyword_check("sunny"),
            length_ok,
        )
        items = [_make_item()]
        results = await local.evaluate(items, eval_name="mixed test")

        assert results.status == "completed"
        assert results.result_counts["passed"] == 1
        assert results.result_counts["failed"] == 0

    @pytest.mark.asyncio
    async def test_evaluator_failure_counted(self):
        @evaluator
        def always_fail(response: str) -> bool:
            return False

        local = LocalEvaluator(always_fail)
        results = await local.evaluate([_make_item()])

        assert results.result_counts["failed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_evaluators(self):
        @evaluator
        def check_a(response: str) -> float:
            return 0.9

        @evaluator
        def check_b(query: str, response: str, expected: str) -> bool:
            return True

        @evaluator(name="check_c")
        def check_c(response: str, conversation: list) -> dict:
            return {"score": 0.8, "reason": "looks good"}

        local = LocalEvaluator(check_a, check_b, check_c)
        results = await local.evaluate([_make_item(expected="test")])

        assert results.result_counts["passed"] == 1
        assert "check_a" in results.per_evaluator
        assert "check_b" in results.per_evaluator
        assert "check_c" in results.per_evaluator


# ---------------------------------------------------------------------------
# async_function_evaluator
# ---------------------------------------------------------------------------

class TestAsyncFunctionEvaluator:
    @pytest.mark.asyncio
    async def test_async_evaluator_in_local(self):
        @async_function_evaluator
        async def async_check(query: str, response: str) -> bool:
            return len(response) > 0

        local = LocalEvaluator(async_check)
        results = await local.evaluate([_make_item()])
        assert results.result_counts["passed"] == 1

    @pytest.mark.asyncio
    async def test_async_with_name(self):
        @async_function_evaluator(name="named_async")
        async def my_async(response: str) -> float:
            return 0.75

        # Async checks return coroutines — must be awaited
        result = await my_async(_make_item())
        assert result.passed is True
        assert result.check_name == "named_async"


# ---------------------------------------------------------------------------
# Auto-wrapping bare checks in evaluate_agent
# ---------------------------------------------------------------------------

class TestAutoWrapEvalChecks:
    @pytest.mark.asyncio
    async def test_bare_check_in_evaluators_list(self):
        """Bare EvalCheck callables are auto-wrapped in LocalEvaluator."""
        from agent_framework._eval import _run_evaluators

        @evaluator
        def is_long(response: str) -> bool:
            return len(response.split()) > 2

        items = [_make_item(response="It is sunny and warm today")]
        results = await _run_evaluators(is_long, items, eval_name="test")
        assert len(results) == 1
        assert results[0].result_counts["passed"] == 1

    @pytest.mark.asyncio
    async def test_mixed_evaluators_and_checks(self):
        """Mix of Evaluator instances and bare checks works."""
        from agent_framework._eval import _run_evaluators

        @evaluator
        def has_words(response: str) -> bool:
            return len(response.split()) > 0

        local = LocalEvaluator(keyword_check("sunny"))

        items = [_make_item(response="It is sunny")]
        results = await _run_evaluators([local, has_words], items, eval_name="test")
        assert len(results) == 2
        assert all(r.result_counts["passed"] == 1 for r in results)

    @pytest.mark.asyncio
    async def test_adjacent_checks_grouped(self):
        """Adjacent bare checks are grouped into a single LocalEvaluator."""
        from agent_framework._eval import _run_evaluators

        @evaluator
        def check_a(response: str) -> bool:
            return True

        @evaluator
        def check_b(response: str) -> bool:
            return True

        items = [_make_item()]
        results = await _run_evaluators([check_a, check_b], items, eval_name="test")
        # Two adjacent checks → one LocalEvaluator → one result
        assert len(results) == 1
        assert results[0].result_counts["passed"] == 1
