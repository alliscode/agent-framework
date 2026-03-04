# Copyright (c) Microsoft. All rights reserved.

"""Tests for the AgentEvalConverter and eval helper functions."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import AgentExecutorResponse, AgentResponse, Content, FunctionTool, Message, WorkflowEvent
from agent_framework._workflows._workflow import WorkflowRunResult

from agent_framework_azure_ai._foundry_evals import (
    AgentEvalConverter,
    EvalResults,
    Evaluators,
    _AgentEvalData,
    _build_item_schema,
    _build_testing_criteria,
    _extract_agent_eval_data,
    _extract_overall_query,
    _resolve_default_evaluators,
    _resolve_evaluator,
    _resolve_openai_client,
    evaluate_dataset,
    evaluate_response,
    evaluate_workflow,
)

# ---------------------------------------------------------------------------
# _resolve_evaluator
# ---------------------------------------------------------------------------


class TestResolveEvaluator:
    def test_short_name(self) -> None:
        assert _resolve_evaluator("relevance") == "builtin.relevance"
        assert _resolve_evaluator("tool_call_accuracy") == "builtin.tool_call_accuracy"
        assert _resolve_evaluator("violence") == "builtin.violence"

    def test_already_qualified(self) -> None:
        assert _resolve_evaluator("builtin.relevance") == "builtin.relevance"
        assert _resolve_evaluator("builtin.custom") == "builtin.custom"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown evaluator 'bogus'"):
            _resolve_evaluator("bogus")


# ---------------------------------------------------------------------------
# AgentEvalConverter.convert_message
# ---------------------------------------------------------------------------


class TestConvertMessage:
    def test_user_text_message(self) -> None:
        msg = Message("user", ["Hello, world!"])
        result = AgentEvalConverter.convert_message(msg)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello, world!"}

    def test_system_message(self) -> None:
        msg = Message("system", ["You are helpful."])
        result = AgentEvalConverter.convert_message(msg)
        assert result[0] == {"role": "system", "content": "You are helpful."}

    def test_assistant_text_message(self) -> None:
        msg = Message("assistant", ["Here is the answer."])
        result = AgentEvalConverter.convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Here is the answer."
        assert "tool_calls" not in result[0]

    def test_assistant_with_tool_call(self) -> None:
        msg = Message(
            "assistant",
            [
                Content.from_function_call(
                    call_id="call_1",
                    name="get_weather",
                    arguments=json.dumps({"location": "Seattle"}),
                ),
            ],
        )
        result = AgentEvalConverter.convert_message(msg)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "Seattle"}

    def test_assistant_text_and_tool_call(self) -> None:
        msg = Message(
            "assistant",
            [
                Content.from_text("Let me check that."),
                Content.from_function_call(
                    call_id="call_2",
                    name="search",
                    arguments={"query": "flights"},
                ),
            ],
        )
        result = AgentEvalConverter.convert_message(msg)
        assert len(result) == 1
        assert result[0]["content"] == "Let me check that."
        assert len(result[0]["tool_calls"]) == 1
        # Dict arguments should be JSON-serialized
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"query": "flights"}

    def test_tool_result_message(self) -> None:
        msg = Message(
            "tool",
            [
                Content.from_function_result(
                    call_id="call_1",
                    result="72°F, sunny",
                ),
            ],
        )
        result = AgentEvalConverter.convert_message(msg)
        assert len(result) == 1
        assert result[0] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "72°F, sunny",
        }

    def test_multiple_tool_results(self) -> None:
        msg = Message(
            "tool",
            [
                Content.from_function_result(call_id="call_1", result="r1"),
                Content.from_function_result(call_id="call_2", result="r2"),
            ],
        )
        result = AgentEvalConverter.convert_message(msg)
        assert len(result) == 2
        assert result[0]["tool_call_id"] == "call_1"
        assert result[1]["tool_call_id"] == "call_2"

    def test_non_string_result_serialized(self) -> None:
        msg = Message(
            "tool",
            [
                Content.from_function_result(
                    call_id="call_1",
                    result={"temp": 72, "unit": "F"},
                ),
            ],
        )
        result = AgentEvalConverter.convert_message(msg)
        parsed = json.loads(result[0]["content"])
        assert parsed == {"temp": 72, "unit": "F"}

    def test_empty_message(self) -> None:
        msg = Message("user", [])
        result = AgentEvalConverter.convert_message(msg)
        assert result[0] == {"role": "user", "content": ""}


# ---------------------------------------------------------------------------
# AgentEvalConverter.convert_messages
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_full_conversation(self) -> None:
        messages = [
            Message("user", ["What's the weather?"]),
            Message(
                "assistant",
                [Content.from_function_call(call_id="c1", name="get_weather", arguments='{"loc": "SEA"}')],
            ),
            Message("tool", [Content.from_function_result(call_id="c1", result="Sunny")]),
            Message("assistant", ["It's sunny in Seattle!"]),
        ]
        result = AgentEvalConverter.convert_messages(messages)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "assistant"
        assert result[3]["content"] == "It's sunny in Seattle!"


# ---------------------------------------------------------------------------
# AgentEvalConverter.extract_tools
# ---------------------------------------------------------------------------


class TestExtractTools:
    def test_extracts_function_tools(self) -> None:
        tool = FunctionTool(
            name="get_weather",
            description="Get weather for a location",
            func=lambda location: f"Sunny in {location}",
        )
        agent = MagicMock()
        agent.default_options = {"tools": [tool]}

        result = AgentEvalConverter.extract_tools(agent)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a location"
        assert "parameters" in result[0]

    def test_skips_non_function_tools(self) -> None:
        agent = MagicMock()
        agent.default_options = {"tools": [{"type": "web_search"}, "some_string"]}

        result = AgentEvalConverter.extract_tools(agent)
        assert len(result) == 0

    def test_no_tools(self) -> None:
        agent = MagicMock()
        agent.default_options = {}
        assert AgentEvalConverter.extract_tools(agent) == []

    def test_no_default_options(self) -> None:
        agent = MagicMock(spec=[])  # No attributes
        assert AgentEvalConverter.extract_tools(agent) == []


# ---------------------------------------------------------------------------
# AgentEvalConverter.to_eval_item
# ---------------------------------------------------------------------------


class TestToEvalItem:
    def test_string_query(self) -> None:
        response = AgentResponse(messages=[Message("assistant", ["The weather is sunny."])])
        item = AgentEvalConverter.to_eval_item(query="What's the weather?", response=response)

        assert item["query"] == "What's the weather?"
        assert item["response"] == "The weather is sunny."
        assert "conversation" in item
        conv = json.loads(item["conversation"])
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"

    def test_message_query(self) -> None:
        input_msgs = [
            Message("system", ["Be helpful."]),
            Message("user", ["Hello"]),
        ]
        response = AgentResponse(messages=[Message("assistant", ["Hi there!"])])
        item = AgentEvalConverter.to_eval_item(query=input_msgs, response=response)

        assert item["query"] == "Hello"  # Only user messages
        conv = json.loads(item["conversation"])
        assert len(conv) == 3  # system + user + assistant

    def test_with_context(self) -> None:
        response = AgentResponse(messages=[Message("assistant", ["Answer."])])
        item = AgentEvalConverter.to_eval_item(
            query="Question?",
            response=response,
            context="Some reference document.",
        )
        assert item["context"] == "Some reference document."

    def test_with_explicit_tools(self) -> None:
        tool = FunctionTool(
            name="search",
            description="Search the web",
            func=lambda q: f"Results for {q}",
        )
        response = AgentResponse(messages=[Message("assistant", ["Found it."])])
        item = AgentEvalConverter.to_eval_item(
            query="Find info",
            response=response,
            tools=[tool],
        )
        assert "tool_definitions" in item
        tool_defs = json.loads(item["tool_definitions"])
        assert len(tool_defs) == 1
        assert tool_defs[0]["name"] == "search"

    def test_with_agent_tools(self) -> None:
        tool = FunctionTool(name="calc", description="Calculate", func=lambda x: str(x))
        agent = MagicMock()
        agent.default_options = {"tools": [tool]}

        response = AgentResponse(messages=[Message("assistant", ["42"])])
        item = AgentEvalConverter.to_eval_item(
            query="What is 6*7?",
            response=response,
            agent=agent,
        )
        tool_defs = json.loads(item["tool_definitions"])
        assert tool_defs[0]["name"] == "calc"

    def test_explicit_tools_override_agent(self) -> None:
        agent_tool = FunctionTool(name="agent_tool", description="from agent", func=lambda: "")
        explicit_tool = FunctionTool(name="explicit_tool", description="explicit", func=lambda: "")

        agent = MagicMock()
        agent.default_options = {"tools": [agent_tool]}

        response = AgentResponse(messages=[Message("assistant", ["Done"])])
        item = AgentEvalConverter.to_eval_item(
            query="Test",
            response=response,
            agent=agent,
            tools=[explicit_tool],
        )
        tool_defs = json.loads(item["tool_definitions"])
        assert len(tool_defs) == 1
        assert tool_defs[0]["name"] == "explicit_tool"


# ---------------------------------------------------------------------------
# _build_testing_criteria
# ---------------------------------------------------------------------------


class TestBuildTestingCriteria:
    def test_without_data_mapping(self) -> None:
        criteria = _build_testing_criteria(["relevance", "coherence"], "gpt-4o")
        assert len(criteria) == 2
        assert criteria[0]["evaluator_name"] == "builtin.relevance"
        assert criteria[0]["initialization_parameters"] == {"deployment_name": "gpt-4o"}
        assert "data_mapping" not in criteria[0]

    def test_with_data_mapping(self) -> None:
        criteria = _build_testing_criteria(["relevance", "groundedness"], "gpt-4o", include_data_mapping=True)
        assert "data_mapping" in criteria[0]
        assert criteria[0]["data_mapping"] == {
            "query": "{{item.query}}",
            "response": "{{item.response}}",
        }
        # Groundedness has an extra context mapping
        assert "context" in criteria[1]["data_mapping"]

    def test_tool_evaluator_includes_tool_definitions(self) -> None:
        criteria = _build_testing_criteria(
            ["relevance", "tool_call_accuracy"], "gpt-4o", include_data_mapping=True
        )
        # relevance should NOT have tool_definitions
        assert "tool_definitions" not in criteria[0]["data_mapping"]
        # tool_call_accuracy should have tool_definitions
        assert criteria[1]["data_mapping"]["tool_definitions"] == "{{item.tool_definitions}}"

    def test_all_tool_evaluators_include_tool_definitions(self) -> None:
        tool_evals = [
            "tool_call_accuracy",
            "tool_selection",
            "tool_input_accuracy",
            "tool_output_utilization",
            "tool_call_success",
        ]
        criteria = _build_testing_criteria(tool_evals, "gpt-4o", include_data_mapping=True)
        for c in criteria:
            assert "tool_definitions" in c["data_mapping"], f"{c['name']} missing tool_definitions"


# ---------------------------------------------------------------------------
# _build_item_schema
# ---------------------------------------------------------------------------


class TestBuildItemSchema:
    def test_without_context(self) -> None:
        schema = _build_item_schema(has_context=False)
        assert "context" not in schema["properties"]
        assert schema["required"] == ["query", "response"]

    def test_with_context(self) -> None:
        schema = _build_item_schema(has_context=True)
        assert "context" in schema["properties"]

    def test_with_tools(self) -> None:
        schema = _build_item_schema(has_tools=True)
        assert "tool_definitions" in schema["properties"]

    def test_with_context_and_tools(self) -> None:
        schema = _build_item_schema(has_context=True, has_tools=True)
        assert "context" in schema["properties"]
        assert "tool_definitions" in schema["properties"]


# ---------------------------------------------------------------------------
# evaluate_dataset (integration shape test with mocks)
# ---------------------------------------------------------------------------


class TestEvaluateDataset:
    @pytest.mark.asyncio
    async def test_calls_evals_api(self) -> None:
        mock_client = MagicMock()

        # Mock evals.create
        mock_eval = MagicMock()
        mock_eval.id = "eval_123"
        mock_client.evals.create.return_value = mock_eval

        # Mock evals.runs.create
        mock_run = MagicMock()
        mock_run.id = "run_456"
        mock_client.evals.runs.create.return_value = mock_run

        # Mock evals.runs.retrieve to return completed
        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 2, "failed": 0}
        mock_completed.report_url = "https://portal.azure.com/eval/run_456"
        mock_completed.per_testing_criteria_results = None
        mock_client.evals.runs.retrieve.return_value = mock_completed

        items = [
            {"query": "Hello", "response": "Hi there!"},
            {"query": "Weather?", "response": "Sunny."},
        ]

        results = await evaluate_dataset(
            items=items,
            evaluators=[Evaluators.RELEVANCE],
            openai_client=mock_client,
            model_deployment="gpt-4o",
        )

        assert isinstance(results, EvalResults)
        assert results.status == "completed"
        assert results.eval_id == "eval_123"
        assert results.run_id == "run_456"
        assert results.report_url == "https://portal.azure.com/eval/run_456"
        assert results.all_passed
        assert results.passed == 2
        assert results.failed == 0

        # Verify evals.create was called with correct structure
        create_call = mock_client.evals.create.call_args
        assert create_call.kwargs["name"] == "Agent Framework Eval"
        assert create_call.kwargs["data_source_config"]["type"] == "custom"

        # Verify evals.runs.create was called with JSONL data source
        run_call = mock_client.evals.runs.create.call_args
        assert run_call.kwargs["data_source"]["type"] == "jsonl"
        content = run_call.kwargs["data_source"]["source"]["content"]
        assert len(content) == 2

    @pytest.mark.asyncio
    async def test_uses_default_evaluators(self) -> None:
        mock_client = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_1"
        mock_client.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_1"
        mock_client.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_client.evals.runs.retrieve.return_value = mock_completed

        await evaluate_dataset(
            items=[{"query": "Hi", "response": "Hello"}],
            openai_client=mock_client,
            model_deployment="gpt-4o",
        )

        # Verify default evaluators were used
        create_call = mock_client.evals.create.call_args
        criteria = create_call.kwargs["testing_criteria"]
        names = {c["name"] for c in criteria}
        assert "relevance" in names
        assert "coherence" in names
        assert "task_adherence" in names


# ---------------------------------------------------------------------------
# Evaluators constants
# ---------------------------------------------------------------------------


class TestEvaluators:
    def test_constants_resolve(self) -> None:
        assert _resolve_evaluator(Evaluators.RELEVANCE) == "builtin.relevance"
        assert _resolve_evaluator(Evaluators.TOOL_CALL_ACCURACY) == "builtin.tool_call_accuracy"
        assert _resolve_evaluator(Evaluators.VIOLENCE) == "builtin.violence"
        assert _resolve_evaluator(Evaluators.INTENT_RESOLUTION) == "builtin.intent_resolution"

    def test_all_constants_are_valid(self) -> None:
        for attr in dir(Evaluators):
            if attr.startswith("_"):
                continue
            value = getattr(Evaluators, attr)
            if isinstance(value, str):
                _resolve_evaluator(value)  # should not raise


# ---------------------------------------------------------------------------
# _resolve_default_evaluators
# ---------------------------------------------------------------------------


class TestResolveDefaultEvaluators:
    def test_explicit_evaluators_passthrough(self) -> None:
        result = _resolve_default_evaluators([Evaluators.VIOLENCE])
        assert result == [Evaluators.VIOLENCE]

    def test_none_gives_defaults(self) -> None:
        result = _resolve_default_evaluators(None)
        assert Evaluators.RELEVANCE in result
        assert Evaluators.COHERENCE in result
        assert Evaluators.TASK_ADHERENCE in result
        assert Evaluators.TOOL_CALL_ACCURACY not in result

    def test_none_with_tool_agent_adds_tool_eval(self) -> None:
        tool = FunctionTool(name="search", description="Search", func=lambda q: q)
        agent = MagicMock()
        agent.default_options = {"tools": [tool]}

        result = _resolve_default_evaluators(None, agent=agent)
        assert Evaluators.TOOL_CALL_ACCURACY in result

    def test_explicit_evaluators_ignore_agent_tools(self) -> None:
        tool = FunctionTool(name="search", description="Search", func=lambda q: q)
        agent = MagicMock()
        agent.default_options = {"tools": [tool]}

        result = _resolve_default_evaluators([Evaluators.RELEVANCE], agent=agent)
        assert result == [Evaluators.RELEVANCE]


# ---------------------------------------------------------------------------
# EvalResults
# ---------------------------------------------------------------------------


class TestEvalResults:
    def test_all_passed_true(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed", result_counts={"passed": 3, "failed": 0, "errored": 0})
        assert r.all_passed
        assert r.passed == 3
        assert r.failed == 0
        assert r.errored == 0
        assert r.total == 3

    def test_all_passed_false_on_failure(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed", result_counts={"passed": 2, "failed": 1, "errored": 0})
        assert not r.all_passed
        assert r.failed == 1

    def test_all_passed_false_on_error(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed", result_counts={"passed": 2, "failed": 0, "errored": 1})
        assert not r.all_passed

    def test_all_passed_false_on_non_completed(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="timeout", result_counts={"passed": 2, "failed": 0, "errored": 0})
        assert not r.all_passed

    def test_all_passed_false_on_empty(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed", result_counts={"passed": 0, "failed": 0, "errored": 0})
        assert not r.all_passed

    def test_assert_passed_succeeds(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed", result_counts={"passed": 1, "failed": 0, "errored": 0})
        r.assert_passed()  # should not raise

    def test_assert_passed_raises(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed", result_counts={"passed": 1, "failed": 1, "errored": 0})
        with pytest.raises(AssertionError, match="1 passed, 1 failed"):
            r.assert_passed()

    def test_assert_passed_custom_message(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="failed")
        with pytest.raises(AssertionError, match="custom error"):
            r.assert_passed("custom error")

    def test_none_result_counts(self) -> None:
        r = EvalResults(eval_id="e", run_id="r", status="completed")
        assert r.passed == 0
        assert r.failed == 0
        assert r.total == 0
        assert not r.all_passed


# ---------------------------------------------------------------------------
# _resolve_openai_client
# ---------------------------------------------------------------------------


class TestResolveOpenAIClient:
    def test_explicit_client(self) -> None:
        mock_client = MagicMock()
        assert _resolve_openai_client(openai_client=mock_client) is mock_client

    def test_project_client(self) -> None:
        mock_oai = MagicMock()
        mock_project = MagicMock()
        mock_project.get_openai_client.return_value = mock_oai

        result = _resolve_openai_client(project_client=mock_project)
        assert result is mock_oai
        mock_project.get_openai_client.assert_called_once()

    def test_explicit_takes_precedence(self) -> None:
        mock_client = MagicMock()
        mock_project = MagicMock()

        result = _resolve_openai_client(openai_client=mock_client, project_client=mock_project)
        assert result is mock_client
        mock_project.get_openai_client.assert_not_called()

    def test_neither_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide either"):
            _resolve_openai_client()

    @pytest.mark.asyncio
    async def test_evaluate_dataset_with_project_client(self) -> None:
        mock_oai = MagicMock()
        mock_project = MagicMock()
        mock_project.get_openai_client.return_value = mock_oai

        mock_eval = MagicMock()
        mock_eval.id = "eval_pc"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_pc"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        results = await evaluate_dataset(
            items=[{"query": "Hi", "response": "Hello"}],
            project_client=mock_project,
            model_deployment="gpt-4o",
        )

        assert results.status == "completed"
        mock_project.get_openai_client.assert_called_once()


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    @pytest.mark.asyncio
    async def test_single_response(self) -> None:
        mock_oai = MagicMock()
        mock_project = MagicMock()
        mock_project.get_openai_client.return_value = mock_oai

        mock_eval = MagicMock()
        mock_eval.id = "eval_resp"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_resp"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = "https://portal.azure.com/eval"
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        response = AgentResponse(
            messages=[Message("assistant", ["The weather is sunny."])],
            response_id="resp_abc123",
        )

        results = await evaluate_response(
            response=response,
            project_client=mock_project,
            model_deployment="gpt-4o",
        )

        assert results.status == "completed"
        assert results.all_passed

        # Verify the response ID was passed through
        run_call = mock_oai.evals.runs.create.call_args
        ds = run_call.kwargs["data_source"]
        assert ds["type"] == "azure_ai_responses"
        content = ds["item_generation_params"]["source"]["content"]
        assert content[0]["item"]["resp_id"] == "resp_abc123"

    @pytest.mark.asyncio
    async def test_multiple_responses(self) -> None:
        mock_oai = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_multi"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_multi"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 2, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        responses = [
            AgentResponse(messages=[Message("assistant", ["Answer 1"])], response_id="resp_1"),
            AgentResponse(messages=[Message("assistant", ["Answer 2"])], response_id="resp_2"),
        ]

        results = await evaluate_response(
            response=responses,
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        assert results.passed == 2
        run_call = mock_oai.evals.runs.create.call_args
        content = run_call.kwargs["data_source"]["item_generation_params"]["source"]["content"]
        assert len(content) == 2

    @pytest.mark.asyncio
    async def test_missing_response_id_without_query_raises(self) -> None:
        mock_oai = MagicMock()
        response = AgentResponse(messages=[Message("assistant", ["Hello"])])

        with pytest.raises(ValueError, match="does not have a response_id"):
            await evaluate_response(
                response=response,
                openai_client=mock_oai,
                model_deployment="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_fallback_to_dataset_with_query(self) -> None:
        """Non-Responses-API: falls back to evaluate_dataset path when query is provided."""
        mock_oai = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_fb"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_fb"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = "https://portal.azure.com/eval"
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        response = AgentResponse(messages=[Message("assistant", ["It's sunny."])])

        results = await evaluate_response(
            response=response,
            query="What's the weather?",
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        assert results.status == "completed"
        assert results.all_passed

        # Should use jsonl data source (dataset path), not azure_ai_responses
        run_call = mock_oai.evals.runs.create.call_args
        ds = run_call.kwargs["data_source"]
        assert ds["type"] == "jsonl"
        content = ds["source"]["content"]
        assert len(content) == 1
        assert content[0]["item"]["query"] == "What's the weather?"
        assert content[0]["item"]["response"] == "It's sunny."

    @pytest.mark.asyncio
    async def test_fallback_with_agent_extracts_tools(self) -> None:
        """Non-Responses-API with agent: tool definitions are included in the eval item."""
        mock_oai = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_tools"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_tools"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        mock_agent = MagicMock()
        mock_agent.default_options = {
            "tools": [FunctionTool(name="my_tool", description="A test tool", func=lambda x: x)]
        }

        response = AgentResponse(messages=[Message("assistant", ["Result."])])

        results = await evaluate_response(
            response=response,
            query="Do the thing",
            agent=mock_agent,
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        assert results.status == "completed"

        run_call = mock_oai.evals.runs.create.call_args
        ds = run_call.kwargs["data_source"]
        content = ds["source"]["content"]
        item = content[0]["item"]
        assert "tool_definitions" in item
        import json
        tool_defs = json.loads(item["tool_definitions"])
        assert any(t["name"] == "my_tool" for t in tool_defs)

    @pytest.mark.asyncio
    async def test_fallback_multiple_responses_with_queries(self) -> None:
        """Non-Responses-API with multiple responses requires matching queries."""
        mock_oai = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_multi_fb"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_multi_fb"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 2, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        responses = [
            AgentResponse(messages=[Message("assistant", ["Answer 1"])]),
            AgentResponse(messages=[Message("assistant", ["Answer 2"])]),
        ]

        results = await evaluate_response(
            response=responses,
            query=["Question 1", "Question 2"],
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        assert results.passed == 2
        run_call = mock_oai.evals.runs.create.call_args
        content = run_call.kwargs["data_source"]["source"]["content"]
        assert len(content) == 2
        assert content[0]["item"]["query"] == "Question 1"
        assert content[1]["item"]["query"] == "Question 2"

    @pytest.mark.asyncio
    async def test_query_response_count_mismatch_raises(self) -> None:
        """Mismatched query and response counts should raise."""
        mock_oai = MagicMock()

        responses = [
            AgentResponse(messages=[Message("assistant", ["A1"])]),
            AgentResponse(messages=[Message("assistant", ["A2"])]),
        ]

        with pytest.raises(ValueError, match="does not match"):
            await evaluate_response(
                response=responses,
                query=["Q1", "Q2", "Q3"],
                openai_client=mock_oai,
                model_deployment="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_responses_api_preferred_when_ids_present(self) -> None:
        """When response_id is present and no query given, uses the fast Responses API path."""
        mock_oai = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_fast"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_fast"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        response = AgentResponse(
            messages=[Message("assistant", ["Answer"])],
            response_id="resp_xyz",
        )

        await evaluate_response(
            response=response,
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        # Verify it used the Responses API path (azure_ai_responses), not jsonl
        run_call = mock_oai.evals.runs.create.call_args
        ds = run_call.kwargs["data_source"]
        assert ds["type"] == "azure_ai_responses"

    @pytest.mark.asyncio
    async def test_tool_evaluators_require_query_even_with_response_id(self) -> None:
        """Tool evaluators need tool_definitions not available via response retrieval."""
        response = AgentResponse(
            messages=[Message("assistant", ["Answer"])],
            response_id="resp_xyz",
        )

        with pytest.raises(ValueError, match="Tool evaluators.*require tool definitions"):
            await evaluate_response(
                response=response,
                evaluators=[Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY],
                openai_client=MagicMock(),
                model_deployment="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_tool_evaluators_with_query_and_agent_uses_dataset_path(self) -> None:
        """Tool evaluators with query+agent bypass fast path and use dataset."""
        mock_oai = MagicMock()

        mock_eval = MagicMock()
        mock_eval.id = "eval_tool"
        mock_oai.evals.create.return_value = mock_eval

        mock_run = MagicMock()
        mock_run.id = "run_tool"
        mock_oai.evals.runs.create.return_value = mock_run

        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = None
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed

        response = AgentResponse(
            messages=[Message("assistant", ["It's sunny"])],
            response_id="resp_xyz",  # Has response_id, but tool evals need dataset path
        )

        agent = MagicMock()
        agent.default_options = {"tools": [
            FunctionTool(name="get_weather", description="Get weather", func=lambda: None),
        ]}

        await evaluate_response(
            response=response,
            query="What's the weather?",
            agent=agent,
            evaluators=[Evaluators.TOOL_CALL_ACCURACY],
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        # Verify it used the dataset path (jsonl), not Responses API path
        run_call = mock_oai.evals.runs.create.call_args
        ds = run_call.kwargs["data_source"]
        assert ds["type"] == "jsonl"

        # Verify tool_definitions are in the data items
        items = ds["source"]["content"]
        assert "tool_definitions" in items[0]["item"]


# ---------------------------------------------------------------------------
# EvalResults.sub_results
# ---------------------------------------------------------------------------


class TestEvalResultsSubResults:
    def test_sub_results_default_empty(self) -> None:
        r = EvalResults(eval_id="e1", run_id="r1", status="completed", result_counts={"passed": 1, "failed": 0})
        assert r.sub_results == {}
        assert r.all_passed

    def test_all_passed_checks_sub_results(self) -> None:
        parent = EvalResults(
            eval_id="e1", run_id="r1", status="completed",
            result_counts={"passed": 2, "failed": 0},
            sub_results={
                "agent-a": EvalResults(
                    eval_id="e2", run_id="r2", status="completed",
                    result_counts={"passed": 1, "failed": 0},
                ),
                "agent-b": EvalResults(
                    eval_id="e3", run_id="r3", status="completed",
                    result_counts={"passed": 1, "failed": 1},
                ),
            },
        )
        assert not parent.all_passed  # agent-b has a failure

    def test_all_passed_with_all_sub_passing(self) -> None:
        parent = EvalResults(
            eval_id="e1", run_id="r1", status="completed",
            result_counts={"passed": 2, "failed": 0},
            sub_results={
                "agent-a": EvalResults(
                    eval_id="e2", run_id="r2", status="completed",
                    result_counts={"passed": 1, "failed": 0},
                ),
            },
        )
        assert parent.all_passed

    def test_assert_passed_includes_failed_agents(self) -> None:
        parent = EvalResults(
            eval_id="e1", run_id="r1", status="completed",
            result_counts={"passed": 2, "failed": 0},
            sub_results={
                "good-agent": EvalResults(
                    eval_id="e2", run_id="r2", status="completed",
                    result_counts={"passed": 1, "failed": 0},
                ),
                "bad-agent": EvalResults(
                    eval_id="e3", run_id="r3", status="completed",
                    result_counts={"passed": 0, "failed": 1},
                ),
            },
        )
        with pytest.raises(AssertionError, match="bad-agent"):
            parent.assert_passed()


# ---------------------------------------------------------------------------
# _extract_agent_eval_data
# ---------------------------------------------------------------------------


def _make_agent_exec_response(
    executor_id: str,
    response_text: str,
    user_messages: list[str] | None = None,
) -> AgentExecutorResponse:
    """Helper to build an AgentExecutorResponse for testing."""
    agent_response = AgentResponse(messages=[Message("assistant", [response_text])])
    full_conv: list[Message] = []
    if user_messages:
        for m in user_messages:
            full_conv.append(Message("user", [m]))
    full_conv.extend(agent_response.messages)
    return AgentExecutorResponse(
        executor_id=executor_id,
        agent_response=agent_response,
        full_conversation=full_conv,
    )


class TestExtractAgentEvalData:
    def test_extracts_single_agent(self) -> None:
        aer = _make_agent_exec_response("planner", "Plan is ready", ["Plan a trip"])

        events = [
            WorkflowEvent.executor_invoked("planner", "Plan a trip"),
            WorkflowEvent.executor_completed("planner", [aer]),
        ]
        result = WorkflowRunResult(events, [])

        data = _extract_agent_eval_data(result)
        assert len(data) == 1
        assert data[0].executor_id == "planner"
        assert data[0].response.text == "Plan is ready"

    def test_extracts_multiple_agents(self) -> None:
        aer1 = _make_agent_exec_response("planner", "Plan done", ["Plan a trip"])
        aer2 = _make_agent_exec_response("booker", "Booked!", ["Book flight"])

        events = [
            WorkflowEvent.executor_invoked("planner", "Plan a trip"),
            WorkflowEvent.executor_completed("planner", [aer1]),
            WorkflowEvent.executor_invoked("booker", "Book flight"),
            WorkflowEvent.executor_completed("booker", [aer2]),
        ]
        result = WorkflowRunResult(events, [])

        data = _extract_agent_eval_data(result)
        assert len(data) == 2
        assert data[0].executor_id == "planner"
        assert data[1].executor_id == "booker"

    def test_skips_internal_executors(self) -> None:
        aer = _make_agent_exec_response("planner", "Done", ["Go"])

        events = [
            WorkflowEvent.executor_invoked("input-conversation", "hello"),
            WorkflowEvent.executor_completed("input-conversation", ["hello"]),
            WorkflowEvent.executor_invoked("planner", "Go"),
            WorkflowEvent.executor_completed("planner", [aer]),
            WorkflowEvent.executor_invoked("end", []),
            WorkflowEvent.executor_completed("end", None),
        ]
        result = WorkflowRunResult(events, [])

        data = _extract_agent_eval_data(result)
        assert len(data) == 1
        assert data[0].executor_id == "planner"

    def test_resolves_agent_from_workflow(self) -> None:
        aer = _make_agent_exec_response("my-agent", "Done", ["Do it"])

        events = [
            WorkflowEvent.executor_invoked("my-agent", "Do it"),
            WorkflowEvent.executor_completed("my-agent", [aer]),
        ]
        result = WorkflowRunResult(events, [])

        # Build a mock workflow with AgentExecutor
        from agent_framework import AgentExecutor
        mock_agent = MagicMock()
        mock_agent.default_options = {"tools": []}
        mock_executor = MagicMock(spec=AgentExecutor)
        mock_executor.agent = mock_agent

        mock_workflow = MagicMock()
        mock_workflow.executors = {"my-agent": mock_executor}

        data = _extract_agent_eval_data(result, mock_workflow)
        assert len(data) == 1
        assert data[0].agent is mock_agent


class TestExtractOverallQuery:
    def test_extracts_string_query(self) -> None:
        events = [WorkflowEvent.executor_invoked("input", "Plan a trip")]
        result = WorkflowRunResult(events, [])
        assert _extract_overall_query(result) == "Plan a trip"

    def test_extracts_message_query(self) -> None:
        msgs = [Message("user", ["What's the weather?"])]
        events = [WorkflowEvent.executor_invoked("input", msgs)]
        result = WorkflowRunResult(events, [])
        assert "What's the weather?" in (_extract_overall_query(result) or "")

    def test_returns_none_for_empty(self) -> None:
        result = WorkflowRunResult([], [])
        assert _extract_overall_query(result) is None


# ---------------------------------------------------------------------------
# evaluate_workflow
# ---------------------------------------------------------------------------


class TestEvaluateWorkflow:
    def _mock_oai_client(self, eval_id: str = "eval_wf", run_id: str = "run_wf") -> MagicMock:
        mock_oai = MagicMock()
        mock_eval = MagicMock()
        mock_eval.id = eval_id
        mock_oai.evals.create.return_value = mock_eval
        mock_run = MagicMock()
        mock_run.id = run_id
        mock_oai.evals.runs.create.return_value = mock_run
        mock_completed = MagicMock()
        mock_completed.status = "completed"
        mock_completed.result_counts = {"passed": 1, "failed": 0}
        mock_completed.report_url = "https://portal.azure.com/eval"
        mock_completed.per_testing_criteria_results = None
        mock_oai.evals.runs.retrieve.return_value = mock_completed
        return mock_oai

    @pytest.mark.asyncio
    async def test_post_hoc_with_workflow_result(self) -> None:
        """Evaluate a workflow result that was already produced."""
        mock_oai = self._mock_oai_client()

        aer1 = _make_agent_exec_response("writer", "Draft written", ["Write about Paris"])
        aer2 = _make_agent_exec_response("reviewer", "Looks good!", ["Review: Draft written"])

        final_output = [Message("assistant", ["Final reviewed output"])]

        events = [
            WorkflowEvent.executor_invoked("input-conversation", "Write about Paris"),
            WorkflowEvent.executor_completed("input-conversation", None),
            WorkflowEvent.executor_invoked("writer", "Write about Paris"),
            WorkflowEvent.executor_completed("writer", [aer1]),
            WorkflowEvent.executor_invoked("reviewer", [aer1]),
            WorkflowEvent.executor_completed("reviewer", [aer2]),
            WorkflowEvent.output("end", final_output),
        ]
        wf_result = WorkflowRunResult(events, [])

        mock_workflow = MagicMock()
        mock_workflow.executors = {}

        results = await evaluate_workflow(
            workflow=mock_workflow,
            workflow_result=wf_result,
            openai_client=mock_oai,
            model_deployment="gpt-4o",
            include_overall=False,
        )

        assert results.status == "completed"
        assert "writer" in results.sub_results
        assert "reviewer" in results.sub_results
        assert len(results.sub_results) == 2

    @pytest.mark.asyncio
    async def test_with_queries_runs_workflow(self) -> None:
        """Passing queries= runs the workflow and evaluates."""
        mock_oai = self._mock_oai_client()

        aer = _make_agent_exec_response("agent", "Response", ["Query"])
        final_output = [Message("assistant", ["Final"])]

        events = [
            WorkflowEvent.executor_invoked("agent", "Test query"),
            WorkflowEvent.executor_completed("agent", [aer]),
            WorkflowEvent.output("end", final_output),
        ]
        wf_result = WorkflowRunResult(events, [])

        mock_workflow = MagicMock()
        mock_workflow.executors = {}
        mock_workflow.run = AsyncMock(return_value=wf_result)

        results = await evaluate_workflow(
            workflow=mock_workflow,
            queries=["Test query"],
            openai_client=mock_oai,
            model_deployment="gpt-4o",
            include_overall=False,
        )

        mock_workflow.run.assert_called_once_with("Test query")
        assert "agent" in results.sub_results

    @pytest.mark.asyncio
    async def test_overall_plus_per_agent(self) -> None:
        """Both overall and per-agent evals run by default."""
        mock_oai = self._mock_oai_client()

        aer = _make_agent_exec_response("planner", "Plan done", ["Plan trip"])
        final_output = [Message("assistant", ["Trip planned!"])]

        events = [
            WorkflowEvent.executor_invoked("input-conversation", "Plan trip"),
            WorkflowEvent.executor_completed("input-conversation", None),
            WorkflowEvent.executor_invoked("planner", "Plan trip"),
            WorkflowEvent.executor_completed("planner", [aer]),
            WorkflowEvent.output("end", final_output),
        ]
        wf_result = WorkflowRunResult(events, [])

        mock_workflow = MagicMock()
        mock_workflow.executors = {}

        results = await evaluate_workflow(
            workflow=mock_workflow,
            workflow_result=wf_result,
            openai_client=mock_oai,
            model_deployment="gpt-4o",
        )

        # Should have per-agent sub_results AND overall
        assert "planner" in results.sub_results
        assert results.status == "completed"
        # evaluate_dataset called twice: once for planner, once for overall
        assert mock_oai.evals.create.call_count == 2

    @pytest.mark.asyncio
    async def test_no_result_or_queries_raises(self) -> None:
        mock_oai = MagicMock()
        mock_workflow = MagicMock()

        with pytest.raises(ValueError, match="Provide either"):
            await evaluate_workflow(
                workflow=mock_workflow,
                openai_client=mock_oai,
                model_deployment="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_per_agent_only(self) -> None:
        """include_overall=False skips the overall eval."""
        mock_oai = self._mock_oai_client()

        aer = _make_agent_exec_response("agent-a", "Done", ["Do stuff"])

        events = [
            WorkflowEvent.executor_invoked("agent-a", "Do stuff"),
            WorkflowEvent.executor_completed("agent-a", [aer]),
        ]
        wf_result = WorkflowRunResult(events, [])

        mock_workflow = MagicMock()
        mock_workflow.executors = {}

        results = await evaluate_workflow(
            workflow=mock_workflow,
            workflow_result=wf_result,
            openai_client=mock_oai,
            model_deployment="gpt-4o",
            include_overall=False,
        )

        assert "agent-a" in results.sub_results
        # Only one eval call (per-agent), no overall
        assert mock_oai.evals.create.call_count == 1
