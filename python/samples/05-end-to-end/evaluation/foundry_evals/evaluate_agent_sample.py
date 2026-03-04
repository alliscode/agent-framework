# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os

from agent_framework import Agent, Message
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_azure_ai import AgentEvalConverter, Evaluators, evaluate_agent, evaluate_dataset, evaluate_response
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

"""
This sample demonstrates evaluating an agent using Azure AI Foundry's built-in evaluators.

It shows three patterns:
1. evaluate_response() — Evaluate a response you already have. Works with any client.
2. evaluate_agent() — Run the agent against test queries and evaluate in one call.
3. evaluate_dataset() — Full control. Run the agent yourself, convert, then evaluate.

Prerequisites:
- An Azure AI Foundry project with a deployed model
- Set AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME in .env

Required components:
- An Agent with tools (the agent to evaluate)
- An AIProjectClient (shared between agent and evals)
"""


# Define a simple tool for the agent
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "seattle": "62°F, cloudy with a chance of rain",
        "london": "55°F, overcast",
        "paris": "68°F, partly sunny",
    }
    return weather_data.get(location.lower(), f"Weather data not available for {location}")


def get_flight_price(origin: str, destination: str) -> str:
    """Get the price of a flight between two cities."""
    return f"Flights from {origin} to {destination}: $450 round-trip"


async def main():
    # 1. Set up the Azure AI project client
    project_client = AIProjectClient(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )

    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")

    # 2. Create an agent with tools
    agent = Agent(
        client=AzureOpenAIResponsesClient(
            project_client=project_client,
            deployment_name=deployment,
        ),
        name="travel-assistant",
        instructions=(
            "You are a helpful travel assistant. "
            "Use your tools to answer questions about weather and flights."
        ),
        tools=[get_weather, get_flight_price],
    )

    # =========================================================================
    # Pattern 1a: evaluate_response() — Responses API (no tool evaluators)
    # =========================================================================
    # For quality evaluators (relevance, coherence, etc.), Responses API
    # responses need only the response — Foundry retrieves the conversation.
    print("=" * 60)
    print("Pattern 1a: evaluate_response() — Responses API")
    print("=" * 60)

    response = await agent.run([Message("user", ["What's the weather like in Seattle?"])])
    print(f"Agent said: {response.text[:100]}...")

    # Responses API: just pass the response — Foundry has the full conversation
    results = await evaluate_response(
        response=response,
        evaluators=[Evaluators.RELEVANCE, Evaluators.COHERENCE],
        project_client=project_client,
        model_deployment=deployment,
    )

    print(f"Status: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")
    results.assert_passed()

    print()
    print("=" * 60)
    print("Pattern 1b: evaluate_response() — With tool evaluators")
    print("=" * 60)

    # Tool evaluators (tool_call_accuracy, etc.) need tool definitions.
    # Provide query= and agent= so the full conversation and tool schemas
    # are included in the evaluation data.
    query = "How much does a flight from Seattle to Paris cost?"
    response = await agent.run([Message("user", [query])])
    print(f"Agent said: {response.text[:100]}...")

    results = await evaluate_response(
        response=response,
        query=query,
        agent=agent,
        evaluators=[Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY],
        project_client=project_client,
        model_deployment=deployment,
    )

    print(f"Status: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")
    results.assert_passed()

    # =========================================================================
    # Pattern 2: evaluate_agent() — Batch test queries
    # =========================================================================
    # Runs the agent against each query, converts the output, and evaluates
    # in a single call. Best for testing against a suite of queries.
    print()
    print("=" * 60)
    print("Pattern 2: evaluate_agent()")
    print("=" * 60)

    results = await evaluate_agent(
        agent=agent,
        queries=[
            "What's the weather like in Seattle?",
            "How much does a flight from Seattle to Paris cost?",
            "What should I pack for London?",
        ],
        evaluators=[Evaluators.RELEVANCE, Evaluators.COHERENCE, Evaluators.TOOL_CALL_ACCURACY],
        project_client=project_client,
        model_deployment=deployment,
    )

    print(f"Status: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")
    results.assert_passed()  # raises AssertionError if any failures

    # =========================================================================
    # Pattern 3: evaluate_dataset() — Manual control
    # =========================================================================
    # Run the agent yourself, convert with AgentEvalConverter, then evaluate.
    # Useful when you want to inspect or modify the data before evaluation,
    # or when you're collecting data from production runs.
    print()
    print("=" * 60)
    print("Pattern 3: evaluate_dataset()")
    print("=" * 60)

    converter = AgentEvalConverter()
    queries = [
        "What's the weather in Paris?",
        "Find me a flight from London to Seattle",
    ]

    items = []
    for query in queries:
        # Run the agent
        response = await agent.run([Message("user", [query])])
        print(f"Query: {query}")
        print(f"Response: {response.text[:100]}...")

        # Convert to eval format — tools are extracted from agent automatically
        item = converter.to_eval_item(query=query, response=response, agent=agent)
        items.append(item)

        # You can inspect the converted data before submitting
        print(f"  Eval item keys: {list(item.keys())}")
        if "tool_definitions" in item:
            tool_defs = json.loads(item["tool_definitions"])
            print(f"  Tools auto-extracted: {[t['name'] for t in tool_defs]}")

    # Submit all items for evaluation at once
    results = await evaluate_dataset(
        items=items,
        evaluators=[Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY, Evaluators.TOOL_OUTPUT_UTILIZATION],
        project_client=project_client,
        model_deployment=deployment,
        eval_name="Travel Assistant Dataset Eval",
    )

    print(f"\nStatus: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")


if __name__ == "__main__":
    asyncio.run(main())


"""
Sample output (with actual Azure AI Foundry project):

============================================================
Pattern 1a: evaluate_response() — Responses API
============================================================
Agent said: The weather in Seattle is currently 62°F, cloudy with a chance of rain...
Status: completed
Results: 1/1 passed
Portal: https://ai.azure.com/...

============================================================
Pattern 1b: evaluate_response() — Any client
============================================================
Agent said: Flights from Seattle to Paris cost $450 round-trip...
Status: completed
Results: 1/1 passed
Portal: https://ai.azure.com/...

============================================================
Pattern 2: evaluate_agent()
============================================================
Status: completed
Results: 3/3 passed
Portal: https://ai.azure.com/...

============================================================
Pattern 3: evaluate_dataset()
============================================================
Query: What's the weather in Paris?
Response: The weather in Paris is currently 68°F, partly sunny...
  Eval item keys: ['query', 'response', 'conversation', 'tool_definitions']
  Tools auto-extracted: ['get_weather', 'get_flight_price']
Query: Find me a flight from London to Seattle
Response: Flights from London to Seattle cost $450 round-trip...
  Eval item keys: ['query', 'response', 'conversation', 'tool_definitions']
  Tools auto-extracted: ['get_weather', 'get_flight_price']

Status: completed
Results: 2/2 passed
Portal: https://ai.azure.com/...
"""
