# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import Agent, AgentEvalConverter, Message, evaluate_agent, evaluate_response
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_azure_ai import Evaluators, FoundryEvals
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

"""
This sample demonstrates evaluating an agent using Azure AI Foundry's built-in evaluators.

It shows three patterns:
1. evaluate_response() — Evaluate a response you already have.
2. evaluate_agent() — Run the agent against test queries and evaluate in one call.
3. FoundryEvals.evaluate() — Full control with direct evaluator access.

Prerequisites:
- An Azure AI Foundry project with a deployed model
- Set AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME in .env

Required components:
- An Agent with tools (the agent to evaluate)
- A FoundryEvals instance (the evaluator)
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

    # 3. Create the evaluator — provider config goes here, once
    evals = FoundryEvals(project_client=project_client, model_deployment=deployment)

    # =========================================================================
    # Pattern 1a: evaluate_response() — quality evaluators
    # =========================================================================
    print("=" * 60)
    print("Pattern 1a: evaluate_response() — quality evaluators")
    print("=" * 60)

    response = await agent.run([Message("user", ["What's the weather like in Seattle?"])])
    print(f"Agent said: {response.text[:100]}...")

    # Quality evaluators: just pass the response — Foundry can retrieve the conversation
    results = await evaluate_response(
        response=response,
        evaluators=evals.select(Evaluators.RELEVANCE, Evaluators.COHERENCE),
    )

    print(f"Status: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")
    if results.all_passed:
        print("✓ All passed")
    else:
        print(f"✗ {results.failed} failed, {results.errored} errored")

    print()
    print("=" * 60)
    print("Pattern 1b: evaluate_response() — with tool evaluators")
    print("=" * 60)

    # Tool evaluators need tool definitions — provide query= and agent=
    query = "How much does a flight from Seattle to Paris cost?"
    response = await agent.run([Message("user", [query])])
    print(f"Agent said: {response.text[:100]}...")

    results = await evaluate_response(
        response=response,
        query=query,
        agent=agent,
        evaluators=evals.select(Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY),
    )

    print(f"Status: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")
    if results.all_passed:
        print("✓ All passed")
    else:
        print(f"✗ {results.failed} failed, {results.errored} errored")

    # =========================================================================
    # Pattern 2: evaluate_agent() — batch test queries
    # =========================================================================
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
        evaluators=evals,  # uses smart defaults (auto-adds tool_call_accuracy)
    )

    print(f"Status: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")
    if results.all_passed:
        print("✓ All passed")
    else:
        print(f"✗ {results.failed} failed, {results.errored} errored")

    # =========================================================================
    # Pattern 3: FoundryEvals.evaluate() — manual control
    # =========================================================================
    print()
    print("=" * 60)
    print("Pattern 3: FoundryEvals.evaluate() — manual control")
    print("=" * 60)

    converter = AgentEvalConverter()
    queries = [
        "What's the weather in Paris?",
        "Find me a flight from London to Seattle",
    ]

    items = []
    for q in queries:
        response = await agent.run([Message("user", [q])])
        print(f"Query: {q}")
        print(f"Response: {response.text[:100]}...")

        item = converter.to_eval_item(query=q, response=response, agent=agent)
        items.append(item)

        print(f"  Has tools: {item.tool_definitions is not None}")
        if item.tool_definitions:
            print(f"  Tools: {[t['name'] for t in item.tool_definitions]}")

    # Submit directly to the evaluator
    tool_evals = evals.select(Evaluators.RELEVANCE, Evaluators.TOOL_CALL_ACCURACY)
    results = await tool_evals.evaluate(items, eval_name="Travel Assistant Eval")

    print(f"\nStatus: {results.status}")
    print(f"Results: {results.passed}/{results.total} passed")
    print(f"Portal: {results.report_url}")


if __name__ == "__main__":
    asyncio.run(main())
