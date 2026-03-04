# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import Agent, Message
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_azure_ai import Evaluators, evaluate_workflow
from agent_framework_orchestrations import SequentialBuilder
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

"""
This sample demonstrates evaluating a multi-agent workflow using Azure AI Foundry evaluators.

It shows two patterns:
1. Post-hoc: Run the workflow, then evaluate the result you already have.
2. Run + evaluate: Pass queries and let evaluate_workflow() run the workflow for you.

Both patterns provide per-agent breakdown in results.sub_results, so you can
identify which agent in the pipeline is underperforming.

Prerequisites:
- An Azure AI Foundry project with a deployed model
- Set AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME in .env
"""


# Simple tools for the agents
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

    client = AzureOpenAIResponsesClient(
        project_client=project_client,
        deployment_name=deployment,
    )

    # 2. Create agents for a sequential workflow
    researcher = Agent(
        client=client,
        name="researcher",
        instructions=(
            "You are a travel researcher. Use your tools to gather weather "
            "and flight information for the destination the user asks about."
        ),
        tools=[get_weather, get_flight_price],
    )

    planner = Agent(
        client=client,
        name="planner",
        instructions=(
            "You are a travel planner. Based on the research provided, "
            "create a concise travel recommendation with packing tips."
        ),
    )

    # 3. Build a sequential workflow: researcher → planner
    workflow = SequentialBuilder(participants=[researcher, planner]).build()

    # =========================================================================
    # Pattern 1: Post-hoc — evaluate a workflow run you already did
    # =========================================================================
    print("=" * 60)
    print("Pattern 1: Post-hoc workflow evaluation")
    print("=" * 60)

    result = await workflow.run("Plan a trip from Seattle to Paris")

    eval_results = await evaluate_workflow(
        workflow=workflow,
        workflow_result=result,
        evaluators=[Evaluators.RELEVANCE, Evaluators.COHERENCE, Evaluators.TOOL_CALL_ACCURACY],
        project_client=project_client,
        model_deployment=deployment,
    )

    print(f"\nOverall: {eval_results.status}")
    print(f"  Passed: {eval_results.passed}/{eval_results.total}")
    print(f"  Portal: {eval_results.report_url}")

    print("\nPer-agent breakdown:")
    for agent_name, agent_eval in eval_results.sub_results.items():
        print(f"  {agent_name}: {agent_eval.passed}/{agent_eval.total} passed")

    return

    # =========================================================================
    # Pattern 2: Run + evaluate — pass queries, workflow runs automatically
    # =========================================================================
    print()
    print("=" * 60)
    print("Pattern 2: Run + evaluate with multiple queries")
    print("=" * 60)

    eval_results = await evaluate_workflow(
        workflow=workflow,
        queries=[
            "Plan a trip from Seattle to Paris",
            "Plan a trip from London to Tokyo",
        ],
        evaluators=[Evaluators.RELEVANCE, Evaluators.TASK_ADHERENCE],
        project_client=project_client,
        model_deployment=deployment,
    )

    print(f"\nOverall: {eval_results.status}")
    print(f"  Passed: {eval_results.passed}/{eval_results.total}")

    print("\nPer-agent breakdown:")
    for agent_name, agent_eval in eval_results.sub_results.items():
        print(f"  {agent_name}: {agent_eval.passed}/{agent_eval.total} passed")

    eval_results.assert_passed()


if __name__ == "__main__":
    asyncio.run(main())


"""
Sample output (with actual Azure AI Foundry project):

============================================================
Pattern 1: Post-hoc workflow evaluation
============================================================

Overall: completed
  Passed: 2/2
  Portal: https://ai.azure.com/...

Per-agent breakdown:
  researcher: 1/1 passed
  planner: 1/1 passed

============================================================
Pattern 2: Run + evaluate with multiple queries
============================================================

Overall: completed
  Passed: 4/4

Per-agent breakdown:
  researcher: 2/2 passed
  planner: 2/2 passed
"""
