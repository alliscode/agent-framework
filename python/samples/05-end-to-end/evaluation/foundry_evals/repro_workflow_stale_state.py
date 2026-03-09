# Copyright (c) Microsoft. All rights reserved.

"""Repro: Running a sequential workflow twice fails with stale state.

Bug: AgentExecutor holds a persistent _session and _cache across
workflow runs. On the second workflow.run(), the Responses API's
previous_response_id from Run 1 leaks into Run 2, sending stale
tool call IDs that have no corresponding tool output in the new
conversation. The API rejects with:

    'No tool output found for function call call_XXXX'

This is the bug that blocks evaluate_workflow(queries=[q1, q2, ...])
since it calls workflow.run() in a loop on the same workflow instance.

To run:
    cd python
    uv run python samples/05-end-to-end/evaluation/foundry_evals/repro_workflow_stale_state.py
"""

import asyncio
import os

from agent_framework import Agent
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_orchestrations import SequentialBuilder
from azure.ai.projects.aio import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"72F and sunny in {location}"


async def main():
    project_client = AIProjectClient(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")

    client = AzureOpenAIResponsesClient(
        project_client=project_client,
        deployment_name=deployment,
    )

    researcher = Agent(
        client=client,
        name="researcher",
        instructions="You are a researcher. Always use your tools to answer.",
        tools=[get_weather]
    )

    summarizer = Agent(
        client=client,
        name="summarizer",
        instructions="Summarize the research in one sentence."
    )

    workflow = SequentialBuilder(participants=[researcher, summarizer]).build()

    # --- Run 1: should succeed ---
    print("Run 1...")
    result1 = await workflow.run("What is the weather in Seattle?")
    outputs1 = result1.get_outputs()
    print(f"  OK: got {len(outputs1)} output(s)")

    # --- Run 2: fails with stale state ---
    # AgentExecutor._session still carries previous_response_id from Run 1.
    # AgentExecutor._cache still has messages with tool call IDs from Run 1.
    # The Responses API rejects them because no matching tool output exists
    # in the new conversation.
    print("\nRun 2 (same workflow instance)...")
    try:
        result2 = await workflow.run("What is the weather in London?")
        outputs2 = result2.get_outputs()
        print(f"  OK: got {len(outputs2)} output(s)")
        print("\n  Bug may be fixed! Both runs succeeded.")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("\n  This is the stale state bug. The AgentExecutor's _session")
        print("  and _cache persist across workflow.run() calls, leaking")
        print("  previous_response_id and old tool call IDs into Run 2.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
