# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Agent Harness Basics

What it does:
- Demonstrates the Agent Harness, a workflow-based runtime for durable agent execution
- Shows turn-based execution with configurable limits (max_turns)
- Demonstrates transcript tracking and harness events
- Shows both simple (AgentHarness) and advanced (HarnessWorkflowBuilder) APIs

The Agent Harness provides:
- Turn-based execution control with max_turns limit
- Transcript tracking for observability (events for each turn)
- Repair of execution invariants (dangling tool calls)
- Layered stop conditions (max turns, agent done, errors)
- Checkpointing support for durability

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run `az login` before executing.
"""

import asyncio

from agent_framework import WorkflowOutputEvent, ai_function
from agent_framework._harness import (
    AgentHarness,
    HarnessResult,
    HarnessWorkflowBuilder,
    RepairTrigger,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


# Define tools for the agent to use
@ai_function
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        Weather information for the specified city.
    """
    # Mock weather data
    weather_data = {
        "seattle": "Cloudy, 55°F, light rain expected",
        "new york": "Sunny, 72°F, clear skies",
        "london": "Overcast, 60°F, chance of showers",
        "tokyo": "Partly cloudy, 68°F, humid",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@ai_function
def get_local_time(city: str) -> str:
    """Get the current local time for a city.

    Args:
        city: The name of the city to get time for.

    Returns:
        Current local time for the specified city.
    """
    # Mock time data
    time_data = {
        "seattle": "3:00 PM PST",
        "new york": "6:00 PM EST",
        "london": "11:00 PM GMT",
        "tokyo": "8:00 AM JST (next day)",
    }
    return time_data.get(city.lower(), f"Time data not available for {city}")


async def demo_simple_harness() -> None:
    """Demonstrate the simple AgentHarness API."""
    print("=" * 60)
    print("Demo 1: Simple AgentHarness API")
    print("=" * 60)

    # Create a chat client and agent with tools
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions=(
            "You are a helpful travel assistant. When asked about a destination, "
            "provide weather and time information using the available tools. "
            "Be concise in your responses."
        ),
        name="travel_assistant",
        tools=[get_weather, get_local_time],
    )

    # Create harness with max_turns=5
    harness = AgentHarness(agent, max_turns=5)

    # Run the harness
    print("\nRunning harness with query: 'What's the weather and time in Seattle?'")
    print("-" * 40)

    async for event in harness.run_stream("What's the weather and time in Seattle?"):
        if isinstance(event, WorkflowOutputEvent):
            result = event.data
            if isinstance(result, HarnessResult):
                print(f"\nHarness completed!")
                print(f"  Status: {result.status.value}")
                print(f"  Turns: {result.turn_count}")
                print(f"  Stop reason: {result.reason.kind if result.reason else 'N/A'}")
                print(f"  Transcript events: {len(result.transcript)}")


async def demo_builder_with_max_turns() -> None:
    """Demonstrate max_turns stop condition using HarnessWorkflowBuilder."""
    print("\n" + "=" * 60)
    print("Demo 2: HarnessWorkflowBuilder with max_turns limit")
    print("=" * 60)

    # Create a chat client and agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions=(
            "You are a creative writer. When asked to write, produce content "
            "and always ask if the user wants more. Keep iterating until stopped."
        ),
        name="creative_writer",
    )

    # Build harness with max_turns=3 to demonstrate turn limiting
    builder = HarnessWorkflowBuilder(agent, max_turns=3)
    workflow = builder.build()

    print("\nRunning harness with max_turns=3")
    print("Query: 'Write a haiku about coding'")
    print("-" * 40)

    result = await workflow.run(
        RepairTrigger(),
        **builder.get_harness_kwargs(),
    )

    outputs = result.get_outputs()
    if outputs:
        harness_result = outputs[0]
        if isinstance(harness_result, HarnessResult):
            print(f"\nHarness completed!")
            print(f"  Status: {harness_result.status.value}")
            print(f"  Turns executed: {harness_result.turn_count}")
            print(f"  Stop reason: {harness_result.reason.kind if harness_result.reason else 'N/A'}")

            # Print transcript summary
            print(f"\n  Transcript ({len(harness_result.transcript)} events):")
            for event in harness_result.transcript:
                print(f"    - {event.event_type}: turn {event.data.get('turn_number', 'N/A')}")


async def demo_transcript_tracking() -> None:
    """Demonstrate transcript tracking for observability."""
    print("\n" + "=" * 60)
    print("Demo 3: Transcript Tracking for Observability")
    print("=" * 60)

    # Create a chat client and agent with tools
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions=(
            "You are a helpful assistant. Answer questions concisely. "
            "Use tools when needed to provide accurate information."
        ),
        name="helpful_assistant",
        tools=[get_weather],
    )

    # Create harness
    builder = HarnessWorkflowBuilder(agent, max_turns=5)
    workflow = builder.build()

    print("\nRunning harness and tracking transcript events...")
    print("Query: 'What is the weather like in London and New York?'")
    print("-" * 40)

    result = await workflow.run(
        RepairTrigger(),
        **builder.get_harness_kwargs(),
    )

    outputs = result.get_outputs()
    if outputs:
        harness_result = outputs[0]
        if isinstance(harness_result, HarnessResult):
            print(f"\n=== Execution Summary ===")
            print(f"Status: {harness_result.status.value}")
            print(f"Total turns: {harness_result.turn_count}")

            print(f"\n=== Full Transcript ===")
            for i, event in enumerate(harness_result.transcript):
                print(f"\nEvent {i + 1}:")
                print(f"  Type: {event.event_type}")
                print(f"  Timestamp: {event.timestamp}")
                for key, value in event.data.items():
                    # Truncate long values
                    str_value = str(value)
                    if len(str_value) > 80:
                        str_value = str_value[:80] + "..."
                    print(f"  {key}: {str_value}")


async def main() -> None:
    """Run all harness demos."""
    print("Agent Harness Demonstration")
    print("=" * 60)
    print("The Agent Harness provides workflow-based infrastructure for")
    print("durable, observable, and controllable agent execution.")
    print()

    # Run demos
    await demo_simple_harness()
    await demo_builder_with_max_turns()
    await demo_transcript_tracking()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
