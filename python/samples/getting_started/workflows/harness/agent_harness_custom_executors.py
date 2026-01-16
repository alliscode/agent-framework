# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Agent Harness with Custom Executors

What it does:
- Shows how to use individual harness executors directly
- Demonstrates building a custom harness workflow
- Shows how to add additional executors to the harness pattern

This sample demonstrates the composability of the harness architecture.
The harness is built on standard workflow primitives, allowing you to:
- Add custom executors to the harness loop
- Modify the flow between executors
- Create specialized harness variants for different use cases

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run `az login` before executing.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework._harness import (
    HarnessResult,
    RepairComplete,
    RepairExecutor,
    RepairTrigger,
    StopDecisionExecutor,
    TurnComplete,
)
from agent_framework._harness._constants import (
    HARNESS_TRANSCRIPT_KEY,
    HARNESS_TURN_COUNT_KEY,
)
from agent_framework._harness._repair_executor import HARNESS_CONFIG_KEY
from agent_framework._harness._state import HarnessEvent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


@dataclass
class TurnSummary:
    """Summary of a completed turn for logging."""

    turn_number: int
    message: str


class LoggingExecutor(Executor):
    """Custom executor that logs turn progress.

    This demonstrates how to add custom executors into the harness loop
    to add observability, validation, or other cross-cutting concerns.
    """

    @handler
    async def log_turn_start(self, repair_complete: RepairComplete, ctx: WorkflowContext[RepairComplete]) -> None:
        """Log when a turn starts and pass through to the agent."""
        turn_count = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
        turn_count = (turn_count or 0) + 1

        print(f"  [Logger] Starting turn {turn_count} (repairs made: {repair_complete.repairs_made})")

        # Pass through to next executor
        await ctx.send_message(repair_complete)


class MockAgentExecutor(Executor):
    """A mock agent executor that simulates agent behavior.

    In a real scenario, this would be AgentTurnExecutor wrapping a real agent.
    For demonstration, we simulate completing after a few turns.
    """

    def __init__(self, complete_after_turns: int = 3, id: str = "mock_agent"):
        super().__init__(id=id)
        self._complete_after_turns = complete_after_turns

    @handler
    async def run_turn(self, trigger: RepairComplete, ctx: WorkflowContext[TurnComplete]) -> None:
        """Simulate an agent turn."""
        # Get current turn count
        turn_count = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)
        turn_count = (turn_count or 0) + 1
        await ctx.set_shared_state(HARNESS_TURN_COUNT_KEY, turn_count)

        # Simulate work
        print(f"  [Agent] Executing turn {turn_count}...")

        # Record event to transcript
        event = HarnessEvent(
            event_type="agent_response",
            data={
                "turn_number": turn_count,
                "message": f"Simulated response for turn {turn_count}",
            },
        )
        transcript = await ctx.get_shared_state(HARNESS_TRANSCRIPT_KEY) or []
        transcript.append(event.to_dict())
        await ctx.set_shared_state(HARNESS_TRANSCRIPT_KEY, transcript)

        # Determine if done
        is_done = turn_count >= self._complete_after_turns
        if is_done:
            print(f"  [Agent] Task completed after {turn_count} turns!")

        await ctx.send_message(TurnComplete(agent_done=is_done))


class ProgressReporter(Executor):
    """Reports progress after each turn."""

    @handler
    async def report(self, turn_complete: TurnComplete, ctx: WorkflowContext[TurnComplete]) -> None:
        """Report turn completion status."""
        turn_count = await ctx.get_shared_state(HARNESS_TURN_COUNT_KEY)

        if turn_complete.agent_done:
            print(f"  [Progress] Turn {turn_count}: Agent signaled completion")
        elif turn_complete.error:
            print(f"  [Progress] Turn {turn_count}: Error - {turn_complete.error}")
        else:
            print(f"  [Progress] Turn {turn_count}: Continuing...")

        # Pass through to stop decision
        await ctx.send_message(turn_complete)


async def demo_custom_harness() -> None:
    """Build a custom harness with additional executors."""
    print("=" * 60)
    print("Custom Harness with Additional Executors")
    print("=" * 60)

    # Build custom harness workflow:
    # repair -> logger -> mock_agent -> progress -> stop_decision -> repair
    workflow = (
        WorkflowBuilder(name="CustomHarness")
        .register_executor(lambda: RepairExecutor(id="repair"), name="repair")
        .register_executor(lambda: LoggingExecutor(id="logger"), name="logger")
        .register_executor(lambda: MockAgentExecutor(complete_after_turns=3, id="agent"), name="agent")
        .register_executor(lambda: ProgressReporter(id="progress"), name="progress")
        .register_executor(lambda: StopDecisionExecutor(id="stop"), name="stop")
        # Wire the loop
        .add_edge("repair", "logger")
        .add_edge("logger", "agent")
        .add_edge("agent", "progress")
        .add_edge("progress", "stop")
        .add_edge("stop", "repair")  # Loop back
        .set_start_executor("repair")
        .set_max_iterations(50)
        .build()
    )

    print("\nRunning custom harness (mock agent completes after 3 turns)...")
    print("-" * 40)

    harness_kwargs: dict[str, Any] = {HARNESS_CONFIG_KEY: {"max_turns": 10}}
    result = await workflow.run(
        RepairTrigger(),
        **harness_kwargs,
    )

    outputs = result.get_outputs()
    if outputs:
        harness_result = outputs[0]
        if isinstance(harness_result, HarnessResult):
            print(f"\n--- Result ---")
            print(f"Status: {harness_result.status.value}")
            print(f"Turns: {harness_result.turn_count}")
            print(f"Stop reason: {harness_result.reason.kind if harness_result.reason else 'N/A'}")


async def demo_harness_with_real_agent() -> None:
    """Demonstrate custom harness with a real Azure agent."""
    print("\n" + "=" * 60)
    print("Custom Harness with Real Agent")
    print("=" * 60)

    # Create Azure agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        instructions="You are a helpful assistant. Answer questions concisely in one sentence.",
        name="concise_assistant",
    )

    # For this demo, we'll use the standard HarnessWorkflowBuilder
    # but you could integrate the real AgentTurnExecutor into a custom workflow
    from agent_framework._harness import HarnessWorkflowBuilder

    builder = HarnessWorkflowBuilder(agent, max_turns=2)
    workflow = builder.build()

    print("\nRunning harness with real agent (max 2 turns)...")
    print("Query: 'What is 2+2?'")
    print("-" * 40)

    result = await workflow.run(
        RepairTrigger(),
        **builder.get_harness_kwargs(),
    )

    outputs = result.get_outputs()
    if outputs:
        harness_result = outputs[0]
        if isinstance(harness_result, HarnessResult):
            print(f"\n--- Result ---")
            print(f"Status: {harness_result.status.value}")
            print(f"Turns: {harness_result.turn_count}")
            print(f"Transcript events: {len(harness_result.transcript)}")


async def main() -> None:
    """Run custom harness demos."""
    print("Custom Agent Harness Demonstration")
    print("=" * 60)
    print("Shows how to build custom harness workflows with")
    print("additional executors for logging, progress, and validation.")
    print()

    await demo_custom_harness()
    await demo_harness_with_real_agent()

    print("\n" + "=" * 60)
    print("Custom harness demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
