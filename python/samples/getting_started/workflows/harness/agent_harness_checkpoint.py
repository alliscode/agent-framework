# Copyright (c) Microsoft. All rights reserved.

"""
Sample: Agent Harness with Checkpointing

What it does:
- Demonstrates checkpointing and resume capabilities of the Agent Harness
- Shows how to persist harness state for durability
- Demonstrates resuming from a checkpoint after interruption

The Agent Harness builds on Workflows checkpointing to provide:
- Automatic checkpointing after each turn
- Resume from any checkpoint
- Preservation of transcript and turn state

Prerequisites:
- Azure OpenAI configured for AzureOpenAIChatClient with required environment variables.
- Authentication via azure-identity. Use AzureCliCredential and run `az login` before executing.
"""

import asyncio
import tempfile
from pathlib import Path

from agent_framework import FileCheckpointStorage, WorkflowOutputEvent, ai_function
from agent_framework._harness import (
    HarnessResult,
    HarnessWorkflowBuilder,
    RepairTrigger,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential


# Simple tool for the agent
@ai_function
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2").

    Returns:
        The result of the calculation.
    """
    try:
        # Simple eval for demo (in production, use a safe math parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"


async def demo_checkpoint_and_resume() -> None:
    """Demonstrate checkpointing and resuming the harness."""
    print("=" * 60)
    print("Agent Harness Checkpoint & Resume Demo")
    print("=" * 60)

    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        storage = FileCheckpointStorage(str(checkpoint_dir))

        # Create a chat client and agent
        chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
        agent = chat_client.create_agent(
            instructions=(
                "You are a math tutor. When given a problem, break it down into steps "
                "and solve each step. Use the calculate tool for each calculation. "
                "Always explain your reasoning."
            ),
            name="math_tutor",
            tools=[calculate],
        )

        # Build harness with checkpointing enabled
        builder = HarnessWorkflowBuilder(
            agent,
            max_turns=10,
            checkpoint_storage=storage,
        )
        workflow = builder.build()

        print("\n--- Phase 1: Initial Run with Checkpointing ---")
        print("Query: 'Calculate (15 + 7) * 3'")
        print("-" * 40)

        checkpoint_ids = []
        async for event in workflow.run_stream(
            RepairTrigger(),
            **builder.get_harness_kwargs(),
        ):
            if isinstance(event, WorkflowOutputEvent):
                result = event.data
                if isinstance(result, HarnessResult):
                    print(f"\nHarness completed!")
                    print(f"  Status: {result.status.value}")
                    print(f"  Turns: {result.turn_count}")

        # List available checkpoints
        checkpoint_ids = await storage.list_checkpoint_ids()
        print(f"\n--- Checkpoints Created: {len(checkpoint_ids)} ---")
        for cp_id in checkpoint_ids[:5]:  # Show first 5
            print(f"  - {cp_id}")
        if len(checkpoint_ids) > 5:
            print(f"  ... and {len(checkpoint_ids) - 5} more")

        # Demonstrate checkpoint inspection
        if checkpoint_ids:
            print("\n--- Checkpoint Inspection ---")
            latest_cp = await storage.load_checkpoint(checkpoint_ids[-1])
            if latest_cp:
                print(f"Latest checkpoint: {latest_cp.checkpoint_id}")
                print(f"  Workflow ID: {latest_cp.workflow_id}")
                print(f"  Iteration: {latest_cp.iteration_count}")
                print(f"  Timestamp: {latest_cp.timestamp}")
                print(f"  Shared state keys: {list(latest_cp.shared_state.keys())}")


async def demo_harness_durability() -> None:
    """Show how harness state persists across workflow instances."""
    print("\n" + "=" * 60)
    print("Harness Durability Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        storage = FileCheckpointStorage(str(checkpoint_dir))

        # Create agent
        chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
        agent = chat_client.create_agent(
            instructions="You are a helpful assistant. Be concise.",
            name="assistant",
        )

        # First workflow instance
        print("\n--- Workflow Instance 1 ---")
        builder1 = HarnessWorkflowBuilder(agent, max_turns=3, checkpoint_storage=storage)
        workflow1 = builder1.build()

        result1 = await workflow1.run(
            RepairTrigger(),
            **builder1.get_harness_kwargs(),
        )

        outputs1 = result1.get_outputs()
        if outputs1 and isinstance(outputs1[0], HarnessResult):
            hr1 = outputs1[0]
            print(f"Instance 1 completed: {hr1.turn_count} turns, status={hr1.status.value}")

        # Get checkpoint count
        checkpoints = await storage.list_checkpoint_ids()
        print(f"Checkpoints after instance 1: {len(checkpoints)}")

        # Second workflow instance (fresh)
        print("\n--- Workflow Instance 2 (Fresh) ---")
        builder2 = HarnessWorkflowBuilder(agent, max_turns=3, checkpoint_storage=storage)
        workflow2 = builder2.build()

        result2 = await workflow2.run(
            RepairTrigger(),
            **builder2.get_harness_kwargs(),
        )

        outputs2 = result2.get_outputs()
        if outputs2 and isinstance(outputs2[0], HarnessResult):
            hr2 = outputs2[0]
            print(f"Instance 2 completed: {hr2.turn_count} turns, status={hr2.status.value}")

        # Final checkpoint count
        checkpoints_final = await storage.list_checkpoint_ids()
        print(f"Total checkpoints: {len(checkpoints_final)}")
        print("\nCheckpoints provide durability for long-running agent tasks!")


async def main() -> None:
    """Run checkpoint demos."""
    print("Agent Harness Checkpointing Demonstration")
    print("=" * 60)
    print("Shows how the harness leverages workflow checkpointing")
    print("for durable, resumable agent execution.")
    print()

    await demo_checkpoint_and_resume()
    await demo_harness_durability()

    print("\n" + "=" * 60)
    print("Checkpoint demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
