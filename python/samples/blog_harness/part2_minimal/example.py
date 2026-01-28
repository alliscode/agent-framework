# Copyright (c) Microsoft. All rights reserved.

"""Part 2 Example: Running the Minimal Harness.

This script demonstrates using the SimpleHarness to run an agent
with turn limits and termination detection.

Usage:
    python example.py
    python example.py --max-turns 5
    python example.py --task "Explain what a harness is in 2 sentences"
"""

import argparse
import asyncio
import math
from datetime import datetime
from typing import Annotated

from agent_framework import ai_function
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from harness import HarnessStatus, SimpleHarness

# ============================================================
# Example Tools
# ============================================================
# Simple tools to demonstrate multi-turn agent behavior.
# Tools use @ai_function decorator to expose them to the agent.


@ai_function
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@ai_function
def calculate(
    expression: Annotated[str, "A mathematical expression like '2 + 2' or 'sqrt(16)'"],
) -> str:
    """Evaluate a mathematical expression safely."""
    # Safe evaluation with math functions available
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# Main
# ============================================================


async def main(task: str, max_turns: int) -> None:
    """Run the harness with a sample task."""
    print(f"Task: {task}")
    print(f"Max turns: {max_turns}")
    print("-" * 50)

    # Create the agent
    client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = client.create_agent(
        name="assistant",
        instructions=(
            "You are a helpful assistant. Use tools when needed. "
            "Be concise in your responses."
        ),
        tools=[get_current_time, calculate],
    )

    # Wrap in harness
    harness = SimpleHarness(agent, max_turns=max_turns)

    # Run with streaming to see progress
    print("\nRunning harness...\n")

    result = None
    async for event in harness.run_stream(task):
        # Print turn progress
        if (
            hasattr(event, "data")
            and isinstance(event.data, dict)
            and "turn" in event.data
        ):
            turn = event.data["turn"]
            response_preview = event.data.get("response", "")[:100]
            print(f"  Turn {turn}: {response_preview}...")

        # Capture final result
        if hasattr(event, "data") and hasattr(event.data, "status"):
            result = event.data

    # Print final result
    print("\n" + "-" * 50)
    if result:
        status_icon = "✓" if result.status == HarnessStatus.COMPLETE else "⚠"
        print(f"{status_icon} Status: {result.status.value}")
        print(f"  Turns used: {result.turn_count}")
        print(f"\nFinal response:\n{result.final_response}")
    else:
        print("No result received!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the minimal harness example")
    parser.add_argument(
        "--task",
        type=str,
        default="What time is it, and what is the square root of 144?",
        help="The task for the agent",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns (default: 10)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.task, args.max_turns))
