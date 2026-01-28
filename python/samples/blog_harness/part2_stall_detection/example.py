# Copyright (c) Microsoft. All rights reserved.

"""Part 3 Example: Stall Detection in Action.

This script demonstrates the stall detection harness with:
1. A normal task that completes successfully
2. A task designed to trigger stall detection

Usage:
    python example.py
    python example.py --stall-test  # Run with a stall-inducing prompt
    python example.py --stall-threshold 2  # Lower threshold for faster stall detection
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


@ai_function
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@ai_function
def calculate(
    expression: Annotated[str, "A mathematical expression like '2 + 2' or 'sqrt(16)'"],
) -> str:
    """Evaluate a mathematical expression safely."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@ai_function
def unreliable_api(
    query: Annotated[str, "The query to send to the API"],
) -> str:
    """An API that always fails (simulates unreliable external service).

    This tool is intentionally broken to demonstrate stall detection
    when an agent keeps retrying a failing operation.
    """
    return "Error: Service temporarily unavailable. Please try again later."


# ============================================================
# Main
# ============================================================

# A task that tends to cause stalls (agent keeps retrying the broken API)
STALL_INDUCING_TASK = """
Use the unreliable_api tool to look up the weather in Seattle.
Keep trying until you get a result.
"""

# A normal task that should complete
NORMAL_TASK = "What time is it, and what is the square root of 256?"


async def main(task: str, max_turns: int, stall_threshold: int) -> None:
    """Run the harness with stall detection."""
    print(f"Task: {task[:80]}...")
    print(f"Max turns: {max_turns}, Stall threshold: {stall_threshold}")
    print("-" * 60)

    # Create the agent
    client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = client.create_agent(
        name="assistant",
        instructions=(
            "You are a helpful assistant. Use tools when needed. Be persistent - if something fails, try again."
        ),
        tools=[get_current_time, calculate, unreliable_api],
    )

    # Wrap in harness with stall detection
    harness = SimpleHarness(
        agent,
        max_turns=max_turns,
        stall_threshold=stall_threshold,
    )

    print("\nRunning harness...\n")

    result = None
    async for event in harness.run_stream(task):
        if hasattr(event, "data") and isinstance(event.data, dict):
            data = event.data

            # Print turn progress
            if "turn" in data:
                turn = data["turn"]
                fingerprint = data.get("fingerprint", "")[:8]
                has_tools = data.get("has_tools", False)
                response_preview = data.get("response", "")[:60]
                tool_indicator = " [tools]" if has_tools else ""
                print(f"  Turn {turn} [{fingerprint}]{tool_indicator}: {response_preview}...")

            # Print stall detection events
            if data.get("event") == "stall_detected":
                stall_count = data["stall_count"]
                print(f"  ⚠️  STALL DETECTED (count: {stall_count}) - injecting continuation prompt")

        # Capture final result
        if hasattr(event, "data") and hasattr(event.data, "status"):
            result = event.data

    # Print final result
    print("\n" + "-" * 60)
    if result:
        status_icons = {
            HarnessStatus.COMPLETE: "✓",
            HarnessStatus.MAX_TURNS: "⚠",
            HarnessStatus.STALLED: "🔄",
            HarnessStatus.ERROR: "✗",
        }
        icon = status_icons.get(result.status, "?")
        print(f"{icon} Status: {result.status.value}")
        print(f"  Turns used: {result.turn_count}")
        print(f"  Stall count: {result.stall_count}")

        if result.status == HarnessStatus.STALLED:
            print("\n  The agent got stuck in a loop and couldn't recover.")
            print("  The harness detected repeated responses and stopped.")

        print(f"\nFinal response:\n{result.final_response[:500]}")
        if len(result.final_response) > 500:
            print("...")
    else:
        print("No result received!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stall detection harness example")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Custom task (default: normal task)",
    )
    parser.add_argument(
        "--stall-test",
        action="store_true",
        help="Run with a stall-inducing task",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns (default: 10)",
    )
    parser.add_argument(
        "--stall-threshold",
        type=int,
        default=3,
        help="Stall threshold (default: 3)",
    )
    args = parser.parse_args()

    # Select task
    if args.task:
        task = args.task
    elif args.stall_test:
        task = STALL_INDUCING_TASK
    else:
        task = NORMAL_TASK

    asyncio.run(main(task, args.max_turns, args.stall_threshold))
