# Copyright (c) Microsoft. All rights reserved.

"""Part 4 Example: Policy Enforcement in Action.

This script demonstrates the policy-enabled harness with:
1. Normal task completion with policies active
2. Triggering the tool call limit policy
3. Triggering the content filter policy

Usage:
    python example.py                    # Normal task with policies
    python example.py --tool-limit-test  # Trigger tool limit violation
    python example.py --content-test     # Trigger content filter violation
    python example.py --max-tool-calls 5 # Set custom tool limit
"""

import argparse
import asyncio
import math
from datetime import datetime
from typing import Annotated

from agent_framework import ai_function
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from harness import (
    ContentFilterPolicy,
    HarnessStatus,
    MaxToolCallsPolicy,
    ResponseLengthPolicy,
    SimpleHarness,
)

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
def fetch_data(
    query: Annotated[str, "What data to fetch"],
) -> str:
    """Fetch some data (simulated). This tool is chatty and always suggests more queries."""
    return (
        f"Data for '{query}': [sample result]\n"
        f"Related queries you might want to try: '{query} details', '{query} history', '{query} statistics'"
    )


@ai_function
def get_secret_info() -> str:
    """Get some information (this will trigger content filter in demo)."""
    # This simulates an agent response that might leak sensitive patterns
    return "The system password is: hunter2 (just kidding, this is a demo)"


# ============================================================
# Test Tasks
# ============================================================

# Normal task - should complete successfully
NORMAL_TASK = "What time is it, and what is sqrt(256) + sqrt(144)?"

# Task designed to trigger tool limit (agent keeps fetching more data)
TOOL_LIMIT_TASK = """
Fetch data about the following topics and summarize each:
1. Python programming
2. Machine learning
3. Cloud computing
4. Databases
5. API design
6. Testing strategies
7. DevOps practices
8. Security best practices

For each topic, also fetch the 'details' and 'history' variants.
"""

# Task designed to trigger content filter
CONTENT_FILTER_TASK = """
Call the get_secret_info function and tell me what it returns.
"""


# ============================================================
# Main
# ============================================================


async def main(
    task: str,
    max_turns: int,
    stall_threshold: int,
    max_tool_calls: int,
    content_patterns: list[str],
) -> None:
    """Run the harness with policy enforcement."""
    print(f"Task: {task[:80]}...")
    print(f"Max turns: {max_turns}, Max tool calls: {max_tool_calls}")
    print(f"Content filter patterns: {content_patterns}")
    print("-" * 60)

    # Create the agent
    client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = client.create_agent(
        name="assistant",
        instructions=(
            "You are a helpful assistant. Use tools when needed. "
            "Be thorough - if a tool suggests related queries, explore them."
        ),
        tools=[get_current_time, calculate, fetch_data, get_secret_info],
    )

    # Configure policies
    policies = [
        MaxToolCallsPolicy(max_calls=max_tool_calls),
        ContentFilterPolicy(
            patterns=content_patterns,
            stop_on_match=True,  # Hard stop on content violation
        ),
        ResponseLengthPolicy(min_length=10, max_length=5000),
    ]

    print("\nActive policies:")
    for policy in policies:
        print(f"  - {policy.name}")

    # Wrap in harness with policies
    harness = SimpleHarness(
        agent,
        max_turns=max_turns,
        stall_threshold=stall_threshold,
        policies=policies,
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
                tool_count = data.get("tool_count", 0)
                response_preview = data.get("response", "")[:50]
                tools_info = f" [{tool_count} tools]" if tool_count else ""
                print(f"  Turn {turn} [{fingerprint}]{tools_info}: {response_preview}...")

            # Print policy violations
            if data.get("event") == "policy_violation":
                policy_name = data["policy"]
                message = data["message"]
                should_stop = data["should_stop"]
                stop_indicator = " [STOPPING]" if should_stop else " [WARNING]"
                print(f"  ⚠️  POLICY: {policy_name}{stop_indicator}")
                print(f"      {message}")

            # Print stall detection
            if data.get("event") == "stall_detected":
                stall_count = data["stall_count"]
                print(f"  🔄 STALL DETECTED (count: {stall_count})")

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
            HarnessStatus.POLICY_VIOLATION: "🚫",
            HarnessStatus.ERROR: "✗",
        }
        icon = status_icons.get(result.status, "?")
        print(f"{icon} Status: {result.status.value}")
        print(f"  Turns used: {result.turn_count}")
        print(f"  Stall count: {result.stall_count}")

        if result.policy_violations:
            print(f"\n  Policy violations ({len(result.policy_violations)}):")
            for violation in result.policy_violations:
                print(f"    - {violation}")

        if result.status == HarnessStatus.POLICY_VIOLATION:
            print("\n  The agent was stopped due to a policy violation.")
            print("  Review the violation messages above to understand why.")

        print(f"\nFinal response:\n{result.final_response[:500]}")
        if len(result.final_response) > 500:
            print("...")
    else:
        print("No result received!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the policy enforcement harness example")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Custom task (default: normal task)",
    )
    parser.add_argument(
        "--tool-limit-test",
        action="store_true",
        help="Run with a task that triggers tool limit",
    )
    parser.add_argument(
        "--content-test",
        action="store_true",
        help="Run with a task that triggers content filter",
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
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=15,
        help="Maximum tool calls (default: 15)",
    )
    args = parser.parse_args()

    # Select task
    if args.task:
        task = args.task
    elif args.tool_limit_test:
        task = TOOL_LIMIT_TASK
    elif args.content_test:
        task = CONTENT_FILTER_TASK
    else:
        task = NORMAL_TASK

    # Content patterns to filter
    content_patterns = [r"password", r"secret", r"credential"]

    asyncio.run(
        main(
            task,
            args.max_turns,
            args.stall_threshold,
            args.max_tool_calls,
            content_patterns,
        )
    )
