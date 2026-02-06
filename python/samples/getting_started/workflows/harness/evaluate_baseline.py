# Copyright (c) Microsoft. All rights reserved.

"""Run the same prompt with the same agent but WITHOUT the harness.

This provides a baseline comparison for evaluating what the harness
adds (work item tracking, artifact separation, turn control, etc.).

Usage:
    python evaluate_baseline.py
    python evaluate_baseline.py --prompt "Your task here..."
    python evaluate_baseline.py --prompt-file prompt.txt
"""

import argparse
import asyncio
from pathlib import Path

from agent_framework._types import ChatMessage
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential

DEFAULT_PROMPT = """\
You are acting as a Staff-level Technical Program Manager at a cloud platform company.

A production incident occurred last week with the following high-level facts:
* 47 minutes of downtime
* Partial data loss for 2 enterprise customers
* Executive escalation and a public status page update

Your task is to produce a complete, internally consistent, cross-functional \
postmortem and prevention plan without using any external tools or references.

Step 1: Incident Reconstruction - Invent a technically plausible root cause \
involving a deployment/config change, a monitoring/alerting failure, and a human \
decision under time pressure. Produce a precise timeline, systems involved, and \
missed signals.

Step 2: Stakeholder Impact Analysis - Analyze impact for Engineering, SRE, \
Security, Legal, Product, and Customers. Each group must have different concerns \
tied to your root cause.

Step 3: Corrective Actions - Create a plan with immediate (0-7d), short-term \
(7-30d), and long-term (30-180d) horizons. Each action maps to a failure and \
a stakeholder concern.

Step 4: Executive Summary - One-page, non-technical, legally safe summary.

Step 5: Consistency Audit - List contradictions, unstated assumptions, and \
legal/PR risks. Revise the Executive Summary to fix issues.
"""

# ANSI codes
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def run_baseline(prompt: str) -> None:
    """Run the agent directly without harness and stream output."""
    print(f"{BOLD}Running baseline (no harness) with prompt ({len(prompt)} chars)...{RESET}")
    print(f"{DIM}{'─' * 70}{RESET}\n")

    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        name="evaluation-agent",
        instructions=(
            "You are an expert analyst. Complete the task thoroughly."
        ),
    )

    thread = agent.get_new_thread()
    messages = [ChatMessage(role="user", text=prompt)]

    total_chars = 0
    async for update in agent.run_stream(messages, thread=thread):
        if hasattr(update, "text") and update.text:
            print(update.text, end="", flush=True)
            total_chars += len(update.text)

    print(f"\n\n{DIM}{'─' * 70}{RESET}")
    print(f"{DIM}  Output: {total_chars} chars | Single turn, no tools{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline agent (no harness) for comparison",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = DEFAULT_PROMPT

    asyncio.run(run_baseline(prompt))


if __name__ == "__main__":
    main()
