# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "agent-framework",
#     "azure-identity",
#     "python-dotenv",
# ]
# ///
# NOTE: also requires agent-framework-eval-harness[gaia] (local workspace package).
# Run from the python/ directory so the workspace environment is used:
#   uv run python samples/05-end-to-end/evaluation/gaia_harness_eval.py
#   uv run python samples/05-end-to-end/evaluation/gaia_harness_eval.py --max-tasks 20 --parallel 4

# Copyright (c) Microsoft. All rights reserved.

"""GAIA Benchmark Evaluation — agent-framework harness vs. published baselines.

Configures ``create_harness_agent`` for GAIA and runs it through
``GAIABenchmark`` via ``EvalHarness``, then prints a comparison-ready summary
against published LangGraph and Claude scores.

Design choices for a fair comparison
─────────────────────────────────────
This sample intentionally mirrors the setup used by the LangGraph Deep Research
Agent and Claude Opus runs so that the scores are directly comparable:

* **Same model (GPT-4o)** — the most common reference model in published GAIA
  leaderboard entries.
* **Web search enabled** — automatic in ``create_harness_agent``.
* **TodoProvider + looping** — the harness's structural advantage: the agent
  creates a plan as todos, then loops autonomously until all todos are done.
  LangGraph achieves a similar effect via its ReAct/plan node graph.
* **File memory and file access disabled** — these features are not available
  in competing systems.  Disabling them keeps the context window clean and
  the comparison fair.
* **FINAL ANSWER extraction** — the agent is instructed to end every response
  with ``FINAL ANSWER: <short answer>``.  A regex extractor pulls that out
  before the official GAIA exact-match scorer runs.  This matches how
  published systems extract the final answer from long agent responses.

Published comparison scores (GPT-4o reference model, GAIA L1 validation):
  LangGraph Deep Research Agent  ~72%
  Claude Opus (Anthropic SDK)    ~75%
  HuggingFace leaderboard top    ~87%

Prerequisites
─────────────
1. HuggingFace token with GAIA dataset access:
     https://huggingface.co/settings/tokens
     https://huggingface.co/datasets/gaia-benchmark/GAIA  (request access)
     export HF_TOKEN="hf_..."

2. Azure AI Foundry project:
     export FOUNDRY_PROJECT_ENDPOINT="https://<your-project>.services.ai.azure.com/..."
     export FOUNDRY_MODEL="gpt-4o"        # reference model for published comparison

3. Install the eval-harness package with GAIA dependencies (workspace package):
     pip install -e "python/packages/eval-harness[gaia]"
   Or from the python/ directory:
     uv run python samples/05-end-to-end/evaluation/gaia_harness_eval.py

Authentication:
     az login
"""

import argparse
import asyncio
import os
import re

from agent_framework import InMemoryHistoryProvider, create_harness_agent, todos_remaining, todos_remaining_message
from agent_framework.foundry import FoundryChatClient
from agent_framework_eval_harness import EvalHarness
from agent_framework_eval_harness.benchmarks import GAIABenchmark
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# ── GAIA-specific agent instructions ─────────────────────────────────────────

GAIA_AGENT_INSTRUCTIONS = """\
## GAIA Benchmark Agent

You are a precise research assistant answering GAIA benchmark questions.

### How to work

Use web search and your available tools to research each question thoroughly.
For multi-step questions, create todos to track each sub-task before executing.
Always verify facts with tools — GAIA questions require specific, current knowledge.

### Answer format

After completing your research, end your response with exactly:

    FINAL ANSWER: <your answer>

The final answer must be short and exact: a number, date, name, short phrase,
or comma-separated list matching precisely what the question asks for.
Do not include units, explanations, or extra punctuation unless they are
part of the expected answer.

Examples:
    FINAL ANSWER: 42
    FINAL ANSWER: Marie Curie
    FINAL ANSWER: 3, 7, 11
    FINAL ANSWER: 1969-07-20
"""

# ── Answer extraction ─────────────────────────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(r"FINAL\s+ANSWER\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE)


def extract_final_answer(response: str) -> str:
    """Extract the ``FINAL ANSWER:`` line from a harness agent response.

    Falls back to the full response text when the pattern is not found,
    which will almost certainly score as incorrect and correctly penalises
    non-compliant responses.
    """
    match = _FINAL_ANSWER_RE.search(response)
    return match.group(1).strip() if match else response.strip()


# ── Main ──────────────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    load_dotenv(override=True)  # load .env if present, but allow env vars to override

    model = os.environ.get("FOUNDRY_MODEL", "gpt-4o")

    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=model,
        credential=AzureCliCredential(),
    )

    # <harness_gaia_agent>
    # Configure the harness agent for GAIA.
    #
    # Key choices for a fair apples-to-apples comparison against LangGraph
    # Deep Research Agent and Claude Opus:
    #   - Web search: enabled automatically by create_harness_agent
    #   - Code interpreter: Foundry-hosted Python sandbox for arithmetic,
    #     sorting, counting — the main gap vs published GAIA scores
    #   - TodoProvider + looping: structured multi-step planning while open todos remain
    #   - File memory/access: disabled (not available in competing systems)
    #   - loop_max_iterations=15: generous cap for complex GAIA tasks
    agent = create_harness_agent(
        client=client,
        max_context_window_tokens=128_000,
        max_output_tokens=8_192,
        name="GaiaHarnessAgent",
        agent_instructions=GAIA_AGENT_INSTRUCTIONS,
        tools=[client.get_code_interpreter_tool()],
        loop_should_continue=todos_remaining(),
        loop_next_message=todos_remaining_message,
        loop_max_iterations=15,
        disable_file_memory=True,
        disable_file_access=True,
        # FoundryChatClient uses the Responses API (server-side history).
        # load_messages=False tells the local provider to stay out of the way
        # and silences the "skipping local history load" warning.
        history_provider=InMemoryHistoryProvider(load_messages=False),
    )
    # </harness_gaia_agent>

    harness = EvalHarness(agent=agent)

    level_str = f"L{args.level}"
    task_str = f"{args.max_tasks} tasks" if args.max_tasks else "all tasks"
    print(f"Running GAIA {level_str} ({task_str}, parallel={args.parallel}, model={model})")
    print()

    # <run_gaia_eval>
    results = await harness.run(
        GAIABenchmark(
            level=args.level,
            max_tasks=args.max_tasks,
            parallel=args.parallel,
            timeout=args.timeout,
            skip_file_attachments=True,  # text-only (matches most published runs)
            answer_extractor=extract_final_answer,
        )
    )
    # </run_gaia_eval>

    harness.print_summary(results)

    # Print a comparison table against published baselines
    if results and results[0].total > 0:
        r = results[0]
        pct = r.passed / r.total * 100

        print("─" * 50)
        print(f"Comparison — GAIA {level_str}, {model}")
        print("─" * 50)
        print(f"  This harness ({task_str}):          {pct:5.1f}%  ({r.passed}/{r.total})")
        print("  LangGraph Deep Research (L1):     ~72.0%  (GPT-4o)")
        print("  Claude Opus – Anthropic SDK (L1): ~75.0%")
        print("  HuggingFace leaderboard top (L1): ~87.0%")
        print()
        print("Note: published scores use the full validation split (165 tasks).")
        print("Scale up with --max-tasks 165 for a full comparison run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GAIA benchmark on the agent-framework harness agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test — 5 tasks
  uv run python samples/05-end-to-end/evaluation/gaia_harness_eval.py --max-tasks 5

  # Full L1 comparison run (matches published scores)
  uv run python samples/05-end-to-end/evaluation/gaia_harness_eval.py --max-tasks 165 --parallel 8

  # Level 2 with per-task timeout
  uv run python samples/05-end-to-end/evaluation/gaia_harness_eval.py --level 2 --max-tasks 20 --timeout 600
        """,
    )
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="GAIA level (default: 1)")
    parser.add_argument("--max-tasks", type=int, default=None, metavar="N", help="Cap tasks (default: all)")
    parser.add_argument("--parallel", type=int, default=1, help="Concurrent agent runs (default: 1)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-task timeout seconds (default: 300)")

    asyncio.run(main(parser.parse_args()))
