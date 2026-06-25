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
from collections.abc import Awaitable, Callable

import aiohttp
from agent_framework import (
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    InMemoryHistoryProvider,
    Message,
    create_harness_agent,
    todos_remaining,
    todos_remaining_message,
    tool,
)
from agent_framework.foundry import FoundryChatClient
from agent_framework_eval_harness import EvalHarness
from agent_framework_eval_harness.benchmarks import GAIABenchmark
from agent_framework_monty import MontyExecuteCodeTool
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

# ── fetch_url tool ────────────────────────────────────────────────────────────

_MAX_PAGE_CHARS = 12_000  # truncate large pages to avoid context overflow


@tool
async def get_youtube_transcript(video_url: str) -> str:
    """Get the spoken transcript of a YouTube video.

    Use this when a question references a YouTube URL. Returns the full captions
    so you can find specific information said in the video.

    Args:
        video_url: Full YouTube URL (e.g. https://www.youtube.com/watch?v=...)
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-untyped]

        vid_match = re.search(r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_-]{11})", video_url)
        if not vid_match:
            return f"Could not extract video ID from: {video_url}"
        transcript = YouTubeTranscriptApi.get_transcript(vid_match.group(1))
        text = " ".join(entry["text"] for entry in transcript)
        return text[:8_000] + ("…[truncated]" if len(text) > 8_000 else "")
    except Exception as exc:
        return f"Error fetching transcript for {video_url}: {exc}"


@tool
async def fetch_url(url: str) -> str:
    """Fetch the text content of a web page.

    Use this after a web search when you need to read the actual page content
    rather than a short search snippet.  Returns the first 12 000 characters
    of the page body text.

    Args:
        url: The full URL to fetch.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GaiaEvalBot/1.0)"}
    try:
        async with aiohttp.ClientSession(headers=headers) as http, http.get(
            url, timeout=aiohttp.ClientTimeout(total=20), allow_redirects=True
        ) as resp:
            resp.raise_for_status()
            html = await resp.text(errors="replace")

        # Very light HTML → text: strip tags, collapse whitespace
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:_MAX_PAGE_CHARS] + ("…[truncated]" if len(text) > _MAX_PAGE_CHARS else "")
    except Exception as exc:
        return f"Error fetching {url}: {exc}"

# ── GAIA-specific agent instructions ─────────────────────────────────────────


GAIA_AGENT_INSTRUCTIONS = """\
## GAIA Benchmark Agent

You are a precise research assistant answering GAIA benchmark questions.

### How to work

Use web search to find relevant pages, then fetch_url to read their full content.
For questions referencing YouTube URLs, use get_youtube_transcript to read the video content.
Search multiple sources and cross-reference before committing to an answer.
Use execute_code for arithmetic, counting, sorting, or data manipulation.
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
# Matches prose boundaries after the answer on the same line.
# Only strip at ". Word" when the preceding token is a full word (≥4 chars),
# not an abbreviation like INT., Mr., Dr., etc.
_PROSE_BOUNDARY_RE = re.compile(
    # ". Word" boundary — only after a full word (≥4 chars), not abbreviations like INT., Mr.
    r"(?:(?<=\w{4})\.\s+(?=[A-Z])"
    # "I ..." or "I've/I'd/I'll ..." — agent continuing with self-reference
    r"|\s+I(?:'[a-z]+)?\s+"
    # Common sentence starters the agent appends after the answer
    r"|\s+(?:The|This|It|Note|Please|Both|All|Here|Understood|Done|Complete|Note)(?:'[a-z]+)?\s*[!,.]?"
    # Comma followed by connective (description, not list): "12 layers, which is..."
    r"|,\s+(?:which|where|that|and|but|or|as|making|giving|so|since|because|meaning|giving)\b"
    r")",
)
# Markdown link: [text](url) or bare (url) — strip the whole thing
_MARKDOWN_LINK_RE = re.compile(r"\s*\[?[^\]]*\]?\(https?://[^\)]+\)")
# Leading markdown bold/italic markers (** or *) the formatter sometimes adds
_MARKDOWN_BOLD_RE = re.compile(r"^\*{1,2}\s*")


def extract_final_answer(response: str) -> str:
    """Extract the ``FINAL ANSWER:`` line from a harness agent response.

    Uses the *last* occurrence when the loop runs multiple iterations — the
    agent's final answer is always the most refined.  Also strips trailing
    prose and markdown citation links the agent may have appended.

    Falls back to the full response text when no FINAL ANSWER is found,
    which will almost certainly score as incorrect and correctly penalises
    non-compliant responses.
    """
    matches = _FINAL_ANSWER_RE.findall(response)
    if not matches:
        return response.strip()
    # Take the last match — the agent's definitive answer after all loop iterations
    answer = matches[-1].strip()
    # Strip leading markdown bold/italic markers
    answer = _MARKDOWN_BOLD_RE.sub("", answer).strip()
    # Strip markdown links (agent sometimes cites sources inline)
    answer = _MARKDOWN_LINK_RE.sub("", answer).strip()
    # Truncate at prose boundary introduced by the agent on the same line
    prose = _PROSE_BOUNDARY_RE.search(answer)
    if prose:
        answer = answer[: prose.start()].strip()
    return answer


# ── Answer formatter middleware ───────────────────────────────────────────────

_FORMATTER_PROMPT = """\
Output the final answer to the question. One line only, exactly:

FINAL ANSWER: <answer>

Rules: number, name, date, or short phrase. No explanation. No units unless required."""


class GaiaAnswerFormatterMiddleware(AgentMiddleware):
    """Ensures every agent response contains a ``FINAL ANSWER:`` line.

    When the agent finishes without the required format, makes a single
    lightweight follow-up call asking it to extract the answer.  This runs
    per loop-iteration so the loop always sees a well-formatted result.
    """

    async def process(self, context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        result = context.result
        if not isinstance(result, AgentResponse):
            return
        if _FINAL_ANSWER_RE.search(result.text or ""):
            return  # already formatted — nothing to do

        # Build the extraction prompt from the current conversation
        extraction_messages = list(result.messages or [])
        extraction_messages.append(Message("user", [_FORMATTER_PROMPT]))
        try:
            extraction = await context.agent.client.get_response(extraction_messages)
            extraction_text = " ".join(
                c.text or ""
                for m in (extraction.messages or [])
                for c in (m.contents or [])
                if getattr(c, "text", None) and m.role == "assistant"
            )
            # Extract and clean from the extraction response, then inject a pristine message
            # so that extract_final_answer always gets a single clean FINAL ANSWER: line.
            clean_matches = _FINAL_ANSWER_RE.findall(extraction_text)
            if clean_matches:
                clean_answer = _MARKDOWN_BOLD_RE.sub("", clean_matches[-1]).strip()
                clean_answer = _MARKDOWN_LINK_RE.sub("", clean_answer).strip()
                prose = _PROSE_BOUNDARY_RE.search(clean_answer)
                if prose:
                    clean_answer = clean_answer[: prose.start()].strip()
                context.result = AgentResponse(
                    messages=list(result.messages or [])
                    + [Message("assistant", [f"FINAL ANSWER: {clean_answer}"])]
                )
            else:
                # Fallback: append raw extraction and let extract_final_answer handle it
                context.result = AgentResponse(
                    messages=list(result.messages or []) + list(extraction.messages or [])
                )
        except Exception:
            pass  # leave original result intact on failure


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
    # Key choices for a fair apples-to-apples comparison against LangGraph:
    #   - Web search: enabled automatically by create_harness_agent
    #   - fetch_url: reads full page content after a search
    #   - MontyExecuteCodeTool: Monty Python sandbox for arithmetic/computation
    #   - GaiaAnswerFormatterMiddleware: ensures every response has FINAL ANSWER:
    #   - TodoProvider + looping: structured multi-step planning
    #   - File memory/access: disabled (not available in competing systems)

    agent = create_harness_agent(
        client=client,
        max_context_window_tokens=128_000,
        max_output_tokens=8_192,
        name="GaiaHarnessAgent",
        agent_instructions=GAIA_AGENT_INSTRUCTIONS,
        tools=[fetch_url, get_youtube_transcript, MontyExecuteCodeTool()],
        middleware=[GaiaAnswerFormatterMiddleware()],
        loop_should_continue=todos_remaining(),
        loop_next_message=todos_remaining_message,
        loop_max_iterations=15,
        disable_file_memory=True,
        disable_file_access=True,
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
            skip_file_attachments=True,
            answer_extractor=extract_final_answer,
            verbose=args.verbose,
            seed=None if args.seed == -1 else args.seed,
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task: question, extracted answer, expected, pass/fail",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0=reproducible; -1=random)")

    asyncio.run(main(parser.parse_args()))
