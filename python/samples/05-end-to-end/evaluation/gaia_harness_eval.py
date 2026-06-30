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

import aiohttp
from agent_framework import (
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

# ── fetch tools ──────────────────────────────────────────────────────────────

_MAX_PAGE_CHARS = 10_000


async def _fetch_page_text(url: str, session: aiohttp.ClientSession, max_chars: int = _MAX_PAGE_CHARS) -> str:
    """Fetch and strip a single page; returns plain text capped at *max_chars*."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20), allow_redirects=True) as resp:
            resp.raise_for_status()
            html = await resp.text(errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars] + ("…[truncated]" if len(text) > max_chars else "")
    except Exception as exc:
        return f"[fetch error: {exc}]"


@tool
async def fetch_url(url: str) -> str:
    """Fetch the full text content of a web page.

    Use after a web search when you need to read the actual page content
    rather than rely on a search snippet. Returns up to 10 000 characters.

    Args:
        url: The full URL to fetch.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GaiaEvalBot/1.0)"}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            return await _fetch_page_text(url, session)
    except Exception as exc:
        return f"Error fetching {url}: {exc}"


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
        video_id = vid_match.group(1)
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        text = " ".join(snippet.text for snippet in transcript)
        return text[:8_000] + ("…[truncated]" if len(text) > 8_000 else "")
    except Exception as exc:
        return f"Error fetching transcript for {video_url}: {exc}"


# ── GAIA-specific agent instructions ─────────────────────────────────────────


GAIA_AGENT_INSTRUCTIONS = """\
## GAIA Benchmark Agent

You are a precise research assistant answering GAIA benchmark questions.

### How to work

Use web search to find relevant pages, then fetch_url to read their full content.
For questions referencing YouTube URLs, use get_youtube_transcript first.
Use execute_code for arithmetic, counting, sorting, or data manipulation.

**Research strategy:**
1. Form 2-3 different search queries from different angles.
2. Use fetch_url to read the full text of the most relevant pages — do not rely on search snippets alone.
3. After finding a candidate answer, **verify it**: re-read the source to confirm the specific value
   is exactly what you found — not a neighboring fact, not a similar concept.
4. If two sources disagree, try a third.

For multi-step questions, create todos to track each sub-task before executing.
Always verify facts with tools — GAIA questions require specific, current knowledge.
4. If two sources disagree, try a third.

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

**Always provide a FINAL ANSWER**, even if uncertain — give your best guess
based on available evidence. Never say "insufficient information" or leave the
answer blank.

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
    # Common sentence starters / acknowledgement words — match word + any trailing punctuation/space
    r"|\s+(?:The|This|It|Note|Please|Both|All|Here|Understood|Done|Complete|However|Therefore|Thus|FINAL)"
    r"(?:'[a-z]+)?(?:\s|[!.,\u2014]|$)"
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


# ── Reformulator ──────────────────────────────────────────────────────────────
#
# Adapted from HuggingFace smolagents prepare_response():
# https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/scripts/reformulator.py
#
# Instead of asking the agent to produce FINAL ANSWER: inline (while still
# reasoning), a separate LLM call reads the entire research transcript with
# fresh eyes and applies precise GAIA formatting rules.  This is the primary
# driver of smolagents' ~55% GAIA L1 score with GPT-4o.

# Verbatim from smolagents reformulator.py
_REFORMULATOR_PROMPT = """\
Read the above conversation and output a FINAL ANSWER to the question. \
The question is repeated here for convenience:

{question}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question \
(e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, \
and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. \
Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether \
the elements are numbers or strings.
Always provide a FINAL ANSWER based on the best available evidence. \
Never output 'Unable to determine' or leave the answer blank — make your best guess."""


def _build_transcript(messages: list[Message], max_chars: int = 30_000) -> str:
    """Extract a concise research transcript from accumulated agent messages.

    Includes assistant reasoning text and tool results; skips function-call
    invocations and user nudge messages (loop internals).
    """
    parts: list[str] = []
    chars_used = 0

    for msg in messages:
        if not msg.contents:
            continue
        role = msg.role

        for content in msg.contents:
            ctype = getattr(content, "type", "")

            if role == "assistant" and ctype == "text":
                text = getattr(content, "text", "") or ""
                if text:
                    entry = f"[assistant]: {text}"
                    parts.append(entry)
                    chars_used += len(entry)

            elif ctype == "function_result":
                result_val = getattr(content, "result", None)
                if result_val:
                    snippet = str(result_val)[:2_000]
                    entry = f"[tool result]: {snippet}"
                    parts.append(entry)
                    chars_used += len(entry)

            if chars_used >= max_chars:
                break
        if chars_used >= max_chars:
            break

    transcript = "\n\n".join(parts)
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "…[truncated]"
    return transcript


def make_reformulator(client: FoundryChatClient):
    """Create an async reformulator callable for ``GAIABenchmark.response_reformulator``.

    Runs ONCE per task after ``agent.run()`` returns (after all loop iterations),
    reading the full accumulated research transcript and applying precise GAIA
    formatting rules via a fresh LLM call.

    Args:
        client: The same ``FoundryChatClient`` used for the agent.

    Returns:
        An async callable ``(question: str, response: AgentResponse) -> AgentResponse``.
    """

    async def reformulate(question: str, response: AgentResponse) -> AgentResponse:  # type: ignore[type-arg]
        transcript = _build_transcript(response.messages or [])
        if not transcript:
            return response

        # Use system + user message structure (like smolagents) rather than passing
        # the raw conversation — avoids the model being confused by prior FINAL ANSWER:
        # lines in the history and echoing them.
        system_content = (
            f"Earlier you were asked the following:\n\n{question}\n\n"
            f"Your research produced the following transcript:\n\n{transcript}"
        )
        reformulator_messages = [
            Message("system", [system_content]),
            Message("user", [_REFORMULATOR_PROMPT.format(question=question)]),
        ]

        try:
            extraction = await client.get_response(reformulator_messages)
            extraction_text = extraction.text or ""

            # The reformulator always produces FINAL ANSWER: <answer>
            m = _FINAL_ANSWER_RE.search(extraction_text)
            if m:
                clean_answer = m.group(1).strip()
                # Inject the clean answer as a new message so extract_final_answer
                # sees a single, prose-free FINAL ANSWER line.
                return AgentResponse(
                    messages=list(response.messages or [])
                    + [Message("assistant", [f"FINAL ANSWER: {clean_answer}"])]
                )
        except Exception:
            pass  # leave original response intact on failure

        return response

    return reformulate


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
    # Key choices for a fair apples-to-apples comparison against smolagents/LangGraph:
    #   - Web search: enabled automatically by create_harness_agent (Bing via Foundry)
    #   - fetch_url: reads full page content when agent has a specific URL
    #   - MontyExecuteCodeTool: Python sandbox for arithmetic/computation
    #   - TodoProvider + looping: structured multi-step planning
    #   - File memory/access: disabled (not available in competing systems)
    #   - No middleware: the reformulator (below) replaces GaiaAnswerFormatterMiddleware

    agent = create_harness_agent(
        client=client,
        max_context_window_tokens=128_000,
        max_output_tokens=8_192,
        name="GaiaHarnessAgent",
        agent_instructions=GAIA_AGENT_INSTRUCTIONS,
        tools=[fetch_url, get_youtube_transcript, MontyExecuteCodeTool()],
        loop_should_continue=todos_remaining(),
        loop_next_message=todos_remaining_message,
        loop_max_iterations=15,
        disable_file_memory=True,
        disable_file_access=True,
        history_provider=InMemoryHistoryProvider(load_messages=False),
    )
    # </harness_gaia_agent>

    # <reformulator>
    # Adapted from HuggingFace smolagents' prepare_response().
    # Runs ONCE per task after all loop iterations complete, reading the
    # full research transcript with fresh eyes and applying precise GAIA
    # formatting rules — rather than relying on inline FINAL ANSWER: text.
    reformulator = make_reformulator(client)
    # </reformulator>

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
            response_reformulator=reformulator,
            verbose=args.verbose,
            seed=None if args.seed == -1 else args.seed,
            results_file=args.results_file,
        )
    )
    # </run_gaia_eval>

    harness.print_summary(results)

    # Print a comparison table against published baselines
    if results and results[0].total > 0:
        r = results[0]
        pct = r.passed / r.total * 100

        print("-" * 50)
        print(f"Comparison -- GAIA {level_str}, {model}")
        print("-" * 50)
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
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Save per-task results as JSON Lines to this file for offline analysis",
    )

    asyncio.run(main(parser.parse_args()))
