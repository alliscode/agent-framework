# Copyright (c) Microsoft. All rights reserved.

"""Evolvable GAIA harness-agent settings used by the Darwin optimization pilot.

Darwin is intentionally restricted to this file. The benchmark runner, task
selection, exact-match scorer, and result persistence remain outside the
evolution surface so candidate fitness remains comparable.
"""

from dataclasses import dataclass

GAIA_AGENT_INSTRUCTIONS = """\
## GAIA Benchmark Agent

You are a precise research assistant answering GAIA benchmark questions.

### How to work

Use web search to find relevant pages, then fetch_url to read their full content.
For questions referencing YouTube URLs, use get_youtube_transcript first.
Use execute_code for arithmetic, counting, sorting, or data manipulation.
For multi-step questions, create todos to track each sub-task before executing.
Always verify facts with tools — GAIA questions require specific, current knowledge.

**Research strategy:**
1. Form 2-3 different search queries from different angles.
2. Use fetch_url to read the full text of the most relevant pages — do not rely on search snippets alone.
3. After finding a candidate answer, verify it against the source.
4. If two sources disagree, try a third.

### Answer format

End the response with exactly:

    FINAL ANSWER: <your answer>

The final answer must be short and exact: a number, date, name, short phrase,
or comma-separated list matching precisely what the question asks for.
Do not include units, explanations, or extra punctuation unless requested.
Always provide a best-effort final answer, even when uncertain.
"""


FINAL_ANSWER_FORMATTER_PROMPT = """\
Output the final answer to the question. One line only, exactly:

FINAL ANSWER: <answer>

Rules: number, name, date, or short phrase. No explanation. No units unless required.
Always provide an answer — never say unable to determine."""


@dataclass(frozen=True)
class GaiaCandidateSettings:
    """Configuration values Darwin may evolve during the pilot."""

    max_context_window_tokens: int = 128_000
    max_output_tokens: int = 8_192
    loop_max_iterations: int = 15
    use_todo_loop: bool = True
    use_answer_formatter: bool = True
    use_reformulator: bool = False
    use_fetch_url: bool = True
    use_youtube_transcript: bool = True
    use_code_execution: bool = True
    use_serpapi_search: bool = False
    disable_file_memory: bool = True
    disable_mode: bool = True
    history_load_messages: bool = False


SETTINGS = GaiaCandidateSettings()
