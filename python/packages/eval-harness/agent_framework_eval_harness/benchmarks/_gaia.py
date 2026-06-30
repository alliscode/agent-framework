# Copyright (c) Microsoft. All rights reserved.

"""GAIA benchmark adapter.

GAIA (General AI Assistants) is a benchmark for general-purpose AI assistants,
covering 466 open-ended tasks across three difficulty levels requiring web
search, tool use, file reading, and multi-step reasoning.

Published comparison scores (approximate, GPT-4o reference model):
  - LangGraph Deep Research Agent: ~72% L1
  - Claude Opus (Anthropic SDK):   ~75% L1
  - HuggingFace leaderboard top:   ~87% L1

Reference:
  https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_framework import AgentResponse, EvalResults, Evaluator, SupportsAgentRun

logger = logging.getLogger(__name__)


# ── scoring ───────────────────────────────────────────────────────────────────


def _normalize_number_str(s: str) -> float:
    for ch in ["$", "%", ","]:
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _split_string(s: str, chars: list[str] | None = None) -> list[str]:
    if chars is None:
        chars = [",", ";"]
    return re.split(f"[{''.join(chars)}]", s)


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", s or "")
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def gaia_scorer(model_answer: str | None, ground_truth: str) -> bool:
    """Official GAIA normalized exact-match scoring function.

    Handles numeric comparison, comma/semicolon-delimited lists, and
    string normalization (whitespace, punctuation, case).

    Args:
        model_answer: The agent's answer.  ``None`` is treated as empty.
        ground_truth: The ground-truth answer from the GAIA dataset.

    Returns:
        ``True`` when the answer matches by GAIA's normalization rules.
    """

    def is_float(x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    if model_answer is None:
        model_answer = "None"

    if is_float(ground_truth):
        return abs(_normalize_number_str(model_answer) - float(ground_truth)) < 1e-6

    if any(ch in ground_truth for ch in [",", ";"]):
        gt_elems = _split_string(ground_truth)
        ma_elems = _split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        return all(
            abs(_normalize_number_str(ma) - float(gt)) < 1e-6
            if is_float(gt)
            else _normalize_str(ma, remove_punct=False) == _normalize_str(gt, remove_punct=False)
            for ma, gt in zip(ma_elems, gt_elems, strict=False)
        )

    return _normalize_str(model_answer) == _normalize_str(ground_truth)


# ── task loading ──────────────────────────────────────────────────────────────


@dataclass
class _GAIATask:
    task_id: str
    question: str
    answer: str
    level: int | None = None
    file_name: str | None = None


def _task_from_record(rec: Any) -> _GAIATask | None:
    if not isinstance(rec, dict):
        return None
    q = rec.get("Question") or rec.get("question") or rec.get("query")
    ans = rec.get("Final answer") or rec.get("answer") or rec.get("final_answer")
    if not isinstance(q, str) or ans is None or str(ans).strip() in ("?", ""):
        return None
    task_id = str(rec.get("task_id") or rec.get("question_id") or rec.get("id") or rec.get("uuid") or q[:32])
    lvl_raw = rec.get("Level") or rec.get("level")
    lvl: int | None = int(lvl_raw) if isinstance(lvl_raw, (int, str)) and str(lvl_raw).isdigit() else None
    fname = rec.get("file_name") or rec.get("filename")
    return _GAIATask(
        task_id=task_id,
        question=q,
        answer=str(ans),
        level=lvl,
        file_name=fname if isinstance(fname, str) else None,
    )


def _load_tasks(
    data_dir: Path,
    *,
    levels: list[int] | None,
    skip_file_attachments: bool,
    max_tasks: int | None,
    seed: int | None = 0,
) -> list[_GAIATask]:
    """Load GAIA tasks from local cache, parquet first then jsonl fallback."""
    import random

    tasks: list[_GAIATask] = []

    # Parquet files (newer format) — prefer validation split (has answers)
    parquet_files = sorted(
        data_dir.rglob("metadata*.parquet"),
        key=lambda p: (0 if "validation" in str(p) else 1, str(p)),
    )
    for p in parquet_files:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]

            for row in pq.read_table(p).to_pylist():
                t = _task_from_record(row)
                if t is None:
                    continue
                if levels and t.level not in levels:
                    continue
                if skip_file_attachments and t.file_name:
                    continue
                tasks.append(t)
        except ImportError:
            logger.warning("pyarrow not installed. Install with: pip install 'agent-framework-eval-harness[gaia]'")
            break
        except Exception:
            logger.warning("Could not load parquet file %s", p, exc_info=True)

    # JSONL fallback (older format)
    if not tasks:
        for p in data_dir.rglob("metadata.jsonl"):
            with p.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    t = _task_from_record(json.loads(line))
                    if t is None:
                        continue
                    if levels and t.level not in levels:
                        continue
                    if skip_file_attachments and t.file_name:
                        continue
                    tasks.append(t)

    rng = random.Random(seed)  # seed=None → truly random; seed=0 → reproducible dev runs  # noqa: S311
    rng.shuffle(tasks)
    return tasks[:max_tasks] if max_tasks is not None else tasks


def _ensure_data(data_dir: Path, hf_token: str | None) -> None:
    """Download the GAIA dataset to *data_dir* if not already cached."""
    has_parquet = any(data_dir.rglob("metadata*.parquet"))
    has_jsonl = any(data_dir.rglob("metadata.jsonl"))
    if data_dir.exists() and (has_parquet or has_jsonl):
        return

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable or hf_token= parameter required to download "
            "the GAIA dataset.\n"
            "  1. Get a token:  https://huggingface.co/settings/tokens\n"
            "  2. Request access: https://huggingface.co/datasets/gaia-benchmark/GAIA\n"
            "  3. Set: export HF_TOKEN='hf_...'"
        )

    try:
        import huggingface_hub  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download GAIA. "
            "Install with: pip install 'agent-framework-eval-harness[gaia]'"
        ) from exc

    logger.info("Downloading GAIA dataset to %s …", data_dir)
    huggingface_hub.snapshot_download(  # nosec B615
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        revision="682dd723ee1e1697e00360edccf2366dc8418dd9",  # pinned for reproducibility
        token=token,
        local_dir=str(data_dir),
        force_download=False,
    )


# ── benchmark ─────────────────────────────────────────────────────────────────


@dataclass
class GAIABenchmark:
    """GAIA benchmark adapter.

    Loads GAIA tasks from HuggingFace (or a local cache), runs the agent on
    each task in parallel, scores with the official GAIA exact-match scorer,
    and returns :class:`~agent_framework.EvalResults`.

    The agent is always constructed by the caller.

    Example:

    .. code-block:: python

        from agent_framework_eval_harness import EvalHarness
        from agent_framework_eval_harness.benchmarks import GAIABenchmark

        harness = EvalHarness(agent=my_agent)
        results = await harness.run(GAIABenchmark(level=1, max_tasks=20))
        harness.print_summary(results)

    Attributes:
        level: GAIA level(s) to run.  1 = simple tool use, 3 = complex
            multi-hop.  Defaults to ``1``.
        max_tasks: Cap on tasks per run.  ``None`` = all tasks in the level.
        skip_file_attachments: Skip ~30% of L1 tasks that include file
            attachments.  Defaults to ``True`` (text-only first pass).
        parallel: Maximum concurrent agent calls.  Defaults to ``1``.
        timeout: Per-task timeout in seconds.  ``None`` = no timeout.
            Defaults to ``300`` (5 min).
        data_dir: Local directory for caching the dataset.  Defaults to a
            system temp directory.
        hf_token: HuggingFace token for downloading.  Falls back to the
            ``HF_TOKEN`` environment variable.
    """

    level: int | list[int] = 1
    max_tasks: int | None = None
    skip_file_attachments: bool = True
    parallel: int = 1
    timeout: float | None = 300.0
    data_dir: str | None = None
    hf_token: str | None = None
    answer_extractor: Callable[[str], str] | None = None
    """Optional function that transforms the agent response before scoring.

    When provided, called with ``item.response`` (the full agent response text)
    and should return a short extracted answer string.  Useful for agents that
    embed the answer in a structured format such as ``"FINAL ANSWER: Paris"``.

    When ``None`` (default), ``item.response`` is passed to ``gaia_scorer``
    directly — suitable for agents that return only the answer.
    """
    verbose: bool = False
    """When True, print per-task diagnostics: question, extracted answer,
    expected answer, and pass/fail.  Useful for debugging extraction failures."""
    seed: int | None = 0
    """Random seed for task shuffling.  Defaults to ``0`` for reproducible
    development runs.  Set to ``None`` for a fresh random draw each time."""
    results_file: str | None = None
    """When set, save per-task results as JSON Lines to this path.  Each line
    contains: task_id, question, expected, extracted, passed, level.
    Useful for offline failure analysis without re-running the benchmark."""
    response_reformulator: Any = None
    """Optional async callable ``(question: str, response: AgentResponse) -> AgentResponse``.

    When provided, called once per task *after* ``agent.run()`` returns (after all loop
    iterations complete). The callable receives the original question and the full
    accumulated ``AgentResponse`` (messages from every loop iteration) and should return
    a new ``AgentResponse`` with a clean, precisely formatted final answer appended.

    Use this to implement a "reformulator" pattern: a separate LLM call that reads
    the full research transcript and extracts the answer with precise formatting
    rules, rather than relying on the agent to produce the right format inline.
    See ``make_reformulator()`` in the GAIA eval sample for a reference implementation
    adapted from HuggingFace smolagents.
    """

    name: str = field(default="GAIA", init=False)

    async def run(
        self,
        agent: SupportsAgentRun,
        *,
        evaluators: Sequence[Evaluator] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        """Run the benchmark and return ``EvalResults``.

        Args:
            agent: The agent to evaluate.
            evaluators: Additional evaluators (e.g. ``FoundryEvals``) run
                alongside the built-in GAIA exact-match scorer.
            eval_name: Display name for the run.  Defaults to
                ``"GAIA L{level}"``.

        Returns:
            A list of ``EvalResults`` — the first entry is always the GAIA
            exact-match result; additional evaluator results follow.
        """
        from agent_framework import (
            AgentEvalConverter,
            EvalItem,
            LocalEvaluator,
            Message,
            evaluator,
        )

        levels = [self.level] if isinstance(self.level, int) else list(self.level)
        run_name = eval_name or f"GAIA L{''.join(str(lv) for lv in levels)}"

        data_dir = Path(self.data_dir) if self.data_dir else Path(__import__("tempfile").gettempdir()) / "data_gaia_hub"
        _ensure_data(data_dir, self.hf_token)

        tasks = _load_tasks(
            data_dir,
            levels=levels,
            skip_file_attachments=self.skip_file_attachments,
            max_tasks=self.max_tasks,
            seed=self.seed,
        )
        if not tasks:
            raise RuntimeError(
                f"No GAIA tasks found for level(s) {levels} in {data_dir}. "
                "Check that the dataset downloaded correctly and the level exists."
            )

        logger.info("Running %d GAIA tasks (level=%s, parallel=%d)", len(tasks), levels, self.parallel)
        start = time.monotonic()

        semaphore = asyncio.Semaphore(self.parallel)
        responses: list[AgentResponse[Any] | None] = list(
            await asyncio.gather(*[self._run_one(t.question, agent, semaphore) for t in tasks])
        )

        logger.info("Finished %d GAIA tasks in %.1fs", len(tasks), time.monotonic() - start)

        @evaluator(name="gaia_exact_match")
        def _gaia_exact_match(response: str, expected_output: str) -> bool:
            answer = self.answer_extractor(response) if self.answer_extractor else response
            return gaia_scorer(answer, expected_output)

        items: list[EvalItem] = []
        for task, response in zip(tasks, responses):
            if response is not None:
                item = AgentEvalConverter.to_eval_item(query=task.question, response=response, agent=agent)
            else:
                item = EvalItem(conversation=[Message("user", [task.question]), Message("assistant", [""])])
            item.expected_output = task.answer
            items.append(item)

        all_evaluators: list[Evaluator] = [LocalEvaluator(_gaia_exact_match)]
        if evaluators:
            all_evaluators.extend(evaluators)

        results = [await ev.evaluate(items, eval_name=run_name) for ev in all_evaluators]

        if self.verbose and results:
            self._print_verbose(tasks, items, results[0])

        if self.results_file and results:
            self._save_results(tasks, items, results[0])

        return results

    def _save_results(self, tasks: list[_GAIATask], items: list[Any], results: EvalResults) -> None:
        """Save per-task results to a JSON Lines file for offline analysis."""
        import json
        from pathlib import Path

        item_map = {r.item_id: r for r in results.items}
        path = Path(self.results_file)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for idx, (task, item) in enumerate(zip(tasks, items)):
                extracted = self.answer_extractor(item.response) if self.answer_extractor else item.response
                item_result = item_map.get(str(idx))
                fh.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "task_id": task.task_id,
                            "level": task.level,
                            "question": task.question,
                            "expected": task.answer,
                            "extracted": extracted,
                            "response": item.response[:2000],  # first 2k chars for diagnosis
                            "passed": item_result.is_passed if item_result else False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _print_verbose(self, tasks: list[_GAIATask], items: list[Any], results: EvalResults) -> None:
        """Print per-task diagnostics for debugging extraction failures."""
        item_map = {r.item_id: r for r in results.items}
        print()
        print(f"{'#':<4} {'Q (truncated)':<55} {'Extracted':<25} {'Expected':<25} {'OK'}")
        print("-" * 120)
        for idx, (task, item) in enumerate(zip(tasks, items)):
            extracted = self.answer_extractor(item.response) if self.answer_extractor else item.response
            expected = task.answer
            item_result = item_map.get(str(idx))
            passed = item_result.is_passed if item_result else False
            mark = "+" if passed else "-"
            q = task.question[:54]
            print(f"{idx:<4} {q:<55} {extracted[:24]:<25} {expected[:24]:<25} {mark}")
        print()

    async def _run_one(
        self,
        query: str,
        agent: SupportsAgentRun,
        semaphore: asyncio.Semaphore,
    ) -> AgentResponse[Any] | None:
        """Run the agent on a single query; returns ``None`` on failure."""
        from agent_framework import Message

        # Create a fresh session per task so ToolApprovalMiddleware (included
        # in create_harness_agent by default) has the session state it requires,
        # and parallel tasks don't share conversation history.
        session = agent.create_session() if hasattr(agent, "create_session") else None

        async with semaphore:
            try:
                if self.timeout is not None:
                    response = await asyncio.wait_for(
                        agent.run([Message("user", [query])], session=session),
                        timeout=self.timeout,
                    )
                else:
                    response = await agent.run([Message("user", [query])], session=session)

                # Apply reformulator if provided — runs ONCE after all loop iterations
                # complete, with the full accumulated transcript.  This is adapted from
                # HuggingFace smolagents' prepare_response() pattern.
                if self.response_reformulator is not None and response is not None:
                    try:
                        response = await self.response_reformulator(query, response)
                    except Exception:
                        logger.warning(
                            "Reformulator failed for task: %.80s…", query, exc_info=True
                        )
                return response

            except asyncio.TimeoutError:
                logger.warning("GAIA task timed out (%.0fs): %.80s…", self.timeout, query)
            except Exception:
                logger.warning("GAIA task failed: %.80s…", query, exc_info=True)
        return None
