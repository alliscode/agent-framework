# Copyright (c) Microsoft. All rights reserved.

"""τ²-bench benchmark adapter.

τ²-bench is a simulation framework for evaluating customer service agents.
A user simulator drives multi-turn conversations against the agent under test;
results are scored by τ²'s pass/fail reward function.

Published comparison scores (airline domain, success rate):
  - gpt-5:        62%
  - gpt-4.1:      60%
  - gpt-4o:       42%

Requirements:
  pip install "agent-framework-lab[tau2]"
  pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench@5ba9e3e56db57c5e4114bf7f901291f09b2c5619"

Reference:
  https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_framework import EvalResults, Evaluator, SupportsAgentRun

logger = logging.getLogger(__name__)

_TAU2_INSTALL_HINT = (
    "tau2-bench is required to run TauBenchmark.\n"
    "Install with:\n"
    "  pip install 'agent-framework-lab[tau2]'\n"
    "  pip install 'tau2 @ git+https://github.com/sierra-research/tau2-bench"
    "@5ba9e3e56db57c5e4114bf7f901291f09b2c5619'"
)


@dataclass
class TauBenchmark:
    """τ²-bench benchmark adapter.

    Runs a multi-turn customer service simulation between the *agent* (as the
    assistant) and a τ²-bench user simulator.  Results are aggregated as
    pass/fail :class:`~agent_framework.EvalResults`.

    The agent is always constructed by the caller.  The assistant chat client
    is extracted from ``agent.client``; pass ``user_client`` explicitly when
    a separate model is preferred for the user simulator (recommended —
    separating simulator and agent models gives more controlled results).

    Requires:
        ``pip install 'agent-framework-lab[tau2]'``
        ``pip install 'tau2 @ git+https://github.com/sierra-research/tau2-bench@...'``

    Example:

    .. code-block:: python

        from agent_framework_eval_harness import EvalHarness
        from agent_framework_eval_harness.benchmarks import TauBenchmark
        from agent_framework.openai import OpenAIChatClient

        user_client = OpenAIChatClient(model="gpt-4o-mini")
        harness = EvalHarness(agent=my_agent)
        results = await harness.run(
            TauBenchmark(domain="airline", user_client=user_client, max_tasks=50),
        )
        harness.print_summary(results)

    Attributes:
        domain: τ²-bench domain.  Only ``"airline"`` is currently supported.
        user_client: Chat client for the user simulator.  If ``None``, falls
            back to ``agent.client`` (same model for both — less controlled).
        max_tasks: Cap on tasks to run.  ``None`` = all tasks in the domain.
        max_steps: Maximum conversation turns.  Defaults to ``50``.
        parallel: Maximum concurrent task runs.  Defaults to ``1``.
    """

    domain: str = "airline"
    user_client: Any = None
    max_tasks: int | None = None
    max_steps: int = 50
    parallel: int = 1

    name: str = field(default="tau2", init=False)

    async def run(
        self,
        agent: SupportsAgentRun,
        *,
        evaluators: Sequence[Evaluator] | None = None,
        eval_name: str | None = None,
    ) -> list[EvalResults]:
        """Run the τ²-bench simulation and return ``EvalResults``.

        Args:
            agent: The agent to evaluate (used as the assistant).  Must have
                a ``.client`` attribute (i.e. an ``Agent`` instance).
            evaluators: Additional evaluators beyond τ²'s built-in pass/fail.
            eval_name: Display name for the run.

        Returns:
            A list of ``EvalResults`` — first entry is τ²-bench pass/fail;
            additional evaluators follow in order.
        """
        try:
            from agent_framework_lab_tau2 import TaskRunner
        except ImportError as exc:
            raise ImportError(_TAU2_INSTALL_HINT) from exc

        if self.domain != "airline":
            raise ValueError(f"Only the 'airline' domain is currently supported; got {self.domain!r}.")

        try:
            from tau2.domains.airline.environment import get_tasks  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(_TAU2_INSTALL_HINT) from exc

        from agent_framework import (
            EvalItem,
            EvalItemResult,
            EvalResults,
            EvalScoreResult,
            Message,
        )

        run_name = eval_name or f"tau2-{self.domain}"

        all_tasks = get_tasks()
        tasks = all_tasks[: self.max_tasks] if self.max_tasks is not None else all_tasks

        assistant_client = getattr(agent, "client", None)
        if assistant_client is None:
            raise ValueError(
                "TauBenchmark requires an agent with a .client attribute "
                "(i.e. an Agent instance, not a raw SupportsAgentRun)."
            )
        user_client = self.user_client or assistant_client

        logger.info(
            "Running %d tau2 tasks (domain=%s, parallel=%d, max_steps=%d)",
            len(tasks),
            self.domain,
            self.parallel,
            self.max_steps,
        )

        runner = TaskRunner(max_steps=self.max_steps)
        semaphore = asyncio.Semaphore(self.parallel)

        raw: list[tuple[float | None, str | None, list[Any]]] = list(
            await asyncio.gather(*[
                self._run_task(task, runner, assistant_client, user_client, semaphore) for task in tasks
            ])
        )

        # Build EvalItemResults from tau2 rewards
        result_items: list[EvalItemResult] = []
        passed = 0
        failed = 0
        errored = 0

        for task, (reward, error, _msgs) in zip(tasks, raw):
            if error:
                status = "error"
                errored += 1
            elif reward is not None and reward > 0.5:
                status = "pass"
                passed += 1
            else:
                status = "fail"
                failed += 1

            result_items.append(
                EvalItemResult(
                    item_id=str(getattr(task, "task_id", id(task))),
                    status=status,
                    scores=[
                        EvalScoreResult(
                            name="tau2_reward",
                            score=reward if reward is not None else 0.0,
                            passed=status == "pass",
                        )
                    ],
                    error_message=error,
                )
            )

        tau2_results = EvalResults(
            provider="tau2",
            eval_id=run_name,
            run_id=run_name,
            status="completed",
            result_counts={"passed": passed, "failed": failed, "errored": errored},
            items=result_items,
        )
        all_results: list[EvalResults] = [tau2_results]

        # Run additional evaluators on the conversation EvalItems
        if evaluators:
            eval_items: list[EvalItem] = []
            for task, (_reward, _err, msgs) in zip(tasks, raw):
                if msgs:
                    eval_items.append(EvalItem(conversation=msgs))
                else:
                    q = getattr(task, "instruction", str(task))
                    eval_items.append(EvalItem(conversation=[Message("user", [q]), Message("assistant", [""])]))
            for ev in evaluators:
                all_results.append(await ev.evaluate(eval_items, eval_name=run_name))

        return all_results

    async def _run_task(
        self,
        task: Any,
        runner: Any,
        assistant_client: Any,
        user_client: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[float | None, str | None, list[Any]]:
        """Run one tau2 task.  Returns ``(reward, error_message, messages)``."""
        async with semaphore:
            try:
                from agent_framework import Message as _Msg

                conversation = await runner.run(task, assistant_client, user_client)
                reward = runner.evaluate(task, conversation, runner.termination_reason)

                # Convert raw conversation to agent_framework Messages for EvalItems
                messages: list[Any] = []
                for msg in conversation or []:
                    if isinstance(msg, dict):
                        messages.append(_Msg(msg.get("role", "user"), [msg.get("content", "")]))

                return float(reward) if reward is not None else 0.0, None, messages
            except Exception:
                logger.warning("tau2 task failed", exc_info=True)
                return None, "task raised an exception", []
