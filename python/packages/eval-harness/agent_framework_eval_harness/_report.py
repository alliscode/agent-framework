# Copyright (c) Microsoft. All rights reserved.

"""Result formatting and summary output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_framework import EvalResults


def print_summary(results: list[EvalResults]) -> None:
    """Print a compact Markdown-style summary table for a list of EvalResults.

    Args:
        results: Eval results returned by :meth:`~agent_framework_eval_harness.EvalHarness.run`.
    """
    if not results:
        print("No results.")
        return

    header = f"{'Provider':<24} {'Status':<12} {'Passed':>7} {'Failed':>7} {'Total':>7} {'%':>7}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    for r in results:
        pct = f"{r.passed / r.total * 100:.1f}%" if r.total else "n/a"
        print(f"{r.provider:<24} {r.status:<12} {r.passed:>7} {r.failed:>7} {r.total:>7} {pct:>7}")

    portal_urls = [(r.provider, r.report_url) for r in results if r.report_url]
    if portal_urls:
        print()
        for provider, url in portal_urls:
            print(f"  {provider}: {url}")

    print()

    for r in results:
        if r.per_evaluator:
            print(f"[{r.provider}] per-evaluator breakdown:")
            for name, counts in r.per_evaluator.items():
                p = counts.get("passed", 0)
                f = counts.get("failed", 0)
                t = p + f
                pct_str = f"{p / t * 100:.1f}%" if t else "n/a"
                print(f"  {name:<30} {p}/{t} ({pct_str})")
            print()
