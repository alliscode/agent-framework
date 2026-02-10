# Copyright (c) Microsoft. All rights reserved.

"""Just-In-Time instruction injection based on execution state."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class JitContext:
    """Execution state available to JIT conditions."""

    turn: int
    max_turns: int
    tool_usage: dict[str, int]
    work_items_complete: int
    work_items_total: int
    compaction_count: int = 0


@dataclass
class JitInstruction:
    """A conditional instruction that fires when its condition is met."""

    id: str
    instruction: str | Callable[[JitContext], str]
    condition: Callable[[JitContext], bool]
    once: bool = True


DEFAULT_JIT_INSTRUCTIONS: list[JitInstruction] = [
    JitInstruction(
        id="no_reads_after_3_turns",
        instruction=(
            "You have been working for several turns but haven't read any files yet. "
            "Use read_file, grep_files, or explore sub-agents to examine source code "
            "before drawing conclusions."
        ),
        condition=lambda ctx: ctx.turn >= 3 and ctx.tool_usage.get("read_file", 0) == 0,
    ),
    JitInstruction(
        id="serial_reads_suggest_parallel",
        instruction=(
            "You have been reading files one at a time across multiple turns. "
            "If you have more files to investigate, consider using parallel explore "
            "sub-agent calls to research multiple modules simultaneously — it's faster "
            "and keeps your context leaner."
        ),
        condition=lambda ctx: (
            ctx.turn >= 4 and ctx.tool_usage.get("read_file", 0) >= 6 and ctx.tool_usage.get("explore", 0) == 0
        ),
    ),
    JitInstruction(
        id="no_deliverable_after_many_reads",
        instruction=(
            "You have read many files but haven't produced a deliverable yet. "
            "Synthesize your findings — present the deliverable inline in your response "
            "if it's under 5KB, or use write_file for larger content."
        ),
        condition=lambda ctx: (
            ctx.turn >= 8
            and ctx.tool_usage.get("read_file", 0) >= 5
            and ctx.tool_usage.get("write_file", 0) == 0
            and ctx.work_items_complete < ctx.work_items_total
        ),
    ),
    JitInstruction(
        id="approaching_turn_limit",
        instruction=(
            "You are approaching the turn limit. Prioritize completing your "
            "most important remaining work items and call work_complete."
        ),
        condition=lambda ctx: ctx.turn >= int(ctx.max_turns * 0.8),
    ),
    JitInstruction(
        id="all_planning_no_execution",
        instruction=(
            "You have created work items but haven't started executing any of them. "
            "Stop planning and begin working on your first item."
        ),
        condition=lambda ctx: (
            ctx.turn >= 3
            and ctx.work_items_total > 0
            and ctx.work_items_complete == 0
            and ctx.tool_usage.get("work_item_add", 0) > 0
            and ctx.tool_usage.get("read_file", 0) == 0
        ),
    ),
    JitInstruction(
        id="post_compaction_guidance",
        instruction=lambda ctx: (
            f"Context compaction has occurred {ctx.compaction_count} time(s) — earlier file "
            "contents and tool results have been summarized to free space. Your completed "
            "work item artifacts are fully preserved. Continue working through your remaining "
            "work items. If you need specific details from a file you already read, you may "
            "re-read the targeted section — but prefer referencing your stored artifacts "
            "when they contain the information you need."
        ),
        condition=lambda ctx: ctx.compaction_count >= 1,
        once=True,
    ),
    JitInstruction(
        id="repeated_compaction_warning",
        instruction=(
            "Context has been compacted multiple times. To work efficiently within "
            "your context budget: complete one work item at a time (read → produce "
            "artifact → mark done) before starting the next. Already-completed "
            "artifacts survive compaction. Avoid holding many files in context "
            "simultaneously — process them sequentially."
        ),
        condition=lambda ctx: ctx.compaction_count >= 2,
        once=True,
    ),
]


@dataclass
class JitInstructionProcessor:
    """Evaluates JIT instructions and returns those that should fire."""

    instructions: list[JitInstruction] = field(default_factory=lambda: list(DEFAULT_JIT_INSTRUCTIONS))
    _fired: set[str] = field(default_factory=set)

    def evaluate(self, context: JitContext) -> list[str]:
        """Return instruction texts for all conditions that are met."""
        results: list[str] = []
        for jit in self.instructions:
            if jit.once and jit.id in self._fired:
                continue
            try:
                if jit.condition(context):
                    text = jit.instruction if isinstance(jit.instruction, str) else jit.instruction(context)
                    results.append(text)
                    if jit.once:
                        self._fired.add(jit.id)
            except Exception:
                logger.warning("JIT instruction '%s' condition raised an error", jit.id, exc_info=True)
        return results
