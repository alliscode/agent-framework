# Copyright (c) Microsoft. All rights reserved.

"""Evaluate Agent Harness output for anti-double-emission and role classification.

This script runs the harness with work items enabled, then analyzes the output
to check whether the agent properly:
- Classifies work items with artifact roles (deliverable/working/control)
- Avoids reproducing artifact content in response text
- Avoids mentioning internal tool names or mechanics in response text
- Uses high-level narrative style in response text

Usage:
    python evaluate_harness_output.py --prompt "Your task here..."
    python evaluate_harness_output.py --prompt-file prompt.txt
    python evaluate_harness_output.py  # uses built-in test prompt
    python evaluate_harness_output.py --rich-display  # demo mode with artifact rendering
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

from agent_framework import AgentRunResponseUpdate, AgentRunUpdateEvent, WorkflowOutputEvent
from agent_framework._harness import (
    AgentHarness,
    HarnessLifecycleEvent,
    HarnessResult,
    get_task_complete_tool,
)
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

# Patterns that indicate meta-references to internal mechanics
META_REFERENCE_PATTERNS = [
    r"\bartifact\b",
    r"\bwork.?item\b",
    r"\bwork_item_\w+\b",
    r"\btask_complete\b",
    r"\bset_artifact\b",
    r"\bstored?.*(as|the)\s+(an?\s+)?artifact\b",
    r"\bsaved?.*(as|the)\s+(an?\s+)?artifact\b",
    r"\bi('ll|.will)\s+(now\s+)?(store|save|record)\s+(this|the|these)\b",
    r"\bi('ve|.have)\s+(now\s+)?(stored|saved|recorded)\b",
    r"\bflag.?revision\b",
    r"\bledger\b",
    r"\btool\s*call\b",
]

# Tool names the agent should never mention
TOOL_NAMES = [
    "work_item_add",
    "work_item_update",
    "work_item_list",
    "work_item_set_artifact",
    "work_item_flag_revision",
    "task_complete",
]


# ── Rich Display Helpers ─────────────────────────────────────────────────────

# ANSI color codes
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
BOX_H = "\u2500"
BOX_V = "\u2502"
BOX_TL = "\u250c"
BOX_TR = "\u2510"
BOX_BL = "\u2514"
BOX_BR = "\u2518"

# Activity verbs shown instead of "turn N" - cycled through for variety
ACTIVITY_VERBS = [
    "reasoning",
    "analyzing",
    "synthesizing",
    "drafting",
    "evaluating",
    "connecting",
    "composing",
    "reviewing",
    "considering",
    "structuring",
]


def word_wrap(text: str, width: int) -> list[str]:
    """Wrap text at word boundaries to fit within width."""
    words = text.split(" ")
    lines: list[str] = []
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + 1 + len(word) <= width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines if lines else [""]


def render_progress_bar(done: int, total: int) -> None:
    """Show work item completion progress."""
    if total == 0:
        return
    bar_width = 30
    progress = done / total
    filled = int(bar_width * progress)
    bar = f"{chr(9608) * filled}{chr(9617) * (bar_width - filled)}"
    pct = int(progress * 100)
    print(f"{DIM}  [{bar}] {done}/{total} items complete ({pct}%){RESET}")


def render_deliverable_block(item_id: str, title: str, content: str) -> None:
    """Render a full deliverable artifact in a formatted block."""
    width = 74
    content_width = width - 4  # 2 for border + 2 for padding

    header = f" {title} "
    pad_total = width - len(header) - 2  # -2 for corners
    pad_left = max(pad_total // 2, 0)
    pad_right = max(pad_total - pad_left, 0)

    print(f"\n{GREEN}{BOX_TL}{BOX_H * pad_left}{header}{BOX_H * pad_right}{BOX_TR}{RESET}")

    for raw_line in content.split("\n"):
        if not raw_line.strip():
            print(f"{GREEN}{BOX_V}{RESET}{'':>{width - 2}}{GREEN}{BOX_V}{RESET}")
            continue
        wrapped = word_wrap(raw_line, content_width)
        for line in wrapped:
            padding = " " * (width - 4 - len(line))
            print(f"{GREEN}{BOX_V}{RESET}  {line}{padding}{GREEN}{BOX_V}{RESET}")

    print(f"{GREEN}{BOX_BL}{BOX_H * (width - 2)}{BOX_BR}{RESET}")
    print(f"{DIM}  [{item_id}]{RESET}")


def render_final_deliverables(deliverables: list[dict]) -> None:
    """Render all deliverables at the end of the run."""
    if not deliverables:
        print(f"\n{YELLOW}  No deliverable artifacts were produced.{RESET}")
        return

    print(f"\n{BOLD}{GREEN}{'=' * 74}{RESET}")
    print(f"{BOLD}{GREEN}  DELIVERABLE ARTIFACTS ({len(deliverables)} items){RESET}")
    print(f"{BOLD}{GREEN}{'=' * 74}{RESET}")

    for d in deliverables:
        render_deliverable_block(d["item_id"], d["title"], d.get("content", ""))

    print(f"\n{GREEN}{'=' * 74}{RESET}\n")


# ── Analysis Functions ───────────────────────────────────────────────────────


def compute_text_overlap(response_text: str, artifact_content: str) -> float:
    """Compute the fraction of artifact content found in response text.

    Uses longest common subsequence ratio to detect content reproduction
    even with minor reformatting.
    """
    if not artifact_content or not response_text:
        return 0.0

    # Check for direct substring matches of significant chunks
    # Split artifact into sentences/lines and check how many appear in response
    artifact_lines = [
        line.strip() for line in artifact_content.split("\n")
        if line.strip() and len(line.strip()) > 20
    ]

    if not artifact_lines:
        return 0.0

    matches = sum(
        1 for line in artifact_lines
        if line in response_text
    )
    return matches / len(artifact_lines)


def find_meta_references(text: str) -> list[dict[str, str]]:
    """Find meta-references to internal mechanics in response text."""
    findings = []
    for pattern in META_REFERENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get surrounding context
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            context = text[start:end].replace("\n", " ")
            findings.append({
                "pattern": pattern,
                "match": match.group(),
                "context": f"...{context}...",
            })

    for tool_name in TOOL_NAMES:
        if tool_name in text:
            idx = text.index(tool_name)
            start = max(0, idx - 40)
            end = min(len(text), idx + len(tool_name) + 40)
            context = text[start:end].replace("\n", " ")
            findings.append({
                "pattern": f"tool_name:{tool_name}",
                "match": tool_name,
                "context": f"...{context}...",
            })

    return findings


def analyze_result(
    response_text: str,
    result: HarnessResult,
    lifecycle_events: list[HarnessLifecycleEvent],
) -> dict:
    """Analyze the harness result for compliance with anti-double-emission rules."""
    analysis = {
        "status": result.status.value,
        "turn_count": result.turn_count,
        "stop_reason": result.reason.kind if result.reason else "unknown",
        "deliverables_count": len(result.deliverables),
        "response_text_length": len(response_text),
        "issues": [],
        "scores": {},
    }

    # --- Score 1: Deliverable Classification ---
    deliverables_updated_events = [
        e for e in lifecycle_events
        if e.event_type == "deliverables_updated"
    ]
    has_deliverables = len(result.deliverables) > 0
    analysis["scores"]["deliverable_classification"] = {
        "has_deliverables": has_deliverables,
        "deliverable_count": len(result.deliverables),
        "lifecycle_events_emitted": len(deliverables_updated_events),
    }
    if not has_deliverables:
        analysis["issues"].append(
            "NO_DELIVERABLES: No items were classified as deliverable. "
            "For a multi-step task, the agent should classify user-facing outputs "
            "with role='deliverable'."
        )

    # --- Score 2: Double Emission ---
    overlap_scores = []
    for d in result.deliverables:
        content = d.get("content", "")
        overlap = compute_text_overlap(response_text, content)
        overlap_scores.append({
            "item_id": d.get("item_id"),
            "title": d.get("title"),
            "content_length": len(content),
            "overlap_ratio": round(overlap, 3),
        })

    max_overlap = max((s["overlap_ratio"] for s in overlap_scores), default=0.0)
    analysis["scores"]["double_emission"] = {
        "max_overlap_ratio": max_overlap,
        "per_deliverable": overlap_scores,
        "pass": max_overlap < 0.3,
    }
    if max_overlap >= 0.3:
        analysis["issues"].append(
            f"DOUBLE_EMISSION: Response text reproduces {max_overlap:.0%} of "
            f"deliverable artifact content. Agent should only describe what it "
            f"produced, not reproduce the content."
        )

    # --- Score 3: Meta-references ---
    meta_refs = find_meta_references(response_text)
    analysis["scores"]["meta_references"] = {
        "count": len(meta_refs),
        "findings": meta_refs[:10],  # cap at 10 for readability
        "pass": len(meta_refs) == 0,
    }
    if meta_refs:
        analysis["issues"].append(
            f"META_REFERENCES: Found {len(meta_refs)} references to internal "
            f"mechanics (tools, artifacts, work items) in response text."
        )

    # --- Score 4: Response Brevity ---
    # High-level narrative should be much shorter than artifact content
    total_artifact_length = sum(len(d.get("content", "")) for d in result.deliverables)
    if total_artifact_length > 0:
        brevity_ratio = len(response_text) / total_artifact_length
        analysis["scores"]["response_brevity"] = {
            "response_length": len(response_text),
            "total_artifact_length": total_artifact_length,
            "ratio": round(brevity_ratio, 2),
            "pass": brevity_ratio < 1.5,
        }
        if brevity_ratio >= 1.5:
            analysis["issues"].append(
                f"VERBOSE_RESPONSE: Response text ({len(response_text)} chars) is "
                f"{brevity_ratio:.1f}x the length of deliverable content "
                f"({total_artifact_length} chars). Agent should be concise."
            )
    else:
        analysis["scores"]["response_brevity"] = {
            "response_length": len(response_text),
            "total_artifact_length": 0,
            "ratio": None,
            "pass": None,
            "note": "No deliverable content to compare against",
        }

    # --- Overall ---
    passing_scores = sum(
        1 for s in analysis["scores"].values()
        if isinstance(s, dict) and s.get("pass") is True
    )
    total_scores = sum(
        1 for s in analysis["scores"].values()
        if isinstance(s, dict) and s.get("pass") is not None
    )
    analysis["overall_pass"] = passing_scores == total_scores
    analysis["pass_rate"] = f"{passing_scores}/{total_scores}"

    return analysis


def print_report(analysis: dict, response_text: str, result: HarnessResult) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("  HARNESS OUTPUT EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Status: {analysis['status']} | Turns: {analysis['turn_count']} | "
          f"Stop: {analysis['stop_reason']}")
    print(f"  Response text: {analysis['response_text_length']} chars")
    print(f"  Deliverables: {analysis['deliverables_count']}")

    print("\n" + "-" * 70)
    print("  SCORES")
    print("-" * 70)

    # Deliverable Classification
    dc = analysis["scores"]["deliverable_classification"]
    status = "PASS" if dc["has_deliverables"] else "FAIL"
    print(f"\n  [{status}] Deliverable Classification")
    print(f"        Items classified as deliverable: {dc['deliverable_count']}")
    print(f"        Lifecycle events emitted: {dc['lifecycle_events_emitted']}")

    # Double Emission
    de = analysis["scores"]["double_emission"]
    status = "PASS" if de["pass"] else "FAIL"
    print(f"\n  [{status}] Anti-Double-Emission (max overlap: {de['max_overlap_ratio']:.0%})")
    for item in de["per_deliverable"]:
        print(f"        [{item['item_id']}] {item['title']}: "
              f"{item['overlap_ratio']:.0%} overlap ({item['content_length']} chars)")

    # Meta-references
    mr = analysis["scores"]["meta_references"]
    status = "PASS" if mr["pass"] else "FAIL"
    print(f"\n  [{status}] No Meta-References ({mr['count']} found)")
    for finding in mr["findings"][:5]:
        print(f"        '{finding['match']}' in: {finding['context']}")

    # Response Brevity
    rb = analysis["scores"]["response_brevity"]
    if rb.get("pass") is not None:
        status = "PASS" if rb["pass"] else "FAIL"
        print(f"\n  [{status}] Response Brevity (ratio: {rb['ratio']}x)")
        print(f"        Response: {rb['response_length']} chars | "
              f"Artifacts: {rb['total_artifact_length']} chars")
    else:
        print(f"\n  [SKIP] Response Brevity - {rb.get('note', 'N/A')}")

    # Issues
    if analysis["issues"]:
        print("\n" + "-" * 70)
        print("  ISSUES")
        print("-" * 70)
        for issue in analysis["issues"]:
            print(f"\n  * {issue}")

    # Overall
    print("\n" + "-" * 70)
    overall = "PASS" if analysis["overall_pass"] else "FAIL"
    print(f"  OVERALL: [{overall}] ({analysis['pass_rate']} checks passed)")
    print("-" * 70)

    # Deliverable previews
    if result.deliverables:
        print("\n" + "-" * 70)
        print("  DELIVERABLE ARTIFACTS")
        print("-" * 70)
        for d in result.deliverables:
            content = d.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"\n  [{d['item_id']}] {d['title']}")
            print(f"  {'~' * 40}")
            for line in preview.split("\n"):
                print(f"    {line}")

    # Response text preview
    print("\n" + "-" * 70)
    print("  RESPONSE TEXT (first 500 chars)")
    print("-" * 70)
    preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
    for line in preview.split("\n"):
        print(f"    {line}")

    print("\n" + "=" * 70)


async def run_evaluation(prompt: str, max_turns: int = 20, rich_display: bool = False) -> dict:
    """Run the harness and evaluate the output.

    Args:
        prompt: The task prompt to send to the agent.
        max_turns: Maximum turns for the harness.
        rich_display: If True, render deliverable progress and content inline.

    Returns:
        Analysis dictionary with scores and issues.
    """
    print(f"Configuring harness (max_turns={max_turns}, work_items=True)...")

    # Create agent with task_complete tool only (no external tools needed)
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        name="evaluation-agent",
        instructions=(
            "You are an expert analyst. Complete the task thoroughly. "
            "Use work items to track each step of your work."
        ),
        tools=[get_task_complete_tool()],
    )

    # Create harness with work items enabled
    harness = AgentHarness(
        agent,
        max_turns=max_turns,
        enable_work_items=True,
        enable_stall_detection=True,
        stall_threshold=4,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
    )

    if rich_display:
        print(f"\n{BOLD}Running harness with prompt ({len(prompt)} chars)...{RESET}")
        print(f"{DIM}{'─' * 70}{RESET}\n")
    else:
        print(f"Running harness with prompt ({len(prompt)} chars)...")
        print("-" * 40)

    # Stream events and collect data
    response_text_parts: list[str] = []
    lifecycle_events: list[HarnessLifecycleEvent] = []
    final_result: HarnessResult | None = None
    turn_texts: list[str] = []
    current_turn_text: list[str] = []
    seen_deliverable_ids: set[str] = set()
    current_turn = 0

    async for event in harness.run_stream(prompt):
        # Capture streaming text
        if isinstance(event, AgentRunUpdateEvent) and event.data:
            update: AgentRunResponseUpdate = event.data
            if hasattr(update, "text") and update.text:
                response_text_parts.append(update.text)
                current_turn_text.append(update.text)
                if rich_display:
                    print(f"{update.text}", end="", flush=True)
                else:
                    print(update.text, end="", flush=True)

        # Capture lifecycle events (HarnessLifecycleEvent extends WorkflowEvent)
        if isinstance(event, HarnessLifecycleEvent):
            lifecycle_events.append(event)

            if rich_display:
                # Show turn starts with an activity verb indicator
                if event.event_type == "turn_started":
                    current_turn = event.data.get("turn_number", current_turn + 1) if event.data else current_turn + 1
                    verb = ACTIVITY_VERBS[(current_turn - 1) % len(ACTIVITY_VERBS)]
                    print(f"\n{DIM}  ● {verb}...{RESET}\n")

                # Show progress bar and render new deliverables inline
                if event.event_type == "deliverables_updated" and event.data:
                    done = event.data.get("done_items", 0)
                    total = event.data.get("total_items", 0)
                    render_progress_bar(done, total)

                    # Render any newly-seen deliverables inline
                    for item in event.data.get("items", []):
                        item_id = item.get("item_id", "")
                        if item_id not in seen_deliverable_ids:
                            seen_deliverable_ids.add(item_id)
                            render_deliverable_block(
                                item_id,
                                item.get("title", "Untitled"),
                                item.get("content", ""),
                            )

        # Capture final result
        if isinstance(event, WorkflowOutputEvent) and isinstance(event.data, HarnessResult):
            final_result = event.data
            if current_turn_text:
                turn_texts.append("".join(current_turn_text))
                current_turn_text = []

        # Track turn boundaries via lifecycle events
        if (isinstance(event, HarnessLifecycleEvent)
                and event.event_type == "turn_completed"
                and current_turn_text):
            turn_texts.append("".join(current_turn_text))
            current_turn_text = []

    if current_turn_text:
        turn_texts.append("".join(current_turn_text))

    if rich_display:
        print(f"\n{DIM}{'─' * 70}{RESET}")
    else:
        print("\n" + "-" * 40)

    if final_result is None:
        print("ERROR: No HarnessResult received!")
        return {"error": "No result", "issues": ["HARNESS_ERROR: No result received"]}

    # Render any deliverables not already shown inline
    if rich_display:
        unseen = [
            d for d in final_result.deliverables
            if d.get("item_id") not in seen_deliverable_ids
        ]
        if unseen:
            render_final_deliverables(unseen)

    # Combine all response text
    response_text = "".join(response_text_parts)

    # Run analysis
    analysis = analyze_result(response_text, final_result, lifecycle_events)
    analysis["turn_texts"] = [
        {"turn": i + 1, "length": len(t), "preview": t[:100]}
        for i, t in enumerate(turn_texts)
    ]

    # Print report (skip in rich display mode unless there are issues)
    if not rich_display or not analysis.get("overall_pass", True):
        print_report(analysis, response_text, final_result)
    elif rich_display:
        # Compact summary in rich mode
        passing = analysis["pass_rate"]
        status = f"{GREEN}PASS{RESET}" if analysis["overall_pass"] else f"{YELLOW}FAIL{RESET}"
        print(f"\n  {BOLD}Evaluation:{RESET} [{status}] ({passing} checks passed)")
        if analysis["issues"]:
            for issue in analysis["issues"]:
                print(f"    {YELLOW}*{RESET} {issue}")
        print()

    return analysis


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Agent Harness output for anti-double-emission compliance"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Task prompt to send to the agent",
    )
    parser.add_argument(
        "--prompt-file", type=str, default=None,
        help="Path to file containing the task prompt",
    )
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum harness turns (default: 20)",
    )
    parser.add_argument(
        "--rich-display", "-r", action="store_true",
        help="Enable rich display mode: show deliverable progress and render artifacts inline",
    )

    args = parser.parse_args()

    # Determine prompt source
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = DEFAULT_PROMPT
        print("Using default test prompt (postmortem scenario).")

    # Run evaluation
    analysis = asyncio.run(run_evaluation(
        prompt, max_turns=args.max_turns, rich_display=args.rich_display,
    ))

    # Exit with non-zero if checks failed
    if not analysis.get("overall_pass", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
