# Copyright (c) Microsoft. All rights reserved.

"""Harness test runner — executes the agent harness with a fixed prompt and
reports structured results for external evaluation.

Sets up the harness identically to devui_harness.py (same CodingTools,
AgentHarness config, task_complete tool) and runs a single prompt through it.
Outputs a JSON report with everything needed to evaluate quality:
response text, files read, files created (with content), tool calls, status,
and turn count.

Usage:
    cd python
    uv run python samples/getting_started/workflows/harness/harness_test_runner.py

Prerequisites:
    - Azure OpenAI configured (AZURE_OPENAI_ENDPOINT env var)
    - Authentication via azure-identity (run `az login`)
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from agent_framework import AgentRunUpdateEvent, WorkflowOutputEvent
from agent_framework._harness import (
    AgentHarness,
    HarnessResult,
    get_task_complete_tool,
)
from agent_framework._harness._compaction import (
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
)
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from coding_tools import CodingTools

# Keep logs quiet so only our structured output matters
logging.basicConfig(level=logging.WARNING)
logging.getLogger("agent_framework").setLevel(logging.WARNING)
# Enable harness debug to see failures
logging.getLogger("agent_framework._harness").setLevel(logging.DEBUG)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SANDBOX = Path(r"C:\Users\bentho\src\alliscode-agent-framework")

PROMPT = (
    "Investigate this repo and find the python based workflow engine. "
    "Research the code and create a detailed architectural design."
)


async def run_test() -> dict:
    """Run the harness once and return a structured result dict."""
    sandbox_dir = SANDBOX.resolve()

    # --- Create tools + agent (mirrors devui_harness.py) ---
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        name="coding-assistant",
        description="A coding assistant that can read/write files and run commands",
        instructions=AGENT_INSTRUCTIONS,
        tools=all_tools,
    )

    harness = AgentHarness(
        agent,
        max_turns=50,
        enable_stall_detection=True,
        stall_threshold=3,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        enable_work_items=True,
        enable_compaction=True,
        compaction_store=InMemoryCompactionStore(),
        artifact_store=InMemoryArtifactStore(),
        summary_cache=InMemorySummaryCache(max_entries=100),
        max_input_tokens=100_000,
        soft_threshold_percent=0.85,
        sandbox_path=str(sandbox_dir),
    )

    # --- Stream and collect everything ---
    response_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[str] = []
    files_read: list[str] = []
    files_written: list[dict] = []
    # Track seen call_ids to deduplicate and take latest (most complete) args
    seen_calls: dict[str, dict] = {}
    turn_count = 0
    final_result: HarnessResult | None = None

    print(f"Running harness with prompt: {PROMPT!r}", file=sys.stderr)
    print("Progress: ", end="", file=sys.stderr, flush=True)

    async for event in harness.run_stream(PROMPT):
        if isinstance(event, AgentRunUpdateEvent) and event.data:
            update = event.data

            # Capture streamed text
            if hasattr(update, "text") and update.text:
                response_parts.append(update.text)
                print(".", end="", file=sys.stderr, flush=True)

            # Capture tool calls from streaming contents (same approach as harness_repl.py)
            if hasattr(update, "contents") and update.contents:
                for content in update.contents:
                    content_type = type(content).__name__
                    if "FunctionCall" in content_type:
                        tool_name = getattr(content, "name", None)
                        tool_args = getattr(content, "arguments", None)
                        call_id = getattr(content, "call_id", None)
                        if tool_name and call_id:
                            was_new = call_id not in seen_calls
                            # Always overwrite — later chunks have more complete args
                            parsed_args = tool_args if isinstance(tool_args, dict) else {}
                            if isinstance(tool_args, str) and tool_args:
                                try:
                                    parsed_args = json.loads(tool_args)
                                except (json.JSONDecodeError, TypeError):
                                    parsed_args = {}
                            seen_calls[call_id] = {"tool": tool_name, "args": parsed_args}
                            if was_new:
                                print(f"[{tool_name}]", end="", file=sys.stderr, flush=True)
                    elif "FunctionResult" in content_type:
                        result_val = getattr(content, "result", "")
                        result_preview = str(result_val)[:200] if result_val else ""
                        tool_results.append(result_preview)

            # Track turns via finish_reason
            if hasattr(update, "finish_reason") and update.finish_reason:
                turn_count += 1

        if isinstance(event, WorkflowOutputEvent) and isinstance(event.data, HarnessResult):
            final_result = event.data

    print(file=sys.stderr)  # newline after dots

    # Build deduplicated tool_calls from seen_calls
    for call_id, call_info in seen_calls.items():
        name = call_info["tool"]
        args = call_info["args"]
        tool_calls.append({"tool": name, "args": args, "call_id": call_id})

    # Count file reads/writes from tool call names (args are unavailable in streaming)
    files_read_count = sum(1 for tc in tool_calls if tc["tool"] == "read_file")
    files_written_count = sum(1 for tc in tool_calls if tc["tool"] == "write_file")
    list_dir_count = sum(1 for tc in tool_calls if tc["tool"] == "list_directory")

    # Scan sandbox for any new .md files the agent created (the deliverable)
    created_file_contents: dict[str, str] = {}
    for f in sandbox_dir.glob("*.md"):
        if f.name in ("README.md", "CODE_OF_CONDUCT.md", "COMMUNITY.md",
                       "CONTRIBUTING.md", "SECURITY.md", "SUPPORT.md",
                       "TRANSPARENCY_FAQ.md"):
            continue  # skip repo files
        rel = str(f.relative_to(sandbox_dir))
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            if len(content) > 100:  # non-trivial file
                files_written.append({"path": rel, "content_length": len(content)})
                created_file_contents[rel] = content
        except Exception:
            pass

    # --- Read any other files the agent may have created ---
    for fw in files_written:
        fpath = sandbox_dir / fw["path"]
        if fw["path"] not in created_file_contents and fpath.exists():
            try:
                created_file_contents[fw["path"]] = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # --- Build report ---
    harness_turn_count = final_result.turn_count if final_result else turn_count
    deliverables = final_result.deliverables if final_result else []
    report = {
        "timestamp": datetime.now().isoformat(),
        "prompt": PROMPT,
        "status": final_result.status.value if final_result else "no_result",
        "stop_reason": final_result.reason.kind if final_result and final_result.reason else "unknown",
        "turn_count": harness_turn_count,
        "response_text": "".join(response_parts),
        "response_text_length": len("".join(response_parts)),
        "tool_calls_count": len(tool_calls),
        "tool_calls_by_type": {
            name: sum(1 for tc in tool_calls if tc["tool"] == name)
            for name in {tc["tool"] for tc in tool_calls}
        },
        "tool_results_count": len(tool_results),
        "read_file_calls": files_read_count,
        "write_file_calls": files_written_count,
        "list_directory_calls": list_dir_count,
        "files_written": files_written,
        "files_written_count": len(files_written),
        "created_file_contents": created_file_contents,
        "deliverables_count": len(deliverables),
        "deliverables": deliverables,
        "agent_instructions": AGENT_INSTRUCTIONS,
    }

    # Include transcript summary if available
    if final_result and final_result.transcript:
        report["transcript_event_types"] = [
            te.event_type for te in final_result.transcript
        ]

    return report


# ---------------------------------------------------------------------------
# Agent instructions — this is the knob we turn between experiments.
# ---------------------------------------------------------------------------
AGENT_INSTRUCTIONS = """\
You are a capable AI coding assistant with access to a local workspace.
You can read and write files, list directories, and run shell commands.
When asked to investigate code, be thorough — read every relevant source file
before drawing conclusions or writing deliverables.
"""


if __name__ == "__main__":
    report = asyncio.run(run_test())

    # Write JSON to stdout for easy piping / reading
    print(json.dumps(report, indent=2))

    # Also save to a file for convenience
    out_path = SANDBOX / "experiment_result.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResult saved to {out_path}", file=sys.stderr)
