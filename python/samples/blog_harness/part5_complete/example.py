# Copyright (c) Microsoft. All rights reserved.

"""Part 5 Example: Complete Harness in Action.

This script demonstrates a real-world use case: an agent that analyzes
data and produces structured JSON output. The harness ensures:
1. The agent doesn't run forever (turn limits)
2. The agent doesn't get stuck (stall detection)
3. The agent doesn't misuse tools (policies)
4. The agent produces valid output (validators)

Usage:
    python example.py                      # Normal analysis task
    python example.py --validation-test    # Trigger validation retry
    python example.py --show-events        # Show all harness events
"""

import argparse
import asyncio
import json
from datetime import datetime
from typing import Annotated

from agent_framework import ai_function
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import AzureCliCredential
from harness import (
    ContentFilterPolicy,
    CustomValidator,
    HarnessBuilder,
    HarnessEvent,
    HarnessStatus,
    JsonSchemaValidator,
    MaxToolCallsPolicy,
)

# ============================================================
# Simulated Data Analysis Tools
# ============================================================


@ai_function
def get_dataset_info(
    dataset_name: Annotated[str, "Name of the dataset to analyze"],
) -> str:
    """Get information about a dataset."""
    datasets = {
        "sales": {
            "rows": 15000,
            "columns": ["date", "product", "quantity", "price", "region"],
            "date_range": "2023-01-01 to 2023-12-31",
        },
        "customers": {
            "rows": 5000,
            "columns": ["id", "name", "email", "signup_date", "tier"],
            "date_range": "2020-01-01 to 2023-12-31",
        },
        "inventory": {
            "rows": 500,
            "columns": ["product_id", "name", "stock", "reorder_point"],
            "date_range": "N/A",
        },
    }

    if dataset_name.lower() in datasets:
        return json.dumps(datasets[dataset_name.lower()], indent=2)
    return f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}"


@ai_function
def run_analysis(
    dataset_name: Annotated[str, "Name of the dataset"],
    analysis_type: Annotated[str, "Type of analysis: 'summary', 'trends', or 'anomalies'"],
) -> str:
    """Run an analysis on a dataset."""
    # Simulated analysis results
    results = {
        "summary": {
            "sales": "Total revenue: $2.5M, Avg order: $167, Top region: West",
            "customers": "Total customers: 5000, Active: 3200, Avg lifetime: 18 months",
            "inventory": "Total SKUs: 500, Low stock: 45, Overstocked: 12",
        },
        "trends": {
            "sales": "Q4 shows 23% growth YoY, Weekend sales up 15%",
            "customers": "New signups increasing 5% monthly, churn down to 2%",
            "inventory": "Seasonal items trending up, Electronics stable",
        },
        "anomalies": {
            "sales": "Spike on 2023-11-24 (Black Friday), Dip on 2023-07-04",
            "customers": "Unusual signup cluster from region=Unknown on 2023-09-15",
            "inventory": "Product SKU-789 has negative stock (data error)",
        },
    }

    dataset = dataset_name.lower()
    analysis = analysis_type.lower()

    if analysis not in results:
        return f"Unknown analysis type. Choose from: {list(results.keys())}"
    if dataset not in results[analysis]:
        return f"Dataset '{dataset}' not available for {analysis} analysis"

    return results[analysis][dataset]


@ai_function
def generate_report_section(
    title: Annotated[str, "Section title"],
    content: Annotated[str, "Section content"],
) -> str:
    """Generate a formatted report section."""
    return f"## {title}\n\n{content}\n"


@ai_function
def get_current_timestamp() -> str:
    """Get the current timestamp for the report."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Test Tasks
# ============================================================

# Real-world task: analyze data and produce structured output
ANALYSIS_TASK = """
Analyze the 'sales' dataset and produce a JSON report with this exact structure:

```json
{
    "report_title": "string",
    "generated_at": "timestamp string",
    "dataset": "dataset name",
    "findings": [
        {"category": "string", "insight": "string", "importance": "high|medium|low"}
    ],
    "recommendations": ["string"],
    "summary": "string"
}
```

Use the available tools to gather data, then output the JSON.
"""

# Task that will trigger validation retry (incomplete output)
VALIDATION_TEST_TASK = """
Give me a quick summary of the sales data.
Just say "Sales look good" - don't worry about the JSON format.
"""


# ============================================================
# Custom Validators
# ============================================================


def validate_has_findings(response: str) -> tuple[bool, list[str]]:
    """Validate that the response contains meaningful findings."""
    errors = []

    # Try to extract JSON
    try:
        json_match = None
        import re

        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response
        data = json.loads(json_str)

        findings = data.get("findings", [])
        if len(findings) < 2:
            errors.append("Report must include at least 2 findings")

        recommendations = data.get("recommendations", [])
        if len(recommendations) < 1:
            errors.append("Report must include at least 1 recommendation")

    except (json.JSONDecodeError, AttributeError):
        errors.append("Could not parse findings from response")

    return len(errors) == 0, errors


# ============================================================
# Main
# ============================================================


async def main(task: str, show_events: bool) -> None:
    """Run the complete harness example."""
    print("=" * 60)
    print("Part 5: Complete Agent Harness Demo")
    print("=" * 60)
    print(f"\nTask: {task[:100]}...")
    print()

    # Create the agent
    client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = client.create_agent(
        name="data-analyst",
        instructions=(
            "You are a data analyst. Use the provided tools to analyze data "
            "and produce structured reports. Always output results in the "
            "exact JSON format requested."
        ),
        tools=[
            get_dataset_info,
            run_analysis,
            generate_report_section,
            get_current_timestamp,
        ],
    )

    # Build the harness using the fluent builder
    harness = (
        HarnessBuilder(agent)
        .with_max_turns(15)
        .with_stall_detection(threshold=3)
        # Policies
        .with_policy(MaxToolCallsPolicy(max_calls=25))
        .with_policy(
            ContentFilterPolicy(
                patterns=[r"password", r"secret", r"api[_-]?key"],
                stop_on_match=True,
            )
        )
        # Validators - ensure output meets requirements
        .with_validator(
            JsonSchemaValidator(
                required_fields=["report_title", "findings", "summary"],
                field_types={"findings": list, "summary": str},
            )
        )
        .with_validator(CustomValidator("HasFindings", validate_has_findings))
        .with_validation_retries(2)
        .build()
    )

    print("Harness Configuration:")
    print("  - Max turns: 15")
    print("  - Stall threshold: 3")
    print("  - Policies: MaxToolCalls(25), ContentFilter")
    print("  - Validators: JsonSchema, HasFindings")
    print("  - Validation retries: 2")
    print()
    print("-" * 60)
    print("Running harness...\n")

    result = None
    turn_count = 0

    async for event in harness.run_stream(task):
        # Extract HarnessEvent from WorkflowEvent
        if hasattr(event, "data"):
            data = event.data

            # Handle HarnessEvent objects
            if isinstance(data, HarnessEvent):
                if show_events:
                    print(f"  [EVENT] {data.event_type}: {data.data}")

                if data.event_type == "turn_started":
                    turn_count = data.turn
                    print(f"  Turn {turn_count} starting...")

                elif data.event_type == "turn_complete":
                    preview = data.data.get("response_preview", "")[:60]
                    tools = data.data.get("tool_call_count", 0)
                    tool_info = f" [{tools} tools]" if tools else ""
                    print(f"  Turn {turn_count} complete{tool_info}: {preview}...")

                elif data.event_type == "policy_violation":
                    policy = data.data.get("policy", "?")
                    msg = data.data.get("message", "")
                    print(f"  ⚠️  Policy [{policy}]: {msg}")

                elif data.event_type == "validation_failed":
                    errors = data.data.get("errors", [])
                    retries = data.data.get("retry_count", 0)
                    print(f"  ❌ Validation failed (retry {retries}):")
                    for err in errors[:3]:  # Show first 3 errors
                        print(f"      - {err}")

                elif data.event_type == "stall_detected":
                    count = data.data.get("stall_count", 0)
                    print(f"  🔄 Stall detected (count: {count})")

                elif data.event_type == "complete":
                    print("  ✓ Agent completed task")

            # Capture final result
            elif isinstance(data, type(result)) if result else hasattr(data, "status"):
                if hasattr(data, "status"):
                    result = data

    # Print final result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    if result:
        status_icons = {
            HarnessStatus.COMPLETE: "✓",
            HarnessStatus.MAX_TURNS: "⚠",
            HarnessStatus.STALLED: "🔄",
            HarnessStatus.POLICY_VIOLATION: "🚫",
            HarnessStatus.VALIDATION_FAILED: "❌",
            HarnessStatus.ERROR: "✗",
        }
        icon = status_icons.get(result.status, "?")

        print(f"\n{icon} Status: {result.status.value}")
        print(f"  Turns used: {result.turn_count}")

        if result.stall_count > 0:
            print(f"  Stall count: {result.stall_count}")

        if result.policy_violations:
            print("\n  Policy violations:")
            for v in result.policy_violations:
                print(f"    - {v}")

        if result.validation_errors:
            print("\n  Validation errors:")
            for e in result.validation_errors:
                print(f"    - {e}")

        print(f"\n{'─' * 60}")
        print("FINAL RESPONSE:")
        print("─" * 60)
        print(result.final_response)

        # If the response contains JSON, pretty-print it
        if "```json" in result.final_response:
            import re

            json_match = re.search(r"```json\s*(.*?)\s*```", result.final_response, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    print("\n" + "─" * 60)
                    print("PARSED JSON OUTPUT:")
                    print("─" * 60)
                    print(json.dumps(parsed, indent=2))
                except json.JSONDecodeError:
                    pass
    else:
        print("\n✗ No result received!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete harness example")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Custom task",
    )
    parser.add_argument(
        "--validation-test",
        action="store_true",
        help="Run with a task that triggers validation",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Show all harness events",
    )
    args = parser.parse_args()

    # Select task
    if args.task:
        task = args.task
    elif args.validation_test:
        task = VALIDATION_TEST_TASK
    else:
        task = ANALYSIS_TASK

    asyncio.run(main(task, args.show_events))
