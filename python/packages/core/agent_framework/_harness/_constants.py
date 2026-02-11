# Copyright (c) Microsoft. All rights reserved.

"""Constants for Agent Harness shared state keys and defaults."""

# SharedState key constants for harness state
HARNESS_TRANSCRIPT_KEY = "harness.transcript"
HARNESS_TURN_COUNT_KEY = "harness.turn_count"
HARNESS_MAX_TURNS_KEY = "harness.max_turns"
HARNESS_STATUS_KEY = "harness.status"
HARNESS_STOP_REASON_KEY = "harness.stop_reason"
HARNESS_PENDING_TOOL_CALLS_KEY = "harness.pending_tool_calls"

# Token budget state key (used by CompactionExecutor)
HARNESS_TOKEN_BUDGET_KEY = "harness.token_budget"  # noqa: S105  # nosec B105

# Compaction state keys (Phase 9 - Production Compaction)
HARNESS_COMPACTION_PLAN_KEY = "harness.compaction_plan"
HARNESS_COMPACTION_METRICS_KEY = "harness.compaction_metrics"

# Initial message key
HARNESS_INITIAL_MESSAGE_KEY = "harness.initial_message"

# Continuation tracking
HARNESS_CONTINUATION_COUNT_KEY = "harness.continuation_count"

# Task contract state keys (Phase 3)
HARNESS_TASK_CONTRACT_KEY = "harness.task_contract"
HARNESS_COVERAGE_LEDGER_KEY = "harness.coverage_ledger"
HARNESS_PROGRESS_TRACKER_KEY = "harness.progress_tracker"
HARNESS_COMPLETION_REPORT_KEY = "harness.completion_report"

# Work item tracking state keys
HARNESS_WORK_ITEM_LEDGER_KEY = "harness.work_item_ledger"

# Work-complete enforcement tracking
HARNESS_WORK_COMPLETE_RETRY_COUNT_KEY = "harness.work_complete_retry_count"

# Default values
DEFAULT_MAX_TURNS = 50
DEFAULT_MAX_INPUT_TOKENS = 128000
DEFAULT_SOFT_THRESHOLD_PERCENT = 0.80
DEFAULT_BLOCKING_THRESHOLD_PERCENT = 0.95
DEFAULT_STALL_THRESHOLD = 3  # Turns without progress before stall detection
DEFAULT_WORK_COMPLETE_MAX_RETRIES = 3  # Max times to retry when agent doesn't call work_complete
