# Copyright (c) Microsoft. All rights reserved.

"""Constants for Agent Harness shared state keys and defaults."""

# SharedState key constants for harness state
HARNESS_TRANSCRIPT_KEY = "harness.transcript"
HARNESS_TURN_COUNT_KEY = "harness.turn_count"
HARNESS_MAX_TURNS_KEY = "harness.max_turns"
HARNESS_STATUS_KEY = "harness.status"
HARNESS_STOP_REASON_KEY = "harness.stop_reason"
HARNESS_PENDING_TOOL_CALLS_KEY = "harness.pending_tool_calls"

# Context pressure state keys (Phase 2)
HARNESS_TOKEN_BUDGET_KEY = "harness.token_budget"
HARNESS_CONTEXT_EDIT_HISTORY_KEY = "harness.context_edit_history"

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

# Default values
DEFAULT_MAX_TURNS = 50
DEFAULT_MAX_INPUT_TOKENS = 100000
DEFAULT_SOFT_THRESHOLD_PERCENT = 0.85
DEFAULT_STALL_THRESHOLD = 3  # Turns without progress before stall detection
