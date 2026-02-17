# Copyright (c) Microsoft. All rights reserved.

"""Shared owner-mode contracts/helpers for harness compaction flow."""

from __future__ import annotations

from typing import Any, Literal, cast

from ._constants import DEFAULT_COMPACTION_OWNER_MODE

CompactionOwnerMode = Literal["agent_turn", "compaction_executor", "shadow"]
VALID_COMPACTION_OWNER_MODES: tuple[CompactionOwnerMode, ...] = ("agent_turn", "compaction_executor", "shadow")


def normalize_compaction_owner_mode(
    value: Any,
    *,
    default: CompactionOwnerMode = cast("CompactionOwnerMode", DEFAULT_COMPACTION_OWNER_MODE),
) -> CompactionOwnerMode:
    """Return a validated owner mode or a safe default."""
    if isinstance(value, str) and value in VALID_COMPACTION_OWNER_MODES:
        return cast("CompactionOwnerMode", value)
    return default


def is_valid_compaction_owner_mode(value: Any) -> bool:
    """Check whether a value is a recognized compaction owner mode."""
    return isinstance(value, str) and value in VALID_COMPACTION_OWNER_MODES
