# Copyright (c) Microsoft. All rights reserved.

"""Stop-policy profile definitions for harness stop-decision behavior."""

from dataclasses import dataclass
from typing import Literal

from ._constants import DEFAULT_STOP_POLICY_PROFILE

StopPolicyProfile = Literal["interactive", "strict_automation"]


@dataclass(frozen=True)
class StopPolicySettings:
    """Resolved stop-policy settings used by StopDecisionExecutor wiring."""

    profile: StopPolicyProfile
    accept_done_after_retries_exhausted: bool


def resolve_stop_policy_settings(profile: str | None) -> StopPolicySettings:
    """Resolve stop-policy profile into concrete stop-decision settings."""
    normalized = profile if profile in ("interactive", "strict_automation") else DEFAULT_STOP_POLICY_PROFILE
    if normalized == "strict_automation":
        return StopPolicySettings(
            profile="strict_automation",
            accept_done_after_retries_exhausted=False,
        )
    return StopPolicySettings(
        profile="interactive",
        accept_done_after_retries_exhausted=True,
    )
