# Copyright (c) Microsoft. All rights reserved.

"""Built-in work completion tool for signaling the agent has finished."""

from typing import Annotated, Callable

from .._tools import ai_function

# Special tool name that the harness recognizes
WORK_COMPLETE_TOOL_NAME = "work_complete"


@ai_function(name=WORK_COMPLETE_TOOL_NAME, approval_mode="never_require")
def work_complete(
    summary: Annotated[str, "Concise description of the outcome and any artifacts produced"],
) -> str:
    """Indicate that all requested work is done and no further actions remain.

    Invoke this once you have finished every outstanding item and verified the
    results. The summary should capture what was delivered.

    Args:
        summary: Concise description of the outcome and any artifacts produced.

    Returns:
        Acknowledgment that the work has been recorded as complete.
    """
    return f"Work recorded as complete: {summary}"


def get_work_complete_tool() -> Callable[..., str]:
    """Get the work_complete tool for adding to an agent's tools.

    Returns:
        The work_complete tool function.
    """
    return work_complete


# Deprecated aliases — use work_complete instead
task_complete = work_complete
TASK_COMPLETE_TOOL_NAME = WORK_COMPLETE_TOOL_NAME
get_task_complete_tool = get_work_complete_tool
