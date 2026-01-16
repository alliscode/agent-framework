# Copyright (c) Microsoft. All rights reserved.

"""Built-in task completion tool for signaling agent is done."""

from typing import Annotated, Callable

from .._tools import ai_function

# Special tool name that the harness recognizes
TASK_COMPLETE_TOOL_NAME = "task_complete"


@ai_function(name=TASK_COMPLETE_TOOL_NAME, approval_mode="never_require")
def task_complete(
    summary: Annotated[str, "Brief summary of what was accomplished"],
) -> str:
    """Signal that the current task is complete.

    Call this tool when you have finished the user's request and have no more
    actions to take. Provide a brief summary of what was accomplished.

    Args:
        summary: Brief summary of what was accomplished.

    Returns:
        Confirmation message.
    """
    return f"Task marked complete: {summary}"


def get_task_complete_tool() -> Callable[..., str]:
    """Get the task_complete tool for adding to an agent's tools.

    Returns:
        The task_complete tool function.
    """
    return task_complete
