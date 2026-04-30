# Copyright (c) Microsoft. All rights reserved.

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_framework import (
    Agent,
    AgentResponse,
    FileSystemPolicy,
    FileSystemTool,
    Message,
)
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

if TYPE_CHECKING:
    from agent_framework import SupportsAgentRun

"""
Demonstration of the built-in FileSystemTool.

This sample shows:
- Sandboxing the agent's filesystem access to a single workspace directory.
- Restricting writes to a sub-path with FileSystemPolicy.write_paths.
- Using the default denylist (blocks `.git`, `.env*`, SSH keys, etc.).
- Wiring up the user-approval flow for destructive operations
  (`fs_delete`, `fs_move`, `fs_rename`), which always require approval.

The sample uses a temporary workspace seeded with a few files so it can run
without a real project.
"""

load_dotenv()


def _seed_workspace(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text(
        "def greet(name):\n    return f'Hello, {name}!'\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Demo Project\n\nA tiny project for the FileSystemTool sample.\n",
        encoding="utf-8",
    )


async def _handle_approvals(query: str, agent: "SupportsAgentRun") -> AgentResponse:
    """Approve every fs_* request interactively (CLI prompt)."""
    result = await agent.run(query)
    while result.user_input_requests:
        new_inputs: list[Any] = [query]
        for req in result.user_input_requests:
            print(
                f"\n[Approval requested] {req.function_call.name}"
                f"\n  args: {req.function_call.arguments}"
            )
            new_inputs.append(Message("assistant", [req]))
            answer = await asyncio.to_thread(input, "Approve? (y/n): ")
            new_inputs.append(
                Message(
                    "user",
                    [req.to_function_approval_response(answer.strip().lower() == "y")],
                )
            )
        result = await agent.run(new_inputs)
    return result


async def main() -> None:
    print("=== FileSystemTool sample ===\n")

    with tempfile.TemporaryDirectory(prefix="fs-tool-demo-") as tmp:
        root = Path(tmp)
        _seed_workspace(root)

        # 1. Configure a security-conservative policy:
        #    - reads anywhere under root (denylist still applies)
        #    - writes restricted to src/ and tests/
        #    - default denylist blocks .git, .env*, SSH keys, etc.
        fs = FileSystemTool(
            root=root,
            policy=FileSystemPolicy(
                write_paths=("src/**", "tests/**"),
            ),
        )

        async with Agent(
            client=OpenAIChatClient(),
            name="FsAgent",
            instructions=(
                "You are a helpful coding assistant. Use the fs_* tools to "
                "inspect and modify files in the workspace. Prefer fs_view "
                "before fs_edit. Ask before deleting anything."
            ),
            tools=fs.as_tools(),
        ) as agent:
            query = (
                "Read src/main.py, then add a docstring to the greet function "
                "explaining what it does. Then list the files in src/."
            )
            print(f"User: {query}\n")
            result = await _handle_approvals(query, agent)
            print(f"\n{agent.name}: {result}\n")

            # Show the resulting file content for verification.
            print("--- src/main.py after edit ---")
            print((root / "src" / "main.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
