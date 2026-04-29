# Harness_Step03_Shell

A Harness sample wiring `Microsoft.Agents.AI.Tools.Shell.LocalShellTool` into
the same provider stack used by `Harness_Step01_Research`:

- `TodoProvider` — agent-managed task list.
- `AgentModeProvider` — plan/execute mode switching.
- `FileMemoryProvider` — durable scratch space across compactions.
- `ToolApprovalAgent` — **the security boundary**. Every shell command surfaces
  as a `ToolApprovalRequest` and runs only after the user approves it.

## What this sample tests

It is the first end-to-end consumer of the .NET shell tool, validating that:

1. The Python `LocalShellTool` public API ports cleanly to .NET
   (`ShellResult`, `ShellPolicy`, `LocalShellTool`, `AsAIFunction`).
2. The shell tool is plug-compatible with the Harness providers (no special
   wiring beyond `shell.AsAIFunction()` in `ChatOptions.Tools`).
3. Approval-in-the-loop semantics work as the security boundary — disabling
   approval requires opting out at the harness level, mirroring the Python
   `acknowledge_unsafe` gate.

## Configuration

| Variable | Description |
|---|---|
| `AZURE_AI_PROJECT_ENDPOINT` | Azure AI Foundry project endpoint. Required. |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` | Model deployment. Defaults to `gpt-5.4`. |
| `AGENT_FRAMEWORK_SHELL` | Optional: override shell binary (e.g. `/usr/bin/zsh`). |

## Running

```bash
cd dotnet/samples/02-agents/Harness/Harness_Step03_Shell
dotnet run
```

Try prompts like:
- "Show me the largest files in my home directory."
- "Find all `.cs` files modified in the last day."
- "Tell me the current git status of this repo."

Each command is presented for approval before execution.

## Known v1 gaps (vs Python)

- Only `ShellMode.Stateless` is implemented. Persistent-session mode (sentinel
  protocol, `cd`/`export` carrying across calls) is the immediate follow-up.
- No `DockerShellTool` yet — sandboxed-shell tier is still Python-only.
- No `acknowledgeUnsafe` constructor gate — for now the harness's
  `ToolApprovalAgent` enforces approval; a future `LocalShellTool` v2 will
  add a parameter so callers wiring up a non-harness pipeline still have
  to opt in to disabling approval explicitly.
