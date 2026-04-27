---
status: proposed
contact: agent-framework maintainers
date: 2026-04-24
deciders: agent-framework maintainers
consulted: CodeAct / Hyperlight authors
informed: Python and .NET SDK consumers
---

# 0026. Built-in tools package with a local shell tool

## Context and Problem Statement

Every shell-capable sample in the repository today re-implements its own
`subprocess.run(..., shell=True, timeout=30)` wrapper and passes it to a
provider-specific `get_shell_tool(func=...)` factory. There is no first-party
executor, no default safety policy, no cross-OS shell selection, no
persistent session, and no output truncation. This is duplicative, unsafe,
and behind competitors such as Claude Code, Anthropic's `bash_20250124`,
LangChain's `BashProcess`, Open Interpreter, and AutoGen's
`LocalCommandLineCodeExecutor`.

We need a first-class local shell tool that ships with the framework, runs
on Windows, Linux and macOS, and applies layered safety defaults — while
leaving room for future built-in tools (file I/O, HTTP fetch, etc.) without
awkward package naming.

## Decision Drivers

- Cross-OS support (Windows PowerShell, bash/sh on Unix).
- Parity across Python and .NET.
- Safe-by-default behaviour (approval, denylist, timeout, truncation,
  workdir confinement).
- Alignment with the existing CodeAct / Hyperlight sandbox vocabulary
  introduced in [ADR 0024](0024-codeact-integration.md) — reuse not fork.
- Room to grow: a home for future first-party tools, not a shell-only
  package.

## Considered Options

1. **Add `LocalShellTool` inside the core `agent-framework` package.**
   - **+** Zero extra dependency for users.
   - **−** Core has no subprocess or platform-specific code today; this
     would widen core's surface area and platform matrix.
2. **Ship a standalone `agent-framework-shell` package.**
   - **+** Isolated; easy to reason about.
   - **−** Forces a parallel package for every future built-in tool.
3. **New generic built-in tools package (`agent-framework-tools` in
   Python, `Microsoft.Agents.AI.Tools` in .NET), with shell as the first
   citizen in a `shell` submodule / namespace.** *(Chosen.)*
4. **Defer to CodeAct / Hyperlight sandbox.**
   - **+** Already has approval, file mounts, network allow-list.
   - **−** CodeAct is a *code* execution model (``execute_code``) with
     microVM cost and setup; shell tools serve a different UX (terminal
     agent, iterative command execution). Not a drop-in replacement.

## Decision Outcome

**Option 3**: introduce `agent-framework-tools` / `Microsoft.Agents.AI.Tools`,
and ship `LocalShellTool` as its first member.

Concrete shape:

- **Modes**: `persistent` (default) and `stateless`. Persistent matches the
  semantics of Anthropic `bash_20250124` and Claude Code's Bash tool — a
  single long-lived shell where `cd`, `export`, and shell functions carry
  across invocations. Stateless matches AutoGen and OpenAI Agents SDK's
  `local_shell` protocol.
- **Cross-OS shell resolution**: `pwsh` → `powershell.exe` on Windows;
  `/bin/bash --noprofile --norc` → `/bin/sh` on Unix. Overridable via
  constructor argument or the `AGENT_FRAMEWORK_SHELL` env var.
- **Persistent-session protocol**: a sentinel printed after each command
  delimits boundaries on plain pipes (no PTY, no pexpect). PowerShell
  emits the sentinel via `[Console]::WriteLine` + explicit flush to dodge
  pipeline buffering; POSIX shells use `printf` after a grouped command.
  The PowerShell command body is base64-encoded and evaluated via
  `Invoke-Expression` so arbitrary multi-line commands work with the
  single-line wrapper required by `pwsh -Command -`.
- **Safety defaults**: approval required, conservative denylist
  (`rm -rf /`, `mkfs`, `dd`, `shutdown`, `curl|sh`, registry/format/fork
  bomb variants), allowlist support, workdir confinement (re-anchors
  `cd` per call), 30 s timeout with process-tree kill, 64 KiB output
  truncation, audit callback.
- **Integration**: `LocalShellTool.as_function()` returns a
  `FunctionTool` with `kind="shell"` so existing provider
  `get_shell_tool(func=…)` factories (OpenAI, Anthropic, GitHub Copilot)
  recognise and advertise it as a shell tool.

### Relationship to CodeAct / Hyperlight

`LocalShellTool` runs **directly on the host** and is deliberately
complementary to `HyperlightCodeActProvider`:

| Tool | Sandbox | Use case |
|---|---|---|
| `HyperlightCodeActProvider.execute_code` | microVM (Hyperlight) | Running untrusted or generated *code*. |
| `LocalShellTool` | Host | Terminal-agent workflows, dev-loop tooling, trusted users. |

We reuse CodeAct's capability vocabulary where it makes sense (approval
semantics). Future work (see follow-up) proposes a `HyperlightShellExecutor`
that routes commands through the same microVM.

### Implementation notes (earned from review)

A rubber-duck critique of the v1 implementation surfaced several
persistent-mode hazards that now have explicit fixes + regression tests:

- **PowerShell rc accuracy.** `$LASTEXITCODE` only tracks native-process
  exits. The wrapper sets `$ErrorActionPreference = 'Stop'`, wraps the
  `Invoke-Expression` in `try/catch/finally`, and derives rc from
  `$LASTEXITCODE`, `$?`, and caught exceptions so cmdlet failures surface.
- **stderr demux.** Per-call `wait_for(stream.read())` loops are brittle
  and can corrupt the session (late stderr attributed to the *next*
  command; `read() called while another coroutine is already waiting`
  errors). The session runs a **single persistent reader task per
  stream** into shared buffers; each `run()` tracks per-call offsets and
  quiesces briefly after the sentinel to capture trailing stderr.
- **Startup race.** A dedicated lifecycle lock guards `start()`/`close()`
  so concurrent first-callers cannot spawn multiple subprocesses.
- **Windows tree-kill.** Timeout recovery invokes `taskkill /T /F /PID`
  to walk the process tree; the previous `proc.kill()` only terminated
  the top-level shell.
- **PowerShell UTF-8.** Both stateless and persistent flows prepend a
  `$OutputEncoding = [Console]::OutputEncoding = UTF8` preamble so
  `powershell.exe` fallback on Windows does not mojibake non-ASCII.
- **Honest `confine_workdir` semantics.** The flag is a per-call
  **re-anchor**, not a hard confinement — `cd /tmp && ...` in a single
  command still wanders. Documented explicitly; callers who need true
  confinement are pointed at CodeAct / future `HyperlightShellExecutor`.

### Threat model

Stating the threat model explicitly so users can reason about residual risk
rather than relying on the word "safe":

**Assets**
- Files reachable from the agent process's effective user.
- Network access from the host.
- Credentials in the host environment / config / keychain.
- The host itself (uptime, integrity).

**Attacker model**
- A model that has been prompt-injected via tool output, retrieved
  documents, or adversarial user input. The model emits arbitrary commands
  to the shell tool.
- An end-user who has an interactive console and may approve or reject
  each command.

**Defenses (in priority order)**

1. **Approval-in-the-loop** (`approval_mode="always_require"`, default).
   Each command surfaces as a `user_input_request`; nothing runs without
   human consent. **This is the only real boundary.** Disabling it
   requires `acknowledge_unsafe=True` to make the risk explicit.
2. **Operator trust / sandbox tier.** `LocalShellTool` is for
   trusted-developer workflows. Untrusted input belongs in
   `HyperlightCodeActProvider` (microVM) or a container.
3. **Process-tree termination on timeout.** Delegated to `psutil` so
   children (`make`, watchers, network tools) cannot survive a timeout.
4. **Output truncation.** Bounded blast radius for runaway commands.
5. **Audit hook.** `on_command` fires for every command that passes
   policy, suitable for SIEM / append-only logs.

**Explicit non-defenses (residual risk classes)**

- The denylist is a **guardrail**, not a boundary. Documented bypasses:
  backslash insertion (`r''m -rf /`), variable expansion
  (`${RM:=rm} -rf /`), interpreter escape (`python -c "…os.system(…)"`),
  base64 smuggling, `eval $(…)`, alternative tools (`find / -delete`),
  PowerShell-native verbs (`Remove-Item -Recurse -Force`). See
  `tests/test_security.py::test_known_denylist_bypasses` — those tests
  *assert the bypasses succeed* to keep the residual risk visible in code.
- `confine_workdir` is a per-call re-anchor, not a sandbox. A single
  command can still `cd /tmp && rm *`.
- Network egress is not restricted.
- Environment-variable leakage to children is not blocked.
- `AGENT_FRAMEWORK_SHELL` env var lets a caller redirect which shell binary
  is launched. An attacker who can set env vars can already do worse.

### Confidence strategy

We do not claim to have eliminated risk. We claim to have:

- A documented threat model (this section).
- Test coverage that **encodes both defenses and non-defenses** so
  regressions in either direction become visible.
- Delegation to a battle-tested library (`psutil`) for the
  security-critical lifecycle primitive.
- No reliance on per-OS hand-rolled flag manipulation for things where a
  canonical implementation exists.

Anything claiming "production-ready for untrusted input" requires a
formal external security review and a sandboxed executor. Today,
`LocalShellTool` is not that thing — `HyperlightShellExecutor` (planned)
will be.

### Non-goals for v1




- Streaming stdout (command-level streaming of partial output). The
  `ShellResult` contract is additive so a future async iterator can be
  added without breaking changes.
- Full containerisation / seccomp / AppArmor. Callers who need that
  should use CodeAct (code) or wait for `HyperlightShellExecutor` (shell).
- Background / long-running shells à la Claude Code's `BashOutput` /
  `KillShell`. Persistent mode covers stateful *foreground* execution
  only.

### Follow-ups

- `HyperlightShellExecutor` ADR (v2). Reuses `FileMount` and
  `AllowedDomain` from `agent_framework_hyperlight._types`; does **not**
  define parallel types.
- `.NET` package parity: `Microsoft.Agents.AI.Tools.Shell.LocalShellTool`
  implementing the same API surface with `System.Diagnostics.Process`.

## More Information

- Python implementation: `python/packages/tools/agent_framework_tools/shell/`
- Quickstart: `python/packages/tools/README.md`
- Samples: `python/samples/02-agents/providers/*/client_with_local_shell.py`
  and `anthropic_with_shell.py` are refreshed to use `LocalShellTool()`.
