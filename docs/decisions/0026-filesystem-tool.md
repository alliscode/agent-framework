---
status: proposed
contact: bentho
date: 2026-04-27
deciders: agent-framework maintainers
---

# Built-in Filesystem Tool

## Context and Problem Statement

`agent-framework` ships several built-in tools (web search, code interpreter,
MCP, skills) but lacks a first-class filesystem tool. A built-in shell tool
is being added in parallel; while a shell can technically run `cat`/`type`,
`grep`/`Select-String`, `sed`/`-replace`, etc., relying on shell for file
operations is unsafe (no sandbox), unportable (POSIX vs PowerShell semantics
diverge sharply), expensive on tokens (unstructured stdout), and fragile for
edits (hard to ensure unique replacements / atomic writes).

Every mature coding-agent stack — Claude Code, Cursor, Cline, Codex CLI,
Continue, GitHub Copilot CLI — ships a typed filesystem tool alongside their
shell tool for these reasons. We need parity, with security as the top
priority.

## Decision Drivers

- **Security first.** Path-traversal, symlink-escape, and over-privileged
  reads/writes must be impossible by construction, not merely discouraged.
- **Cross-OS portability.** Identical behavior on Windows, macOS, and Linux.
- **Token efficiency.** Structured returns the model can consume reliably.
- **Parity with competing frameworks** for safe file inspection and editing.
- **No new heavy dependencies on `agent-framework-core`.**
- **Cross-language consistency.** API shape should be portable to .NET.

## Considered Options

1. **Rely on the shell tool only.** Document recipes (`cat`, `Get-Content`,
   `sed`, `grep`, etc.).
2. **Recommend an external MCP filesystem server.**
3. **Build a dedicated, sandboxed `FileSystemTool` in core (this proposal).**

## Decision Outcome

Chosen option: **3 — built-in `FileSystemTool` in core**, gated as
`ExperimentalFeature.FILESYSTEM`.

It is the only option that delivers (a) a sandbox enforced in our process
(critical for security since the agent's tool calls cannot bypass it),
(b) cross-OS structured results, and (c) zero per-invocation overhead.
Options 1 and 2 fall short on at least one of these.

### Tool Surface

Six non-destructive ops + three destructive ops, all returning
:class:`TypedDict` results:

| Op | Approval | Purpose |
| --- | --- | --- |
| `fs_view` | never_require | Read a UTF-8 text file with line numbers and optional `view_range`. |
| `fs_create` | never_require | Create a new file (fails if it exists). |
| `fs_edit` | never_require | Unique-match `old_str` → `new_str` (atomic). |
| `fs_multi_edit` | never_require | Atomic batch of unique-match edits applied sequentially to one file. |
| `fs_glob` | never_require | POSIX-style glob with `**` support. |
| `fs_grep` | never_require | Regex search; ripgrep when present, Python fallback otherwise. |
| `fs_list_dir` | never_require | Bounded-depth directory listing. |
| `fs_delete` | **always_require** | Delete a single file (no directories, no symlinks). |
| `fs_move` | **always_require** | Move within the workspace. |
| `fs_rename` | **always_require** | Rename within the workspace. |

The destructive ops are exposed with `approval_mode="always_require"` and
this is **not configurable** — hosts must wire up the agent approval flow
(ADR-0006) to use them.

### Security Model

1. **Mandatory workspace root.** `FileSystemTool(root=...)` is required;
   there is no implicit "current working directory". All paths resolve
   against `root`, and any path that escapes it is rejected.
2. **Symlink discipline.** Any intermediate symlink under `root` causes the
   op to fail. Reuses the same `_path_security` helpers used by skills.
3. **Default denylist.** Patterns blocking `.git`, `.env*`, `*.pem`,
   `*.key`, SSH keys, `.aws/credentials`, `.npmrc`, `.pypirc`, `.netrc`
   are applied even inside `root`. Configurable via
   `FileSystemPolicy(denylist=...)`.
4. **Read/write split.** Optional `read_paths` and `write_paths` allowlists
   on `FileSystemPolicy`. Both default to "anywhere under root, except
   denylist".
5. **Size and result limits.** `max_file_bytes` (5 MiB default),
   `max_results` (1000 default), `max_view_lines` (2000 default).
6. **Atomic writes.** `tempfile.mkstemp` + `os.fsync` + `os.replace` in
   the same directory.
7. **Encoding.** UTF-8 strict; binary files (NUL byte sniff) are refused.
8. **Edit safety.** `fs_edit` requires the match count to equal `1` (or
   an explicit `count`); ambiguous edits fail loudly rather than guessing.
9. **Audit logging.** Every successful op logs at INFO; destructive ops at
   WARNING; denials at WARNING.
10. **Defense in depth on grep.** When ripgrep is invoked, we pass
    `--no-config` to ignore user/global rg config and we forward our
    denylist via `--glob '!pattern'` so rg honors it as well.

### Grep Backend

Pure-Python by default. If `rg` is found on `PATH` at construction time,
we use it transparently for substantially faster searches on large repos.
The tool falls back to Python automatically if rg returns a non-zero
status, times out, or fails to spawn.

This matches Aider's behavior and avoids adding a binary or new dependency
on `agent-framework-core`. Ripgrep also honors `.gitignore` natively, so
when it is available, gitignore filtering is essentially free.

### Gitignore Filtering

`FileSystemPolicy.respect_gitignore` (default `True`) makes `fs_glob`,
`fs_grep` (Python backend), and `fs_list_dir` skip files matched by
`.gitignore` rules discovered at the workspace root and any nested
directories. Direct reads/writes via `fs_view`, `fs_edit`, etc. are
**not** affected — gitignore controls *discovery*, not *access*. The
denylist still wins over `!`-negations so users cannot un-ignore secrets.

The matcher is intentionally minimal: it covers `*`, `?`, `**`, character
classes, leading `/` for anchored patterns, trailing `/` for directory-only
matches, leading `!` for negation, and `#` comments. For full git fidelity
on large repositories, ripgrep is the recommended backend.

### Cross-Language

The Python implementation is the reference. A .NET `FileSystemTool` with
the same operation set, return shapes, and security model is tracked as a
follow-up.

## Consequences

- Agents have a safe, structured, portable filesystem capability that
  works without admin privileges, in restricted environments, and across
  Windows/macOS/Linux identically.
- The shell tool is freed to focus on its actual purpose (running
  commands like builds, tests, git, package managers) without being
  pressed into service for file I/O.
- New experimental surface area: `FileSystemTool`, `FileSystemPolicy`,
  result `TypedDict` types. Promoted out of experimental once the API
  stabilizes and a .NET counterpart ships.

## Out of Scope

- File watching / change notifications.
- Recursive directory copy / archive operations.
- Notebook (`.ipynb`) cell-level editing.
- MCP filesystem server interop (separate effort).
