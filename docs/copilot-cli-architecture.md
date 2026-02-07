# GitHub Copilot CLI Internal Architecture

> Reverse-engineered from `@github/copilot` npm package v0.0.405 (15.6 MB bundled JS).
> The `github/copilot-cli` GitHub repo is a shell (README + install script only).
> All source lives in the compiled npm package distributed via `npm install -g @github/copilot`.

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [System Prompt Architecture](#system-prompt-architecture)
3. [Agentic Loop — Two Levels](#agentic-loop--two-levels)
4. [Quality Enforcement Mechanisms](#quality-enforcement-mechanisms)
5. [Compaction / Context Management](#compaction--context-management)
6. [Token Counting](#token-counting)
7. [Sub-Agent System](#sub-agent-system)
8. [Custom Instructions & Skills](#custom-instructions--skills)
9. [Hooks System](#hooks-system)
10. [Tool Permission & Filtering](#tool-permission--filtering)
11. [JIT (Just-In-Time) Instructions](#jit-just-in-time-instructions)
12. [Feature Flags](#feature-flags)
13. [Implications for Our Harness](#implications-for-our-harness)

---

## Package Structure

```
@github/copilot@0.0.405
├── index.js                  # 15.6 MB — main bundle (all CLI logic)
├── sdk/index.js              # 11.5 MB — SDK bundle (session/transport layer)
├── sdk/index.d.ts            # 496 KB  — SDK type definitions
├── definitions/
│   ├── explore.agent.yaml    # 2.8 KB  — Explore sub-agent definition
│   ├── task.agent.yaml       # 2.0 KB  — Task sub-agent definition
│   └── code-review.agent.yaml # 4.2 KB — Code review sub-agent definition
├── clipboard/                # Native clipboard bindings
├── sharp/                    # Image processing (wasm)
├── prebuilds/                # winpty agent binaries
├── worker/                   # conout socket worker
├── npm-loader.js             # Package loader
├── README.md
├── LICENSE.md
└── changelog.json
```

---

## System Prompt Architecture

The system prompt is assembled from composable template parts using a tagged template literal system (`Cd()` template function). The main prompt structure:

```
identity
  → "You are the GitHub Copilot CLI, a terminal assistant built by GitHub."
  → Tone and style rules (concise, direct, 3 sentences max)
  → Tool usage efficiency (PARALLEL TOOL CALLING emphasis)
  → Version information
  → Environment context (cwd, OS, directory listing)
  → Job description ("perform the task... smallest possible changes")

code_change_instructions (wrapped in XML tags)
  → rules_for_code_changes
  → linting_building_testing
  → additional_instructions (security, CodeQL if enabled)
  → style rules

guidelines
  → self_documentation (fetch_copilot_cli_documentation tool)
  → using_sql_tool (per-session SQLite with todos/todo_deps tables)
  → todo_tracking (kebab-case IDs, status workflow)
  → tips_and_tricks
  → reporting_progress (report_intent tool)

environment_limitations
  → allowed_actions
  → disallowed_actions
  → prohibited_actions (security/privacy rules)

tools (per-tool usage instructions)
  → bash/powershell (sync, async, detach modes)
  → store_memory
  → edit (batch edits)
  → grep (ripgrep-based)
  → glob
  → task (sub-agent launcher)
  → code_search_tools
  → ask_user
  → fetch_copilot_cli_documentation
  → MCP server instructions

customInstructions
  → AGENTS.md, CLAUDE.md, GEMINI.md (from repo root)
  → .github/copilot-instructions.md
  → ~/.copilot/copilot-instructions.md
  → .github/instructions/ directory
  → Nested AGENTS.md files

additionalInstructions
lastInstructions
  → "Your thinking should be thorough, so it's fine if it's very long."
```

### Key System Prompt Details

**Identity section** includes Windows-specific path instructions injected conditionally:
```
CRITICAL: Since you're running on Windows, always use Windows-style paths
with backslashes (\) as the path separator.
```

**Task completion instructions** (injected when in task/autopilot mode):
```
* A task is not complete until the expected outcome is verified and persistent
* After configuration changes, run the necessary commands to apply them
* After starting a background process, verify it is running and responsive
* If an initial approach fails, try alternative tools or methods
```

**Validation instructions** (for task agents):
```
* Validate that your changes accomplish the requested task.
* Always validate that your changes don't break existing behavior,
  EXCEPT when a custom agent has completed the work.
* When a custom agent has completed work, do NOT validate or review
  their changes. Accept their work as final.
```

---

## Agentic Loop — Two Levels

### Inner Loop: `getCompletionWithTools`

This is the core turn-by-turn execution loop. Multiple implementations exist for different providers (OpenAI, Anthropic, local), but they share the same pattern:

```javascript
// Simplified from the decompiled source
async *getCompletionWithTools(systemPrompt, messages, tools, options) {
    let turn = options.initialTurnCount ?? 0;
    let stopped = false;
    let rejected = false;

    for (; !stopped && !rejected; ) {
        // --- Pre-request phase ---
        yield { kind: "turn_started", turn };

        for (let processor of options.processors.preRequest) {
            // CompactionProcessor, JitInstructionsProcessor,
            // ImmediatePromptProcessor, PremiumRequestProcessor, etc.
            for await (let event of processor.preRequest({ turn, messages, modelInfo, ... })) {
                yield event;  // compaction_started, compaction_completed, etc.
            }
        }

        // --- LLM call with retry logic ---
        let response = await this.makeRequest(client, model, messages, requestOptions, tools);

        // --- Process response ---
        let assistantMessage = response.choices[0].message;
        messages.push(assistantMessage);
        yield { kind: "message", turn, message: assistantMessage };

        if (this.isToolCallResponse(response)) {
            // --- Tool execution phase ---
            let toolCalls = response.choices.flatMap(c => c.message.tool_calls);

            // Pre-tools-execution hooks (preToolUse can deny/modify)
            for (let processor of options.processors.preToolsExecution) {
                let denials = await processor.preToolsExecution({ turn, toolCalls });
                // denied tools get immediate results without execution
            }

            // Execute tools (parallel or sequential)
            for await (let result of this.mergeToolResults(...)) {
                if (result.toolResult.resultType === "rejected") rejected = true;
                messages.push({ role: "tool", content: result.toolResult.textResultForLlm });
                yield { kind: "tool_execution", turn, toolCallId, toolResult };
            }

            // Post-tool-execution hooks
            for (let processor of options.processors.postToolExecution) {
                await processor.postToolExecution({ toolCall, toolResult, turn });
            }
        } else {
            // No tool calls = agent wants to stop
            yield { kind: "response", turn, response: assistantMessage };
            stopped = true;  // EXIT INNER LOOP
        }

        yield { kind: "turn_ended", turn };
        turn++;
    }
}
```

**Key observations:**
- The loop exits ONLY when the LLM returns a response with NO tool calls (just text)
- OR when a tool result has `resultType: "rejected"` (e.g., permission denied)
- Processors are invoked at every stage, providing interception points
- Tool calls can run in parallel (`executeToolsInParallel` flag)
- Retry logic handles 429, 503, 500, 400, 499 with exponential backoff
- On 413 (too large), images are stripped from the request and retried

### Outer Loop: `runAgenticLoop` (Session Level)

The session wraps the inner loop with hook-based continuation logic:

```javascript
// Simplified from session class
async runAgenticLoop(prompt, attachments, displayPrompt, billable) {
    // ... setup: build settings, tools, system prompt, callbacks ...

    let rejected = false;
    let hadToolCalls = false;

    // Run the inner loop
    let completionStream = client.getCompletionWithTools(
        systemPrompt, messages, tools, {
            processors: {
                preRequest: [this.compactionProcessor, jitProcessor, immediatePromptProcessor, ...],
                preToolsExecution: [new PreToolUseHooksProcessor(this.hooks, this.workingDir)],
                postToolExecution: [fileContentsProcessor],
                onRequestError: [fileContentsProcessor, this.compactionProcessor, jitProcessor],
                onStreamingChunk: [streamingProcessor, intentProcessor]
            },
            executeToolsInParallel: true,
            stream: this.enableStreaming,
        }
    );

    // Process all events from the inner loop
    for await (let event of completionStream) {
        if (this.abortController?.signal.aborted) break;

        switch (event.kind) {
            case "message":
                // Track if response has tool calls
                hadToolCalls = message.tool_calls?.length > 0;
                // Emit events to UI...
                break;
            case "tool_execution":
                if (event.toolResult.resultType === "rejected") rejected = true;
                // Track todo content updates, plan file updates, etc.
                break;
            case "compaction_started":
            case "compaction_completed":
                // Emit to UI...
                break;
            // ... other event types ...
        }
    }

    // ========== POST-COMPLETION QUALITY GATE ==========
    if (!rejected && !this.abortController?.signal.aborted && !hadToolCalls) {
        let hookResult = await invokeHooks(this.hooks?.agentStop, {
            timestamp: Date.now(),
            cwd: this.workingDir,
            sessionId: this.sessionId,
            transcriptPath: this.transcriptPath,
            stopReason: "end_turn"
        });

        if (hookResult?.decision === "block" && hookResult.reason) {
            // FORCE CONTINUATION: enqueue a new user message
            this.enqueueUserMessage({ prompt: hookResult.reason }, /* prepend */ true);
            // processQueuedItems will pick this up and call runAgenticLoop again
            return;
        }
    }

    // Session truly idle — emit session.idle event
}
```

**Key observations:**
- The `agentStop` hook fires AFTER the inner loop exits naturally
- The hook receives full session context (sessionId, transcriptPath, cwd)
- External code can inspect the transcript and decide to block completion
- Blocking enqueues a new user message that restarts the agentic loop
- The `processQueuedItems` method processes queued messages sequentially
- `session.idle` is only emitted when no more items are in the queue

---

## Quality Enforcement Mechanisms

### 1. `task_complete` Tool (AUTOPILOT_MODE)

When the `AUTOPILOT_MODE` feature flag is enabled, a `task_complete` tool is added to the tool set:

```javascript
function createTaskCompleteTool(onTaskComplete) {
    return {
        name: "task_complete",
        title: "Task complete",
        description: [
            "Mark the current task as complete.",
            "Use this tool ONLY when you are confident the task is fully done.",
            "",
            "Guidelines:",
            "- Call this tool only after verifying your changes work correctly",
            "- Include a brief summary of what was accomplished",
            "- If you're unsure whether the task is complete, continue working",
            "- Err on the side of caution - it's better to do extra verification",
            "  than to mark incomplete work as done"
        ].join("\n"),
        input_schema: { summary: "string - A brief summary of what was accomplished" },
        instructions: [
            "Use the task_complete tool to explicitly mark when you have finished.",
            "",
            "IMPORTANT: Only call this tool when you are confident the task is fully complete.",
            "",
            "When calling this tool:",
            "- First, provide a brief 1-sentence summary in your message text",
            "- Then call the tool with a more detailed summary",
            "",
            "When to call:",
            "- After you have completed ALL requested changes",
            "- After tests pass (if applicable)",
            "- After you have verified the solution works correctly",
            "- When you are confident no further work is needed",
            "",
            "When NOT to call:",
            "- If you encountered errors you haven't resolved",
            "- If there are remaining steps to complete",
            "- If you're waiting for user input or clarification",
            "- If you haven't verified your changes work",
            "- If you're unsure whether the task is fully done",
            "",
            "CRITICAL: Do NOT mark the task complete prematurely.",
            "If in doubt, keep working. Only call this tool when you have",
            "high confidence the task is truly finished and verified."
        ].join("\n")
    };
}
```

**Continuation nudge** (injected when agent stops without calling `task_complete`):

```
You have not yet marked the task as complete using the task_complete tool.
If you were planning, stop planning and start implementing.
You aren't done until you have fully completed the task.

IMPORTANT: Do NOT call task_complete if:
- You have open questions or ambiguities - use your best judgment and continue
- You encountered an error - try to resolve it or find an alternative approach
- There are remaining steps - complete them first

Keep working autonomously until the task is truly finished,
then call task_complete with a summary.
```

### 2. `agentStop` Hook (Post-Completion Gate)

After the inner loop exits, the session checks the `agentStop` hook:

```javascript
// Only fires when: no rejection, no abort, no pending tool calls
if (!rejected && !aborted && !hadToolCalls) {
    let result = await hooks.agentStop({
        timestamp: Date.now(),
        cwd: this.workingDir,
        sessionId: this.sessionId,
        transcriptPath: this.transcriptPath,
        stopReason: "end_turn"
    });

    if (result?.decision === "block" && result.reason) {
        this.enqueueUserMessage({ prompt: result.reason }, true);
        return; // Will be picked up by processQueuedItems
    }
}
```

### 3. `onSubagentStop` Hook (Sub-Agent Gate)

For custom agent tool invocations, a similar hook exists:

```javascript
// In the custom agent outer loop
for (;;) {
    let { messages } = await agent.agent(location, systemPrompt, history, userMessage, {}, tools);
    history = messages;

    if (onSubagentStop) {
        let result = await onSubagentStop({
            sessionId, transcriptPath, agentName, agentDisplayName
        });
        if (result?.decision === "block" && result.reason) {
            // Continue the agent with a new instruction
            userMessage = result.reason;
            continue;
        }
    }

    // Extract last assistant text and break
    break;
}
```

### 4. `preToolUse` Hook (Pre-Execution Gate)

Can deny tool calls or modify arguments before execution:

```javascript
class PreToolUseHooksProcessor {
    async preToolsExecution({ toolCalls }) {
        let denials = new Map();
        for (let hook of this.hooks.preToolUse) {
            for (let toolCall of toolCalls) {
                let result = await hook({
                    timestamp: Date.now(),
                    cwd: this.workingDir,
                    toolName: toolCall.function.name,
                    toolArgs: toolCall.function.arguments
                });

                if (result?.permissionDecision === "deny") {
                    denials.set(toolCall.id, {
                        textResultForLlm: `Denied by preToolUse hook: ${result.permissionDecisionReason}`,
                        resultType: "denied"
                    });
                } else if (result?.modifiedArgs !== undefined) {
                    toolCall.function.arguments = JSON.stringify(result.modifiedArgs);
                }
            }
        }
        return denials.size > 0 ? denials : undefined;
    }
}
```

---

## Compaction / Context Management

### CompactionProcessor

Runs as a `preRequest` processor before every LLM call:

```javascript
class CompactionProcessor {
    // Thresholds
    static DEFAULT_BACKGROUND_THRESHOLD = 0.80;   // Start background compaction
    static DEFAULT_BUFFER_EXHAUSTION_THRESHOLD = 0.95; // Block and wait
    static DEFAULT_MIN_MESSAGES = 4;

    async *preRequest({ messages, modelInfo, toolDefinitions, turn, client, tools }) {
        let maxTokens = modelInfo.capabilities.limits.max_prompt_tokens
                     || modelInfo.capabilities.limits.max_context_window_tokens
                     || 128000;

        let messageTokens = fR(messages, modelInfo.name);     // Count message tokens
        let toolTokens = OIe(toolDefinitions, modelInfo.name); // Count tool schema tokens
        let totalTokens = messageTokens + toolTokens;
        let utilization = totalTokens / maxTokens;

        // Case 1: Pending compaction from previous turn
        if (this.pendingCompaction) {
            let completed = this.checkCompactionCompleted();
            if (completed) {
                yield* this.applyCompactionResult(context);
                return;
            }
            if (utilization >= this.bufferExhaustionThreshold) {
                // BLOCK: Wait for compaction to finish
                yield* this.waitForCompactionAndApply(context);
                return;
            }
            // Continue with buffer space remaining
            return;
        }

        // Case 2: Below threshold — no action needed
        if (utilization < this.backgroundThreshold) {
            return; // "Utilization XX.X% below threshold 80%"
        }

        // Case 3: Skip if too few messages
        if (messages.length < this.minMessagesForCompaction) {
            return;
        }

        // Case 4: Start background compaction
        yield { kind: "compaction_started", turn };
        this.startBackgroundCompaction(context, totalTokens);

        // If forced, wait for completion
        if (this.forceCompactionOnNextRequest) {
            yield* this.waitForCompactionAndApply(context);
        }
    }
}
```

### Background Compaction Process

```javascript
startBackgroundCompaction(context, currentTokens) {
    let { messages, modelInfo } = context;
    let checkpointMessages = [...messages]; // Snapshot

    let promise = (async () => {
        // Prepare messages for summarization (filter system messages)
        let filteredMessages = prepareForCompaction(messages, this.includeCheckpointTitle);
        let planContent = await this.getPlanContent?.();
        let estimatedTokens = countTokens(messages, modelInfo.name);

        // Call LLM to summarize
        return await compactThread(context.client, systemPrompt, filteredMessages, context.tools, logger);
    })();

    this.pendingCompaction = {
        promise,
        checkpointMessages,
        startTurn: context.turn,
        checkpointTokens: currentTokens,
        completed: false,
        cancelled: false,
        originalUserMessages: this.getOriginalUserMessages?.(),
        todoContent: this.getTodoContent?.()
    };

    promise.then(result => {
        this.pendingCompaction.completed = true;
        this.pendingCompaction.result = result;
        this.onCompactionComplete({ success: true, summaryContent: result.content, ... });
    }).catch(error => {
        this.pendingCompaction.completed = true;
        this.pendingCompaction.error = error;
        this.onCompactionComplete({ success: false, error });
    });
}
```

### Applying Compaction Results

After compaction completes, the message array is rebuilt:

```javascript
// sGe function — rebuilds messages after compaction
function rebuildMessages(messages, summaryContent, originalUserMessages, options) {
    let newMessages = [];

    // 1. Keep the system message
    // 2. Insert compaction summary as a user/assistant pair
    // 3. Preserve original user messages (for context continuity)
    // 4. If plan content exists, inject it
    // 5. If todo content exists, inject it
    // 6. If invoked skills exist, note them
    // 7. Append recent messages (post-compaction-checkpoint)

    return newMessages;
}
```

### Error Recovery

On 413 (request too large) errors, the CompactionProcessor forces compaction on retry:

```javascript
async onRequestError({ status, retry }) {
    let isContextLimitError = status === 413 || isProviderContextLimitError(error);
    if (retry === 0 && isContextLimitError) {
        this.forceCompactionOnNextRequest = true;
    }
}
```

---

## Token Counting

### Per-Message Token Counting

Copilot uses tiktoken to count tokens accurately per message:

```javascript
// $Jo — tokenizer factory (caches by model name)
function getTokenizer(model) {
    let cached = tokenizers.get(model);
    if (!cached) {
        try { cached = tiktoken_for_model(model); }
        catch { cached = getDefaultTokenizer(); }
        tokenizers.set(model, cached);
    }
    return cached;
}

// yR — count tokens for a text string
function countTextTokens(text, model) {
    let encoded = getTokenizer(model).encode(text, [], []);
    let scaleFactor = getScaleFactor(model);
    return Math.ceil(encoded.length * scaleFactor);
}

// fR — count total tokens for a message array
function countMessageTokens(messages, model) {
    let overhead = Math.ceil(3 * getScaleFactor(model)); // Per-request overhead
    return overhead + messages.reduce((sum, msg) => countSingleMessage(msg, model) + sum, 0);
}

// l7 — count tokens for a single message
function countSingleMessage(message, model) {
    let tokens = 0;
    let perMessageOverhead = 3; // role + separators

    if (message.role) perMessageOverhead++;
    if (message.role !== "function" && message.role !== "tool" && message.name) {
        tokens += countTextTokens(message.name, model) + Math.ceil(1 * getScaleFactor(model));
    }

    // Content tokens
    if (message.content) {
        if (typeof message.content === "string") {
            tokens += countTextTokens(message.content, model);
        } else {
            // Array content (multimodal)
            for (let part of message.content) {
                if (part.type === "text") tokens += countTextTokens(part.text, model);
                else if (part.type === "refusal") tokens += countTextTokens(part.refusal, model);
                else if (part.type === "image_url") tokens += countImageTokens(part, model);
                else tokens += countTextTokens(JSON.stringify(part), model);
            }
        }
    }

    // Tool call tokens (assistant messages)
    if (message.role === "assistant" && message.tool_calls) {
        for (let toolCall of message.tool_calls) {
            tokens += countTextTokens(toolCall.function.name, model);
            tokens += countTextTokens(toolCall.function.arguments, model);
            tokens += 3; // tool call overhead
        }
    }

    return tokens + perMessageOverhead;
}

// OIe — count tool schema tokens
function countToolSchemaTokens(toolDefinitions, model) {
    // Serializes each tool's schema and counts tokens
    // Includes name, description, parameters JSON schema
}
```

### Scale Factors

The `zMe(model)` function returns a per-model scaling factor. Most models use 1.0, but some model families have different ratios to account for tokenizer differences.

---

## Sub-Agent System

Copilot uses specialized sub-agents defined in YAML files (`definitions/*.agent.yaml`):

### Explore Agent (`explore.agent.yaml`)

```yaml
name: explore
displayName: Explore Agent
model: claude-haiku-4.5
tools: [grep, glob, view, lsp]
prompt: |
  You are an exploration agent specialized in rapid codebase analysis.
  CRITICAL: Keep your answer under 300 words
  CRITICAL: MAXIMIZE PARALLEL TOOL CALLING
  Aim to answer questions in 1-3 tool calls when possible
```

### Task Agent (`task.agent.yaml`)

```yaml
name: task
displayName: Task Agent
model: claude-haiku-4.5
tools: ["*"]  # All tools
prompt: |
  You are a command execution agent that runs development commands.
  On SUCCESS: Return brief one-line summary ("All 247 tests passed")
  On FAILURE: Return full error output for debugging
  Do NOT attempt to fix errors or make suggestions - just execute and report
```

### Code Review Agent (`code-review.agent.yaml`)

```yaml
name: code-review
displayName: Code Review Agent
model: claude-sonnet-4.5
tools: ["*"]  # All tools (for investigation only)
prompt: |
  You are a code review agent with an extremely high bar for feedback.
  Your guiding principle: finding your feedback should feel like finding
  a $20 bill in your jeans after doing laundry.

  ONLY surface: bugs, security vulns, race conditions, memory leaks,
  missing error handling, breaking API changes, measurable perf issues.

  NEVER comment on: style, formatting, naming, grammar, "consider" suggestions,
  minor refactoring, code organization, missing docs, "best practices".

  CRITICAL: You Must NEVER Modify Code.
  If you find NO issues: "No significant issues found."
```

### General-Purpose Agent

Launched via the `task` tool with `agent_type: "general-purpose"`:
- Uses the full system prompt (same as main agent)
- Runs in a separate context window
- Gets `claude-sonnet-4.5` (or user-selected model)
- Has access to all tools
- Results are returned to the parent agent as tool results

### Sub-Agent Invocation Pattern

```javascript
// From the main agent's perspective, sub-agents are tools
{
    name: "task",
    description: "Custom agent: Launch specialized agents...",
    input_schema: {
        prompt: "string - The task for the agent to perform",
        description: "string - A short (3-5 word) description",
        agent_type: "enum - explore | task | general-purpose | code-review",
        model: "string - Optional model override",
        mode: "enum - sync | background"
    }
}
```

The main agent delegates to sub-agents to keep its own context clean:
- **Explore** agents for codebase questions (safe to run in parallel)
- **Task** agents for builds/tests (brief success output, verbose failure)
- **Code review** for analyzing diffs
- **General-purpose** for complex multi-step work (separate context window)

---

## Custom Instructions & Skills

### Instruction File Resolution Order

Copilot loads custom instructions from multiple sources with this priority:

1. **User-level**: `~/.copilot/copilot-instructions.md`
2. **Repo-level**: `.github/copilot-instructions.md`
3. **AGENTS.md** (repo root)
4. **CLAUDE.md** (repo root)
5. **GEMINI.md** (repo root)
6. **VS Code instructions**: `.github/instructions/` directory
7. **Nested AGENTS.md**: Found in subdirectories
8. **Environment variable**: `COPILOT_CUSTOM_INSTRUCTIONS_DIRS`

All found instructions are wrapped in `<custom_instruction>` XML tags and concatenated.

### Deduplication

If `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` have identical content (via `realpathSync` check), they are merged into a single instruction block with a combined source path.

### Skills System

Skills are domain-specific instruction sets loaded from `skillDirectories`. They provide additional context and instructions that can be invoked by the agent during execution.

---

## Hooks System

The SDK exposes lifecycle hooks that external code can register:

```typescript
interface SessionHooks {
    userPromptSubmitted?: Array<(event: {
        timestamp: number;
        cwd: string;
        prompt: string;
    }) => Promise<{ modifiedPrompt?: string } | void>>;

    preToolUse?: Array<(event: {
        timestamp: number;
        cwd: string;
        toolName: string;
        toolArgs: any;
    }) => Promise<{
        permissionDecision?: "allow" | "deny";
        permissionDecisionReason?: string;
        modifiedArgs?: any;
        additionalContext?: string;
    } | void>>;

    postToolUse?: Array<(event: {
        timestamp: number;
        cwd: string;
        toolName: string;
        toolArgs: any;
        toolResult: ToolResult;
    }) => Promise<{
        modifiedResult?: ToolResult;
    } | void>>;

    agentStop?: Array<(event: {
        timestamp: number;
        cwd: string;
        sessionId: string;
        transcriptPath: string;
        stopReason: "end_turn";
    }) => Promise<{
        decision?: "allow" | "block";
        reason?: string;
    } | void>>;

    subagentStop?: Array<(event: {
        timestamp: number;
        cwd: string;
        sessionId: string;
        transcriptPath: string;
        agentName: string;
        agentDisplayName: string;
        stopReason: "end_turn";
    }) => Promise<{
        decision?: "allow" | "block";
        reason?: string;
    } | void>>;

    sessionStart?: Array<(event: { timestamp: number; cwd: string }) => Promise<void>>;
    sessionEnd?: Array<(event: {
        timestamp: number;
        cwd: string;
        reason: "complete" | "error";
        error?: Error;
    }) => Promise<void>>;

    errorOccurred?: Array<(event: {
        timestamp: number;
        cwd: string;
        error: Error;
        errorContext: string;
        recoverable: boolean;
    }) => Promise<void>>;
}
```

### Hook Invocation Pattern

```javascript
// uB — universal hook invoker
async function invokeHooks(hookArray, event, logger) {
    if (!hookArray || hookArray.length === 0) return undefined;
    let result;
    for (let hook of hookArray) {
        try {
            result = await hook(event);
        } catch (error) {
            logger.error(`Hook execution failed: ${error}`);
        }
    }
    return result; // Returns last non-undefined result
}
```

---

## Tool Permission & Filtering

### Shell Command Safety

Shell tools use a safety assessor system:

```javascript
shellConfig.withScriptSafetyAssessor(undefined, async function(script, interpreter, identifier) {
    return {
        result: "completed",
        commands: [{ identifier, readOnly: false }],
        possiblePaths: [],
        possibleUrls: [],
        hasWriteFileRedirection: false,
        canOfferSessionApproval: false
    };
});
```

### Tool Filtering

Tools can be filtered at multiple levels:
- `isToolEnabled(toolName)` — session-level filter
- `filterTool` callback — passed to sub-agent context
- `--allow-tool` / `--deny-tool` CLI flags (glob patterns)
- `disabledSkills` configuration

---

## JIT (Just-In-Time) Instructions

The `JitInstructionsProcessor` (`a7e` class) injects user messages at specific points during execution based on conditions:

```javascript
class JitInstructionsProcessor {
    async *preRequest({ turn, messages }) {
        // Check each JIT instruction config
        for (let [key, config] of Object.entries(this.jitInstructions)) {
            // Check if within time threshold (percentage of timeout remaining)
            let timeCheck = checkTimeRemaining(this.settings, config.percentRemainingOfTimeout);
            if (!timeCheck.isWithin) continue;

            // Check condition: only inject when no paths changed (if configured)
            if (config.whenNoPathsChanged) {
                let changedPaths = await this.gitHandler.getChangedPaths(this.location, "HEAD", this.baseCommit);
                if (changedPaths.length > 0) continue;
            }

            // Get instruction text (can be dynamic function)
            let instruction = typeof config.instruction === "function"
                ? config.instruction(this.location)
                : config.instruction;

            // Skip if already emitted
            if (this.emittedJitInstructions.has(instruction)) continue;

            // Inject as user message
            this.emittedJitInstructions.add(instruction);
            let message = { role: "user", content: instruction };
            messages.push(message);
            yield { kind: "message", turn, message, source: "jit-instruction" };
        }
    }
}
```

JIT instructions are used for:
- Reminding the agent about remaining work when time is running out
- Nudging the agent when no file changes have been made
- Injecting context-specific guidance based on execution state

---

## Feature Flags

```javascript
const FEATURE_FLAGS = {
    AUTOPILOT_MODE: "experimental",  // Enable autopilot mode for autonomous agent operation
    FLEET_COMMAND: "experimental",   // Enable /fleet for parallel execution of large work
    LSP_TOOLS: "on",                 // Language Server Protocol tools
    VSCODE_INTEGRATION: "staff",     // VS Code integration features
    QUEUED_COMMANDS: "staff",        // Command queueing
    SKILLS_INSTRUCTIONS: "staff",    // Skills system
    DISABLE_WEB_TOOLS: "off",       // Web search/fetch tools
    IMAGE_PASTE_FALLBACK: "staff",  // Image paste handling
};
```

Access levels: `"on"` (everyone), `"experimental"` (opt-in), `"staff"` (GitHub employees only), `"off"` (disabled).

---

## Implications for Our Harness

### What Copilot Does That We Should Replicate

1. **Explicit completion tool**: The `task_complete` pattern forces the agent to explicitly declare "I'm done" rather than silently stopping when it runs out of ideas. Our harness should make the stop decision a conscious agent action, not just the absence of tool calls.

2. **Post-completion hook / quality gate**: The `agentStop` hook inspects session state after the agent tries to stop and can force continuation with a specific reason. Our `_stop_decision_executor.py` should check tool usage metrics (read_file count, write_file count, turns) and reject premature stops with actionable instructions.

3. **Continuation nudge text**: Copilot's nudge is assertive and specific — "If you were planning, stop planning and start implementing." Our nudges should be equally direct and tell the agent exactly what to do next.

4. **Token counting from wire format**: Copilot counts tokens from the actual OpenAI message format (content strings, tool call name + arguments, tool results), not internal representations. Uses tiktoken with per-model scaling.

5. **Background compaction**: Compaction runs asynchronously (80% threshold) and only blocks when truly needed (95% threshold). This keeps the agent running smoothly even under memory pressure.

6. **JIT instructions**: Injecting context-sensitive guidance during execution (not just at the start) based on conditions like time remaining or files changed.

7. **Sub-agent delegation**: Heavy lifting is delegated to specialized sub-agents with their own context windows, keeping the main agent's context clean.

### What Copilot Does NOT Do

1. **No minimum-effort enforcement**: There's no built-in "you must read at least N files" or "you must use at least N tools" gate. Quality comes from the system prompt quality and the `task_complete` forcing explicit intent.

2. **No LLM-as-judge**: There's no secondary LLM call to evaluate the quality of the agent's work. Quality enforcement is entirely rule-based (hooks, flags, token thresholds).

3. **No work item system**: Copilot uses a simpler todo table (SQLite) rather than a structured work item system with statuses and artifacts. The agent manages its own task breakdown.

4. **No per-turn state injection**: Unlike our harness's work item state injection, Copilot relies on the conversation history and JIT instructions rather than injecting structured state summaries between turns.
