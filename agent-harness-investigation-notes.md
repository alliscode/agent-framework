Skip to content
Chat history



You said:
It seems the new trend in AI development is the "agent harness". Can you do a deep dive on this topic and help me understand what an agent harness is, what it contains, what the structure of it is, how it's used, etc.
ChatGPT said:

Thought for 8s
“Agent harness” is becoming the name for the non-model part of an agent system: the scaffolding that turns an LLM from “a chat completion” into “a long-running, tool-using, stateful worker that can be evaluated and shipped.”

You’ll see slightly different emphases depending on who’s using the term:

Product/engineering usage: “everything except the model” that makes an agent reliable in the real world. 

LangChain / DeepAgents usage: an opinionated bundle of built-in capabilities (planning tool, filesystem, subagents, etc.) around a standard tool-calling loop. 

Evaluation/benchmark usage: the harness as the orchestrator for reproducible rollouts (VMs/sandboxes, tool traces, logging, scoring). 

Below is a “deep dive” view that reconciles these into a single mental model.

The shortest useful definition
An agent harness is the runtime + infrastructure wrapper around an LLM that handles:

Control flow (the agent loop, planning, stopping conditions)

Tooling (registration, invocation, permissions, retries, timeouts)

Context engineering (what goes into the prompt, what gets summarized, what gets persisted)

State & memory (durable state, files, scratchpads, checkpoints)

Safety & governance (policies, approvals, audit logs)

Observability & evaluation (traces, metrics, replay, tests)

This “wraps around the model” framing is common across recent writeups. 

Why it’s a “trend” now
As models got better at tool calling, teams discovered the hard part isn’t “can the model call tools?”—it’s:

Can it make progress across many steps and many context windows without drifting? (long-running reliability) 

Can you constrain it to do safe/approved actions, and prove what happened? (governance + audit) 

Can you reproduce and evaluate behavior like you do with normal software? (eval harnesses, rollouts, replay) 

So “agent harness” emerges as the explicit name for the software discipline around those concerns. 

What an agent harness contains (the anatomy)
Think in layers.

1) The agent loop (control plane)
This is the “driver”:

input → plan → select tool/action → execute → observe → update state → repeat

termination rules (done / max steps / low confidence / needs human)

optional: multi-agent delegation / subagents

DeepAgents describes itself as this same core loop but with built-in capabilities on top. 

2) Tooling substrate (capability plane)
A harness typically standardizes:

tool registry (names, schemas, auth scopes)

execution adapters (HTTP, DB, shell, browser, file ops)

reliability (retries, backoff, circuit breakers)

permissions (allow/deny lists, per-tool budgets, sandboxing)

You’ll also see “provider hot-swapping” harnesses that unify multiple agent SDKs behind a consistent interface. 

3) Context engineering (prompt + compression + retrieval)
This is where harnesses differ most, and why they’re often described as “everything except the model.”

Typical components:

message/history management (what to include/exclude)

summarization/compaction and “state extraction”

retrieval (RAG) + caching

“injection points” (system reminders, constraints, rubrics) at the right time—not constantly

Recent commentary ties harness design directly to context engineering strategies like compaction/offloading/isolation into sub-agents. 

4) State, memory, and durability (data plane)
To be long-running, you need state that outlives a single prompt window:

durable run state: step counter, goals, plan, pending approvals

artifacts: files, patches, datasets

checkpoints: resume after crashes / context rollover

provenance: what tool outputs were used to decide what

Anthropic’s “long-running agent harness” framing focuses heavily on incremental progress across many context windows (which implies durable state + compaction). 

5) Governance: safety, approvals, and policies
In production, the harness is where you enforce:

human-in-the-loop gates (approve spend, approve destructive actions)

policy checks (PII, security rules, compliance)

tool-level constraints (read-only filesystem, no network, restricted domains)

6) Observability & evaluation (engineering plane)
A harness usually bakes in:

traces: every model call + tool call + intermediate state

structured logs/events (for replay)

evaluators (unit-like checks, LLM judges, task-specific scorers)

sandbox orchestration for reproducible runs (VMs/containers, deterministic seeds)

The Holistic Agent Leaderboard paper explicitly calls out a standardized evaluation harness that orchestrates large-scale rollouts. 

LangChain also notes deep agents need more bespoke test logic and evaluation scaffolding than “single-shot” LLM apps. 

A concrete “structure” (how you’d draw it)
Here’s a useful reference architecture you can map most harnesses onto:

┌─────────────────────────────── App / Product ────────────────────────────────┐
│ UI / API  ──>  Session/Task Router  ──>  Agent Harness (this is the “thing”) │
└──────────────────────────────────────────────────────────────────────────────┘

Agent Harness
├─ Control Loop
│   ├─ Planner / Decomposer
│   ├─ Stepper (tool-call loop)
│   ├─ Stopper (done/limits/uncertainty)
│   └─ Subagent Manager (optional)
├─ Context Engine
│   ├─ Prompt assembly (policies, instructions, tools)
│   ├─ Memory injection (episodic / semantic)
│   ├─ Compaction / summarization
│   └─ Budgeting (tokens, time, $)
├─ Tool Runtime
│   ├─ Tool registry + schemas
│   ├─ Auth + secrets
│   ├─ Sandbox (fs, shell, browser, VM)
│   └─ Reliability (retry/timeout/limits)
├─ State & Artifacts
│   ├─ Durable run state (checkpoints)
│   ├─ Filesystem / object store
│   └─ Provenance (what led to what)
└─ Observability & Evals
    ├─ Tracing (model+tool calls)
    ├─ Replay (deterministic-ish)
    ├─ Evaluators (tests, judges)
    └─ Reports / dashboards
If you compare “framework vs runtime vs harness,” this is basically: harness = opinionated packaging across those layers. That distinction is explicitly discussed in the “framework vs runtime vs harness” discourse around DeepAgents. 

How it’s used in practice (end-to-end flow)
A) Building an agent with a harness
You usually:

Declare tools (schemas, permissions)

Configure context strategy (memory sources, compaction rules, budgets)

Choose a loop style (react loop, plan-then-execute, hierarchical subagents)

Set governance (approval points, restricted actions)

Turn on tracing + evaluations

DeepAgents is explicitly positioned as providing those built-in capabilities (planning, filesystem, subagents) so you’re not re-implementing the “OS” around the agent. 

B) Running a long task
At runtime:

The harness repeatedly calls the model, executes tools, writes artifacts, compacts context, and checkpoints progress.

If the task exceeds the context window, the harness uses compaction + persistence to continue. 

C) Debugging / iterating
Because the harness records structured traces, you can:

replay a run to reproduce failures

inspect where drift happened (bad tool output? wrong context injection? missing constraints?)

add targeted evaluators to prevent regressions 

Two common meanings you should keep separate
People overload “harness” in two adjacent ways:

1) Agent harness (runtime harness)
The thing described above: scaffolding for running agents reliably.

2) Evaluation harness
A special case focused on testing agents:

standardized environments

fixed tasks and scoring

reproducible rollouts at scale 

They overlap heavily (a good runtime harness makes evaluation easier), but the evaluation harness often adds orchestration and scoring infrastructure.

What makes something “a harness” vs “a framework”?
A decent rule of thumb:

Framework: gives you primitives (agents, tools, memory abstractions).

Runtime: executes graphs/loops with concurrency/state semantics.

Harness: ships an opinionated, batteries-included environment that covers the messy edges: filesystem, planning, subagents, compaction, guardrails, observability, eval hooks.

That “batteries included around the loop” idea is exactly how DeepAgents positions itself. 

A practical checklist (what to look for when someone says “agent harness”)
If you’re evaluating a harness (or designing your own), ask:

Statefulness: durable checkpoints? resumable runs?

Context lifecycle: compaction, memory injection, budgets?

Tool discipline: schema enforcement, retries, permissions, sandboxing?

Governance: approvals, audit logs, policy checks?

Observability: traces, replay, artifact diffing?

Eval readiness: can you run 1,000 rollouts and score them consistently?

The recent long-running harness work emphasizes multi-window progress + the infrastructure required to make that dependable. 

If you want, tell me what kind of agent you’re building (coding agent, ops agent, business workflow, research) and whether you care more about durable execution, tool safety, or evaluation/benchmarking—and I’ll map this into a concrete harness design with recommended module boundaries and “must-have” interfaces.

You said:
Here are some notes I've taken on Lang Chain's Deep Agents harness:

- Large Tool Result eviction. This is clever - large tool results are written to the file system and the result message is replaced with a summary and a reference to the file. Agents that want to get more details can choose to read the file.
- Sub Agents - Can be created by the top agent using the Task tool. Sub agents are stateless and return a single report message.
- Conversation compaction. Default behavior seem to be:
    - Triggers at 85% of max_input_token (if known)
    - Keeps most recent 10% of conversation
    - Summarizes older messages
- Dangling tool call repair -0 Fixes chat history when tool calls never complete, etc.
- Built in to-do list tracking using write_todos tool available to agents. Todos have a status.
- HIL for tools calls etc. Configurable.
- Prompt caching where the service supports it (Anthropic).

Question, can you see if a similar analysis can be performed on the Claude Agent SDK?
ChatGPT said:

Thought for 1m 24s
An agent harness is basically the “runtime shell” around an LLM that turns it from a single completion into a durable worker: it owns the loop, the state, the tool plumbing, and all the guardrails you need for long-running, tool-heavy work.

Think of it like the difference between:

calling a function (raw chat/completions), vs

running a process (agent harness: lifecycle, filesystem, permissions, logs, recovery).

What an agent harness usually contains
1) The agent loop (control plane)
The harness implements the repeated cycle:

build next prompt from state

call model

detect tool requests / subagent dispatch

execute tools

append tool results

apply safety/permissions/HIL

compact/evict context if needed

continue until stop criteria

Claude Agent SDK is explicitly positioned as this kind of “general-purpose agent harness” (Anthropic uses that term) and emphasizes long-running operation + context management (“compaction”). 

2) State model (data plane)
Harnesses typically separate state into:

conversation transcript (messages / events)

working memory (summaries, decisions, scratchpads)

artifacts (files, logs, datasets, patches, reports)

execution context (cwd, environment vars, credentials, permissions)

Claude Agent SDK is designed to run as a long-running process with a persistent shell and a working directory (that’s already a big “harness” signal). 

3) Tool runtime + tool UX
A harness standardizes tool invocation, retries, error handling, and often:

tool allow/deny lists

permission prompts / approvals

hooks around tool calls

structured tool results and formatting correctness

Claude Agent SDK’s quickstart describes this clearly: you stream messages while the SDK handles tool execution + context management + retries. 

It also has a rich hooks system to intercept tool use and lifecycle events (PreToolUse/PostToolUse/PermissionRequest/etc.). 

4) Context management (the “staying alive” layer)
This is where modern “agent harness” work is concentrated:

summarization / compaction

eviction / tool-result clearing

caching

persistence across sessions

Your LangChain Deep Agents notes are exactly in this category.

For Claude / Anthropic, you can map those ideas to a few distinct mechanisms:

A) Conversation compaction (summarize-and-reset):
Anthropic’s cookbook describes compaction_control as: monitor token usage → inject a summary prompt → replace history with the summary → keep going. 

Claude Code / Agent SDK also exposes a user-facing /compact slash command to compact history. 

B) Tool-result eviction / clearing (evict-but-keep-working):
Anthropic has a server-side “context editing” strategy called tool result clearing (clear_tool_uses_20250919) that clears older tool results (optionally tool inputs too) and replaces them with placeholders. 

This is conceptually very similar to your “Large Tool Result eviction”, just implemented as an API-level mechanism rather than “write to disk + reference” (though writing to disk is still a common harness pattern when you control a filesystem).

C) Prompt caching:
Anthropic documents prompt caching via cache_control blocks. 

In practice, Claude Code / Agent SDK environments also tend to benefit from caching behavior (and there’s active discussion about cache control in the SDK ecosystem), but the canonical mechanism is still the platform-level cache_control. 

5) Safety + control: permissions and HIL
Harnesses are where you implement:

human-in-the-loop approvals (especially for writes, network, payments)

sandboxing / isolation

credential boundaries

policy enforcement

Claude Agent SDK’s hosting guidance is very explicit: run inside container-based sandboxing; isolate; manage network access; etc. 

Subagents can also run with restricted permissions/tools (in Claude Code subagent configs you can constrain tools + permission modes). 

Mapping your Deep Agents harness notes → Claude Agent SDK
Here’s the closest “feature parity” view:

1) Large tool result eviction
Deep Agents: write big tool results to filesystem, replace message with summary + reference.

Claude / Anthropic equivalents:

Tool result clearing at the API level clears older tool results and inserts placeholders. 

Separately, because Agent SDK sessions run with a working directory, you can choose to externalize large artifacts to files and have the agent read them later (that’s a harness pattern, and Claude’s runtime is compatible with it). 

2) Subagents
Deep Agents: top agent spawns stateless subagents returning a report.

Claude Agent SDK: explicit subagents support:

define programmatically (agents parameter),

define via filesystem (.claude/agents/),

and there’s a built-in general-purpose subagent invokable via the Task tool. 

Also, Claude Code subagents can be restricted by tool access + permission modes. 

3) Conversation compaction
Deep Agents: threshold-based summarization; keep recent slice; summarize older.

Claude / Anthropic:

Agent-style automatic compaction is described as “inject summary prompt → clear history → resume with summary”. 

/compact exists as an operational control to compact. 

(I didn’t find a public doc stating the exact 85% / keep-10% defaults you observed in Deep Agents; Claude’s docs focus more on the mechanism and configuration knobs than those specific heuristics.)

4) Dangling tool call repair
Deep Agents: repairs history when tool calls don’t complete, etc.

Claude / Anthropic:

The underlying Messages/tool-use protocol is strict: tool results must immediately follow tool_use blocks, otherwise you get errors. 

The Agent SDK’s value prop is that it runs the loop and handles tool execution/retries so you don’t end up with malformed tool history as often. 

Hooks also let you detect tool failures and implement your own recovery policies (e.g., retry, escalate, request human approval). 

So: Claude’s approach is less “repair the transcript after the fact” (as a headline feature) and more “own the loop so the transcript stays valid; add hooks for failure handling.”

5) Built-in todo list tracking
Deep Agents: write_todos tool, statuses.

Claude Agent SDK: has built-in Todo Lists with a lifecycle (pending → in_progress → completed → removed). 

6) HIL for tool calls
Deep Agents: configurable.

Claude Agent SDK: permissions + hooks + secure deployment patterns are first-class. In Claude Code subagents you can also configure permission modes (e.g., plan/read-only, auto-accept edits, etc.). 

7) Prompt caching
Deep Agents: uses Anthropic caching where supported.

Claude / Anthropic: prompt caching is a documented platform feature via cache_control. 

And context editing interacts with caching (tool result clearing can invalidate cached prefixes). 

So what’s “the harness” in Claude Agent SDK, structurally?
If you were to sketch Claude Agent SDK as components (conceptually), it’s something like:

Session runtime

persistent shell + working directory 

streaming agent loop (query() yields events) 

Tools

built-in file/command/edit tools 

MCP tool expansion & naming conventions 

Control/guardrails

permissions + secure deployment guidance 

hooks for interception and policy 

Context management

compaction patterns / controls 

tool-result clearing (API-level eviction) 

prompt caching 

Workflow support primitives

subagents 

todo tracking 

file checkpointing / rewind for safe iteration 

If you want, I can go one level deeper
Two useful next cuts (no extra questions needed, just tell me which you want):

“Harness design patterns”: a reference architecture (interfaces, data model, event stream schema) that maps Deep Agents + Claude SDK concepts into a single unified harness abstraction (very relevant to your workflow-engine thinking).

“Claude Agent SDK teardown”: enumerate every major capability surface (sessions, hooks, subagents, checkpointing, structured outputs, slash commands, MCP integration, hosting patterns), and compare it head-to-head with what you observed in LangChain Deep Agents—calling out what’s native, what’s API-level, and what you’d have to implement yourself.

You said:
Question about the server side tool result clearing. Are you saying that when I use Claude code or Agent SDK, the conversation and compaction are manager server side? I suppose this is a reason when Claude Agent SDK is really designed to only target Claude models?
ChatGPT said:

Thought for 22s
Not quite. There are two different mechanisms here, and only one of them is “server-side.”

1) Server-side tool result clearing ≠ “server manages my conversation”
Tool result clearing is part of Anthropic’s server-side context editing: when your request is too large, the platform can edit the request you send by clearing older tool results (replacing them with placeholders) so it fits. That’s “server-side” in the sense that the API applies the edit, not your code. 

But it’s not “Anthropic keeps your conversation and compacts it for you” in a persistent, magical way. It’s more like: for this request, the platform can apply an editing strategy to the conversation payload you provided.

2) Compaction in Claude Agent SDK is explicitly an SDK feature (client-side)
Anthropic draws a clear line here:

Compaction is an SDK feature: the SDK notices token pressure, asks Claude to summarize, then replaces earlier history with that summary and continues. 

Anthropic even says compaction is unlike server-side context editing (which clears content); compaction is “summarize then replace.” 

So for Claude Agent SDK: the harness is managing the conversation on the client/runtime side (your process), using Claude calls to generate summaries.

For Claude Code: it’s still a harness (the Claude Code app) doing the same kind of management; it’s just “their client” rather than “your client.” The /compact command is evidence that compaction is an operation the client performs (it triggers a compaction behavior). 

Why this pushes the Claude Agent SDK toward Claude-only (your hypothesis)
Yes—this is a big reason the Claude Agent SDK is naturally “Claude-first,” because it leans on Claude-/Anthropic-specific primitives and guarantees, for example:

Context editing behaviors like clear_tool_uses_20250919 that the platform understands and can apply. 

Prompt caching via cache_control (also Anthropic-specific in shape/semantics). 

The exact tool-use message protocol + strict sequencing expectations that Claude tooling is designed around. 

You could build an “Agent SDK”-like harness that targets multiple model providers—but you’d have to:

re-implement compaction policies,

implement your own “tool result eviction” strategy for providers without server-side context editing,

abstract tool-call protocols and edge cases,

and normalize caching / tracing semantics.

Anthropic’s SDK can be much more “batteries included” because it can assume those Claude platform capabilities exist.

If you want, I can sketch a provider-neutral harness abstraction that treats:

compaction (summarize/replace),

eviction (clear/placeholder),

artifact externalization (write-to-fs + reference),
as interchangeable “context pressure strategies”—and show exactly which ones are Claude-native vs portable.

You said:
Yes I would love to see your provider neutral harness abstraction!
ChatGPT said:
Awesome — here’s a provider-neutral harness abstraction that treats context pressure, tool protocols, state, and eval/observability as first-class “ports,” so you can swap Anthropic/OpenAI/Bedrock/etc. without rewriting the whole runtime.

I’ll give you:

Core interfaces (what a harness is)

Event model (so everything is observable + replayable)

Context pressure strategies (your compaction/eviction/externalization trio)

Tool protocol normalization (dangling tool calls, strict sequencing, etc.)

Provider adapters (Claude vs others: what’s native vs emulated)

A concrete execution sketch (pseudocode architecture)

1) The harness in one sentence
A provider-neutral harness is a deterministic state machine that advances a run by consuming events, producing a model request, and applying tool results — while delegating provider quirks to adapters and delegating context pressure to pluggable strategies.

2) Core types and interfaces
2.1 The “Run” object: everything is state
type RunId = string;

interface RunState {
  runId: RunId;

  // “authoritative truth” for replay/debug
  transcript: Transcript;            // ordered events/messages
  workingMemory: WorkingMemory;      // summaries, plans, todo list, etc.
  artifacts: ArtifactIndex;          // files/blobs/patches + metadata
  toolLedger: ToolLedger;            // tool calls in-flight, results, failures

  budgets: Budgets;                  // tokens, time, $; per-tool quotas
  policies: PolicySet;               // permissions, HIL gates, redaction rules
  providerHints: ProviderHints;      // caching, context editing flags, etc.
}
Key point: transcript is the source of truth. Everything else can be derived or rebuilt.

2.2 Provider-neutral model API (“LLM port”)
interface ModelProvider {
  id: string;
  capabilities(): ProviderCapabilities;

  // The only “model call” the harness needs.
  generate(req: ModelRequest, ctx: CallContext): AsyncIterable<ModelEvent>;
}

interface ProviderCapabilities {
  maxInputTokens?: number;
  supportsToolCalls: boolean;

  // Provider-specific “extras”
  supportsPromptCaching?: boolean;
  supportsServerContextEditing?: boolean;  // e.g., tool-result clearing
  supportsStructuredOutput?: boolean;      // JSON schema / grammar
}
2.3 Tool runtime (“Tools port”)
interface ToolRuntime {
  listTools(): ToolSpec[];

  invoke(call: ToolCall, ctx: ToolContext): Promise<ToolResult>;
}
2.4 Context engineering (“Context engine port”)
This is the big one for your question.

interface ContextEngine {
  // Build the next model request from state.
  buildRequest(state: RunState): ModelRequest;

  // Handle context pressure *before* calling the model.
  ensureWithinBudget(state: RunState, provider: ProviderCapabilities): Promise<ContextEditPlan>;

  // Apply the edit plan (summaries, clearing, externalization).
  applyEdits(state: RunState, plan: ContextEditPlan): Promise<void>;
}
2.5 Governance + HIL (“Policy port”)
interface PolicyEngine {
  evaluateToolCall(call: ToolCall, state: RunState): Promise<PolicyDecision>;
  evaluateOutput(output: ModelOutput, state: RunState): Promise<PolicyDecision>;
}
2.6 Observability (“Trace port”)
interface Tracer {
  emit(event: HarnessEvent): void;
}
3) The event model (replay-friendly)
Everything becomes an event. This is how you get determinism and “eval harness” compatibility.

type HarnessEvent =
  | { type: "user_input"; content: string; ts: number }
  | { type: "model_request"; request: ModelRequest; ts: number }
  | { type: "model_output"; output: ModelOutput; ts: number }
  | { type: "tool_call"; call: ToolCall; ts: number }
  | { type: "tool_result"; result: ToolResult; ts: number }
  | { type: "context_edit"; plan: ContextEditPlan; ts: number }
  | { type: "policy_gate"; decision: PolicyDecision; ts: number }
  | { type: "error"; error: HarnessError; ts: number }
  | { type: "checkpoint"; snapshotRef: string; ts: number };
Transcript is just HarnessEvent[] with invariants.

This gives you:

tracing

replay

test assertions (“did we ever call network tool without approval?”)

4) Context pressure strategies (the provider-neutral abstraction you want)
You want to treat “Deep Agents eviction,” “Claude server tool clearing,” and “compaction” as interchangeable strategies.

4.1 ContextEditPlan: a sequence of edits
interface ContextEditPlan {
  reason: "token_budget" | "cost_budget" | "time_budget" | "policy";
  estimatedTokenReduction: number;
  edits: ContextEdit[];
}

type ContextEdit =
  | { kind: "summarize"; scope: TranscriptRange; summaryType: "bullet"|"narrative"|"state_json" }
  | { kind: "externalize"; scope: TranscriptRange; artifactKind: "file"|"blob"; pointerStyle: "path"|"artifact_id" }
  | { kind: "clear_tool_results"; scope: TranscriptRange; mode: "results_only"|"inputs_and_results"; placeholderTemplate: string }
  | { kind: "drop"; scope: TranscriptRange } // last resort
  | { kind: "reframe_system"; additions: string[] }; // tighten instructions if drifted
4.2 Strategy selection policy
This is the portable logic:

interface ContextPressureStrategy {
  name: string;

  // Can this strategy run given provider/runtime capabilities?
  isApplicable(provider: ProviderCapabilities, state: RunState): boolean;

  // Propose edits for current pressure.
  propose(state: RunState, provider: ProviderCapabilities): Promise<ContextEditPlan | null>;
}
Then your ContextEngine.ensureWithinBudget() is:

estimate tokens of next request

if over threshold → try strategies in order

return the first plan that fits

4.3 Recommended default strategy ordering
In provider-neutral land, I’d use:

Externalize large tool outputs (cheapest fidelity loss; best for logs/files)

Clear tool results (if provider supports server-side editing, or you emulate)

Compaction summarization (good general-purpose; some fidelity loss)

Drop (last resort)

This maps cleanly to your Deep Agents notes:

“write to filesystem + summary + reference” == externalize

Claude tool clearing == clear_tool_results

conversation compaction == summarize

(dangling tool repair is separate; see next section)

4.4 Provider-specific realization
Same abstract plan, different execution:

If provider supports server-side context editing: clear_tool_results becomes a provider hint in request

Otherwise: you implement it by rewriting transcript locally (replace old tool results with placeholders)

That’s the core trick: don’t bake the mechanism into the policy.

5) Tool protocol normalization (dangling tool calls, strict sequencing)
Providers differ a lot here. You want a ToolProtocolAdapter that enforces transcript invariants.

5.1 Invariants (provider-neutral)
Every tool_call has an eventual tool_result OR a terminal failure record.

Tool call IDs are unique.

Tool results are associated with exactly one tool call.

A model step cannot “assume” tool output without it existing in transcript.

5.2 Adapter
interface ToolProtocolAdapter {
  normalizeModelOutput(output: ModelOutput, state: RunState): NormalizedActions;

  // Repair strategies for malformed history (dangling tools, etc.)
  repairTranscript(state: RunState): RepairPlan | null;
}

interface NormalizedActions {
  assistantMessages: AssistantMessage[];
  toolCalls: ToolCall[];
  stopReason?: "done"|"need_user"|"max_steps"|"policy_block";
}
5.3 Dangling tool call repair (provider-neutral)
Deep Agents does this explicitly. In your abstraction, it becomes:

type RepairPlan =
  | { kind: "insert_tool_error_results"; toolCallIds: string[]; error: string }
  | { kind: "remove_unmatched_tool_calls"; toolCallIds: string[] }
  | { kind: "rewind_to_checkpoint"; checkpointRef: string };
Your harness can run repairTranscript():

on startup (resume)

after crashes

before building a request

This makes “dangling tool call repair” portable.

6) Provider adapters: what’s native vs emulated
Here’s how I’d implement adapters for common providers without tying your harness to any of them.

6.1 Anthropic/Claude adapter
Tool call parsing: native

Prompt caching: native (cache_control)

Server-side context editing: native (clear_tool_uses_* style) when enabled

Compaction: harness-managed (client), but Claude is good at producing “state summaries”

So in Claude mode:

prefer clear_tool_results as a hint if you want it

still keep externalization/compaction as fallback

6.2 OpenAI adapter (Responses/Assistants style)
Tool calling: native

Prompt caching: different semantics (provider-specific; sometimes none)

Server-side context editing: typically not an equivalent “clear tool uses” mechanism

So you emulate clearing/externalization/compaction in harness

6.3 Bedrock / others
Same story: tool calling may exist, server-side editing likely doesn’t → harness does it.

Net: Provider neutrality is achievable if you treat server-side editing as an optimization, not as your only plan.

7) Putting it together: execution sketch
Here’s the harness step loop with all the ports:

async function runUntilStop(state: RunState) {
  while (true) {
    // 0) Repair invariants first (portable dangling tool repair)
    const repair = toolProtocol.repairTranscript(state);
    if (repair) applyRepair(state, repair);

    // 1) Build request
    let req = contextEngine.buildRequest(state);

    // 2) Budget check + context pressure strategy
    const plan = await contextEngine.ensureWithinBudget(state, provider.capabilities());
    if (plan) {
      tracer.emit({ type: "context_edit", plan, ts: Date.now() });
      await contextEngine.applyEdits(state, plan);
      req = contextEngine.buildRequest(state); // rebuild after edits
    }

    // 3) Model call (streaming)
    tracer.emit({ type: "model_request", request: req, ts: Date.now() });

    const actions = await collectAndNormalize(provider.generate(req), state);

    // 4) Record assistant messages
    appendAssistantMessages(state, actions.assistantMessages);

    // 5) Handle tool calls
    for (const call of actions.toolCalls) {
      const decision = await policy.evaluateToolCall(call, state);
      tracer.emit({ type: "policy_gate", decision, ts: Date.now() });

      if (decision.kind === "require_human") {
        await waitForApproval(decision);
      }
      if (decision.kind === "deny") {
        appendToolError(state, call, "Denied by policy");
        continue;
      }

      tracer.emit({ type: "tool_call", call, ts: Date.now() });

      const result = await tools.invoke(call, mkToolContext(state));
      tracer.emit({ type: "tool_result", result, ts: Date.now() });

      appendToolResult(state, result);
    }

    // 6) Stop?
    if (actions.stopReason) return actions.stopReason;

    // 7) Checkpoint (optional)
    if (shouldCheckpoint(state)) {
      const ref = await saveCheckpoint(state);
      tracer.emit({ type: "checkpoint", snapshotRef: ref, ts: Date.now() });
    }
  }
}
Notice what’s provider-neutral:

transcript invariants

context pressure plans

policy gating

event stream

Notice what’s provider-specific:

provider.generate() implementation

tool-call parsing

optional request hints (cache, server context editing, JSON schema)

8) A “portable” mapping of your Deep Agents features into this abstraction
Large tool result eviction → externalize edit

Subagents → a tool call into an SubagentRuntime (which itself is just another harness with different config)

Compaction → summarize edit with your 85% threshold policy

Dangling tool repair → ToolProtocolAdapter.repairTranscript()

Todos → workingMemory.todos with a tool write_todos (provider-independent)

HIL → PolicyEngine returning require_human

Prompt caching → providerHints.cache in ModelRequest when supported

9) If you want a “minimum viable harness spec”
If you’re thinking about standardizing this (like a portable harness contract), the minimum spec I’d write down is:

Transcript event schema (the HarnessEvent union)

Tool call schema + invariants

ContextEditPlan schema

ProviderCapabilities flags

A conformance suite:

run + replay equivalence

invariants always hold

context pressure strategies reduce tokens as claimed

policy gates are enforced

That spec is what makes “agent harness” a real product surface vs a pile of ad hoc glue.

If you want to take this one more step: I can express the whole thing as a TypeSpec/JSON Schema “Harness Contract” (events, tool calls, edit plans, run state snapshot format). That’s usually the easiest way to make multiple teams/providers implement the same harness semantics.

You said:
Question, when it comes to server managed conversations such as Foundry, Bedrock, Assistants or even Responses, what needs to change? The stateful server component means we cannot reliably edit the conversation, compact it, etc. right?
ChatGPT said:

Thought for 14s
Yeah — once the provider owns the thread, the harness has to change shape. The core shift is:

You stop treating “the provider conversation” as your editable source of truth, and instead treat it as an opaque execution log (or a black-box state machine). Your harness keeps its own canonical state and only projects what it wants the model to see.

That’s the provider-neutral way to stay in control.

What breaks (or becomes unreliable) with server-managed threads
When the server keeps the conversation (Assistants/Threads, Bedrock Agents sessions, Foundry-like server state):

You often can’t delete / rewrite / reorder earlier messages.

You may not know exactly what the server will include in the next context window (truncation heuristics, tool-result retention, hidden system messages).

“Compaction” becomes hard because compaction is rewriting history, and server threads usually don’t let you rewrite history—only append.

“Eviction” (like dropping large tool results) is similarly hard unless the platform has a first-class feature for it.

So yes: you typically cannot reliably edit the conversation in place.

But: you can still get almost all the benefits of compaction/eviction — you just do it differently.

The harness pattern that works: Canonical state + Projection
1) Maintain your own canonical transcript
Your harness keeps an internal event log (messages, tool calls/results, artifacts) as the source of truth.

2) Treat provider thread as one of these modes
Pick one:

Mode A — Opaque memory (black box):
You send incremental messages and accept that the provider decides what it remembers. This is simplest but least controllable.

Mode B — Append-only log:
You use the server thread for persistence/traceability, but you don’t trust it for prompt assembly. You still keep canonical state client-side.

Mode C — Stateless projection:
Even if the provider can store state, you choose not to rely on it for context. Each step you send a curated prompt built from your state (this is the most harness-y approach, but sometimes fights the product shape).

What changes in compaction / eviction strategies
Compaction (summarize-and-replace)
In server-managed threads you usually can’t “replace,” so you do summarize-and-supersede:

Generate a state summary (or “run summary”) message.

Append it to the thread.

From that point on, instruct the agent to treat the summary as authoritative and older messages as archival.

This works, but it’s not perfectly reliable because the model might still see old stuff if the server includes it. So you strengthen it by:

Putting the summary in a system/developer slot if the platform supports that kind of high-priority instruction.

Using a structured summary format (facts, goals, decisions, open questions) and referencing it explicitly.

Eviction / Large tool outputs
If you can’t remove old tool results from the thread:

Externalize big outputs to artifacts (files/blob store) and only append a short summary + pointer.

Prefer tools that return handles rather than massive payloads.

Design tools with “paged read” (e.g., read_lines(file, start, count)).

This makes “eviction” effectively happen at the tool boundary, not via thread editing.

Tool-result clearing
If the platform has a native “clear tool results” or “context editing” feature, you can use it as an optimization. If it doesn’t, you emulate it via artifact pointers (above).

The “forking” escape hatch (when you truly need rewrite)
If you need true compaction semantics (replace history) and the server thread is immutable:

Start a new thread/session (“fork”)

Seed it with:

the summary

the minimal active artifacts/pointers

any durable run state you need

This restores full control at the cost of continuity inside the provider thread.

This is the most reliable way to guarantee the model isn’t being influenced by old stuff you can’t delete.

Practical rule of thumb
If your agent needs strong guarantees (replayability, deterministic-ish debugging, rigorous budgets):

✅ Keep a canonical harness transcript and treat provider threads as storage/transport only.
✅ Use projection + artifact pointers + forking when needed.

If your agent is more conversational/productivity and less safety-critical:

✅ Let the provider thread be the memory, and just add summarization messages occasionally.

What to change in your harness abstraction (concretely)
Earlier we had ContextEngine.applyEdits(state, plan) that rewrites transcript. With server-managed threads:

Replace “edit transcript” with “edit projection”
Add a layer:

Canonical transcript (your store; editable)

Provider thread (append-only; not trusted for exact prompt assembly)

Projection (what you intend the model to see next)

So your ContextEngine becomes:

buildProjection(state) -> PromptBundle

appendToProviderThread(delta) (optional; for traceability)

forkProviderThread(seedBundle) when you must “reset”

And “compaction” becomes one of:

update canonical transcript (always possible)

append summary to provider thread (supersede)

fork thread (true reset)

The punchline
You’re exactly right about the core tension: server-side statefulness reduces your ability to rewrite history.

The way out is: don’t make the server thread your truth. Make your harness truth, and treat the server as either an append-only log or a runtime you can fork.

If you tell me which platform shape you’re targeting (Foundry-like: durable run + triggers; or Assistants-like: thread-per-user; or Bedrock Agents-like: session-per-task), I can sketch the best “projection + forking + artifact” strategy for that shape with a concrete state diagram.

You said:
Back to the architecture proposal... I'm thinking this should be able to be build using the agent-framework Workflows since they are stateful and have a runtime that can keep things alive. Can you take a look at how we could model the engine this way?
ChatGPT said:
Yep — a provider-neutral harness maps really cleanly onto a stateful workflow runtime like agent-framework Workflows. The key move is to treat the “agent harness” not as an ad-hoc loop, but as a first-class workflow with durable state, typed events, and pluggable strategies.

Below is a concrete way to model it.

1) How the harness maps onto a stateful workflow engine
Harness = a long-running workflow instance
Workflow instance state = your canonical RunState

Workflow steps = “plan → model → tools → observe → context-pressure → checkpoint”

Events = the transcript (your replay log)

Policies/HIL = gates/branches that pause/resume the workflow

So the “agent loop” becomes a repeatable subgraph in the workflow runtime.

2) Proposed state model inside Workflows
Create a single durable state object (versioned) that lives with the workflow instance:

RunState:
  runId: string
  phase: enum [idle, running, waiting_human, completed, failed]

  transcript:  # append-only canonical log
    - eventId: string
      type: enum [user_input, model_request, model_output, tool_call, tool_result, context_edit, policy_gate, checkpoint, error]
      payload: object
      ts: datetime

  workingMemory:
    summary: string
    todos: [ {id, title, status} ]
    plan: object
    facts: object

  artifacts:
    - artifactId: string
      kind: enum [file, blob, patch]
      uri: string
      meta: object

  toolLedger:
    inflight: [ {callId, toolName, args, ts} ]
    completed: [ {callId, status, resultRef, ts} ]

  budgets:
    tokenBudget: {maxInput, maxOutput, softThreshold}
    costBudget: {maxDollars}
    timeBudget: {deadline}

  provider:
    providerId: string
    capabilities: object
    hints: object # caching/context editing knobs if supported
This matches the harness abstraction we discussed, but “stateful workflow engine” makes it durable-by-default.

3) Model the harness as a workflow graph
At a high level:

Repair invariants (dangling tool calls, partially written steps)

Build projection (what to send to provider)

Apply context pressure strategy (externalize/clear/compact)

Call model (streaming)

Normalize actions (messages + tool calls)

Policy gate (HIL approvals)

Execute tools

Append results

Stop? else loop

In workflow terms: a loop with durable checkpoints between every phase.

4) Concrete node types you’d want in agent-framework Workflows
A) Deterministic “reducers” (pure state transforms)
These nodes read RunState and append/modify it deterministically.

RepairTranscriptNode

BuildProjectionNode

ApplyContextEditPlanNode

NormalizeModelOutputNode

AppendToolResultNode

CheckpointNode

These should be pure (no IO) when possible, so replay is stable.

B) Effectful nodes (IO)
ModelCallNode(providerAdapter)

ToolInvokeNode(toolRuntime)

ArtifactWriteNode(storage)

ArtifactReadNode(storage) (rare; mostly tools do reads)

C) Gates / waits
PolicyEvaluateNode(policyEngine)

HumanApprovalWaitNode (pause/resume)

5) Where “provider-neutral” lives: adapters + strategies
Provider Adapter (per model backend)
Implements:

Generate(projection) -> stream(ModelEvent)

tool-call protocol parsing/normalization

provider-specific knobs (caching, server context editing, structured outputs)

This is the only part that’s “Claude vs Responses vs Bedrock.”

Context Pressure Strategy (pluggable policy)
Implements:

compute token estimate (or approximate)

propose ContextEditPlan:

externalize(toolResultRange)

clear_tool_results(range) (if provider supports; otherwise emulate)

summarize(range) / compaction

drop(range) (last resort)

This is portable, and it’s exactly where your Deep Agents vs Claude differences get unified.

6) Modeling the loop in Workflows
If your workflow DSL supports a while/until construct, great. If it doesn’t, use a “superstep” pattern:

Each iteration is one Turn sub-workflow

It schedules itself again if not done

State is checkpointed at boundaries

Turn subworkflow
repair

context_pressure

model_call

normalize

tool_phase

checkpoint

branch: done or continue

This maps to Pregel-ish “superstep barrier” semantics you’ve already thought about.

7) HIL and safe tool execution becomes first-class workflow pausing
This is where stateful workflows shine.

PolicyEvaluateNode returns decisions:

allow

deny

require_human(approvalSpec)

If require_human, workflow transitions to waiting_human

A trigger/event resumes it with approval outcome

This gives you:

deterministic audit trails (approval is an event)

resumability

external integration (Teams/email/etc.) without hacking the agent loop

8) “Server-managed conversation” compatibility (Foundry/Assistants/Bedrock)
Workflows let you pick your mode per provider:

Mode 1: Canonical transcript in workflow state (recommended)
Provider thread is treated as optional/opaque

You send a projection each time (curated prompt)

If provider insists on thread IDs, you still store them, but don’t trust them as “truth”

Mode 2: Append-only provider thread + workflow truth
Write important events to provider thread for continuity

Still keep canonical transcript in RunState

Compaction becomes “summarize-and-supersede” messages and/or “fork thread” (new session) when needed

Workflows make “forking” a clean branch: ForkSessionNode creates new provider thread and updates provider.sessionId in state.

9) A minimal workflow spec sketch (pseudo-YAML)
This is illustrative (not your exact syntax):

workflow: AgentHarness
state: RunState

nodes:
  - id: repair
    type: RepairTranscriptNode

  - id: propose_context_edits
    type: ProposeContextEditsNode
    inputs: { state: $.state, providerCaps: $.state.provider.capabilities }

  - id: apply_context_edits
    type: ApplyContextEditPlanNode
    when: $.plan != null

  - id: build_projection
    type: BuildProjectionNode

  - id: call_model
    type: ModelCallNode
    inputs: { projection: $.projection, provider: $.state.provider }

  - id: normalize
    type: NormalizeModelOutputNode

  - id: policy_gate
    type: PolicyEvaluateNode
    foreach: $.normalized.toolCalls

  - id: wait_human
    type: HumanApprovalWaitNode
    when: $.policyDecision.kind == require_human

  - id: invoke_tool
    type: ToolInvokeNode
    foreach: $.approvedToolCalls

  - id: checkpoint
    type: CheckpointNode

  - id: decide
    type: DecideContinueNode

edges:
  - repair -> propose_context_edits -> apply_context_edits -> build_projection -> call_model -> normalize -> policy_gate
  - policy_gate -> wait_human -> invoke_tool -> checkpoint -> decide
  - decide (continue) -> repair
  - decide (done) -> end
This is literally the harness loop, expressed as a durable workflow.

10) What you get “for free” from agent-framework Workflows
If your Workflows runtime already has:

durable state

pause/resume

event triggers

structured inputs/outputs

execution history

…then your harness becomes much simpler. You mostly implement:

provider adapters

tool runtimes

context pressure strategies

projection builder

invariant repair

Everything else is workflow semantics.

11) The two design choices that matter most
Choice A: Transcript as events vs messages
Use events. Messages are just one event type.
This makes replay/debugging/eval far better and avoids provider protocol coupling.

Choice B: Where “projection” is produced
Projection should be a deterministic function of state:

deterministic ordering

deterministic selection of memory/artifacts

deterministic “what to include” policy

That’s how you stay provider-neutral and testable.

If you want, I can take a stab at modeling this in your workflow schema style (agents/nodes/orchestration/error handling), including:

the exact RunState schema

ContextEditPlan schema

the “Turn” subworkflow node set

error handling paths (tool failure, model failure, approval timeout)

and how to represent streaming model events in your event stream format.

You said:
Tell me about the repair step. What is that doing?
ChatGPT said:
Great question — the repair step is subtle but foundational once you move to a durable, stateful, replayable harness.

I’ll explain it in layers:

What “repair” means conceptually

Why it’s mandatory in a stateful workflow engine

What invariants it enforces

Concrete classes of repairs (with examples)

What repair is not

How to structure it as a deterministic workflow node

1) What the repair step is
The repair step is a deterministic reconciliation pass that runs before every turn of the agent loop to ensure:

“The current RunState is internally consistent, complete, and safe to advance.”

It does not generate new reasoning, call the model, or make creative decisions.

It simply answers:

“If we resumed execution right now, would the next model call see a coherent world?”

If the answer is “no,” repair fixes it.

2) Why repair is mandatory in a stateful workflow engine
Once you have:

durable state

pause / resume

retries

crashes

human-in-the-loop waits

provider retries / partial streams

…you will get mid-turn state corruption.

Examples:

A workflow crashes after writing a tool_call event but before the tool_result.

A model stream is interrupted halfway through.

A tool invocation times out after the call is recorded.

A human approval never arrives.

A provider returns malformed tool calls.

A previous run is resumed after a code upgrade.

Without a repair step, these become undefined behavior bugs.

With repair, they become recoverable states.

Think of repair as fsck for your agent state.

3) The invariants repair enforces
Your harness should define hard invariants that must be true before a turn can proceed.

Typical invariants:

Transcript invariants
Events are append-only and ordered.

Every tool_call has:

a matching tool_result, OR

a terminal failure record.

No “future” events relative to current phase.

No duplicate tool call IDs.

No assistant messages that reference tool outputs that don’t exist.

Tool ledger invariants
In-flight tool calls reflect reality.

No tool is both “inflight” and “completed.”

Failed tool calls are marked terminal.

Phase invariants
If phase == waiting_human, no model calls may occur.

If phase == completed, no further events may be appended.

Budget invariants
Budgets are monotonically decreasing.

Overruns are explicitly marked.

Repair exists to restore these invariants.

4) What repair actually does (concrete repair classes)
Below are the main categories of repair, in increasing order of severity.

A) Dangling tool call repair (most common)
Problem

tool_call(id=123, name=write_file)
# crash happens here
No result ever recorded.

Repair options

Insert a synthetic tool_result:

tool_result(id=123, status=error, message="Execution interrupted")
Or rewind to a checkpoint before the tool call.

Or mark tool call as abandoned + forbid retry.

Rule of thumb

Never let a model see a tool call without a terminal outcome.

B) Partial / malformed model output repair
Problem

Streaming response ended mid-token.

Tool call JSON is invalid.

Model produced an assistant message and a tool call but protocol order is broken.

Repair actions

Truncate incomplete assistant output.

Convert malformed tool calls into explicit errors.

Insert a model_error event.

Optionally retry the model call (policy-dependent).

Repair makes the transcript syntactically valid again.

C) Phase mismatch repair
Problem

State says waiting_human, but no pending approvals exist.

State says running, but a human approval is outstanding.

Repair

Recompute phase from transcript + policy state.

Correct phase to match reality.

This is critical when resuming after crashes or restarts.

D) Ledger reconciliation
Problem

Tool ledger says call is inflight, but transcript shows result.

Transcript shows tool result, but ledger missing entry.

Repair

Rebuild ledger from transcript (authoritative).

Ledger is derived state; transcript is truth.

E) Budget reconciliation
Problem

Token/cost counters are missing or inconsistent after restart.

Repair

Recompute estimates from transcript.

Apply conservative defaults if unknown.

Mark “budget uncertain” if exact reconstruction isn’t possible.

F) Context-pressure aftermath repair
Problem

A compaction or eviction plan partially applied before crash.

Repair

Detect incomplete ContextEditPlan.

Either:

roll it back (restore from checkpoint), or

complete it deterministically.

G) Version / schema migration repair (advanced but important)
Problem

Workflow engine upgraded.

State schema changed.

Repair

Migrate state to new schema.

Insert state_migration event.

Preserve replayability.

This is huge for long-lived agents.

5) What repair is not
Repair does not:

Re-plan tasks

Decide next actions

Call the LLM

“Fix” bad reasoning

Change intent or goals

Those belong to planning or policy nodes.

Repair only restores mechanical correctness.

A good litmus test:

If two engineers ran repair independently on the same state, would they produce the same result?

If “yes,” it belongs in repair.

6) How to structure repair as a workflow node
Inputs
RunState

Outputs
Updated RunState

Optional RepairReport

Properties
Pure (no IO)

Deterministic

Idempotent

Fast

Pseudocode sketch
function repair(state: RunState): RunState {
  state = repairDanglingTools(state);
  state = repairMalformedModelOutputs(state);
  state = reconcileToolLedger(state);
  state = recomputePhase(state);
  state = reconcileBudgets(state);
  return state;
}
Emit repair events (optional but recommended)
Each repair should emit structured events:

{
  "type": "repair",
  "kind": "dangling_tool_call",
  "toolCallId": "123",
  "action": "insert_error_result"
}
This gives you:

auditability

debugging

evaluation hooks (“how often do we need repair?”)

7) Why repair becomes more important with server-managed conversations
When providers manage threads:

You may not fully control ordering.

Tool results may be injected asynchronously.

Provider retries may duplicate messages.

Repair becomes your shield:

reconcile what you think happened vs what actually happened

normalize provider quirks into your invariant model

8) Mental model
Think of the harness as a transactional system:

Model call + tool execution = a transaction

Crashes happen mid-transaction

Repair is your transaction recovery log

Without it, stateful agents are brittle.
With it, they are resumable processes.

If you want, next we can:

define a formal invariant list for your harness,

or design a RepairReport schema you can feed into evals (“repair rate as a quality signal”),

or walk through a real failure timeline and show how repair keeps the agent alive step by step.

You said:
Got it. Let’s walk through the agent loop again. One thing I’m trying to grasp is how much of the loop should be predetermined by the workflow and how much should be within the model loop. When I say model loop I mean the loop of the model inferencing and calling tools. At one extreme you could just give the model a tool for each the loop states and let it manage everything. That’s probably not going to work well at this point but you get my point.
ChatGPT said:
Yep — this is the design tension: workflow-determined control vs model-determined control.

A good way to think about it is: the workflow should own anything that needs guarantees (durability, budgets, safety, ordering, observability), and the model should own anything that benefits from flexibility (reasoning, choosing tools, composing steps, writing outputs).

If you push too much into the model, you get chaos. If you push too much into the workflow, you get a brittle “agent-shaped state machine” that’s expensive to author.

Here’s a clean way to slice it.

The two loops
A) The workflow loop (outer loop)
This is your “OS scheduler”:

defines phases

persists state

gates risky actions

enforces budgets & stop conditions

ensures transcript correctness (repair)

provides resumability / HIL / triggers

B) The model loop (inner loop)
This is the “CPU”:

reasons about what to do next

selects tools

interprets tool outputs

produces intermediate working notes and final artifacts

You can run multiple inner-loop steps per outer-loop turn, or just one, depending on how you want to checkpoint.

Design spectrum (from “workflow owns everything” to “model owns everything”)
1) Model-as-controller (not great today)
Workflow just calls the model, model calls “advance_state(next_state)” tools.

Pros: minimal workflow authoring
Cons: no guarantees; hard to debug; prone to drift; can deadlock; unsafe.

This is the extreme you described.

2) Hybrid harness (recommended)
Workflow owns guardrails + lifecycle, model owns micro-decisions.

The workflow exposes capability tools (search, write file, create patch, run tests), not “state transition” tools.

3) Workflow-as-controller (too rigid)
Workflow is a big deterministic state machine: plan → research → draft → test → finalize.
Model is used as a subroutine inside each state.

Pros: predictable
Cons: authoring burden; brittle; doesn’t generalize across tasks.

A concrete partition that works well
Workflow should predetermine:
1) Transaction boundaries

what constitutes an atomic step you can checkpoint and replay

typically: “one model call + all tool calls it requested + results recorded”

2) Invariants

dangling tool calls fixed

consistent ledger

phase correctness

3) Gates

HIL approvals (writes, network, spending, destructive ops)

policy enforcement and redaction

4) Budgets

max steps, token/cost/time

per-tool quotas (e.g., browser: 10 calls)

5) Context pressure policy

when to compact/evict/externalize

but the model can generate the summary content

6) Stop reasons

done / need user / blocked / max budget

The model should decide:
1) Which tools to call and with what args
2) How to interpret tool results
3) How to decompose tasks (within allowed budget)
4) What artifacts to produce (report, patch, files)

The “right” inner loop size: 1-step vs multi-step
Option A: 1 model call per workflow iteration (tight control)
Outer loop:

build prompt

model call

execute returned tool calls

checkpoint

repeat

Pros: maximum durability, best observability
Cons: more latency, more provider calls

This is usually the safest default for early systems.

Option B: N model calls per workflow iteration (faster, less durable)
Outer loop calls a “model micro-loop” node that may run several think/act cycles before returning.

Pros: faster, fewer workflow transitions
Cons: if you crash mid micro-loop, repair is harder; less HIL granularity

A good compromise is:

keep the micro-loop bounded (e.g., 3 steps max or 30s max)

checkpoint at the end

A pattern I like: “Plan once, Act many, but checkpoint every Act”
Make workflow structure small but meaningful:

Initialize

Plan (one model call; produce a plan + tool budget)

Act loop (repeat small steps)

Finalize (compose answer / artifacts)

Done

Where Act loop is the classic tool-calling loop, but each “Act” is transactional.

This gives you enough structure to avoid drift, but doesn’t over-prescribe.

Don’t give the model “state transition” tools (usually)
Instead of tools like:

enter_planning()

enter_execution()

finish()

Give it tools like:

search_web(query)

read_file(path)

write_patch(diff)

run_tests()

ask_human(question) (or request approval)

Then the workflow decides when phases start/end based on:

what the model outputs

budgets

policies

This is a critical harness principle: tools should be capabilities, not control.

How the full loop looks in this hybrid design
Outer workflow step (transaction)
Repair invariants

Apply context pressure strategy if needed (externalize/compact)

Build projection

Call model

Normalize into:

assistant message(s)

tool calls

stop signal (done/need-user)

Policy gate tool calls (HIL)

Execute tools, record results

Stop? else checkpoint and loop

Inner model behavior
The model sees:

current goal

current summary

relevant artifacts/pointers

tool list + constraints
…and emits tool calls.

How to decide “how much predetermined workflow”
Use this rule:

If a mistake here would be expensive/unsafe or would break replay/debugging, the workflow owns it.

So workflow owns:

spending money

destructive writes

external network

long-running loops

retries/backoff

checkpoint boundaries

Model owns:

reasoning

which tool

how to interpret

content generation

A nice compromise for your “state tools” thought experiment
If you do want the model to influence phases, do it via suggestions, not commands.

Example:

model can output {"suggested_phase":"research"} in structured output

workflow treats it as advisory and can accept/reject

This gives you the flexibility of “model knows what to do next” without giving it the keys to the car.

If you want, I can propose an explicit phase schema that fits agent-framework Workflows (like Plan → Act → Reflect → Finalize) plus the exact stop conditions and what the model is allowed to do in each phase. That tends to make the division crisp and implementable.

You said:
Ok that’s good info. You said “8.	Stop? else checkpoint and loop”… Answering the question of when to stop isn’t that simple. What strategy do you recommend? What context do you have to keep around to ensure this works?
ChatGPT said:
You’re right: “stop?” is not a boolean. In practice you want layered stop logic with a strongly-typed notion of “done-ness” plus enough retained context to validate it.

A good strategy is: declare the contract of completion, keep a compact “task ledger” to measure it, and require an explicit finalization handshake (model proposes done; harness verifies).

Recommended stop strategy: 3 layers
Layer 1: Hard stops (workflow-owned)
These don’t require model judgment.

step limit reached

token/cost/time budget exceeded

repeated no-progress loop detected

policy block (needs approval, denied)

tool failures exceed threshold

If any of these triggers, you stop with a reason (and usually ask user / escalate).

Layer 2: Contract check (harness-verifiable)
You define “done” as a set of verifiable conditions.

Examples (depending on task type):

“Produced artifact X at path Y”

“All required fields in output schema are present and valid”

“Tests pass” / “build succeeds”

“Answered all user questions” (mapped to checklist items)

“No open todos remain” (or only allowed ones remain)

This is the most important layer for reliability: done-ness isn’t vibes, it’s a checklist.

Layer 3: Model judgment (advisory)
Let the model propose done when it believes:

the contract is satisfied

there are no unresolved uncertainties worth spending more budget on

But the model’s “done” signal is only accepted if Layer 2 passes.

The “finalization handshake”
Make “final answer” a distinct phase:

Model emits propose_done with a completion report

Harness runs verification

If verified → finalize and stop

If not → generate a “gap list” and continue

That gap list becomes next-step guidance.

Completion report schema (example)
Have the model output something like:

{
  "stop_recommendation": "done|need_user|continue",
  "deliverables": [
    {"type":"file","path":"out/report.md","description":"..."},
    {"type":"answer","description":"Summary in chat"}
  ],
  "requirements_coverage": [
    {"req_id":"R1","status":"met","evidence":"..."},
    {"req_id":"R2","status":"met","evidence":"..."}
  ],
  "open_questions": [],
  "confidence": 0.78
}
The harness can validate:

deliverables exist

required req_ids are all met

open_questions empty or acceptable

confidence above threshold (optional)

What context you must keep around for stopping to work
1) A “Task Contract” (explicit completion criteria)
This is the biggest missing piece in most agents.

Create a durable object early (either inferred from user input or provided by workflow):

TaskContract:
  goal: string
  required_outputs:
    - id: R1
      type: artifact|response|action
      predicate: "file_exists('...')" | "json_schema_valid('...')" | "tool_result('tests').passed"
  must_ask_user_if_missing:
    - id: Q1
      question: "..."
  acceptability:
    max_known_unknowns: 0
    min_confidence: 0.6
If you don’t keep this around, the agent has no stable target for “done.”

2) Coverage ledger (what has been satisfied and why)
A compact, persistent ledger mapping criteria → evidence:

CoverageLedger:
  R1:
    status: met
    evidence:
      - eventRef: e123
        kind: artifact
        value: "out/report.md"
  R2:
    status: unmet
This is what makes stop decisions robust across compaction.

3) Open loops / outstanding commitments
Agents drift because they forget what they promised.

Keep:

open todos (with owner + status)

pending tool calls / approvals

unanswered user questions

This should be separate from the transcript so it survives compaction.

4) “Progress signals” to detect spinning
Keep a rolling window of:

last N actions (tool calls, summaries)

a hash of “state of the world” (e.g., artifact set + ledger)

last N model “intents” / selected tools

If nothing changes across several iterations, you’re looping.

A simple, effective heuristic:

define progress_fingerprint = hash(coverageLedger + artifactIndex + openTodos)

if fingerprint unchanged for K turns → stop with “stalled” and ask user / re-plan

5) Uncertainty register (known unknowns)
If the model says “done” but has unresolved uncertainties, it shouldn’t stop silently.

Persist:

assumptions made

unknowns that could change output correctness

what would resolve them (tool / user)

Then “done” requires either:

unknowns = 0, or

unknowns accepted explicitly by user (“ship with assumptions”).

How compaction interacts with stopping
If you compact, you must preserve the contract + ledger + open items outside the chat history.

So your compaction summary should explicitly include:

goal

current plan (short)

open todos

open questions

ledger status summary

But more importantly: those should live in structured state, not only in text.

A practical stop algorithm (works well)
Each turn:

If any hard stop → stop(reason)

If model recommends need_user → stop(reason=need_user, questions)

If model recommends done:

run verifiers:

contract predicates

schema validation

artifact existence

tests pass (if applicable)

if pass → stop(done)

else → continue with a generated gap list

Else if progress_fingerprint unchanged K times → stop(stalled)

Else continue

This keeps the model in the loop but makes stopping reliable.

Minimal set of “stop context” you should always persist
If you only keep 5 things, keep these:

TaskContract (completion criteria)

CoverageLedger (criteria → evidence)

OpenTodos / OpenQuestions

ArtifactIndex (what exists + pointers)

ProgressFingerprint history (for stall detection)

Everything else can be compacted aggressively.

If you want, I can turn this into:

a concrete TaskContract schema

a StopDecisionNode design for Workflows (inputs/outputs, verifiers)

and a set of default predicates/verifiers (file exists, json schema validate, tool result contains, etc.) suitable for a provider-neutral harness.




No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.
