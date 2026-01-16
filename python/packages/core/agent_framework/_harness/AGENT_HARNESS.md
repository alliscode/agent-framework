# Agent Harness: Concepts and Architecture

## What is an Agent Harness?

An **Agent Harness** is the runtime control plane that enables reliable, long-running agent execution. It is *not* a specific agent, nor a specific set of tools—it's the infrastructure layer that any agent can run within.

The harness provides:
- The **outer loop** that orchestrates agent turns (the agent itself has an inner loop of inference → tool calls → inference)
- **State and durability** (checkpoints, resume, memory)
- **Stop conditions** (turn limits, stall detection, task completion verification)
- **Governance** (permissions, human-in-the-loop, policies)
- **Context management** (compaction, eviction, externalization)
- **Observability** (traces, transcripts, replay)

You can implement all of this with no filesystem, no shell, no code execution—the harness is independent of what capabilities the agent has access to.

## The Three-Layer Model

A clean architecture separates three concerns that are often conflated:

```
┌─────────────────────────────────────────────────────────────────┐
│                         PERSONA                                 │
│         Goals, style, tool preferences, safety posture          │
│                                                                 │
│   "Coding Assistant"    "Customer Support"    "Research Agent"  │
├─────────────────────────────────────────────────────────────────┤
│                        ENVIRONMENT                              │
│            Capabilities available to the agent                  │
│                                                                 │
│   Filesystem?  Shell?  Browser?  APIs?  Database?  Artifacts?   │
├─────────────────────────────────────────────────────────────────┤
│                         HARNESS                                 │
│                    Runtime control plane                        │
│                                                                 │
│   Loop • State • Repair • Context • Policy • Observability      │
└─────────────────────────────────────────────────────────────────┘
```

### Harness (Control Plane)
Always present. Manages *how* the agent executes:
- Turn-based execution loop
- Checkpoint and recovery
- Stop condition evaluation
- Context pressure management
- Policy enforcement
- Event logging and tracing

### Environment (Capability Plane)
Pluggable. Defines *what* the agent can do:
- Filesystem access (or not)
- Shell execution (or not)
- Browser automation (or not)
- API integrations
- Artifact storage

Different deployments have different environments. A coding agent needs filesystem and shell. A customer support agent needs CRM APIs. A research agent needs web search. The harness works with all of them.

### Persona (Configuration)
Defines *who* the agent is:
- System instructions and goals
- Which tools are visible/enabled
- Risk tolerance and autonomy level
- What "done" means for this agent
- Style and interaction patterns

## Why This Separation Matters

### Filesystem is Not Definitional

Many reference implementations bundle filesystem + shell + editor and call it "an agent harness." But that's conflating harness with environment.

Filesystem is common because it cheaply solves several problems:
- Artifact persistence (store outputs, patches, reports)
- Large-result eviction (write to file, keep pointer in context)
- Intermediate workspace (code generation, test artifacts)
- Replay and debug (diff artifacts, compare states)

But filesystem is a *capability*, not a requirement. A customer support agent with only CRM tools still needs a harness—it just doesn't need filesystem tools.

### Shell is a High-Risk Capability

Shell access appears in coding harnesses because agents need to run tests, linters, and builds. But shell is arbitrary code execution and must be prohibited in many environments:
- SaaS multi-tenant backends
- Regulated environments (HIPAA, PCI, FedRAMP)
- Customer-facing copilots
- Production infrastructure agents

A harness that *requires* shell is unusable in these deployments. Shell should be:
- Explicitly declared as a capability
- Policy-gated and budgeted
- Logged with full provenance
- Disabled by default in restricted environments

## Persona Capability Matrix

Different agent personas need different environmental capabilities:

| Capability | Chat-Only | API Agent | Research | Coding | Ops/Infra |
|------------|-----------|-----------|----------|--------|-----------|
| **Artifact Store** | Optional | Yes | Yes | Yes | Yes |
| **Filesystem** | No | No | Optional | Yes | Optional |
| **Shell** | No | No | No | Yes (sandboxed) | Restricted |
| **Web Search** | No | Optional | Yes | Optional | No |
| **Browser** | No | No | Yes | No | No |
| **HTTP APIs** | No | Yes | Yes | Optional | Yes |
| **Database** | No | Optional | Optional | Optional | Yes |

The harness remains constant across all these personas. What changes is the environment and configuration.

## Implementing the Layered Model

### Harness as Workflow

The harness is implemented as a workflow—a composition of specialized executors in a loop:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HARNESS OUTER LOOP                                │
│                                                                             │
│   ┌────────┐     ┌─────────────────┐     ┌────────────┐     ┌───────────┐  │
│   │ Repair │ ──▶ │ Context Pressure│ ──▶ │ Agent Turn │ ──▶ │   Stop    │  │
│   │        │     │   (optional)    │     │            │     │ Decision  │  │
│   └────────┘     └─────────────────┘     └────────────┘     └───────────┘  │
│        ▲                                                          │        │
│        │                                                          │        │
│        │                    ┌──────────────────┐                  │        │
│        └────────────────────│    continue?     │◀─────────────────┘        │
│                             └──────────────────┘                           │
│                                     │ no                                   │
│                                     ▼                                      │
│                             ┌──────────────────┐                           │
│                             │   Yield Result   │                           │
│                             └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘

Each executor's responsibility:

┌─────────────────────────────────────────────────────────────────────────────┐
│ REPAIR                                                                      │
│ • Fix dangling tool calls (call recorded but no result)                     │
│ • Initialize harness state on first turn                                    │
│ • Emit harness_started lifecycle event                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTEXT PRESSURE (optional)                                                 │
│ • Check token budget against threshold                                      │
│ • Apply reduction strategies if needed:                                     │
│   - Summarize old turns                                                     │
│   - Clear tool results                                                      │
│   - Externalize to artifacts                                                │
│   - Drop oldest messages                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT TURN                                                                  │
│ • Increment turn counter                                                    │
│ • Run agent's inner loop (inference → tool calls → inference)               │
│ • Track tool calls for repair detection                                     │
│ • Handle continuation prompts if agent stops without completing             │
│ • Emit turn_started/turn_completed lifecycle events                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STOP DECISION (layered evaluation)                                          │
│                                                                             │
│ Layer 1 - Hard stops:                                                       │
│   • Max turns reached → STOP                                                │
│   • Agent error → STOP (failed)                                             │
│                                                                             │
│ Layer 2 - Stall detection (optional):                                       │
│   • No progress for N turns → STOP (stalled)                                │
│                                                                             │
│ Layer 3 - Contract verification (optional):                                 │
│   • Agent signals done → verify contract                                    │
│   • Contract satisfied → STOP (done)                                        │
│   • Contract not satisfied → CONTINUE with gap report                       │
│                                                                             │
│ Layer 4 - Agent signal:                                                     │
│   • Agent signals done (no contract) → STOP (done)                          │
│                                                                             │
│ Default: CONTINUE to next turn                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

This provides checkpointing, event streaming, and composition for free by leveraging existing workflow infrastructure.

### Environment as Tool Collections

Each environment is a collection of tools appropriate for that deployment:

```python
# Coding environment - full capabilities
coding_tools = [
    filesystem.read_file,
    filesystem.write_file,
    shell.run_command,
    editor.apply_diff,
]

# API environment - no filesystem or shell
api_tools = [
    crm.get_customer,
    crm.update_ticket,
    email.send_message,
]

# Research environment - read-only + web
research_tools = [
    web.search,
    web.fetch_page,
    artifacts.store_note,
]
```

### Persona as Configuration

The persona configures which tools are visible, how the agent behaves, and what "done" means:

```python
# Coding assistant persona
coding_persona = AgentPersona(
    instructions="You are a skilled developer...",
    tools=coding_tools,
    task_contract=TaskContract(...),  # Formal completion criteria
    autonomy="high",  # Can execute without approval
)

# Customer support persona
support_persona = AgentPersona(
    instructions="You help customers with their issues...",
    tools=api_tools,
    task_contract=None,  # Model judgment for completion
    autonomy="low",  # Requires approval for actions
)
```

### Context Engineering

The harness manages context pressure independently of tools:

```python
harness = AgentHarness(
    agent,
    # Context management (harness concern)
    max_input_tokens=100000,
    context_strategies=[
        SummarizeOldTurns(),
        ExternalizeToArtifacts(),
        DropToolResults(),
    ],
)
```

Externalization strategies work with whatever artifact storage the environment provides—local files, blob storage, database, or even in-memory with a pointer.

## Summary

| Layer | Responsibility | Varies By |
|-------|---------------|-----------|
| **Harness** | Loop, state, durability, context, policy, observability | Deployment (cloud vs edge vs embedded) |
| **Environment** | Available capabilities (filesystem, shell, APIs, etc.) | Security posture, platform, regulations |
| **Persona** | Agent identity, visible tools, completion criteria | Use case, user, task type |

The harness is the constant. It enables any agent to run reliably, durably, and observably. What capabilities that agent has access to—and how it's configured to use them—is a separate concern.

This separation allows:
- **One harness implementation** that works across all agent types
- **Pluggable environments** for different security and platform requirements
- **Configurable personas** for different use cases and user needs
- **Clean policy boundaries** between infrastructure and capabilities
