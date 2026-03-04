# Foundry Agent Evals — Integration Paths

Agent Framework integration with Azure AI Foundry Evals. Four paths depending on your scenario.

## Paths

### Path 1: Trace-Based Evaluation

Evaluate any agent by pointing Foundry at OTel traces in App Insights. Works with any model provider (OpenAI, Anthropic, Ollama, etc.). Zero changes to agent code — evaluate after the fact. Can also evaluate specific Responses API responses by ID.

### Path 2: Foundry-Managed Evaluation

Foundry invokes the target, captures output, and evaluates. For agents or models registered in Foundry. Enables scheduled recurring evals, multi-turn red teaming, and CI/CD quality gates — all unattended.

### Path 3: Dataset Evaluation

Run your agent locally against test cases, convert the output, and send to Foundry for evaluation. The dev inner loop — "pytest for agent quality." The only path that combines controlled inputs with custom agent code.

### Path 4: Continuous Evaluation

One-time setup of an evaluation rule in Foundry. Every response from the agent is automatically evaluated. Requires the agent to be registered in Foundry with responses using `agent_reference`.

## Decision Tree

```
What are you trying to do?
│
├─ "Evaluate my agent during development against specific test cases"
│   │
│   ├─ Agent is registered in Foundry? ──→ Path 2 (Foundry-managed)
│   │                                       Foundry runs it for you
│   │
│   └─ Agent is not in Foundry? ───────────→ Path 3 (Dataset eval)
│                                           Run locally, eval in cloud
│
├─ "Evaluate what my agent already did"
│   │
│   ├─ Agent has OTel → App Insights? ──→ Path 1 (Traces)
│   │  (or Responses API response IDs)    Easiest, works with any agent
│   │
│   └─ No OTel / no App Insights? ──────→ ⚠ Not supported today.
│                                           Set up OTel tracing to
│                                           enable Path 1.
│
├─ "Set up recurring/scheduled evaluation"
│   │
│   ├─ Agent is registered in Foundry? ──→ Path 2 (Foundry-managed)
│   │                                       Foundry schedules it natively
│   │
│   └─ Agent is not in Foundry? ─────────→ ⚠ Not supported natively.
│                                           Foundry only schedules evals
│                                           for targets it can invoke.
│
├─ "Monitor production quality continuously"
│   │
│   ├─ Agent is registered in Foundry ──→ Path 4 (Continuous eval)
│   │  and responses use agent_reference?  Auto-eval on every response
│   │
│   └─ Agent is not registered ─────────→ ⚠ Not supported natively.
│      in Foundry?
│
└─ "Red team my agent for safety"
    │
    ├─ Agent is registered in Foundry? ──→ Path 2 (Foundry-managed)
    │                                       Multi-turn adversarial testing
    │
    └─ Agent is not registered ──────────→ ⚠ Not supported natively.
       in Foundry?                           Foundry needs to invoke the
                                             target for red teaming.
```

## Quick Reference

| I want to... | My agent is... | Use |
|---|---|---|
| Evaluate past runs | Any agent with OTel | **Path 1** — Traces |
| Evaluate specific responses | Using Responses API | **Path 1** — Traces (with response IDs) |
| Run test cases in CI | Registered in Foundry | **Path 2** — Foundry-managed |
| Run test cases locally | Custom code | **Path 3** — Dataset eval |
| Scheduled recurring evals | Registered in Foundry | **Path 2** — Foundry-managed |
| Scheduled recurring evals | Not in Foundry | ⚠ Not supported natively |
| Monitor production | Registered in Foundry | **Path 4** — Continuous eval |
| Red team for safety | Registered in Foundry | **Path 2** — Foundry-managed |
| Red team for safety | Not in Foundry | ⚠ Not supported natively |
