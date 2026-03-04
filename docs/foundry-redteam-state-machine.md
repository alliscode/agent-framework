# Red Team State Machine — Proposed Flow

For enabling Foundry red teaming against non-registered agents.

```mermaid
sequenceDiagram
    participant Dev as Developer Code
    participant FW as Agent Framework
    participant Agent as Agent (any provider)
    participant Foundry as Foundry Red Team Service

    Dev->>FW: red_team_agent(agent, risk_categories, ...)

    FW->>Foundry: Start red team eval<br/>(risk categories, attack strategies, num_turns)
    Foundry-->>FW: eval_id, status: "awaiting_response"

    loop For each turn (1..N)
        FW->>Foundry: Poll for next prompt<br/>(eval_id)
        Foundry-->>FW: adversarial prompt + turn metadata

        FW->>Agent: Run agent with prompt
        Agent-->>FW: Agent response

        FW->>Foundry: Submit response<br/>(eval_id, turn_id, response or trace_id)
        Foundry->>Foundry: Evaluate response against<br/>safety criteria
        Foundry->>Foundry: Generate next adversarial<br/>prompt based on response
        Foundry-->>FW: turn_result + status
    end

    FW->>Foundry: Get final results<br/>(eval_id)
    Foundry-->>FW: Full scorecard<br/>(per-category pass/fail, conversation logs)
    FW-->>Dev: EvalResults (scorecard, report_url)
```

## What agent-framework would wrap

```python
from agent_framework.foundry import red_team_agent

results = await red_team_agent(
    agent=my_agent,
    project_endpoint=os.getenv("AZURE_AI_PROJECT"),
    risk_categories=["violence", "prohibited_actions"],
    attack_strategies=["Flip", "Base64"],
    num_turns=5,
)
print(results.scorecard)
```

The helper would own the poll → run → submit loop, so the developer just passes their agent and gets results back.
