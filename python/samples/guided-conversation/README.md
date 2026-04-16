# Guided Conversation

Add form-filling behavior to any Agent with a single context provider. Define your "form" as a Pydantic model, attach a `GuidedConversationProvider`, and the agent manages the conversation to fill it out — tracking progress, capturing evidence, and knowing when the form is complete.

## Concept

Traditional structured output is single-shot: one LLM call → one JSON object. But many real-world data collection scenarios are conversational — insurance claims, medical intake, customer onboarding. The user doesn't dump all the data at once; they tell a story, answer questions, clarify details.

**Guided Conversation** bridges this gap:

1. **Schema defines the goal** — a Pydantic model with required/optional fields, descriptions, and validators
2. **Agent tracks progress** — knows which fields are filled, which remain, what's required vs optional
3. **Conversation is optimized** — the agent asks targeted questions, batches related fields, handles clarifications
4. **Evidence is captured** — each field value is paired with the user's exact words that produced it
5. **Completion is automatic** — required fields filled + user confirmation = done

## Quick Start

```python
from pydantic import BaseModel, Field
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from guided_conversation import GuidedConversationProvider

# 1. Define your form as a Pydantic model
class ContactInfo(BaseModel):
    """contact information"""
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(default=None, description="Phone number")

# 2. Create the provider and attach it to any Agent
form = GuidedConversationProvider(ContactInfo)
agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    instructions="You are a friendly receptionist.",
    context_providers=[form],
)

# 3. Run a normal conversation loop — you own the UX
session = agent.create_session()
while True:
    user_input = input("You: ")
    response = await agent.run(user_input, session=session)
    print(f"Agent: {response.text}")

    if form.is_complete(session):
        result = form.get_result(session)
        print(result.data)       # ContactInfo(name="Alice", email="alice@example.com", phone=None)
        print(result.evidence)   # {"name": "User said 'My name is Alice'", ...}
        break
```

## Optional Features

All optional — off by default. Mix and match as needed.

### Turn Budget (`max_turns`)

Limit conversation length. The agent sees a countdown and plans accordingly. When turns run out, the form auto-submits (if all required fields are done) or abandons with partial data.

```python
form = GuidedConversationProvider(
    InsuranceClaim,
    max_turns=10,
)

# Check remaining turns from your loop:
remaining = form.turns_remaining(session)  # int or None
```

### Conversation Flow (`conversation_flow`)

Free-form pacing guidance injected as instructions. Tells the agent what order to ask questions in, how to group topics, etc. No enforcement — purely advisory.

```python
form = GuidedConversationProvider(
    MedicalIntake,
    conversation_flow=(
        "Start with easy demographics (name, DOB, phone). "
        "Then visit reason and chief complaint with follow-ups. "
        "Medical history next. Optional fields last."
    ),
)
```

### Final Review Pass (`final_update`)

Two-phase submit: the first `submit_form` call triggers a review ("check for any fields you missed"), the second actually completes. Helps catch values mentioned in conversation but not recorded.

```python
form = GuidedConversationProvider(
    MedicalIntake,
    final_update=True,
)
# No changes to your loop — it's handled inside the tools.
```

### State Persistence (`save_state` / `load_state`)

Serialize form state to JSON for storage/resume. You pick the backend (file, DB, Redis).

```python
# Save
json_str = form.save_state(session)
Path("state.json").write_text(json_str)

# Load (e.g., after process restart)
json_str = Path("state.json").read_text()
session = agent.create_session()
form.load_state(session, json_str)
# Continue the conversation where you left off
```

## Architecture

```
┌───────────────────────────────────────────┐
│  Your Agent (unchanged)                   │
│                                           │
│  context_providers:                       │
│    - GuidedConversationProvider[T]         │
│        ↳ injects tools:                   │
│            update_form, get_form_status,   │
│            submit_form                    │
│        ↳ injects progress instructions    │
│        ↳ injects conversation flow        │
│        ↳ injects turn budget              │
│        ↳ manages FormState in session     │
│    - InMemoryHistoryProvider (auto)       │
│                                           │
│  You own the conversation loop.           │
│  Provider exposes is_complete() and       │
│  get_result() for checking state.         │
└───────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| `GuidedConversationProvider[T]` | The single class you need — context provider that injects tools, progress, and manages state |
| `FormState[T]` | Internal: introspects Pydantic model, tracks filled/missing fields, evidence |
| `FormResult[T]` | Output: the completed form data, evidence, and status |

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `form_type` | `type[T]` | *required* | Pydantic model defining the form schema |
| `source_id` | `str` | `"guided_conversation"` | Provider identifier in session state |
| `max_turns` | `int \| None` | `None` | Maximum conversation turns (None = unlimited) |
| `conversation_flow` | `str \| None` | `None` | Free-form pacing guidance for the agent |
| `final_update` | `bool` | `False` | Enable two-phase submit with review pass |

### Public Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_complete(session)` | `bool` | Whether the form has been submitted |
| `get_result(session)` | `FormResult[T]` | Form data, evidence, status, validation errors |
| `get_form_state(session)` | `FormState[T] \| None` | Raw form state for inspection |
| `turns_remaining(session)` | `int \| None` | Remaining turns (None if no limit) |
| `save_state(session)` | `str` | Serialize form state to JSON |
| `load_state(session, data)` | `None` | Restore form state from JSON |

## File Structure

```
guided_conversation/
├── __init__.py              # Public API
├── _form_state.py           # FormState[T], FormResult[T], FieldInfo
└── _provider.py             # GuidedConversationProvider (tools, context, state)
```

## Running the Examples

```bash
# Set environment variables (or use a .env file)
export FOUNDRY_PROJECT_ENDPOINT="..."
export FOUNDRY_MODEL="gpt-4o"

# Insurance claim — demonstrates max_turns + conversation_flow
cd python/samples/guided-conversation
python example_insurance.py

# Medical intake — demonstrates final_update + conversation_flow
python example_medical_intake.py
```

## Future Directions

- **Nested models**: Support Pydantic models with nested sub-models (sub-forms)
- **Workflow upgrade**: Migrate to `Workflow` + `Executor` for checkpointing/suspend-resume
- **Multi-valued fields**: Better handling of list fields (witnesses, items, etc.)
- **Streaming**: Stream agent responses for better UX in long-running turns
