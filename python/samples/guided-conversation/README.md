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
    client=OpenAIChatClient(model_id="gpt-4o"),
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

### How It Works

1. **Setup**: `GuidedConversationProvider` introspects the Pydantic model to learn field names, types, required/optional status, and descriptions. It creates form tools bound to the session state.

2. **Each turn** (all automatic via `before_run()`):
   - Progress summary is injected into the agent's context (what's filled, what's missing)
   - Three form tools are injected (`update_form`, `get_form_status`, `submit_form`)
   - The agent responds conversationally and calls `update_form` when it extracts values
   - Values are validated and coerced using Pydantic's `TypeAdapter` — invalid values are rejected with helpful errors

3. **Completion**: When all required fields are filled, the agent summarizes and calls `submit_form`, which validates the complete model. Your loop detects this via `form.is_complete(session)`.

## File Structure

```
guided_conversation/
├── __init__.py              # Public API
├── _form_state.py           # FormState[T], FormResult[T], FieldInfo
└── _provider.py             # GuidedConversationProvider (tools, context, state)
```

## Running the Examples

```bash
# Set environment variables
export AZURE_AI_PROJECT_ENDPOINT="..."
export AZURE_AI_MODEL_DEPLOYMENT_NAME="gpt-4o"

# Run the insurance claim example (7 fields, simple)
cd python/samples/guided-conversation
python example_insurance.py

# Run the medical intake example (19 fields, complex)
python example_medical_intake.py
```

## Future Directions

- **Nested models**: Support Pydantic models with nested sub-models (sub-forms)
- **Workflow upgrade**: Migrate to `Workflow` + `Executor` for checkpointing/suspend-resume
- **Confirmation step**: Built-in review step before final submission
- **Multi-valued fields**: Better handling of list fields (witnesses, items, etc.)
- **Streaming**: Stream agent responses for better UX in long-running turns
