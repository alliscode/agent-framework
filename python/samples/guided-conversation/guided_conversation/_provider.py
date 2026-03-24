# Copyright (c) Microsoft. All rights reserved.

"""Guided Conversation Provider.

A single BaseContextProvider that adds guided form-filling behavior to any Agent.
Injects tools, progress instructions, and manages form state — no new concepts needed.

Usage:
    form = GuidedConversationProvider(InsuranceClaim)

    agent = Agent(
        client=client,
        instructions="You are an insurance claims assistant.",
        context_providers=[form],
    )

    session = agent.create_session()
    while True:
        user_input = input("You: ")
        response = await agent.run(user_input, session=session)
        print(response.text)

        if form.is_complete(session):
            result = form.get_result(session)
            break
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel

from agent_framework import BaseContextProvider, FunctionTool
from agent_framework._middleware import FunctionInvocationContext

from ._form_state import FormResult, FormState

try:
    from typing import TypeVar
except ImportError:
    from typing_extensions import TypeVar

if TYPE_CHECKING:
    from agent_framework import AgentSession
    from agent_framework._agents import SupportsAgentRun
    from agent_framework._sessions import SessionContext

logger = logging.getLogger("guided_conversation")

T = TypeVar("T", bound=BaseModel)

FORM_STATE_KEY = "form_state"

GUIDANCE_INSTRUCTIONS = """
## Guided Conversation Instructions

You are conducting a guided conversation to collect information for a form.
Your goal is to naturally collect all required fields through conversation.

**Guidelines:**
- Ask about one or two related fields at a time, not all at once.
- Be conversational and natural — don't make it feel like an interrogation.
- When the user provides information, use the `update_form` tool to record it immediately.
- Always provide evidence: quote or paraphrase the user's exact words as the evidence parameter.
- If a value seems ambiguous, ask for clarification before recording it.
- After collecting all required fields, summarize what you've collected and ask for confirmation.
- Once confirmed, call `submit_form` to complete the conversation.
- You may ask about optional fields after required ones are done, but don't push too hard.
- If the user wants to correct a previously recorded value, use `update_form` again to overwrite it.
"""


class GuidedConversationProvider(BaseContextProvider, Generic[T]):
    """A context provider that adds guided form-filling to any Agent.

    Attach this to an Agent's context_providers and it will:
    - Inject form-tracking tools (update_form, get_form_status, submit_form)
    - Inject dynamic progress instructions before each turn
    - Track form completion state in session.state
    - Validate and coerce field values using Pydantic

    Example:
        >>> form = GuidedConversationProvider(InsuranceClaim)
        >>> agent = Agent(client=client, instructions="...", context_providers=[form])
        >>> session = agent.create_session()
        >>> response = await agent.run("Hi, I need to file a claim", session=session)
        >>> form.is_complete(session)
        False
    """

    def __init__(
        self,
        form_type: type[T],
        source_id: str = "guided_conversation",
    ) -> None:
        super().__init__(source_id=source_id)
        self._form_type = form_type
        self._tools = _create_form_tools(form_type, source_id)

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject form tools and progress instructions before each turn."""
        # Initialize form state on first run
        if FORM_STATE_KEY not in state:
            state[FORM_STATE_KEY] = FormState.from_model(self._form_type)

        form_state: FormState = state[FORM_STATE_KEY]

        # Inject tools
        context.tools.extend(self._tools)

        # Inject progress instructions
        context.instructions.append(GUIDANCE_INSTRUCTIONS)
        context.instructions.append(form_state.progress_summary())

    # -- Public API for checking state from the outer loop --

    def get_form_state(self, session: AgentSession) -> FormState[T] | None:
        """Get the current form state from a session."""
        state = session.state.get(self.source_id)
        if isinstance(state, dict):
            return state.get(FORM_STATE_KEY)
        return None

    def is_complete(self, session: AgentSession) -> bool:
        """Check if the form has been submitted successfully."""
        form_state = self.get_form_state(session)
        return form_state is not None and form_state.submitted

    def get_result(self, session: AgentSession) -> FormResult[T]:
        """Get the form result (data + evidence) from a session."""
        form_state = self.get_form_state(session)
        if form_state is None:
            return FormResult(status="in_progress")
        return FormResult.from_form_state(form_state, self._form_type)


# -- Tool creation (private to this module) --


def _get_form_state_from_context(
    ctx: FunctionInvocationContext, source_id: str
) -> FormState | None:
    """Extract FormState from the FunctionInvocationContext's session state."""
    if ctx.session is None:
        return None
    provider_state = ctx.session.state.get(source_id)
    if isinstance(provider_state, dict):
        return provider_state.get(FORM_STATE_KEY)
    return None


def _create_form_tools(form_type: type[BaseModel], source_id: str) -> list[FunctionTool]:
    """Create the three form tools, bound to a specific provider source_id."""
    field_descriptions = _build_field_descriptions(form_type)

    def update_form(ctx: FunctionInvocationContext, field_name: str, value: Any, evidence: str) -> str:
        """Update a form field with a value extracted from the conversation.

        Call this tool whenever you learn the value for a form field from the user's
        response. Provide the field name, the value, and the evidence (quote or
        reasoning from the user's words that supports this value).

        Args:
            ctx: Injected function invocation context (provides session access).
            field_name: The name of the field to update. Must be one of the valid field names.
            value: The value to set. Must match the expected type for the field.
            evidence: A quote or summary from the user's response that justifies this value.

        Returns:
            Confirmation message or error description.
        """
        form_state = _get_form_state_from_context(ctx, source_id)
        if form_state is None:
            return "Error: Form state not available."

        error = form_state.set_field(field_name, value, evidence)
        if error:
            return f"Error: {error}"

        remaining = len(form_state.missing_required)
        if remaining > 0:
            return (
                f"✓ Recorded {field_name} = {value!r}. "
                f"{remaining} required field(s) remaining: {', '.join(form_state.missing_required)}"
            )
        return (
            f"✓ Recorded {field_name} = {value!r}. "
            "All required fields collected! You can ask about optional fields or call submit_form."
        )

    def get_form_status(ctx: FunctionInvocationContext) -> str:
        """Get the current form completion status.

        Call this to see which fields have been filled, which are still needed,
        and the overall progress percentage.

        Args:
            ctx: Injected function invocation context (provides session access).

        Returns:
            A summary of the current form state.
        """
        form_state = _get_form_state_from_context(ctx, source_id)
        if form_state is None:
            return "Error: Form state not available."

        return form_state.progress_summary()

    def submit_form(ctx: FunctionInvocationContext) -> str:
        """Submit the completed form.

        Call this when all required fields have been collected and the user has
        confirmed the information is correct. The form will be validated against
        the schema.

        Args:
            ctx: Injected function invocation context (provides session access).

        Returns:
            Success confirmation or validation error details.
        """
        form_state = _get_form_state_from_context(ctx, source_id)
        if form_state is None:
            return "Error: Form state not available."

        if not form_state.is_complete:
            missing = ", ".join(form_state.missing_required)
            return f"Cannot submit: missing required fields: {missing}"

        try:
            form_state.build(form_type)
        except Exception as e:
            return f"Validation error: {e}"

        form_state.submitted = True
        return "✓ Form submitted successfully! All data has been validated."

    update_description = (
        "Update a form field with a value extracted from the conversation. "
        f"Valid fields:\n{field_descriptions}"
    )

    return [
        FunctionTool(name="update_form", description=update_description, func=update_form),
        FunctionTool(
            name="get_form_status",
            description="Get the current form completion status showing filled and remaining fields.",
            func=get_form_status,
        ),
        FunctionTool(
            name="submit_form",
            description=(
                "Submit the completed form after all required fields are collected "
                "and the user confirms the information is correct."
            ),
            func=submit_form,
        ),
    ]


def _build_field_descriptions(form_type: type[BaseModel]) -> str:
    """Build a formatted string describing all form fields for the tool description."""
    lines: list[str] = []
    for name, field in form_type.model_fields.items():
        required = field.is_required()
        tag = "REQUIRED" if required else "optional"
        desc = field.description or "No description"
        annotation = field.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        lines.append(f"  - {name} ({type_name}, {tag}): {desc}")
    return "\n".join(lines)
