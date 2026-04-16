# Copyright (c) Microsoft. All rights reserved.

"""
Guided Conversation — Insurance Claim Form

Demonstrates guided form-filling by adding a GuidedConversationProvider
to a standard Agent. The provider injects tools and progress instructions
so the agent knows what data to collect and tracks progress automatically.

Environment variables:
  FOUNDRY_PROJECT_ENDPOINT     — Your Azure AI Foundry project endpoint
  FOUNDRY_MODEL                — Model deployment name (e.g. gpt-4o)
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

from guided_conversation import GuidedConversationProvider

load_dotenv()


# -- Define the form as a Pydantic model --

class InsuranceClaim(BaseModel):
    """an auto insurance claim"""

    policyholder_name: str = Field(description="Full legal name of the policyholder")
    policy_number: str = Field(description="Policy number, format: POL-XXXXX")
    incident_date: str = Field(description="Date of the incident in YYYY-MM-DD format")
    incident_description: str = Field(
        description="Detailed description of what happened, including location and circumstances"
    )
    police_report_filed: bool = Field(description="Whether a police report was filed for this incident")
    damage_estimate: float | None = Field(
        default=None, description="Estimated damage amount in USD, if known"
    )
    witnesses: str | None = Field(
        default=None, description="Names and contact info of any witnesses, if available"
    )


async def main() -> None:
    client = FoundryChatClient(
        credential=AzureCliCredential(),
    )

    # 1. Create the guided conversation provider with optional features
    form = GuidedConversationProvider(
        InsuranceClaim,
        max_turns=15,
        conversation_flow=(
            "Start with policyholder identification (name, policy number), "
            "then move to incident details (date, description, police report), "
            "and finally ask about optional information (damage estimate, witnesses)."
        ),
    )

    # 2. Attach it to a standard Agent — no new concepts
    agent = Agent(
        client=client,
        instructions=(
            "You are a friendly and empathetic insurance claims assistant. "
            "Help the user file their auto insurance claim. Be understanding — "
            "they may be stressed. Ask clear questions and confirm important details."
        ),
        context_providers=[form],
    )

    # 3. Kick off the conversation — agent speaks first
    session = agent.create_session()
    response = await agent.run(
        "I need to file an auto insurance claim.", session=session
    )
    print(f"Agent: {response.text}\n")

    while not form.is_complete(session):
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Conversation ended by user.")
            break

        response = await agent.run(user_input, session=session)
        print(f"\nAgent: {response.text}\n")

        # Show remaining turns if budget is set
        remaining = form.turns_remaining(session)
        if remaining is not None and remaining <= 3:
            print(f"  ⏱ {remaining} turn(s) remaining\n")

    # 4. Print results
    result = form.get_result(session)
    print("\n" + "=" * 60)
    print(f"Form Status: {result.status.value}")

    if result.data:
        print("\nCollected Data:")
        for field_name, value in result.data.model_dump().items():
            print(f"  {field_name}: {value}")

        print("\nEvidence:")
        for field_name, evidence in result.evidence.items():
            print(f"  {field_name}: {evidence}")
    else:
        print("\nForm was not completed.")
        if result.validation_errors:
            print("Validation errors:")
            for error in result.validation_errors:
                print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
