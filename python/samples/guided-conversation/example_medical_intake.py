# Copyright (c) Microsoft. All rights reserved.

"""
Guided Conversation — Medical Intake Questionnaire

A more complex example demonstrating guided form-filling with a 19-field
medical intake form. Shows how GuidedConversationProvider handles many fields,
mixed required/optional, and domain-specific conversational guidelines.

Environment variables:
  AZURE_AI_PROJECT_ENDPOINT                — Your Azure AI Foundry project endpoint
  AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME   — Model deployment name (e.g. gpt-4o)
"""

import asyncio
import os
from enum import Enum
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_framework import Agent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential

from guided_conversation import GuidedConversationProvider

load_dotenv()


# -- Supporting types --


class BiologicalSex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class VisitReason(str, Enum):
    NEW_SYMPTOMS = "new_symptoms"
    FOLLOW_UP = "follow_up"
    ANNUAL_CHECKUP = "annual_checkup"
    REFERRAL = "referral"
    PRESCRIPTION_REFILL = "prescription_refill"
    OTHER = "other"


# -- The form --


class MedicalIntake(BaseModel):
    """a new patient medical intake questionnaire"""

    # --- Demographics (required) ---
    full_name: str = Field(
        description="Patient's full legal name as it appears on their ID"
    )
    date_of_birth: str = Field(
        description="Date of birth in YYYY-MM-DD format"
    )
    biological_sex: str = Field(
        description=(
            "Biological sex assigned at birth. "
            "Accepted values: male, female, intersex, prefer_not_to_say. "
            "Be sensitive when asking — explain it's for medical accuracy."
        )
    )
    phone_number: str = Field(
        description="Primary contact phone number with area code"
    )

    # --- Visit info (required) ---
    reason_for_visit: str = Field(
        description=(
            "Primary reason for today's visit. "
            "Accepted values: new_symptoms, follow_up, annual_checkup, "
            "referral, prescription_refill, other. "
            "If 'other', ask the patient to describe."
        )
    )
    chief_complaint: str = Field(
        description=(
            "Main symptom or concern in the patient's own words. "
            "This should be a detailed description — ask follow-up questions "
            "about onset, duration, severity (1-10 scale), and what makes it better/worse."
        )
    )
    symptom_duration: str = Field(
        description=(
            "How long the patient has been experiencing their chief complaint. "
            "Be specific: 'about 3 weeks' is better than 'a while'."
        )
    )

    # --- Medical history (required) ---
    current_medications: str = Field(
        description=(
            "List of all current medications including dosage and frequency. "
            "Ask about prescription meds, over-the-counter, vitamins, and supplements. "
            "If none, record 'none'."
        )
    )
    known_allergies: str = Field(
        description=(
            "All known allergies — medications, food, environmental. "
            "For medication allergies, ask about the reaction type "
            "(rash, breathing difficulty, etc.). If none, record 'NKDA' (no known drug allergies)."
        )
    )
    chronic_conditions: str = Field(
        description=(
            "Any chronic or ongoing medical conditions "
            "(e.g., diabetes, hypertension, asthma, arthritis). "
            "If none, record 'none'."
        )
    )

    # --- Optional but valuable ---
    previous_surgeries: str | None = Field(
        default=None,
        description=(
            "Any previous surgeries or major medical procedures, "
            "including approximate year. If none, record 'none'."
        ),
    )
    family_medical_history: str | None = Field(
        default=None,
        description=(
            "Relevant family medical history — especially heart disease, "
            "diabetes, cancer, or mental health conditions in parents or siblings."
        ),
    )
    tobacco_use: bool | None = Field(
        default=None,
        description="Whether the patient currently uses tobacco products"
    )
    alcohol_use: str | None = Field(
        default=None,
        description=(
            "Alcohol consumption frequency — e.g., 'none', 'social/occasional', "
            "'1-2 drinks per week', 'daily'. Be non-judgmental when asking."
        ),
    )
    emergency_contact_name: str | None = Field(
        default=None,
        description="Name of emergency contact person"
    )
    emergency_contact_phone: str | None = Field(
        default=None,
        description="Phone number of emergency contact"
    )
    insurance_provider: str | None = Field(
        default=None,
        description="Name of health insurance provider, if insured"
    )
    insurance_member_id: str | None = Field(
        default=None,
        description="Insurance member/policy ID number"
    )
    additional_notes: str | None = Field(
        default=None,
        description=(
            "Anything else the patient wants the doctor to know — "
            "concerns, questions, context about their situation."
        ),
    )


async def main() -> None:
    credential = AzureCliCredential()
    client = AzureOpenAIResponsesClient(
        project_endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        deployment_name=os.environ["AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"],
        credential=credential,
    )

    # 1. Create the guided conversation provider
    form = GuidedConversationProvider(MedicalIntake)

    # 2. Attach it to a standard Agent
    agent = Agent(
        client=client,
        instructions=(
            "You are a warm, professional medical intake coordinator at a doctor's office. "
            "Your job is to collect the patient's information before their appointment.\n\n"
            "Guidelines:\n"
            "- Be warm and reassuring — patients may be anxious or in pain.\n"
            "- Group related questions naturally (don't jump between topics).\n"
            "- Start with easy questions (name, DOB) to build rapport before sensitive ones.\n"
            "- For the chief complaint, ask good follow-up questions: When did it start? "
            "How severe is it on a 1-10 scale? What makes it better or worse?\n"
            "- Be sensitive about biological sex, substance use, and mental health topics.\n"
            "- For medications and allergies, probe gently — patients often forget OTC meds "
            "and supplements.\n"
            "- Don't overwhelm — ask about 2-3 fields at a time, then record what you learn.\n"
            "- After collecting all required fields, briefly summarize and ask if anything "
            "needs correction before moving to optional fields.\n"
            "- For optional fields, frame them as 'helpful but not required' — don't push.\n"
            "- If the patient seems to want to skip optional fields, that's fine — submit."
        ),
        context_providers=[form],
    )

    # 3. Kick off the conversation — agent speaks first
    session = agent.create_session()
    response = await agent.run(
        "I'm here for my appointment and need to fill out the intake form.",
        session=session,
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

    # 4. Print results
    result = form.get_result(session)
    print("\n" + "=" * 60)
    print(f"INTAKE STATUS: {result.status.value.upper()}")
    print("=" * 60)

    if result.data:
        data = result.data.model_dump()

        print("\n📋 Patient Information:")
        print("-" * 40)

        # Group the output for readability
        sections = {
            "Demographics": ["full_name", "date_of_birth", "biological_sex", "phone_number"],
            "Visit Information": ["reason_for_visit", "chief_complaint", "symptom_duration"],
            "Medical History": [
                "current_medications", "known_allergies", "chronic_conditions",
                "previous_surgeries", "family_medical_history",
            ],
            "Lifestyle": ["tobacco_use", "alcohol_use"],
            "Emergency Contact": ["emergency_contact_name", "emergency_contact_phone"],
            "Insurance": ["insurance_provider", "insurance_member_id"],
            "Additional": ["additional_notes"],
        }

        for section_name, fields in sections.items():
            section_values = {f: data[f] for f in fields if data.get(f) is not None}
            if section_values:
                print(f"\n  {section_name}:")
                for field_name, value in section_values.items():
                    label = field_name.replace("_", " ").title()
                    print(f"    {label}: {value}")

        print(f"\n📝 Evidence Trail ({len(result.evidence)} items):")
        print("-" * 40)
        for field_name, evidence in result.evidence.items():
            label = field_name.replace("_", " ").title()
            print(f"  {label}:")
            print(f"    → {evidence}")
    else:
        print("\nIntake was not completed.")
        if result.validation_errors:
            print("Validation errors:")
            for error in result.validation_errors:
                print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
