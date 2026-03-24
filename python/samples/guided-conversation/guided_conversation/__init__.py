# Copyright (c) Microsoft. All rights reserved.

"""Guided Conversation — form-filling via context provider.

Add GuidedConversationProvider to any Agent to guide conversations
toward collecting structured data defined by a Pydantic model.
"""

from ._form_state import FieldEntry, FieldInfo, FormResult, FormState, FormStatus
from ._provider import GuidedConversationProvider

__all__ = [
    "FieldEntry",
    "FieldInfo",
    "FormResult",
    "FormState",
    "FormStatus",
    "GuidedConversationProvider",
]
