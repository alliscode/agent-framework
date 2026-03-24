# Copyright (c) Microsoft. All rights reserved.

"""Form state tracking for guided conversations.

This module provides the core data structures for tracking form completion
progress during a guided conversation.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Generic, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

try:
    from typing import TypeVar
except ImportError:
    from typing_extensions import TypeVar

logger = logging.getLogger("guided_conversation")

T = TypeVar("T", bound=BaseModel)


class FormStatus(str, Enum):
    """Status of the guided conversation form."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    PARTIAL = "partial"
    ABANDONED = "abandoned"


class FieldInfo(BaseModel):
    """Metadata about a single form field, derived from the Pydantic model."""

    name: str
    field_type: str
    required: bool
    description: str = ""
    default: Any = None
    has_default: bool = False

    model_config = {"arbitrary_types_allowed": True}


class FieldEntry(BaseModel):
    """A collected field value with evidence."""

    value: Any
    evidence: str = ""

    model_config = {"arbitrary_types_allowed": True}


class FormState(BaseModel, Generic[T]):
    """Tracks form completion progress during a guided conversation.

    Introspects the target Pydantic model to understand required/optional
    fields, then tracks which have been filled and the evidence for each.
    """

    field_metadata: dict[str, FieldInfo] = Field(default_factory=dict)
    collected: dict[str, FieldEntry] = Field(default_factory=dict)
    submitted: bool = False
    _model_type: type | None = None
    _field_validators: dict[str, TypeAdapter] = {}

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_model(cls, model_type: type[T]) -> FormState[T]:
        """Create a FormState by introspecting a Pydantic model's fields."""
        field_metadata: dict[str, FieldInfo] = {}
        field_validators: dict[str, TypeAdapter] = {}

        for name, field in model_type.model_fields.items():
            required = field.is_required()
            description = field.description or ""
            has_default = not required
            default = field.default if has_default else None

            # Build a human-readable type string
            field_type = _type_display(field.annotation)

            field_metadata[name] = FieldInfo(
                name=name,
                field_type=field_type,
                required=required,
                description=description,
                default=default,
                has_default=has_default,
            )

            # Build a TypeAdapter for per-field validation & coercion
            if field.annotation is not None:
                field_validators[name] = TypeAdapter(field.annotation)

        state = cls(field_metadata=field_metadata)
        state._model_type = model_type
        state._field_validators = field_validators
        return state

    # -- Query methods --

    @property
    def all_fields(self) -> list[str]:
        return list(self.field_metadata.keys())

    @property
    def required_fields(self) -> list[str]:
        return [n for n, f in self.field_metadata.items() if f.required]

    @property
    def optional_fields(self) -> list[str]:
        return [n for n, f in self.field_metadata.items() if not f.required]

    @property
    def filled_fields(self) -> list[str]:
        return list(self.collected.keys())

    @property
    def missing_required(self) -> list[str]:
        return [f for f in self.required_fields if f not in self.collected]

    @property
    def missing_optional(self) -> list[str]:
        return [f for f in self.optional_fields if f not in self.collected]

    @property
    def is_complete(self) -> bool:
        """All required fields have been collected."""
        return len(self.missing_required) == 0

    @property
    def completion_fraction(self) -> float:
        """Fraction of all fields (required + optional) that are filled."""
        total = len(self.field_metadata)
        if total == 0:
            return 1.0
        return len(self.collected) / total

    # -- Mutation methods --

    def set_field(self, field_name: str, value: Any, evidence: str = "") -> str | None:
        """Set a field value with evidence. Returns an error string on failure, None on success.

        The value is validated and coerced against the field's type annotation
        using Pydantic's TypeAdapter. For example, "3500" will be coerced to
        3500.0 for a float field, but "a lot" will return a validation error.
        """
        if field_name not in self.field_metadata:
            valid = ", ".join(self.field_metadata.keys())
            return f"Unknown field '{field_name}'. Valid fields: {valid}"

        # Validate and coerce the value against the field's type
        validator = self._field_validators.get(field_name)
        if validator is not None:
            try:
                value = validator.validate_python(value)
            except ValidationError as e:
                meta = self.field_metadata[field_name]
                errors = "; ".join(err["msg"] for err in e.errors())
                return (
                    f"Invalid value for '{field_name}' (expected {meta.field_type}): {errors}. "
                    f"Please ask the user to clarify or provide a valid {meta.field_type} value."
                )

        self.collected[field_name] = FieldEntry(value=value, evidence=evidence)
        return None

    def clear_field(self, field_name: str) -> str | None:
        """Clear a previously set field. Returns error string on failure."""
        if field_name not in self.field_metadata:
            return f"Unknown field '{field_name}'."
        self.collected.pop(field_name, None)
        return None

    # -- Build methods --

    def to_values_dict(self) -> dict[str, Any]:
        """Return collected values as a plain dict (field_name -> value)."""
        return {name: entry.value for name, entry in self.collected.items()}

    def to_evidence_dict(self) -> dict[str, str]:
        """Return evidence as a plain dict (field_name -> evidence string)."""
        return {name: entry.evidence for name, entry in self.collected.items()}

    def build(self, model_type: type[T]) -> T:
        """Validate and construct the final Pydantic model.

        Raises ValidationError if required fields are missing or values are invalid.
        """
        return model_type.model_validate(self.to_values_dict())

    # -- Display methods --

    def progress_summary(self) -> str:
        """Human-readable summary of form progress for agent instructions."""
        lines: list[str] = []
        lines.append("## Form Progress\n")

        # Filled fields
        if self.filled_fields:
            lines.append("**Completed fields:**")
            for name in self.filled_fields:
                entry = self.collected[name]
                meta = self.field_metadata[name]
                req_tag = " (required)" if meta.required else " (optional)"
                lines.append(f"  - {name}{req_tag}: {entry.value}")
            lines.append("")

        # Missing required
        if self.missing_required:
            lines.append("**Required fields still needed:**")
            for name in self.missing_required:
                meta = self.field_metadata[name]
                desc = f" — {meta.description}" if meta.description else ""
                lines.append(f"  - {name} ({meta.field_type}){desc}")
            lines.append("")

        # Missing optional
        if self.missing_optional:
            lines.append("**Optional fields (nice to have):**")
            for name in self.missing_optional:
                meta = self.field_metadata[name]
                desc = f" — {meta.description}" if meta.description else ""
                lines.append(f"  - {name} ({meta.field_type}){desc}")
            lines.append("")

        # Status
        pct = int(self.completion_fraction * 100)
        filled = len(self.filled_fields)
        total = len(self.all_fields)
        lines.append(f"**Progress:** {filled}/{total} fields ({pct}%)")

        if self.is_complete:
            lines.append(
                "**Status:** All required fields collected! "
                "You may ask about optional fields or call submit_form to complete."
            )
        else:
            lines.append(
                "**Status:** Still collecting required fields. "
                "Focus on the required fields listed above."
            )

        return "\n".join(lines)


class FormResult(BaseModel, Generic[T]):
    """The output of a completed guided conversation."""

    data: Any = None  # Will be T when complete
    evidence: dict[str, str] = Field(default_factory=dict)
    status: FormStatus = FormStatus.IN_PROGRESS
    validation_errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_form_state(cls, form_state: FormState[T], model_type: type[T]) -> FormResult[T]:
        """Build a FormResult from the current form state."""
        evidence = form_state.to_evidence_dict()

        if not form_state.is_complete:
            return cls(
                data=None,
                evidence=evidence,
                status=FormStatus.PARTIAL,
            )

        try:
            data = form_state.build(model_type)
            return cls(
                data=data,
                evidence=evidence,
                status=FormStatus.COMPLETE,
            )
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            return cls(
                data=None,
                evidence=evidence,
                status=FormStatus.PARTIAL,
                validation_errors=errors,
            )


def _type_display(annotation: Any) -> str:
    """Convert a type annotation to a human-readable string."""
    if annotation is None:
        return "any"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional (Union with None)
    if origin is type(int | str):  # types.UnionType for X | Y
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f"{_type_display(non_none[0])} (optional)"
        return " | ".join(_type_display(a) for a in non_none)

    # Handle typing.Union
    try:
        import typing

        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return f"{_type_display(non_none[0])} (optional)"
            return " | ".join(_type_display(a) for a in non_none)
    except AttributeError:
        pass

    # Handle list, dict, etc.
    if origin is list:
        inner = _type_display(args[0]) if args else "any"
        return f"list[{inner}]"
    if origin is dict:
        k = _type_display(args[0]) if args else "str"
        v = _type_display(args[1]) if len(args) > 1 else "any"
        return f"dict[{k}, {v}]"

    # Simple types
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation)
