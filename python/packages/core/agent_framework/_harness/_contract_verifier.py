# Copyright (c) Microsoft. All rights reserved.

"""Contract verification for Agent Harness.

This module provides the ContractVerifier class that evaluates predicates
to determine if task requirements have been satisfied.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from ._task_contract import (
    CoverageLedger,
    Evidence,
    Predicate,
    PredicateType,
    RequiredOutput,
    RequirementStatus,
    TaskContract,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a single predicate.

    Attributes:
        success: Whether the predicate was satisfied.
        message: Human-readable message about the result.
        evidence: Evidence if the predicate was satisfied.
    """

    success: bool
    message: str
    evidence: Evidence | None = None


@dataclass
class ContractVerificationResult:
    """Result of verifying an entire contract.

    Attributes:
        satisfied: Whether all required predicates were satisfied.
        results: Results for each requirement.
        unmet_requirements: List of unmet requirement IDs.
        message: Overall verification message.
    """

    satisfied: bool
    results: dict[str, VerificationResult]
    unmet_requirements: list[str]
    message: str


class ContractVerifier:
    """Verifies task contract requirements.

    The verifier evaluates predicates against current state to determine
    if requirements have been satisfied.
    """

    def __init__(
        self,
        *,
        working_directory: str | None = None,
        transcript: list[dict[str, Any]] | None = None,
        artifacts: dict[str, str] | None = None,
    ):
        """Initialize the verifier.

        Args:
            working_directory: Base directory for file checks.
            transcript: The harness transcript for context.
            artifacts: Map of artifact IDs to paths/values.
        """
        self._working_directory = working_directory or os.getcwd()
        self._transcript = transcript or []
        self._artifacts = artifacts or {}

    def verify_predicate(
        self,
        predicate: Predicate,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify a single predicate.

        Args:
            predicate: The predicate to verify.
            context: Additional context for verification.

        Returns:
            VerificationResult indicating success/failure.
        """
        context = context or {}

        try:
            if predicate.type == PredicateType.ALWAYS_TRUE:
                return self._verify_always_true(predicate)
            if predicate.type == PredicateType.FILE_EXISTS:
                return self._verify_file_exists(predicate)
            if predicate.type == PredicateType.CONTAINS_TEXT:
                return self._verify_contains_text(predicate, context)
            if predicate.type == PredicateType.TOOL_RESULT_SUCCESS:
                return self._verify_tool_success(predicate)
            if predicate.type == PredicateType.JSON_SCHEMA_VALID:
                return self._verify_json_schema(predicate, context)
            if predicate.type == PredicateType.CUSTOM:
                return self._verify_custom(predicate, context)
            return VerificationResult(
                success=False,
                message=f"Unknown predicate type: {predicate.type}",
            )
        except Exception as e:
            logger.warning(f"Predicate verification failed: {e}")
            return VerificationResult(
                success=False,
                message=f"Verification error: {e!s}",
            )

    def verify_requirement(
        self,
        requirement: RequiredOutput,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify a single requirement.

        Args:
            requirement: The requirement to verify.
            context: Additional context for verification.

        Returns:
            VerificationResult indicating success/failure.
        """
        result = self.verify_predicate(requirement.predicate, context)

        if result.success and result.evidence:
            # Add requirement ID to evidence
            result.evidence = Evidence(
                event_ref=result.evidence.event_ref,
                kind=result.evidence.kind,
                value=f"{requirement.id}: {result.evidence.value}",
            )

        return result

    def verify_contract(
        self,
        contract: TaskContract,
        ledger: CoverageLedger | None = None,
        context: dict[str, Any] | None = None,
    ) -> ContractVerificationResult:
        """Verify an entire contract.

        Args:
            contract: The contract to verify.
            ledger: Optional ledger to update with results.
            context: Additional context for verification.

        Returns:
            ContractVerificationResult with overall status.
        """
        results: dict[str, VerificationResult] = {}
        unmet: list[str] = []

        for requirement in contract.required_outputs:
            # Check if already met in ledger
            if ledger:
                coverage = ledger.coverage.get(requirement.id)
                if coverage and coverage.status == RequirementStatus.MET:
                    results[requirement.id] = VerificationResult(
                        success=True,
                        message="Already satisfied (from ledger)",
                    )
                    continue

            # Verify the requirement
            result = self.verify_requirement(requirement, context)
            results[requirement.id] = result

            # Update ledger if provided
            if ledger and result.success and result.evidence:
                ledger.mark_met(requirement.id, result.evidence)

            # Track unmet non-optional requirements
            if not result.success and not requirement.optional:
                unmet.append(requirement.id)

        satisfied = len(unmet) == 0

        if satisfied:
            message = "All requirements satisfied"
        else:
            message = f"{len(unmet)} requirement(s) not satisfied: {', '.join(unmet)}"

        return ContractVerificationResult(
            satisfied=satisfied,
            results=results,
            unmet_requirements=unmet,
            message=message,
        )

    def _verify_always_true(self, predicate: Predicate) -> VerificationResult:
        """Verify an always_true predicate (soft requirement)."""
        return VerificationResult(
            success=True,
            message=predicate.description or "Soft requirement (model judgment)",
            evidence=Evidence(
                event_ref="model_judgment",
                kind="response",
                value="Accepted based on model judgment",
            ),
        )

    def _verify_file_exists(self, predicate: Predicate) -> VerificationResult:
        """Verify a file_exists predicate."""
        path = predicate.args.get("path", "")

        # Handle relative paths
        if not os.path.isabs(path):
            path = os.path.join(self._working_directory, path)

        if os.path.exists(path):
            return VerificationResult(
                success=True,
                message=f"File exists: {path}",
                evidence=Evidence(
                    event_ref="file_check",
                    kind="artifact",
                    value=path,
                ),
            )
        return VerificationResult(
            success=False,
            message=f"File not found: {path}",
        )

    def _verify_contains_text(
        self, predicate: Predicate, context: dict[str, Any]
    ) -> VerificationResult:
        """Verify a contains_text predicate."""
        pattern = predicate.args.get("pattern", "")
        field = predicate.args.get("field", "response")

        # Get text to search
        text = ""
        if field == "response":
            # Search in recent transcript responses
            for event in reversed(self._transcript):
                if event.get("event_type") == "agent_response":
                    text = str(event.get("data", {}).get("message", ""))
                    break
        elif field == "transcript":
            # Search entire transcript
            import json
            text = json.dumps(self._transcript)
        else:
            # Check context
            text = str(context.get(field, ""))

        if re.search(pattern, text, re.IGNORECASE):
            return VerificationResult(
                success=True,
                message=f"Found pattern: {pattern}",
                evidence=Evidence(
                    event_ref="text_search",
                    kind="response",
                    value=f"Pattern '{pattern}' found in {field}",
                ),
            )
        return VerificationResult(
            success=False,
            message=f"Pattern not found: {pattern}",
        )

    def _verify_tool_success(self, predicate: Predicate) -> VerificationResult:
        """Verify a tool_result_success predicate."""
        tool_name = predicate.args.get("tool_name", "")

        # Search transcript for successful tool results
        for event in reversed(self._transcript):
            if event.get("event_type") == "tool_result":
                data = event.get("data", {})
                if data.get("tool_name") == tool_name:
                    # Check for success (no error field)
                    if not data.get("error"):
                        return VerificationResult(
                            success=True,
                            message=f"Tool {tool_name} succeeded",
                            evidence=Evidence(
                                event_ref=event.get("event_id", "unknown"),
                                kind="tool_result",
                                value=f"Tool {tool_name} completed successfully",
                            ),
                        )

        return VerificationResult(
            success=False,
            message=f"No successful result found for tool: {tool_name}",
        )

    def _verify_json_schema(
        self, predicate: Predicate, context: dict[str, Any]
    ) -> VerificationResult:
        """Verify a json_schema_valid predicate."""
        # Basic JSON schema validation
        # For full validation, would need jsonschema library
        schema = predicate.args.get("schema", {})
        data = context.get("data")

        if data is None:
            return VerificationResult(
                success=False,
                message="No data provided for schema validation",
            )

        # Basic type checking
        expected_type = schema.get("type")
        if expected_type:
            type_map: dict[str, type | tuple[type, ...]] = {
                "object": dict,
                "array": list,
                "string": str,
                "number": (int, float),
                "boolean": bool,
            }
            expected = type_map.get(expected_type)
            if expected is not None and not isinstance(data, expected):
                return VerificationResult(
                    success=False,
                    message=f"Expected type {expected_type}, got {type(data).__name__}",
                )

        # Check required fields for objects
        if isinstance(data, dict) and "required" in schema:
            missing = [f for f in schema["required"] if f not in data]
            if missing:
                return VerificationResult(
                    success=False,
                    message=f"Missing required fields: {', '.join(missing)}",
                )

        return VerificationResult(
            success=True,
            message="Schema validation passed",
            evidence=Evidence(
                event_ref="schema_check",
                kind="response",
                value="Data matches expected schema",
            ),
        )

    def _verify_custom(
        self, predicate: Predicate, context: dict[str, Any]
    ) -> VerificationResult:
        """Verify a custom predicate.

        Custom predicates can use a 'check' function in args that
        takes context and returns bool.
        """
        check_fn = predicate.args.get("check")

        if check_fn is None:
            # No check function - treat as always true with warning
            logger.warning("Custom predicate has no 'check' function, defaulting to True")
            return VerificationResult(
                success=True,
                message="Custom predicate (no check function)",
            )

        if callable(check_fn):
            try:
                result = check_fn(context)
                if result:
                    return VerificationResult(
                        success=True,
                        message="Custom check passed",
                        evidence=Evidence(
                            event_ref="custom_check",
                            kind="response",
                            value=predicate.description or "Custom verification passed",
                        ),
                    )
                return VerificationResult(
                    success=False,
                    message="Custom check failed",
                )
            except Exception as e:
                return VerificationResult(
                    success=False,
                    message=f"Custom check error: {e!s}",
                )

        return VerificationResult(
            success=False,
            message="Invalid custom predicate configuration",
        )
