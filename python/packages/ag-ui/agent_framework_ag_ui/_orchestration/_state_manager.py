# --- PATCHED FOR RELIABILITY (Durability/fallbacks) ---
# Copyright (c) Microsoft. All rights reserved.
"""State orchestration utilities."""

import json
from typing import Any

from ag_ui.core import CustomEvent, EventType
from agent_framework import ChatMessage, TextContent

class StateManager:
    """Coordinates state defaults, snapshots, and structured updates."""

    def __init__(
        self,
        state_schema: dict[str, Any] | None,
        predict_state_config: dict[str, dict[str, str]] | None,
        require_confirmation: bool,
    ) -> None:
        self.state_schema = state_schema or {}
        self.predict_state_config = predict_state_config or {}
        self.require_confirmation = require_confirmation
        self.current_state: dict[str, Any] = {}
        self._state_from_input: bool = False

    def initialize(self, initial_state: dict[str, Any] | None) -> dict[str, Any]:
        """Initialize state with schema defaults."""
        self._state_from_input = initial_state is not None
        # --- PATCH: Defensive fallback for malformed state ---
        if initial_state is not None and not isinstance(initial_state, dict):
            # Allow only dicts as input; forcibly reset and log
            import logging
            logging.warning(f"StateManager: initial_state was {type(initial_state)} - ignoring and using defaults.")
            initial_state = None
        # --- END PATCH ---
        self.current_state = (initial_state or {}).copy()
        self._apply_schema_defaults()
        return self.current_state

    def predict_state_event(self) -> CustomEvent | None:
        if not self.predict_state_config:
            return None
        predict_state_value = [
            {
                "state_key": state_key,
                "tool": config["tool"],
                "tool_argument": config["tool_argument"],
            }
            for state_key, config in self.predict_state_config.items()
        ]
        return CustomEvent(
            type=EventType.CUSTOM,
            name="PredictState",
            value=predict_state_value,
        )

    def initial_snapshot_event(self, event_bridge: Any) -> Any:
        if not self.state_schema:
            return None
        self._apply_schema_defaults()
        return event_bridge.create_state_snapshot_event(self.current_state)

    def state_context_message(self, is_new_user_turn: bool, conversation_has_tool_calls: bool) -> ChatMessage | None:
        if not self.current_state or not self.state_schema:
            return None
        if not is_new_user_turn:
            return None
        if conversation_has_tool_calls and not self._state_from_input:
            return None
        state_json = json.dumps(self.current_state, indent=2)
        return ChatMessage(
            role="system",
            contents=[
                TextContent(
                    text=(
                        "Current state of the application:\n"
                        f"{state_json}\n\n"
                        "When modifying state, you MUST include ALL existing data plus your changes.\n"
                        "For example, if adding one new item to a list, include ALL existing items PLUS the one new item.\n"
                        "Never replace existing data - always preserve and append or merge."
                    )
                )
            ],
        )

    def extract_state_updates(self, response_dict: dict[str, Any]) -> dict[str, Any]:
        if self.state_schema:
            return {key: response_dict[key] for key in self.state_schema.keys() if key in response_dict}
        return {k: v for k, v in response_dict.items() if k != "message"}

    def apply_state_updates(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        self.current_state.update(updates)

    def _apply_schema_defaults(self) -> None:
        for key, schema in self.state_schema.items():
            if key in self.current_state:
                continue
            if isinstance(schema, dict) and schema.get("type") == "array":  # type: ignore
                self.current_state[key] = []
            else:
                self.current_state[key] = {}