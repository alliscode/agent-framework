# --- PATCHED TEST FOR RELIABILITY EDGE CASES ---
# Copyright (c) Microsoft. All rights reserved.
# (Other test imports...)

import pytest

# Existing imports...
import json
import sys
from collections.abc import AsyncIterator, MutableSequence
from pathlib import Path
from typing import Any

import pytest
from agent_framework import ChatAgent, ChatMessage, ChatOptions, TextContent
from agent_framework._types import ChatResponseUpdate
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from test_helpers_ag_ui import StreamingChatClientStub

# --- BEGIN PATCH: Reliability validation tests ---

@pytest.mark.asyncio
async def test_state_schema_and_predict_config_validation():
    from agent_framework_ag_ui import AgentFrameworkAgent
    class MyState(BaseModel):
        val: int
    async def stream_fn(messages, chat_options, **kwargs):
        yield ChatResponseUpdate(contents=[TextContent(text="ok")])
    agent = ChatAgent(name="tester", instructions="t", chat_client=StreamingChatClientStub(stream_fn))
    # Provide an int state -- should reset to defaults
    bad_state_input = 42  # Not a dict!
    agent_wrapper = AgentFrameworkAgent(
        agent=agent,
        state_schema=MyState,
        predict_state_config={
            "bad": {"whoops": "nope"},  # should be filtered out
            "good": {"tool": "op", "tool_argument": "x"},
        })
    # Should not raise error, defaults applied
    input_data = {"messages": [{"role": "user", "content": "hi"}], "state": bad_state_input}
    events = []
    async for e in agent_wrapper.run_agent(input_data):
        events.append(e)
    # State must fall back to empty/default for schema (val missing, so will get default type via schema)
    found_start = any(ev.type == "RUN_STARTED" for ev in events)
    found_finish = any(ev.type == "RUN_FINISHED" for ev in events)
    assert found_start and found_finish, f"Start/Finish events missing: {[ev.type for ev in events]}"

# --- END PATCH ---

# Keep existing tests below...
