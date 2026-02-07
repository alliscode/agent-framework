# Copyright (c) Microsoft. All rights reserved.

"""Run the Agent Harness in DevUI or SMS mode.

This script demonstrates running a harness-wrapped agent in DevUI for
interactive testing and debugging, or via SMS/WhatsApp messaging. The harness provides:
- Turn limits and stall detection
- Continuation prompts to verify task completion
- Task contract verification (optional)
- Context compaction (optional) - production-quality context management
- Work item tracking (optional) - self-critique loop for multi-step tasks
- MCP tool integration (optional) - connect to MCP servers for additional tools
- SMS/WhatsApp messaging (optional) - communicate via SMS relay service

Usage:
    python devui_harness.py [--sandbox PATH] [--port PORT] [--compaction] [--work-items] [--mcp NAME COMMAND ARGS...]
    python devui_harness.py --sms --relay-url URL --api-key KEY [--phone NUMBER] [--sms-mode websocket|polling]

Examples:
    # Basic DevUI usage
    python devui_harness.py

    # With context compaction enabled
    python devui_harness.py --compaction

    # With work item tracking (self-critique loop)
    python devui_harness.py --work-items

    # With both compaction and work items
    python devui_harness.py --compaction --work-items

    # With an MCP stdio server
    python devui_harness.py --mcp compose compose mcp /path/to/project.json

    # With multiple MCP servers
    python devui_harness.py --mcp filesystem npx -y @modelcontextprotocol/server-filesystem /tmp --mcp compose compose mcp /path/to/project.json

    # SMS mode - connect to relay and listen for all messages
    python devui_harness.py --sms --relay-url http://localhost:8080 --api-key my-secret

    # SMS mode - only handle messages from a specific number
    python devui_harness.py --sms --relay-url http://localhost:8080 --api-key my-secret --phone +15551234567

    # SMS mode with context compaction for long conversations
    python devui_harness.py --sms --relay-url http://localhost:8080 --api-key my-secret --compaction

Environment variables for SMS mode:
    SMS_RELAY_URL: URL of the SMS relay service
    SMS_RELAY_API_KEY: API key for the relay service
    SMS_PHONE_NUMBER: Phone number to listen for (optional, listens to all if not set)
"""

import argparse
import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from agent_framework import MCPStdioTool
from agent_framework._harness import (
    AgentHarness,
    InMemoryArtifactStore,
    InMemoryCompactionStore,
    InMemorySummaryCache,
    MarkdownRenderer,
    get_task_complete_tool,
    render_stream,
)
from agent_framework._workflows._events import AgentRunUpdateEvent
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework_devui import serve
from azure.identity import AzureCliCredential
from coding_tools import CodingTools
from dotenv import load_dotenv

# Load .env file for SMS configuration
load_dotenv()

# Enable debug logging for harness and mapper
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("agent_framework._harness").setLevel(logging.DEBUG)
logging.getLogger("agent_framework._workflows").setLevel(logging.DEBUG)
logging.getLogger("agent_framework_devui._mapper").setLevel(logging.DEBUG)
logging.getLogger("agent_framework_devui._executor").setLevel(logging.DEBUG)
# Also log raw events from the harness
logging.getLogger("agent_framework_devui._server").setLevel(logging.DEBUG)


AGENT_INSTRUCTIONS = """You are a capable AI coding assistant with access to a local workspace.
You can read and write files, list directories, and run shell commands.
When asked to investigate code, be thorough — read every relevant source file
before drawing conclusions or writing deliverables.
"""

SMS_AGENT_INSTRUCTIONS = """You are a helpful messaging assistant. You receive text messages via SMS or WhatsApp and respond via the same channel.

IMPORTANT CONSTRAINTS:
- Keep responses SHORT - SMS messages over 160 characters get split and cost more
- Be concise and direct
- If a task requires multiple steps, summarize progress briefly
- Avoid code blocks, bullet points, and formatting that doesn't work in text messages

GUIDELINES:
1. Understand the user's request from their message
2. Provide a helpful, concise response
3. If you need clarification, ask a simple question
4. When done with a task, respond with a brief summary

STYLE:
- Conversational but efficient
- No emojis unless the user uses them
- Plain text only - no markdown
"""


class RenderedHarness:
    """Wrapper that applies MarkdownRenderer to harness run_stream output.

    This wraps an AgentHarness and intercepts run_stream() calls to apply
    the MarkdownRenderer, which formats progress indicators and deliverables
    as markdown for DevUI display.
    """

    def __init__(self, harness: AgentHarness, use_renderer: bool = True):
        self._harness = harness
        self._renderer = MarkdownRenderer() if use_renderer else None
        # Copy attributes DevUI needs for discovery
        self.id = harness.id
        self.name = harness.name
        self.description = harness.description

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying harness."""
        return getattr(self._harness, name)

    async def run_stream(self, message: str, **kwargs: Any):
        """Run the harness with optional markdown rendering."""
        if self._renderer:
            async for event in render_stream(self._harness, message, self._renderer, **kwargs):
                yield event
        else:
            async for event in self._harness.run_stream(message, **kwargs):
                yield event


# =============================================================================
# SMS Relay Client and Harness
# =============================================================================

logger = logging.getLogger(__name__)


class SmsRelayClient:
    """Client for communicating with the SMS Relay service."""

    def __init__(self, relay_url: str, api_key: str):
        """Initialize the relay client.

        Args:
            relay_url: Base URL of the SMS relay service.
            api_key: API key for authentication.
        """
        self.relay_url = relay_url.rstrip("/")
        self.api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with auth headers."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-API-Key": self.api_key},
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_sms(self, to_number: str, message: str) -> bool:
        """Send an SMS via the relay service.

        Args:
            to_number: The recipient's phone number.
            message: The message content.

        Returns:
            True if successful, False otherwise.
        """
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.relay_url}/send",
                json={"to": to_number, "message": message},
            ) as response:
                result = await response.json()
                if result.get("success"):
                    logger.info(f"SMS sent to {to_number}: {message[:50]}...")
                    return True
                logger.error(f"SMS failed to {to_number}: {result.get('error')}")
                return False
        except Exception as e:
            logger.error("Error sending SMS via relay: %s", e)
            return False

    async def send_whatsapp(self, to_number: str, message: str) -> bool:
        """Send a WhatsApp message via the relay service.

        Args:
            to_number: The recipient's phone number.
            message: The message content.

        Returns:
            True if successful, False otherwise.
        """
        try:
            session = await self._get_session()

            async with session.post(
                f"{self.relay_url}/send/whatsapp",
                json={"to": to_number, "message": message},
            ) as response:
                result = await response.json()
                if result.get("success"):
                    logger.info(f"WhatsApp sent to {to_number}: {message[:50]}...")
                    return True
                error = result.get("error", "Unknown error")
                hint = result.get("hint", "")
                logger.error("WhatsApp failed to %s: %s", to_number, error)
                if hint:
                    logger.info("Hint: %s", hint)
                return False
        except Exception as e:
            logger.error("Error sending WhatsApp via relay: %s", e)
            return False

    async def send_message(self, to_number: str, message: str, channel: str = "sms") -> bool:
        """Send a message via the appropriate channel.

        Args:
            to_number: The recipient's phone number.
            message: The message content.
            channel: The channel to use ("sms" or "whatsapp").

        Returns:
            True if successful, False otherwise.
        """
        if channel == "whatsapp":
            return await self.send_whatsapp(to_number, message)
        return await self.send_sms(to_number, message)

    async def poll_messages(
        self,
        from_number: str | None = None,
        ack_ids: list[str] | None = None,
    ) -> list[dict]:
        """Poll for new messages from the relay.

        Args:
            from_number: Optional phone number to filter by.
            ack_ids: Optional list of message IDs to acknowledge.

        Returns:
            List of message dictionaries.
        """
        try:
            session = await self._get_session()
            params = {"undelivered": "true"}
            if from_number:
                params["from"] = from_number
            if ack_ids:
                params["ack"] = ",".join(ack_ids)

            async with session.get(
                f"{self.relay_url}/messages",
                params=params,
            ) as response:
                result = await response.json()
                return result.get("messages", [])
        except Exception as e:
            logger.error("Error polling messages: %s", e)
            return []

    async def stream_messages(
        self,
        from_number: str | None = None,
    ):
        """Stream messages via WebSocket.

        Args:
            from_number: Optional phone number to filter by.

        Yields:
            Message dictionaries as they arrive.
        """
        import json as json_module

        ws_url = self.relay_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?api_key={self.api_key}"
        if from_number:
            ws_url = f"{ws_url}&from={from_number}"

        session = await self._get_session()

        while True:
            try:
                async with session.ws_connect(ws_url) as ws:
                    logger.info("WebSocket connected to relay")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json_module.loads(msg.data)
                            yield data
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.info("WebSocket closed")
                            break
            except aiohttp.ClientError as e:
                logger.error("WebSocket connection error: %s", e)
                await asyncio.sleep(5)  # Retry after delay
            except Exception as e:
                logger.error("Unexpected WebSocket error: %s", e)
                await asyncio.sleep(5)


class SmsConversation:
    """Tracks conversation state for a phone number."""

    def __init__(self, phone_number: str, harness: AgentHarness):
        self.phone_number = phone_number
        self.harness = harness
        self.created_at = datetime.now()
        self.last_activity = datetime.now()


class SmsHarness:
    """SMS agent harness that communicates via SMS Relay service."""

    def __init__(
        self,
        relay_url: str,
        api_key: str,
        phone_filter: str | None = None,
        enable_compaction: bool = False,
    ):
        """Initialize the SMS harness.

        Args:
            relay_url: URL of the SMS relay service.
            api_key: API key for the relay service.
            phone_filter: Optional phone number to filter incoming messages.
            enable_compaction: Whether to enable context compaction.
        """
        self.relay_url = relay_url
        self.phone_filter = phone_filter
        self._enable_compaction = enable_compaction

        # Initialize relay client
        self._relay = SmsRelayClient(relay_url, api_key)

        # Initialize chat client for agents
        self._chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())

        # Track conversations by phone number
        self._conversations: dict[str, SmsConversation] = {}

        logger.info("SMS Harness initialized. Relay: %s", relay_url)
        if phone_filter:
            logger.info("Filtering for phone: %s", phone_filter)

    def _create_harness_for_conversation(self) -> AgentHarness:
        """Create a new harness instance for a conversation."""
        agent = self._chat_client.create_agent(
            name="sms-assistant",
            description="SMS assistant",
            instructions=SMS_AGENT_INSTRUCTIONS,
            tools=[get_task_complete_tool()],
        )

        compaction_kwargs: dict[str, Any] = {}
        if self._enable_compaction:
            compaction_kwargs = {
                "enable_compaction": True,
                "compaction_store": InMemoryCompactionStore(),
                "artifact_store": InMemoryArtifactStore(),
                "summary_cache": InMemorySummaryCache(max_entries=50),
                "max_input_tokens": 50_000,
                "soft_threshold_percent": 0.80,
            }

        return AgentHarness(
            agent,
            max_turns=5,  # Keep SMS interactions short
            enable_stall_detection=True,
            stall_threshold=2,
            enable_continuation_prompts=False,  # No continuation for SMS
            **compaction_kwargs,
        )

    def _get_or_create_conversation(self, phone_number: str) -> SmsConversation:
        """Get existing conversation or create new one."""
        if phone_number not in self._conversations:
            harness = self._create_harness_for_conversation()
            self._conversations[phone_number] = SmsConversation(phone_number, harness)
            logger.info("New conversation started with %s", phone_number)
        else:
            self._conversations[phone_number].last_activity = datetime.now()

        return self._conversations[phone_number]

    async def handle_incoming_sms(self, from_number: str, message: str) -> str:
        """Handle an incoming SMS and generate a response.

        Args:
            from_number: The sender's phone number.
            message: The SMS message content.

        Returns:
            The agent's response text.
        """
        logger.info("Incoming SMS from %s: %s", from_number, message)

        # Get or create conversation
        conversation = self._get_or_create_conversation(from_number)

        # Run the harness with the message
        response_text = ""
        try:
            async for event in conversation.harness.run_stream(message):
                # Handle AgentRunUpdateEvent - this contains the streaming text
                if isinstance(event, AgentRunUpdateEvent) and event.data:
                    update = event.data
                    if hasattr(update, "text") and update.text:
                        response_text += update.text

            # Clean up response for SMS
            response_text = response_text.strip()
            logger.info(f"Generated response ({len(response_text)} chars): {response_text[:100]}...")

            # If empty response, provide a default
            if not response_text:
                response_text = "I received your message but couldn't generate a response. Please try again."

        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)
            response_text = "Sorry, I encountered an error. Please try again."

        return response_text

    async def process_and_respond(self, from_number: str, message: str, channel: str = "sms"):
        """Process incoming message and send response via the same channel."""
        response = await self.handle_incoming_sms(from_number, message)
        await self._relay.send_message(from_number, response, channel=channel)

    async def run_polling(self, poll_interval: float = 2.0):
        """Run the harness using polling mode.

        Args:
            poll_interval: Seconds between polls.
        """
        logger.info("Starting polling mode (interval: %ss)", poll_interval)

        while True:
            try:
                messages = await self._relay.poll_messages(from_number=self.phone_filter)

                for msg in messages:
                    if msg.get("direction") == "inbound":
                        from_number = msg.get("from")
                        text = msg.get("message")
                        msg_id = msg.get("id")
                        channel = msg.get("channel", "sms")  # Default to SMS for backward compat

                        if from_number and text:
                            # Process the message and respond on the same channel
                            await self.process_and_respond(from_number, text, channel=channel)

                            # Acknowledge the message
                            await self._relay.poll_messages(ack_ids=[msg_id])

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Polling error: %s", e)
                await asyncio.sleep(poll_interval)

    async def run_websocket(self):
        """Run the harness using WebSocket streaming mode."""
        logger.info("Starting WebSocket mode")

        async for msg in self._relay.stream_messages(from_number=self.phone_filter):
            try:
                if msg.get("direction") == "inbound":
                    from_number = msg.get("from")
                    text = msg.get("message")
                    channel = msg.get("channel", "sms")  # Default to SMS for backward compat

                    if from_number and text:
                        await self.process_and_respond(from_number, text, channel=channel)

            except Exception as e:
                logger.error("Error processing WebSocket message: %s", e)

    async def close(self):
        """Clean up resources."""
        await self._relay.close()

    def reset_conversation(self, phone_number: str):
        """Reset conversation state for a phone number."""
        if phone_number in self._conversations:
            del self._conversations[phone_number]
            logger.info("Conversation reset for %s", phone_number)


# =============================================================================
# Agent Creation
# =============================================================================

def create_harness_agent(
    sandbox_dir: Path,
    enable_compaction: bool = True,
    enable_work_items: bool = True,
    mcp_tools: list[MCPStdioTool] | None = None,
) -> AgentHarness:
    """Create a harness-wrapped agent with coding tools.

    Returns an AgentHarness that can be registered with DevUI.
    The AgentHarness has a run_stream(message) method that DevUI
    can call directly with the user's input.

    Args:
        sandbox_dir: Directory for agent workspace.
        enable_compaction: Whether to enable production context compaction (default: True).
        enable_work_items: Whether to enable work item tracking (default: True).
        mcp_tools: Optional list of connected MCP tools to add to the agent.
    """
    # Create tools sandboxed to the directory
    tools = CodingTools(sandbox_dir)
    all_tools = tools.get_tools() + [get_task_complete_tool()]

    # Add MCP tool functions if provided
    if mcp_tools:
        for mcp_tool in mcp_tools:
            all_tools.extend(mcp_tool.functions)
            print(f"Added {len(mcp_tool.functions)} tools from MCP server '{mcp_tool.name}'")

    # Create the underlying agent
    chat_client = AzureOpenAIChatClient(credential=AzureCliCredential())
    agent = chat_client.create_agent(
        name="coding-assistant",
        description="A coding assistant that can read/write files and run commands",
        instructions=AGENT_INSTRUCTIONS,
        tools=all_tools,
    )

    # Configure compaction stores if enabled
    compaction_kwargs: dict[str, Any] = {}
    if enable_compaction:
        compaction_kwargs = {
            "enable_compaction": True,
            "compaction_store": InMemoryCompactionStore(),
            "artifact_store": InMemoryArtifactStore(),
            "summary_cache": InMemorySummaryCache(max_entries=100),
            "max_input_tokens": 100_000,
            "soft_threshold_percent": 0.85,
        }

    # Wrap in harness with all features enabled
    harness = AgentHarness(
        agent,
        max_turns=50,
        enable_stall_detection=True,
        stall_threshold=3,
        enable_continuation_prompts=True,
        max_continuation_prompts=2,
        enable_work_items=enable_work_items,
        sandbox_path=str(sandbox_dir),
        **compaction_kwargs,
    )

    # Add id, name and description for DevUI discovery
    # DevUI requires both id and name for agent-like entities
    harness.id = "coding-harness"
    harness.name = "coding-harness"
    compaction_status = "enabled" if enable_compaction else "disabled"
    work_items_status = "enabled" if enable_work_items else "disabled"
    harness.description = (
        f"Coding assistant with harness infrastructure. "
        f"Sandbox: {sandbox_dir}. Compaction: {compaction_status}. "
        f"Work items: {work_items_status}"
    )

    return harness


def _connect_mcp_tools_sync(mcp_tools: list[MCPStdioTool]) -> list[MCPStdioTool]:
    """Connect MCP tools synchronously before starting the event loop.

    This runs the async connect() in an event loop, keeping the connections
    alive for use with DevUI's event loop.

    Args:
        mcp_tools: List of MCP tools to connect.

    Returns:
        List of successfully connected MCP tools.
    """
    connected: list[MCPStdioTool] = []

    async def connect_all():
        for mcp_tool in mcp_tools:
            try:
                await mcp_tool.connect()
                connected.append(mcp_tool)
                print(f"Connected to MCP server '{mcp_tool.name}' - {len(mcp_tool.functions)} tools available")
            except Exception as e:
                print(f"Error connecting to MCP server '{mcp_tool.name}': {e}")

    # Get or create event loop and run connection
    # Note: We don't close the loop to keep MCP connections alive
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(connect_all())

    return connected


def main():
    """Entry point for the DevUI or SMS harness."""
    parser = argparse.ArgumentParser(
        description="Run coding agent harness in DevUI or SMS mode",
    )

    # Common arguments
    parser.add_argument(
        "--compaction",
        action="store_true",
        help="Enable production context compaction (Phase 9)",
    )

    # DevUI-specific arguments
    devui_group = parser.add_argument_group("DevUI options")
    devui_group.add_argument(
        "--sandbox",
        type=Path,
        default=None,
        help="Directory for agent workspace (default: temp directory)",
    )
    devui_group.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for DevUI server (default: 8080)",
    )
    devui_group.add_argument(
        "--work-items",
        action="store_true",
        help="Enable work item tracking (self-critique loop)",
    )
    devui_group.add_argument(
        "--mcp",
        action="append",
        nargs="+",
        metavar=("NAME", "COMMAND"),
        help="Add MCP stdio server: --mcp NAME COMMAND [ARGS...] (can be repeated)",
    )

    # SMS-specific arguments
    sms_group = parser.add_argument_group("SMS options")
    sms_group.add_argument(
        "--sms",
        action="store_true",
        help="Run in SMS mode instead of DevUI (connects to SMS relay service)",
    )
    sms_group.add_argument(
        "--relay-url",
        type=str,
        default=os.environ.get("SMS_RELAY_URL", "http://localhost:8080"),
        help="URL of the SMS relay service (default: http://localhost:8080)",
    )
    sms_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the SMS relay service (or set SMS_RELAY_API_KEY env var)",
    )
    sms_group.add_argument(
        "--phone",
        type=str,
        default=os.environ.get("SMS_PHONE_NUMBER"),
        help="Phone number to filter incoming messages (optional)",
    )
    sms_group.add_argument(
        "--sms-mode",
        choices=["websocket", "polling"],
        default="websocket",
        help="SMS connection mode: websocket (real-time) or polling (default: websocket)",
    )
    sms_group.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds (only for polling mode, default: 2.0)",
    )

    args = parser.parse_args()

    # Run in SMS mode or DevUI mode
    if args.sms:
        _run_sms_mode(args)
    else:
        _run_devui_mode(args)


async def _run_sms_harness(args):
    """Run the SMS harness asynchronously."""
    harness = SmsHarness(
        relay_url=args.relay_url,
        api_key=args.api_key,
        phone_filter=args.phone,
        enable_compaction=args.compaction,
    )

    try:
        if args.sms_mode == "websocket":
            await harness.run_websocket()
        else:
            await harness.run_polling(poll_interval=args.poll_interval)
    finally:
        await harness.close()


def _run_sms_mode(args):
    """Run in SMS messaging mode."""
    # Get API key
    api_key = args.api_key or os.environ.get("SMS_RELAY_API_KEY")
    if not api_key:
        print("Error: API key required. Use --api-key or set SMS_RELAY_API_KEY env var")
        return

    args.api_key = api_key

    print(f"\n{'=' * 60}")
    print("SMS/WhatsApp Agent Harness Starting")
    print(f"{'=' * 60}")
    print(f"Relay URL: {args.relay_url}")
    print(f"Mode: {args.sms_mode}")
    if args.phone:
        print(f"Phone filter: {args.phone}")
    else:
        print("Phone filter: ALL (no filter)")
    if args.compaction:
        print("Context compaction: ENABLED")
    print("Channels: SMS and WhatsApp (auto-detected from incoming messages)")
    print(f"{'=' * 60}\n")

    try:
        asyncio.run(_run_sms_harness(args))
    except KeyboardInterrupt:
        print("\nShutting down...")


def _run_devui_mode(args):
    """Run in DevUI mode (original behavior)."""
    # Determine sandbox directory
    if args.sandbox:
        sandbox_dir = args.sandbox.resolve()
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using sandbox: {sandbox_dir}")
    else:
        # Use temporary directory
        temp_dir = tempfile.mkdtemp(prefix="harness_devui_")
        sandbox_dir = Path(temp_dir)
        print(f"Using temp sandbox: {sandbox_dir}")

    # Parse and create MCP tools
    mcp_tools: list[MCPStdioTool] = []
    if args.mcp:
        for mcp_args in args.mcp:
            if len(mcp_args) < 2:
                print(f"Error: --mcp requires at least NAME and COMMAND, got: {mcp_args}")
                return
            name = mcp_args[0]
            command = mcp_args[1]
            command_args = mcp_args[2:] if len(mcp_args) > 2 else []
            mcp_tool = MCPStdioTool(
                name=name,
                command=command,
                args=command_args,
                description=f"MCP server: {name}",
                approval_mode="never_require",
            )
            mcp_tools.append(mcp_tool)
            print(f"Configured MCP server '{name}': {command} {' '.join(command_args)}")

    # Connect MCP tools before starting DevUI
    connected_tools: list[MCPStdioTool] = []
    if mcp_tools:
        connected_tools = _connect_mcp_tools_sync(mcp_tools)

    # Create the harness-wrapped agent with MCP tools
    harness = create_harness_agent(
        sandbox_dir,
        enable_compaction=args.compaction,
        enable_work_items=args.work_items,
        mcp_tools=connected_tools or None,
    )

    # Wrap with MarkdownRenderer when work items are enabled
    if args.work_items:
        harness = RenderedHarness(harness, use_renderer=True)

    print(f"\nStarting DevUI on port {args.port}...")
    print(f"Sandbox directory: {sandbox_dir}")
    print("Harness config: max_turns=20, stall_threshold=3")
    if args.compaction:
        print("Context compaction: ENABLED (100K tokens, 85% threshold)")
    else:
        print("Context compaction: disabled")
    if args.work_items:
        print("Work item tracking: ENABLED (self-critique loop)")
        print("Markdown renderer: ENABLED (activity verbs, progress bars)")
    else:
        print("Work item tracking: disabled")
    if connected_tools:
        total_mcp_tools = sum(len(t.functions) for t in connected_tools)
        print(f"MCP servers: {len(connected_tools)} connected ({total_mcp_tools} tools)")

    # Launch DevUI with the harness
    serve(
        entities=[harness],
        port=args.port,
        auto_open=True,
        mode="developer",
    )


if __name__ == "__main__":
    main()
