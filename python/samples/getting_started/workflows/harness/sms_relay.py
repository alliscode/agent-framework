# Copyright (c) Microsoft. All rights reserved.

"""Messaging Relay Service - Standalone SMS and WhatsApp gateway using Azure Communication Services.

This service provides a centralized messaging relay that can be used by multiple
agent instances. It handles:
- Receiving SMS and WhatsApp messages via Azure Event Grid webhooks
- Sending SMS via Azure Communication Services SMS API
- Sending WhatsApp messages via Azure Communication Services Advanced Messaging API
- Message queuing and delivery to connected clients via WebSocket or polling

Usage:
    python sms_relay.py [--port PORT] [--api-key SECRET]

Environment variables required:
    AZURE_COMMUNICATION_CONNECTION_STRING: Connection string for Azure Communication Services
    AZURE_COMMUNICATION_SMS_FROM_NUMBER: Phone number to send SMS from (E.164 format)
    SMS_RELAY_API_KEY: Shared secret for API authentication (optional but recommended)

Optional environment variables for WhatsApp:
    WHATSAPP_CHANNEL_ID: Channel Registration ID for WhatsApp Business Account

API Endpoints:
    POST /sms                    - Azure Event Grid webhook for SMS (no auth required)
    POST /whatsapp               - Azure Event Grid webhook for WhatsApp (no auth required)
    POST /send                   - Send an SMS message (requires API key)
    POST /send/whatsapp          - Send a WhatsApp text message (requires API key)
    POST /send/whatsapp/template - Send a WhatsApp template message (requires API key)
    GET  /messages               - Poll for new messages (requires API key)
    WS   /ws                     - WebSocket for real-time message streaming (requires API key)
    GET  /health                 - Health check (no auth required)

WhatsApp Notes:
    - The first message to a user MUST be a template message (WhatsApp Business API requirement)
    - After the user replies, you can send regular text messages for 24 hours
    - Template messages must be pre-approved in your WhatsApp Business Account

Authentication:
    All protected endpoints require the X-API-Key header:
    curl -H "X-API-Key: your-secret" http://localhost:8080/send ...

Examples:
    # Start with auto-generated API key
    python sms_relay.py --port 8080

    # Start with specific API key
    python sms_relay.py --port 8080 --api-key my-secret-key

    # Send an SMS via the API
    curl -X POST http://localhost:8080/send \
        -H "Content-Type: application/json" \
        -H "X-API-Key: your-secret" \
        -d '{"to": "+15551234567", "message": "Hello!"}'

    # Send a WhatsApp template message (required for first contact)
    curl -X POST http://localhost:8080/send/whatsapp/template \
        -H "Content-Type: application/json" \
        -H "X-API-Key: your-secret" \
        -d '{
            "to": "+15551234567",
            "template": "appointment_reminder",
            "values": {
                "first_name": "John",
                "last_name": "Doe",
                "business": "Lamna Healthcare",
                "datetime": "January 15, 2026 at 10:00 AM"
            }
        }'

    # Send a WhatsApp text message (after user has replied)
    curl -X POST http://localhost:8080/send/whatsapp \
        -H "Content-Type: application/json" \
        -H "X-API-Key: your-secret" \
        -d '{"to": "+15551234567", "message": "Hello via WhatsApp!"}'
"""

import argparse
import asyncio
import json
import logging
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aiohttp import WSMsgType, web
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API Key header name
API_KEY_HEADER = "X-API-Key"


@dataclass
class SmsMessage:
    """Represents an SMS or WhatsApp message."""

    id: str
    direction: str  # "inbound" or "outbound"
    from_number: str
    to_number: str
    message: str
    timestamp: datetime
    delivered: bool = False
    delivery_status: str | None = None
    channel: str = "sms"  # "sms" or "whatsapp"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "direction": self.direction,
            "from": self.from_number,
            "to": self.to_number,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "delivered": self.delivered,
            "delivery_status": self.delivery_status,
            "channel": self.channel,
        }


@dataclass
class MessageQueue:
    """Thread-safe message queue for a phone number."""

    phone_number: str
    messages: list[SmsMessage] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)

    async def add_message(self, msg: SmsMessage):
        """Add a message and notify all subscribers."""
        self.messages.append(msg)
        # Notify WebSocket subscribers
        for queue in self.subscribers:
            await queue.put(msg)

    def get_undelivered(self) -> list[SmsMessage]:
        """Get all undelivered messages."""
        return [m for m in self.messages if not m.delivered]

    def mark_delivered(self, message_ids: list[str]):
        """Mark messages as delivered."""
        for msg in self.messages:
            if msg.id in message_ids:
                msg.delivered = True


class SmsRelay:
    """SMS and WhatsApp Relay service using Azure Communication Services."""

    def __init__(
        self,
        connection_string: str | None = None,
        from_phone_number: str | None = None,
        whatsapp_channel_id: str | None = None,
    ):
        """Initialize the messaging relay.

        Args:
            connection_string: Azure Communication Services connection string.
            from_phone_number: Phone number to send SMS from (E.164 format).
            whatsapp_channel_id: Channel Registration ID for WhatsApp Business Account.
        """
        self.connection_string = connection_string or os.environ.get(
            "AZURE_COMMUNICATION_CONNECTION_STRING"
        )
        self.from_phone_number = from_phone_number or os.environ.get(
            "AZURE_COMMUNICATION_SMS_FROM_NUMBER"
        )
        whatsapp_id = whatsapp_channel_id or os.environ.get("WHATSAPP_CHANNEL_ID")
        # Strip quotes and whitespace that might come from .env file
        self.whatsapp_channel_id = whatsapp_id.strip().strip('"').strip("'") if whatsapp_id else None
        
        logger.info(f"WhatsApp Channel ID from config: {self.whatsapp_channel_id}")

        if not self.connection_string:
            raise ValueError(
                "Connection string required. Set AZURE_COMMUNICATION_CONNECTION_STRING"
            )
        if not self.from_phone_number:
            raise ValueError(
                "From phone number required. Set AZURE_COMMUNICATION_SMS_FROM_NUMBER"
            )

        # Lazy import to allow running without azure SDK for testing
        from azure.communication.sms import SmsClient

        self._sms_client = SmsClient.from_connection_string(self.connection_string)

        # Initialize WhatsApp client if channel ID is provided
        self._whatsapp_client = None
        if self.whatsapp_channel_id:
            try:
                from azure.communication.messages import NotificationMessagesClient

                self._whatsapp_client = NotificationMessagesClient.from_connection_string(
                    self.connection_string
                )
                logger.info(f"WhatsApp client initialized. Channel ID: {self.whatsapp_channel_id}")
            except ImportError:
                logger.warning(
                    "azure-communication-messages package not installed. "
                    "WhatsApp support disabled. Install with: pip install azure-communication-messages"
                )
            except Exception as e:
                logger.error(f"Failed to initialize WhatsApp client: {e}")
        else:
            logger.warning("No WHATSAPP_CHANNEL_ID configured - WhatsApp sending disabled")

        # Message queues by phone number (for inbound messages from that number)
        self._queues: dict[str, MessageQueue] = {}

        # Global queue for all messages (for broadcast subscribers)
        self._global_subscribers: list[asyncio.Queue] = []

        # Message history (for debugging/replay)
        self._all_messages: list[SmsMessage] = []

        logger.info(f"SMS Relay initialized. From: {self.from_phone_number}")
        if self.whatsapp_channel_id:
            logger.info(f"WhatsApp enabled. Channel ID: {self.whatsapp_channel_id}")

    def _get_or_create_queue(self, phone_number: str) -> MessageQueue:
        """Get or create a message queue for a phone number."""
        if phone_number not in self._queues:
            self._queues[phone_number] = MessageQueue(phone_number)
        return self._queues[phone_number]

    async def handle_inbound_message(
        self, from_number: str, to_number: str, message: str, channel: str = "sms"
    ) -> SmsMessage:
        """Handle an inbound SMS or WhatsApp message.

        Args:
            from_number: The sender's phone number.
            to_number: The recipient's phone number (our number) or channel ID.
            message: The message content.
            channel: The channel type ("sms" or "whatsapp").

        Returns:
            The created SmsMessage.
        """
        msg = SmsMessage(
            id=str(uuid.uuid4()),
            direction="inbound",
            from_number=from_number,
            to_number=to_number,
            message=message,
            timestamp=datetime.utcnow(),
            channel=channel,
        )

        # Add to phone-specific queue
        queue = self._get_or_create_queue(from_number)
        await queue.add_message(msg)

        # Add to global history
        self._all_messages.append(msg)

        # Notify global subscribers
        for sub_queue in self._global_subscribers:
            await sub_queue.put(msg)

        logger.info(f"Inbound {channel.upper()} from {from_number}: {message[:50]}...")
        return msg

    async def handle_inbound_sms(self, from_number: str, to_number: str, message: str) -> SmsMessage:
        """Handle an inbound SMS message (backward compatibility wrapper).

        Args:
            from_number: The sender's phone number.
            to_number: The recipient's phone number (our number).
            message: The SMS message content.

        Returns:
            The created SmsMessage.
        """
        return await self.handle_inbound_message(from_number, to_number, message, channel="sms")

    def send_sms(self, to_number: str, message: str) -> SmsMessage | None:
        """Send an SMS message.

        Args:
            to_number: The recipient's phone number.
            message: The message content.

        Returns:
            The SmsMessage if successful, None otherwise.
        """
        try:
            # Truncate very long messages
            if len(message) > 1600:
                message = message[:1597] + "..."

            response = self._sms_client.send(
                from_=self.from_phone_number,
                to=to_number,
                message=message,
                enable_delivery_report=True,
            )

            result = response[0]
            msg = SmsMessage(
                id=result.message_id or str(uuid.uuid4()),
                direction="outbound",
                from_number=self.from_phone_number,
                to_number=to_number,
                message=message,
                timestamp=datetime.utcnow(),
                delivered=result.successful,
                delivery_status="sent" if result.successful else result.error_message,
            )

            self._all_messages.append(msg)

            if result.successful:
                logger.info(f"SMS sent to {to_number}: {message[:50]}...")
            else:
                logger.error(f"SMS failed to {to_number}: {result.error_message}")

            return msg

        except Exception as e:
            logger.error(f"Error sending SMS to {to_number}: {e}")
            return None

    def send_whatsapp(self, to_number: str, message: str) -> SmsMessage | None:
        """Send a WhatsApp message.

        Args:
            to_number: The recipient's phone number (E.164 format).
            message: The message content.

        Returns:
            The SmsMessage if successful, None otherwise.
        """
        if not self._whatsapp_client:
            logger.error("WhatsApp client not initialized. Set WHATSAPP_CHANNEL_ID.")
            return None

        if not self.whatsapp_channel_id:
            logger.error("WhatsApp channel ID not configured.")
            return None

        try:
            from azure.communication.messages.models import TextNotificationContent

            # Truncate very long messages (WhatsApp limit is ~4096 characters)
            if len(message) > 4000:
                message = message[:3997] + "..."

            text_content = TextNotificationContent(
                channel_registration_id=self.whatsapp_channel_id,
                to=[to_number],
                content=message,
            )

            response = self._whatsapp_client.send(text_content)
            receipt = response.receipts[0]

            msg = SmsMessage(
                id=receipt.message_id or str(uuid.uuid4()),
                direction="outbound",
                from_number=self.whatsapp_channel_id,
                to_number=to_number,
                message=message,
                timestamp=datetime.utcnow(),
                delivered=True,
                delivery_status="sent",
                channel="whatsapp",
            )

            self._all_messages.append(msg)
            logger.info(f"WhatsApp sent to {to_number}: {message[:50]}...")
            return msg

        except Exception as e:
            logger.error(f"Error sending WhatsApp to {to_number}: {e}")
            # Store the last error for better error reporting
            self._last_whatsapp_error = str(e)
            return None

    def get_last_whatsapp_error(self) -> str | None:
        """Get the last WhatsApp error message for debugging."""
        return getattr(self, "_last_whatsapp_error", None)

    def send_whatsapp_template(
        self,
        to_number: str,
        template_name: str,
        template_language: str = "en_US",
        template_values: dict[str, str] | None = None,
    ) -> SmsMessage | None:
        """Send a WhatsApp template message.

        Template messages are required for initiating conversations with users.
        The template must be pre-approved in your WhatsApp Business Account.

        Args:
            to_number: The recipient's phone number (E.164 format).
            template_name: The name of the approved template.
            template_language: The language code for the template (default: "en_US").
            template_values: Dictionary mapping parameter names to values.
                For the appointment_reminder template, use:
                {"first_name": "...", "last_name": "...", "business": "...", "datetime": "..."}

        Returns:
            The SmsMessage if successful, None otherwise.
        """
        if not self._whatsapp_client:
            logger.error("WhatsApp client not initialized. Set WHATSAPP_CHANNEL_ID.")
            return None

        if not self.whatsapp_channel_id:
            logger.error("WhatsApp channel ID not configured.")
            return None

        try:
            from azure.communication.messages.models import (
                MessageTemplate,
                MessageTemplateText,
                TemplateNotificationContent,
                WhatsAppMessageTemplateBindings,
                WhatsAppMessageTemplateBindingsComponent,
            )

            # Create the template
            template = MessageTemplate(name=template_name, language=template_language)

            # Build template values and bindings if provided
            if template_values:
                template_text_values = []
                binding_components = []

                for name, text in template_values.items():
                    template_text = MessageTemplateText(name=name, text=text)
                    template_text_values.append(template_text)
                    binding_components.append(
                        WhatsAppMessageTemplateBindingsComponent(ref_value=name)
                    )

                template.template_values = template_text_values
                template.bindings = WhatsAppMessageTemplateBindings(body=binding_components)

            template_content = TemplateNotificationContent(
                channel_registration_id=self.whatsapp_channel_id,
                to=[to_number],
                template=template,
            )

            response = self._whatsapp_client.send(template_content)
            receipt = response.receipts[0]

            # Build a summary message for logging
            message_summary = f"[Template: {template_name}]"
            if template_values:
                message_summary += f" values: {template_values}"

            msg = SmsMessage(
                id=receipt.message_id or str(uuid.uuid4()),
                direction="outbound",
                from_number=self.whatsapp_channel_id,
                to_number=to_number,
                message=message_summary,
                timestamp=datetime.utcnow(),
                delivered=True,
                delivery_status="sent",
                channel="whatsapp",
            )

            self._all_messages.append(msg)
            logger.info(f"WhatsApp template '{template_name}' sent to {to_number}")
            return msg

        except Exception as e:
            logger.error(f"Error sending WhatsApp template to {to_number}: {e}")
            return None

    def get_messages(
        self,
        from_number: str | None = None,
        since: datetime | None = None,
        undelivered_only: bool = False,
    ) -> list[SmsMessage]:
        """Get messages, optionally filtered.

        Args:
            from_number: Filter by sender phone number.
            since: Only return messages after this time.
            undelivered_only: Only return undelivered messages.

        Returns:
            List of matching messages.
        """
        if from_number:
            queue = self._queues.get(from_number)
            if not queue:
                return []
            messages = queue.get_undelivered() if undelivered_only else queue.messages
        else:
            messages = self._all_messages

        if since:
            messages = [m for m in messages if m.timestamp > since]

        if undelivered_only and not from_number:
            messages = [m for m in messages if not m.delivered]

        return messages

    def mark_delivered(self, message_ids: list[str], from_number: str | None = None):
        """Mark messages as delivered.

        Args:
            message_ids: List of message IDs to mark as delivered.
            from_number: If specified, only mark in that queue.
        """
        if from_number:
            queue = self._queues.get(from_number)
            if queue:
                queue.mark_delivered(message_ids)
        else:
            for msg in self._all_messages:
                if msg.id in message_ids:
                    msg.delivered = True

    def subscribe_global(self) -> asyncio.Queue:
        """Subscribe to all inbound messages.

        Returns:
            A queue that will receive all inbound messages.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._global_subscribers.append(queue)
        return queue

    def unsubscribe_global(self, queue: asyncio.Queue):
        """Unsubscribe from global messages."""
        if queue in self._global_subscribers:
            self._global_subscribers.remove(queue)

    def subscribe_phone(self, phone_number: str) -> asyncio.Queue:
        """Subscribe to messages from a specific phone number.

        Returns:
            A queue that will receive messages from that number.
        """
        queue_obj = self._get_or_create_queue(phone_number)
        sub_queue: asyncio.Queue = asyncio.Queue()
        queue_obj.subscribers.append(sub_queue)
        return sub_queue

    def unsubscribe_phone(self, phone_number: str, queue: asyncio.Queue):
        """Unsubscribe from a phone number's messages."""
        queue_obj = self._queues.get(phone_number)
        if queue_obj and queue in queue_obj.subscribers:
            queue_obj.subscribers.remove(queue)


def create_relay_app(relay: SmsRelay, api_key: str) -> web.Application:
    """Create aiohttp web application for the SMS relay.

    Args:
        relay: The SmsRelay instance.
        api_key: The API key for authentication.
    """

    def check_api_key(request: web.Request) -> bool:
        """Verify the API key from request headers."""
        provided_key = request.headers.get(API_KEY_HEADER)
        return secrets.compare_digest(provided_key or "", api_key)

    def require_auth(handler):
        """Decorator to require API key authentication."""

        async def wrapper(request: web.Request) -> web.Response:
            if not check_api_key(request):
                logger.warning(f"Unauthorized request to {request.path} from {request.remote}")
                return web.json_response(
                    {"error": "Unauthorized", "message": f"Missing or invalid {API_KEY_HEADER} header"},
                    status=401,
                )
            return await handler(request)

        return wrapper

    async def handle_webhook(request: web.Request) -> web.Response:
        """Handle incoming SMS events from Azure Event Grid."""
        try:
            # CloudEvents OPTIONS validation
            if request.method == "OPTIONS":
                webhook_origin = request.headers.get("WebHook-Request-Origin", "")
                logger.info(f"CloudEvents OPTIONS validation from: {webhook_origin}")
                return web.Response(
                    status=200,
                    headers={
                        "WebHook-Allowed-Origin": webhook_origin or "*",
                        "WebHook-Allowed-Rate": "100",
                    },
                )

            body = await request.json()
            logger.debug(f"Webhook received: {json.dumps(body)[:500]}")

            # Event Grid schema (array)
            if isinstance(body, list) and len(body) > 0:
                event = body[0]
                event_type = event.get("eventType", "")

                # Validation handshake
                if event_type == "Microsoft.EventGrid.SubscriptionValidationEvent":
                    validation_code = event["data"]["validationCode"]
                    logger.info(f"Event Grid validation - code: {validation_code}")
                    return web.json_response({"validationResponse": validation_code})

                # SMS received
                if event_type == "Microsoft.Communication.SMSReceived":
                    data = event.get("data", {})
                    from_number = data.get("from", "")
                    to_number = data.get("to", relay.from_phone_number)
                    message = data.get("message", "")

                    if from_number and message:
                        await relay.handle_inbound_sms(from_number, to_number, message)

                    return web.Response(status=200, text="OK")

                # Delivery report
                if event_type == "Microsoft.Communication.SMSDeliveryReportReceived":
                    data = event.get("data", {})
                    logger.info(
                        f"Delivery report: {data.get('deliveryStatus')} for {data.get('to')}"
                    )
                    return web.Response(status=200, text="OK")

            # CloudEvents schema (single object)
            elif isinstance(body, dict):
                event_type = body.get("type", "")

                if event_type == "Microsoft.EventGrid.SubscriptionValidationEvent":
                    validation_code = body.get("data", {}).get("validationCode", "")
                    logger.info(f"CloudEvents validation - code: {validation_code}")
                    return web.json_response({"validationResponse": validation_code})

                if event_type == "Microsoft.Communication.SMSReceived":
                    data = body.get("data", {})
                    from_number = data.get("from", "")
                    to_number = data.get("to", relay.from_phone_number)
                    message = data.get("message", "")

                    if from_number and message:
                        await relay.handle_inbound_sms(from_number, to_number, message)

                    return web.Response(status=200, text="OK")

            return web.Response(status=200, text="OK")

        except json.JSONDecodeError:
            logger.error("Invalid JSON in webhook request")
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return web.Response(status=500, text="Internal error")

    async def handle_whatsapp_webhook(request: web.Request) -> web.Response:
        """Handle incoming WhatsApp events from Azure Event Grid."""
        try:
            # CloudEvents OPTIONS validation
            if request.method == "OPTIONS":
                webhook_origin = request.headers.get("WebHook-Request-Origin", "")
                logger.info(f"CloudEvents OPTIONS validation (WhatsApp) from: {webhook_origin}")
                return web.Response(
                    status=200,
                    headers={
                        "WebHook-Allowed-Origin": webhook_origin or "*",
                        "WebHook-Allowed-Rate": "100",
                    },
                )

            body = await request.json()
            logger.debug(f"WhatsApp webhook received: {json.dumps(body)[:500]}")

            # Event Grid schema (array)
            if isinstance(body, list) and len(body) > 0:
                event = body[0]
                event_type = event.get("eventType", "")

                # Validation handshake
                if event_type == "Microsoft.EventGrid.SubscriptionValidationEvent":
                    validation_code = event["data"]["validationCode"]
                    logger.info(f"Event Grid validation (WhatsApp) - code: {validation_code}")
                    return web.json_response({"validationResponse": validation_code})

                # WhatsApp message received
                if event_type == "Microsoft.Communication.AdvancedMessageReceived":
                    data = event.get("data", {})
                    from_number = data.get("from", "")
                    to_number = data.get("to", relay.whatsapp_channel_id or "")
                    content = data.get("content", "")
                    channel_type = data.get("channelType", "whatsapp")

                    if from_number and content:
                        await relay.handle_inbound_message(
                            from_number, to_number, content, channel=channel_type
                        )

                    return web.Response(status=200, text="OK")

                # WhatsApp delivery status update
                if event_type == "Microsoft.Communication.AdvancedMessageDeliveryStatusUpdated":
                    data = event.get("data", {})
                    status = data.get("status", "")
                    message_id = data.get("messageId", "")
                    logger.info(f"WhatsApp delivery status: {status} for message {message_id}")
                    return web.Response(status=200, text="OK")

            # CloudEvents schema (single object)
            elif isinstance(body, dict):
                event_type = body.get("type", "")

                if event_type == "Microsoft.EventGrid.SubscriptionValidationEvent":
                    validation_code = body.get("data", {}).get("validationCode", "")
                    logger.info(f"CloudEvents validation (WhatsApp) - code: {validation_code}")
                    return web.json_response({"validationResponse": validation_code})

                if event_type == "Microsoft.Communication.AdvancedMessageReceived":
                    data = body.get("data", {})
                    from_number = data.get("from", "")
                    to_number = data.get("to", relay.whatsapp_channel_id or "")
                    content = data.get("content", "")
                    channel_type = data.get("channelType", "whatsapp")

                    if from_number and content:
                        await relay.handle_inbound_message(
                            from_number, to_number, content, channel=channel_type
                        )

                    return web.Response(status=200, text="OK")

                if event_type == "Microsoft.Communication.AdvancedMessageDeliveryStatusUpdated":
                    data = body.get("data", {})
                    status = data.get("status", "")
                    message_id = data.get("messageId", "")
                    logger.info(f"WhatsApp delivery status: {status} for message {message_id}")
                    return web.Response(status=200, text="OK")

            return web.Response(status=200, text="OK")

        except json.JSONDecodeError:
            logger.error("Invalid JSON in WhatsApp webhook request")
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            logger.error(f"WhatsApp webhook error: {e}")
            return web.Response(status=500, text="Internal error")

    async def handle_send(request: web.Request) -> web.Response:
        """Send an SMS message (requires API key)."""
        if not check_api_key(request):
            return web.json_response(
                {"error": "Unauthorized", "message": f"Missing or invalid {API_KEY_HEADER} header"},
                status=401,
            )
        try:
            body = await request.json()
            to_number = body.get("to")
            message = body.get("message")

            if not to_number:
                return web.json_response({"error": "Missing 'to' field"}, status=400)
            if not message:
                return web.json_response({"error": "Missing 'message' field"}, status=400)

            result = relay.send_sms(to_number, message)

            if result:
                return web.json_response({
                    "success": True,
                    "message_id": result.id,
                    "delivered": result.delivered,
                })
            else:
                return web.json_response({"success": False, "error": "Failed to send"}, status=500)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_send_whatsapp(request: web.Request) -> web.Response:
        """Send a WhatsApp message (requires API key)."""
        if not check_api_key(request):
            return web.json_response(
                {"error": "Unauthorized", "message": f"Missing or invalid {API_KEY_HEADER} header"},
                status=401,
            )

        if not relay.whatsapp_channel_id:
            return web.json_response(
                {"error": "WhatsApp not configured", "message": "Set WHATSAPP_CHANNEL_ID environment variable"},
                status=503,
            )

        try:
            body = await request.json()
            to_number = body.get("to")
            message = body.get("message")

            if not to_number:
                return web.json_response({"error": "Missing 'to' field"}, status=400)
            if not message:
                return web.json_response({"error": "Missing 'message' field"}, status=400)

            result = relay.send_whatsapp(to_number, message)

            if result:
                return web.json_response({
                    "success": True,
                    "message_id": result.id,
                    "channel": "whatsapp",
                    "delivered": result.delivered,
                })
            else:
                error_detail = relay.get_last_whatsapp_error() or "Failed to send WhatsApp message"
                return web.json_response({
                    "success": False,
                    "error": error_detail,
                    "hint": "If 24h window expired, send a template message first via /send/whatsapp/template",
                }, status=500)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_send_whatsapp_template(request: web.Request) -> web.Response:
        """Send a WhatsApp template message (requires API key).

        Template messages are required for initiating conversations with new users.
        
        Request body:
            {
                "to": "+15551234567",
                "template": "appointment_reminder",
                "language": "en_US",  // optional, defaults to "en_US"
                "values": {
                    "first_name": "John",
                    "last_name": "Doe",
                    "business": "Lamna Healthcare",
                    "datetime": "January 15, 2026 at 10:00 AM"
                }
            }
        """
        if not check_api_key(request):
            return web.json_response(
                {"error": "Unauthorized", "message": f"Missing or invalid {API_KEY_HEADER} header"},
                status=401,
            )

        if not relay.whatsapp_channel_id:
            return web.json_response(
                {"error": "WhatsApp not configured", "message": "Set WHATSAPP_CHANNEL_ID environment variable"},
                status=503,
            )

        try:
            body = await request.json()
            to_number = body.get("to")
            template_name = body.get("template")
            template_language = body.get("language", "en_US")
            template_values = body.get("values", {})

            if not to_number:
                return web.json_response({"error": "Missing 'to' field"}, status=400)
            if not template_name:
                return web.json_response({"error": "Missing 'template' field"}, status=400)

            result = relay.send_whatsapp_template(
                to_number=to_number,
                template_name=template_name,
                template_language=template_language,
                template_values=template_values,
            )

            if result:
                return web.json_response({
                    "success": True,
                    "message_id": result.id,
                    "channel": "whatsapp",
                    "template": template_name,
                    "delivered": result.delivered,
                })
            else:
                return web.json_response({"success": False, "error": "Failed to send WhatsApp template"}, status=500)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_messages(request: web.Request) -> web.Response:
        """Get messages (polling endpoint, requires API key)."""
        if not check_api_key(request):
            return web.json_response(
                {"error": "Unauthorized", "message": f"Missing or invalid {API_KEY_HEADER} header"},
                status=401,
            )
        from_number = request.query.get("from")
        undelivered_only = request.query.get("undelivered", "true").lower() == "true"
        ack = request.query.get("ack", "").split(",")
        ack = [a.strip() for a in ack if a.strip()]

        # Acknowledge previously received messages
        if ack:
            relay.mark_delivered(ack, from_number)

        messages = relay.get_messages(
            from_number=from_number,
            undelivered_only=undelivered_only,
        )

        return web.json_response({
            "messages": [m.to_dict() for m in messages],
            "count": len(messages),
        })

    async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint for real-time message streaming (requires API key)."""
        # Check API key from query param (WebSocket can't use headers easily from browser)
        ws_api_key = request.query.get("api_key", "")
        header_api_key = request.headers.get(API_KEY_HEADER, "")

        if not (secrets.compare_digest(ws_api_key, api_key) or secrets.compare_digest(header_api_key, api_key)):
            logger.warning(f"Unauthorized WebSocket connection attempt from {request.remote}")
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.close(code=4001, message=b"Unauthorized")
            return ws

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Check for phone number filter
        phone_number = request.query.get("from")

        if phone_number:
            queue = relay.subscribe_phone(phone_number)
            logger.info(f"WebSocket connected for phone: {phone_number}")
        else:
            queue = relay.subscribe_global()
            logger.info("WebSocket connected (global)")

        try:
            # Send any pending undelivered messages
            pending = relay.get_messages(from_number=phone_number, undelivered_only=True)
            for msg in pending:
                await ws.send_json(msg.to_dict())

            # Listen for new messages and client pings
            while True:
                # Wait for either a new message or a client message
                done, pending_tasks = await asyncio.wait(
                    [
                        asyncio.create_task(queue.get()),
                        asyncio.create_task(ws.receive()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending_tasks:
                    task.cancel()

                for task in done:
                    result = task.result()

                    # New SMS message to send
                    if isinstance(result, SmsMessage):
                        await ws.send_json(result.to_dict())
                        relay.mark_delivered([result.id], phone_number)

                    # Client WebSocket message
                    elif isinstance(result, web.WSMessage):
                        if result.type == WSMsgType.TEXT:
                            # Handle client commands (e.g., send SMS or WhatsApp)
                            try:
                                data = json.loads(result.data)
                                if data.get("action") == "send":
                                    channel = data.get("channel", "sms")
                                    if channel == "whatsapp":
                                        sent = relay.send_whatsapp(data["to"], data["message"])
                                    else:
                                        sent = relay.send_sms(data["to"], data["message"])
                                    await ws.send_json({
                                        "type": "send_result",
                                        "success": sent is not None,
                                        "message_id": sent.id if sent else None,
                                        "channel": channel,
                                    })
                                elif data.get("action") == "ack":
                                    relay.mark_delivered(data.get("ids", []), phone_number)
                            except json.JSONDecodeError:
                                pass

                        elif result.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                            break

        except asyncio.CancelledError:
            pass
        finally:
            if phone_number:
                relay.unsubscribe_phone(phone_number, queue)
            else:
                relay.unsubscribe_global(queue)
            logger.info("WebSocket disconnected")

        return ws

    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "from_number": relay.from_phone_number,
            "whatsapp_enabled": relay.whatsapp_channel_id is not None,
            "whatsapp_channel_id": relay.whatsapp_channel_id,
            "active_queues": len(relay._queues),
            "total_messages": len(relay._all_messages),
            "global_subscribers": len(relay._global_subscribers),
        })

    app = web.Application()
    app.router.add_route("*", "/sms", handle_webhook)
    app.router.add_route("*", "/whatsapp", handle_whatsapp_webhook)
    app.router.add_post("/send", handle_send)
    app.router.add_post("/send/whatsapp", handle_send_whatsapp)
    app.router.add_post("/send/whatsapp/template", handle_send_whatsapp_template)
    app.router.add_get("/messages", handle_messages)
    app.router.add_get("/ws", handle_websocket)
    app.router.add_get("/health", handle_health)

    return app


def main():
    parser = argparse.ArgumentParser(description="Run SMS Relay Service")
    parser.add_argument(
        "--connection-string",
        type=str,
        default=None,
        help="Azure Communication Services connection string",
    )
    parser.add_argument(
        "--from-number",
        type=str,
        default=None,
        help="Phone number to send SMS from (E.164 format)",
    )
    parser.add_argument(
        "--whatsapp-channel-id",
        type=str,
        default=None,
        help="WhatsApp Channel Registration ID (or set WHATSAPP_CHANNEL_ID env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the relay server (default: 8080)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (or set SMS_RELAY_API_KEY env var)",
    )
    args = parser.parse_args()

    # Get or generate API key
    api_key = args.api_key or os.environ.get("SMS_RELAY_API_KEY")
    if not api_key:
        api_key = secrets.token_urlsafe(32)
        logger.warning("No API key provided, generated a random one")

    try:
        relay = SmsRelay(
            connection_string=args.connection_string,
            from_phone_number=args.from_number,
            whatsapp_channel_id=args.whatsapp_channel_id,
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set the required environment variables:")
        print("  export AZURE_COMMUNICATION_CONNECTION_STRING='your-connection-string'")
        print("  export AZURE_COMMUNICATION_SMS_FROM_NUMBER='+15551234567'")
        print("\nOptional for WhatsApp:")
        print("  export WHATSAPP_CHANNEL_ID='your-channel-id-guid'")
        return

    app = create_relay_app(relay, api_key)

    print(f"\n{'='*60}")
    print("Messaging Relay Service Starting")
    print(f"{'='*60}")
    print(f"From Number: {relay.from_phone_number}")
    if relay.whatsapp_channel_id:
        print(f"WhatsApp Channel ID: {relay.whatsapp_channel_id}")
    else:
        print("WhatsApp: Not configured (set WHATSAPP_CHANNEL_ID to enable)")
    print(f"\n** API Key: {api_key} **")
    print(f"   (Use header: {API_KEY_HEADER}: {api_key})")
    print(f"\nEndpoints (🔒 = requires API key):")
    print(f"  SMS Webhook:       POST http://0.0.0.0:{args.port}/sms                    (open)")
    print(f"  WhatsApp Webhook:  POST http://0.0.0.0:{args.port}/whatsapp               (open)")
    print(f"  Send SMS:          POST http://0.0.0.0:{args.port}/send                   🔒")
    print(f"  Send WhatsApp:     POST http://0.0.0.0:{args.port}/send/whatsapp          🔒")
    print(f"  Send WA Template:  POST http://0.0.0.0:{args.port}/send/whatsapp/template 🔒")
    print(f"  Messages:          GET  http://0.0.0.0:{args.port}/messages               🔒")
    print(f"  WebSocket:         WS   ws://0.0.0.0:{args.port}/ws                       🔒")
    print(f"  Health:            GET  http://0.0.0.0:{args.port}/health                 (open)")
    print(f"{'='*60}")
    print("\nConfigure Azure Event Grid webhooks:")
    print(f"  SMS Endpoint:      https://<your-domain>/sms")
    print(f"  SMS Event:         Microsoft.Communication.SMSReceived")
    print(f"  WhatsApp Endpoint: https://<your-domain>/whatsapp")
    print(f"  WhatsApp Event:    Microsoft.Communication.AdvancedMessageReceived")
    print(f"\nExample usage:")
    print(f"  # Send SMS")
    print(f"  curl -X POST http://localhost:{args.port}/send \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -H '{API_KEY_HEADER}: {api_key}' \\")
    print(f"    -d '{{\"to\": \"+15551234567\", \"message\": \"Hello\"}}'")  
    if relay.whatsapp_channel_id:
        print(f"\n  # Send WhatsApp Template (required for first message to a user)")
        print(f"  curl -X POST http://localhost:{args.port}/send/whatsapp/template \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -H '{API_KEY_HEADER}: {api_key}' \\")
        print(f"    -d '{{")
        print(f'      "to": "+15551234567",')
        print(f'      "template": "appointment_reminder",')
        print(f'      "values": {{')
        print(f'        "first_name": "John", "last_name": "Doe",')
        print(f'        "business": "Lamna Healthcare",')
        print(f'        "datetime": "January 15, 2026 at 10:00 AM"')
        print(f"      }}}}'")
        print(f"\n  # Send WhatsApp (after user has replied)")
        print(f"  curl -X POST http://localhost:{args.port}/send/whatsapp \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -H '{API_KEY_HEADER}: {api_key}' \\")
        print(f"    -d '{{\"to\": \"+15551234567\", \"message\": \"Hello via WhatsApp!\"}}'")  
    print(f"{'='*60}\n")

    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
