import json
import logging
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from pydantic_core import ValidationError
from core.schemas import RedisToProcessCtx
from core.tasks import process_segments
from dramatiq_abort import abort
from redis.asyncio import from_url as redis_from_url
from django.conf import settings


class SegmentConsumer(AsyncJsonWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("django")
        self.redis = redis_from_url(settings.REDIS_URL)

    async def connect(self):
        self.logger.info(f"WebSocket connected with channel name: {self.channel_name}")
        self.processing_msg_ids = set()
        self.authenticated = False
        await self.redis.set(
            "active_channels",
            json.dumps(
                json.loads(await self.redis.get("active_channels") or "[]")
                + [self.channel_name]
            ),
        )
        await self.accept()

    async def disconnect(self, code):
        for msg_id in self.processing_msg_ids:
            try:
                abort(msg_id)
            except Exception as e:
                self.logger.error(f"Failed to abort message {msg_id}: {e}")

        await self.redis.set(
            "active_channels",
            json.dumps(
                [
                    channel
                    for channel in json.loads(
                        await self.redis.get("active_channels") or "[]"
                    )
                    if channel != self.channel_name
                ]
            ),
        )
        self.logger.info(f"WS disconnected with code {code}")

    async def receive_json(self, content, **kwargs):
        self.logger.info(f"WS received: {content}")

        if self.authenticated:
            await self.send_json(
                {"type": "info", "content": "WebSocket is already authenticated"}
            )
            return

        ws_key = content.get("ws_key")
        if not ws_key:
            await self.send_json({"error": "Missing websocket key"})
            await self.close()
            return
        raw = await self.redis.get(f"ws_key:{ws_key}")
        if not raw:
            await self.send_json({"error": "Invalid or expired websocket key"})
            await self.close()
            return

        try:
            data = RedisToProcessCtx.model_validate_json(raw)
        except ValidationError as e:
            self.logger.warning(f"Schema validation failed: {e}")
            await self.send_json({"error": "Corrupt websocket key data"})
            await self.close()
            return

        self.authenticated = True
        msg = process_segments.send(data.segments, data.model, self.channel_name)
        self.processing_msg_ids.add(msg.message_id)
        await self.send_json(
            {
                "type": "info",
                "content": "WebSocket authenticated and started scoring segments",
            }
        )

    async def worker_response(self, payload):
        # Redirect to the websocket
        # self.logger.info(f"Redirecting worker response: {payload}")
        await self.send_json(payload["content"])

        if payload["content"]["type"] == "success":
            await self.close()
