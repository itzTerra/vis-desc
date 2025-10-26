import json
import logging
import asyncio
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from pydantic_core import ValidationError
from core.schemas import RedisToProcessCtx
from core.tasks import process_segment_batch
from dramatiq_abort import abort
from redis.asyncio import from_url as redis_from_url
from django.conf import settings
from core.tools.evaluate import EVALUATOR_TO_BATCH_SIZE, evaluate_segments


consumer_resources = {}
if not settings.ENABLE_DRAMATIQ:
    from core.schemas import Evaluator
    from core.tools.evaluators.nli_roberta import NLIRoberta
    from core.tools.evaluators.minilm_svm import MiniLMSVMEvaluator
    from core.tools.evaluators.random_eval import RandomEvaluator
    from core.tools.text2features import FeatureService
    from core.tools.text2features_paths import (
        FEATURE_PIPELINE_RESOURCES,
    )

    consumer_resources["feature_service"] = FeatureService(
        feature_pipeline_resources=FEATURE_PIPELINE_RESOURCES,
        cache_dir=settings.MODEL_CACHE_DIR,
    )

    consumer_resources["evaluators"] = {
        Evaluator.minilm_svm: MiniLMSVMEvaluator(
            feature_service=consumer_resources["feature_service"]
        ),
        Evaluator.nli_roberta: NLIRoberta(cache_dir=settings.MODEL_CACHE_DIR),
        Evaluator.random: RandomEvaluator(),
    }


class SegmentConsumer(AsyncJsonWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("django")
        self.redis = redis_from_url(settings.REDIS_URL)
        # Batching related attributes
        self._pending_messages: list[dict] = []
        self._flush_task: asyncio.Task | None = None
        self._finished_batches = 0

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
        # Start background flushing loop for batched worker responses
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def disconnect(self, code):
        for msg_id in getattr(self, "processing_msg_ids", []):
            try:
                abort(msg_id)
            except Exception as e:
                self.logger.error(f"Failed to abort message {msg_id}: {e}")

        # Cancel background flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

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

        # Distribute segments to process across workers
        batch_size = EVALUATOR_TO_BATCH_SIZE.get(data.model, 32)
        worker_batches = [
            data.segments[i : i + batch_size]
            for i in range(0, len(data.segments), batch_size)
        ]
        self._finished_batches = 0
        self._batches_to_process = len(worker_batches)

        if settings.ENABLE_DRAMATIQ:
            for i, batch in enumerate(worker_batches):
                msg = process_segment_batch.send(
                    batch, i, data.model, self.channel_name
                )
                self.processing_msg_ids.add(msg.message_id)
        await self.send_json(
            {
                "type": "info",
                "content": "WebSocket authenticated and started scoring segments",
            }
        )
        if not settings.ENABLE_DRAMATIQ:
            for i, batch in enumerate(worker_batches):
                evaluator = consumer_resources["evaluators"].get(data.model)

                res = [s for s in evaluate_segments(evaluator, batch)]

                await self.send_json(
                    {
                        "type": "batch",
                        "content": res,
                    }
                )
                self._finished_batches += 1

    async def worker_response(self, payload):
        # Accumulate worker responses for periodic batching
        content = payload["content"]
        match content.get("type"):
            case "segment":
                self._pending_messages.append(content["content"])
            case "batch":
                try:
                    await self.send_json(
                        {
                            "type": "batch",
                            "content": content["content"],
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Failed to send batch: {e}")
            case "success":
                self._finished_batches += 1

    async def _flush_loop(self):
        try:
            while True:
                await asyncio.sleep(settings.WS_RESPONSE_INTERVAL_SEC)
                if not self._pending_messages:
                    if self._finished_batches == self._batches_to_process:
                        # No pending messages but closure requested (edge case)
                        await self.close()
                        return
                    continue

                to_send = self._pending_messages
                self._pending_messages = []
                try:
                    await self.send_json(
                        {
                            "type": "batch",
                            "content": to_send,
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Failed to send batched messages: {e}")

                if self._finished_batches == self._batches_to_process:
                    try:
                        await self.send_json(
                            {
                                "type": "success",
                                "content": "Processing complete",
                            }
                        )
                    except Exception:
                        pass
                    await self.close()
                    return
        except asyncio.CancelledError:
            # Attempt a final flush before shutting down, if there is anything pending
            if self._pending_messages:
                try:
                    await self.send_json(
                        {
                            "type": "batch",
                            "content": self._pending_messages,
                        }
                    )
                except Exception:
                    pass
            raise
