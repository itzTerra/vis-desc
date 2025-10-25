import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from core.schemas import Evaluator
from core.tools.evaluate import evaluate_segments
import dramatiq
from core.dramatiq.middleware import worker_resources
from dramatiq.middleware import TimeLimitExceeded


def check_consumer_status(redis, channel_name: str):
    """
    Check if the consumer is still connected.
    """
    active_channels = json.loads(redis.get("active_channels") or "[]")
    return channel_name in active_channels


@dramatiq.actor(
    time_limit=10 * 60 * 1000,
    max_retries=0,
    max_age=10 * 60 * 1000,
    throws=(TimeLimitExceeded,),
)
def process_segments(segments: list[str], model: Evaluator, channel_name: str):
    logger = process_segments.logger  # type: ignore
    actor_name = process_segments.actor_name
    if not check_consumer_status(worker_resources["redis"], channel_name):
        logger.info(f"Worker {actor_name} aborted")
        return

    channel_layer = get_channel_layer()
    assert channel_layer is not None, "Channel layer is not configured"

    count = 0
    segment_count = len(segments)
    evaluator = worker_resources["evaluators"].get(model)

    for res in evaluate_segments(evaluator, segments):
        if not check_consumer_status(worker_resources["redis"], channel_name):
            logger.info(f"Worker {actor_name} aborted")
            return
        logger.info(f"Worker {actor_name} sent segment {count} of {segment_count}")
        async_to_sync(channel_layer.send)(
            channel_name,
            {
                "type": "worker.response",
                "content": {
                    "type": "segment",
                    "content": res,
                },
            },
        )
        count += 1

    logger.info(f"Worker {actor_name} finished")
    async_to_sync(channel_layer.send)(
        channel_name,
        {
            "type": "worker.response",
            "content": {
                "type": "success",
                "content": f"Worker {actor_name} finished",
            },
        },
    )


@dramatiq.actor(
    time_limit=10 * 60 * 1000,
    max_retries=0,
    max_age=10 * 60 * 1000,
    throws=(TimeLimitExceeded,),
)
def process_segment_batch(
    segments: list[str], batch_index: int, model: Evaluator, channel_name: str
):
    logger = process_segment_batch.logger  # type: ignore
    actor_name = process_segment_batch.actor_name
    if not check_consumer_status(worker_resources["redis"], channel_name):
        logger.info(f"Worker {actor_name} aborted")
        return

    channel_layer = get_channel_layer()
    assert channel_layer is not None, "Channel layer is not configured"

    segment_count = len(segments)
    evaluator = worker_resources["evaluators"].get(model)

    res = [s for s in evaluate_segments(evaluator, segments)]

    if not check_consumer_status(worker_resources["redis"], channel_name):
        logger.info(f"Worker {actor_name} aborted")
        return

    async_to_sync(channel_layer.send)(
        channel_name,
        {
            "type": "worker.response",
            "content": {
                "type": "batch",
                "content": res,
            },
        },
    )
    logger.info(
        f"Worker {actor_name} sent {segment_count} segments of batch {batch_index}"
    )
    async_to_sync(channel_layer.send)(
        channel_name,
        {
            "type": "worker.response",
            "content": {
                "type": "success",
                "content": f"Worker {actor_name} finished",
            },
        },
    )
