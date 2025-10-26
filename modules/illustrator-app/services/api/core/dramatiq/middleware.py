from django.conf import settings
from core.schemas import Evaluator

# from core.tools.evaluators.nli_roberta import NLIRoberta
from dramatiq.middleware import Middleware
import dramatiq
from core.tools.evaluators.minilm_svm import MiniLMSVMEvaluator
from core.tools.evaluators.random_eval import RandomEvaluator
from core.tools.redis import get_redis_client
from core.tools.text2features import FeatureService
from core.tools.text2features_paths import (
    FEATURE_PIPELINE_RESOURCES,
)

worker_resources = {}


class WorkerInitializationMiddleware(Middleware):
    def before_worker_boot(self, broker, worker: dramatiq.Worker):
        worker_resources["redis"] = get_redis_client()
        worker_resources["feature_service"] = FeatureService(
            feature_pipeline_resources=FEATURE_PIPELINE_RESOURCES,
            cache_dir=settings.MODEL_CACHE_DIR,
        )

        worker_resources["evaluators"] = {
            Evaluator.minilm_svm: MiniLMSVMEvaluator(
                feature_service=worker_resources["feature_service"]
            ),
            # Evaluator.nli_roberta: NLIRoberta(cache_dir=settings.MODEL_CACHE_DIR),
            Evaluator.random: RandomEvaluator(),
        }

        worker.logger.info("Worker initialized resources!")
