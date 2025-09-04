from core.schemas import Evaluator
from core.tools.evaluators.deberta_mnli import DebertaMNLIEvaluator
from dramatiq.middleware import Middleware
import dramatiq
from core.tools.evaluators.random_eval import RandomEvaluator
from core.tools.redis import get_redis_client

worker_resources = {}


class WorkerInitializationMiddleware(Middleware):
    def before_worker_boot(self, broker, worker: dramatiq.Worker):
        worker_resources["redis"] = get_redis_client()
        worker_resources["evaluators"] = {
            Evaluator.deberta_mnli: DebertaMNLIEvaluator(create_model=True),
            Evaluator.random: RandomEvaluator()
        }


        worker.logger.info("Worker initialized resources!")
