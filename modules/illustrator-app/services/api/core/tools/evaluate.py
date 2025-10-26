from core.schemas import Evaluator
from core.tools.evaluators.base import BaseEvaluator

EVALUATOR_TO_BATCH_SIZE: dict[Evaluator, int] = {
    Evaluator.minilm_svm: 32,
    Evaluator.nli_roberta: 32,
    Evaluator.random: 256,
}


def evaluate_segments(classifier: BaseEvaluator, segments: list[str]):
    for segment, score in classifier.evaluate_segments(segments):
        yield {"text": segment, "score": score}
