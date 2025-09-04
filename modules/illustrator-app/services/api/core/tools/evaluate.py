from typing import Iterable

from core.tools.evaluators.base import BaseEvaluator


def evaluate_segments(classifier: BaseEvaluator, segments: list[str]):
    for segment, score in classifier.evaluate_segments(segments):
        yield {"segment": segment, "score": score}
