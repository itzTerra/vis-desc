import random

from core.tools.evaluators.base import BaseEvaluator


class RandomEvaluator(BaseEvaluator):
    def evaluate_segments(self, segments: list[str]):
        return [(segment, random.uniform(0, 1)) for segment in segments]
