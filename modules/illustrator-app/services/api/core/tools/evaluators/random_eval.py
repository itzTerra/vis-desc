import random


class RandomEvaluator:
    def evaluate_segments(self, segments: list[str]):
        return [(segment, random.uniform(0, 1)) for segment in segments]
