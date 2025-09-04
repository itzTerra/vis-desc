from typing import Iterable


class BaseEvaluator:
    def evaluate_segments(self, segments: list[str]) -> Iterable[tuple[str, float]]:
        raise NotImplementedError
