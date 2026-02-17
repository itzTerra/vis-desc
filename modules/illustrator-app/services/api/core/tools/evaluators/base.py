import abc
from typing import Iterable


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate_segments(self, segments: list[str]) -> Iterable[tuple[str, float]]:
        pass

    @abc.abstractmethod
    def uses_dramatiq(self) -> bool:
        pass
