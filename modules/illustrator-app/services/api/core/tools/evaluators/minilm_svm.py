from typing import Iterable
import numpy as np
from core.tools.evaluators.base import BaseEvaluator, OnnxMixin
from core.tools.text2features import FeatureService
from core.utils import MODEL_DIR


class MiniLMSVMEvaluator(BaseEvaluator, OnnxMixin):
    MODEL_PATH = MODEL_DIR / "minilm_svm.onnx"

    def __init__(self, feature_service: FeatureService):
        self.feature_service = feature_service
        self.init_inference_session(self.MODEL_PATH)

    def evaluate_segments(self, segments: list[str]) -> Iterable[tuple[str, float]]:
        features = self.feature_service.get_features(
            segments, "sentence-transformers/all-MiniLM-L6-v2"
        )
        scores = self.predict(features)
        # Clamp the scores between 0 and 5
        scores = np.clip(scores, 0, 5)
        # Inverse lerp from [0,5] to [0,1]
        scores = scores / 5.0
        return list(zip(segments, scores.tolist()))
