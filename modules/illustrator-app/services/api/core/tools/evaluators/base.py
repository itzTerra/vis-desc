from typing import Iterable
import onnxruntime as rt
import numpy as np
from pathlib import Path


class BaseEvaluator:
    def evaluate_segments(self, segments: list[str]) -> Iterable[tuple[str, float]]:
        raise NotImplementedError


class OnnxMixin:
    def init_inference_session(self, model_path: str | Path) -> None:
        sess = rt.InferenceSession(model_path)
        self.input_name = sess.get_inputs()[0].name
        self.output_names = [out.name for out in sess.get_outputs()]
        self.sess = sess

    def predict(self, X: np.ndarray) -> np.ndarray:
        onnx_inputs = {self.input_name: X}
        outputs = self.sess.run(self.output_names, onnx_inputs)[0]
        return outputs.flatten()
