from .nli_base import NLIZeroshotClassifier
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from typing import Iterable


class NLIRoberta(NLIZeroshotClassifier):
    """https://huggingface.co/richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"""

    def __init__(self):
        super().__init__(
            candidate_labels=["detailed", "not detailed"],
            hypothesis_template="This text is {} in terms of visual details of characters, setting, or environment.",
        )

    def create_model(self):
        if self.classifier is None:
            # Load the quantized model
            model = ORTModelForSequenceClassification.from_pretrained(
                "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX",
                file_name="model_quantized.onnx",
            )

            tokenizer = AutoTokenizer.from_pretrained(
                "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"
            )

            # Patch the model's forward method to handle token_type_ids
            original_forward = model.forward

            def patched_forward(
                input_ids=None, attention_mask=None, token_type_ids=None, **kwargs
            ):
                return original_forward(
                    input_ids=input_ids, attention_mask=attention_mask, **kwargs
                )

            model.forward = patched_forward

            # Create zero-shot classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU inference
            )

    def evaluate_segments(self, segments: list[str]) -> Iterable[tuple[str, float]]:
        results = super().evaluate_segments(segments)
        return list(zip(segments, [scores[0] for scores in results]))
