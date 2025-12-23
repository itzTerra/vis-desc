from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


class NLIZeroshotClassifier:
    def __init__(
        self, candidate_labels=None, hypothesis_template=None, create_model=True
    ):
        self.candidate_labels = candidate_labels
        self.hypothesis_template = hypothesis_template
        self.classifier = None
        if create_model:
            self.create_model()

    # Resource handler usable with the 'with' statement
    class Opts:
        def __init__(
            self,
            parent_classifier: "NLIZeroshotClassifier",
            candidate_labels=None,
            hypothesis_template=None,
        ):
            self.parent_classifier = parent_classifier
            self.candidate_labels = candidate_labels
            self.hypothesis_template = hypothesis_template

        def __enter__(self):
            if self.candidate_labels is not None:
                self.old_candidate_labels = self.parent_classifier.candidate_labels
                self.parent_classifier.candidate_labels = self.candidate_labels
            if self.hypothesis_template is not None:
                self.old_hypothesis_template = (
                    self.parent_classifier.hypothesis_template
                )
                self.parent_classifier.hypothesis_template = self.hypothesis_template
            return self.parent_classifier

        def __exit__(self, exc_type, exc_value, traceback):
            if self.candidate_labels is not None:
                self.parent_classifier.candidate_labels = self.old_candidate_labels
            if self.hypothesis_template is not None:
                self.parent_classifier.hypothesis_template = (
                    self.old_hypothesis_template
                )

    def set_options(self, candidate_labels=None, hypothesis_template=None):
        return self.Opts(self, candidate_labels, hypothesis_template)

    def create_model(self):
        raise NotImplementedError

    def evaluate_segments(self, segments: list[str]) -> list[tuple[str, list[float]]]:
        self.create_model()

        results = [
            [
                result["scores"][result["labels"].index(label)]
                for label in self.candidate_labels
            ]
            for result in self.classifier(
                segments,
                self.candidate_labels,
                hypothesis_template=self.hypothesis_template,
            )
        ]
        return list(zip(segments, results))


class ORTNLIZeroshotClassifier(NLIZeroshotClassifier):
    def get_model_name(self) -> str:
        raise NotImplementedError

    def get_model_file_name(self) -> str | None:
        return None

    def create_model(self):
        if self.classifier is not None:
            return

        # Load the quantized model
        model = ORTModelForSequenceClassification.from_pretrained(
            self.get_model_name(),
            file_name=self.get_model_file_name(),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.get_model_name(),
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
        )


class NLIRoberta(ORTNLIZeroshotClassifier):
    """https://huggingface.co/richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"""

    def get_model_name(self):
        return "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX"

    def get_model_file_name(self):
        return "model_quantized.onnx"


class NLIDebertaLarge(ORTNLIZeroshotClassifier):
    """https://huggingface.co/richardr1126/deberta-v3-large-zeroshot-v2.0-ONNX"""

    def get_model_name(self):
        return "richardr1126/deberta-v3-large-zeroshot-v2.0-ONNX"

    def get_model_file_name(self):
        return "model_quantized.onnx"


class NLIModernBERTLarge(ORTNLIZeroshotClassifier):
    """https://huggingface.co/onnx-community/ModernBERT-large-zeroshot-v2.0-ONNX"""

    def get_model_name(self):
        return "onnx-community/ModernBERT-large-zeroshot-v2.0-ONNX"

    def get_model_file_name(self):
        return "model_int8.onnx"
