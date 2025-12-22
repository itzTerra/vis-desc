from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from scipy.stats import pearsonr
from dataclasses import dataclass
from utils import DATA_DIR

METRICS_DIR = DATA_DIR / "metrics" / "nli"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class NLIConfig:
    candidate_labels: list[str]
    hypothesis_template: str

    def __str__(self) -> str:
        return f"labels={self.candidate_labels} | template={self.hypothesis_template}"


def probs_to_scores(probs: list[float]) -> float:
    """Interpolate a probability vector to a regression score in [0, 1].

    Assumes candidate labels correspond to uniformly spaced points across the
    regression range.

    Returns:
        A float in [0, 1] representing the expected position.
    """
    k = len(probs)
    if k == 0:
        return 0

    total = sum(probs)
    if total <= 0:
        return 0

    probs = [s / total for s in probs]

    # Single-label edge case: treat the lone label as the high end.
    if k == 1:
        return probs[0]

    # Uniform positions from 0 to 1 for label indices 0..k-1
    step = 1.0 / (k - 1)
    expected_pos = sum(p * (i * step) for i, p in enumerate(probs))

    # Guard against minor numerical drift
    if expected_pos < 0:
        return 0
    if expected_pos > 1:
        return 1
    return expected_pos


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


def get_corrs(probs_list: list[list[float]], labels: list[float]) -> float:
    scores = [probs_to_scores(probs) for probs in probs_list]
    # Calculate correlation between scores and true labels
    correlation, _ = pearsonr(scores, labels)
    return correlation
