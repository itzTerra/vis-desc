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


def get_corrs(probs_list: list[list[float]], labels: list[float]) -> float:
    scores = [probs_to_scores(probs) for probs in probs_list]
    # Calculate correlation between scores and true labels
    correlation, _ = pearsonr(scores, labels)
    return correlation
