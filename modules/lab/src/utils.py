from pathlib import Path
import json
import os
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from scipy.stats import pearsonr
import numpy as np
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
BOOK_DIR = ROOT_DIR / "data" / "books"
BOOK_META_DIR = BOOK_DIR / "meta"
SEGMENT_DIR = ROOT_DIR / "data" / "segments"
PROCESSED_BOOKS_PATH = ROOT_DIR / "data" / "processed_books.json"
TO_ANNOTATE_DIR = ROOT_DIR / "data" / "to-annotate"
DATA_DIR = ROOT_DIR / "data"
IMAG_DATA_DIR = DATA_DIR / "datasets" / "concreteness"


# https://stackoverflow.com/questions/1229068/with-python-can-i-keep-a-persistent-dictionary-and-modify-it
class PersistentDict(dict):
    def __init__(self, filename: str | Path, *args, **kwargs):
        self.filename = filename
        self._load()

    def _load(self):
        if os.path.isfile(self.filename) and os.path.getsize(self.filename) > 0:
            with open(self.filename, "r") as fh:
                super().update(json.load(fh))

    def _dump(self):
        with open(self.filename, "w") as fh:
            json.dump(self, fh)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        self._dump()

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return "%s(%s)" % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
        self._dump()


def get_device_name() -> str:
    """Get the device name (GPU or CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Get CPU name from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass

    return "CPU"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive metrics for predictions.

    Returns:
        dict with keys:
        - mse: Mean squared error
        - accuracy: Accuracy of rounded scores
        - corr: Pearson correlation coefficient
        - precision: Precision for each class (0-5)
        - recall: Recall for each class (0-5)
        - f1: F1-score for each class (0-5)
        - support: Support for each class (0-5)
        - confusion_matrix: Confusion matrix for rounded scores
        - predictions: Model predictions
    """
    if np.any(np.isnan(y_pred)):
        return {
            "mse": float("nan"),
            "accuracy": float("nan"),
            "corr": float("nan"),
            "precision": [float("nan")] * 6,
            "recall": [float("nan")] * 6,
            "f1": [float("nan")] * 6,
            "support": [0] * 6,
            "confusion_matrix": [[0] * 6 for _ in range(6)],
            "predictions": y_pred.tolist(),
        }
    # Round predictions and true values to nearest integer (0-5)
    y_true_rounded = np.clip(np.round(y_true), 0, 5).astype(int)
    y_pred_rounded = np.clip(np.round(y_pred), 0, 5).astype(int)

    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true_rounded, y_pred_rounded)
    corr, _ = pearsonr(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_rounded, y_pred_rounded, labels=[0, 1, 2, 3, 4, 5], zero_division=0
    )
    cm = confusion_matrix(y_true_rounded, y_pred_rounded, labels=[0, 1, 2, 3, 4, 5])

    return {
        "mse": float(mse),
        "accuracy": float(accuracy),
        "corr": float(corr),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
    }


def average_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {
            "mse": float("nan"),
            "accuracy": float("nan"),
            "precision": [float("nan")] * 6,
            "recall": [float("nan")] * 6,
            "f1": [float("nan")] * 6,
            "support": [0] * 6,
            "confusion_matrix": [[0] * 6 for _ in range(6)],
        }

    cms = np.array([m["confusion_matrix"] for m in metrics], dtype=float)
    cm_avg = np.nanmean(cms, axis=0)
    cm_avg = np.nan_to_num(cm_avg, nan=0.0)

    result = {
        "mse": float(np.nanmean([m["mse"] for m in metrics])),
        "accuracy": float(np.nanmean([m["accuracy"] for m in metrics])),
        "precision": np.nanmean([m["precision"] for m in metrics], axis=0).tolist(),
        "recall": np.nanmean([m["recall"] for m in metrics], axis=0).tolist(),
        "f1": np.nanmean([m["f1"] for m in metrics], axis=0).tolist(),
        "support": np.sum([m["support"] for m in metrics], axis=0).tolist(),
        "confusion_matrix": np.rint(cm_avg).astype(int).tolist(),
    }

    # Optionally include corr if present
    if all("corr" in m for m in metrics):
        result["corr"] = float(np.nanmean([m["corr"] for m in metrics]))

    return result
