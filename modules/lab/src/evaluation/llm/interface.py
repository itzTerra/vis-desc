"""Conversion functions for LLM model metrics to core visualization data structures."""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluation.core import AggregatedModelData, DatasetMetrics


def to_int_0_5(values: np.ndarray) -> np.ndarray:
    """Round continuous scores to discrete 0-5 labels.

    Args:
        values: Continuous values (typically in 0-5 range)

    Returns:
        Array of rounded integer values between 0-5
    """
    return np.clip(np.round(values), 0, 5).astype(int)


def compute_metrics_from_llm_data(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics from LLM predictions and true labels.

    Args:
        predictions: Predicted ratings (continuous values, typically 0-5 range)
        labels: True labels (discrete values 0-5)

    Returns:
        Dictionary with mse, accuracy, precision, recall, f1, support, confusion_matrix
    """
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)

    pred_labels = to_int_0_5(predictions)
    true_labels = to_int_0_5(labels)

    mse = float(mean_squared_error(true_labels, pred_labels))
    accuracy = float(accuracy_score(true_labels, pred_labels))

    precision_arr = precision_score(
        true_labels, pred_labels, labels=list(range(6)), average=None, zero_division=0
    )
    recall_arr = recall_score(
        true_labels, pred_labels, labels=list(range(6)), average=None, zero_division=0
    )
    f1_arr = f1_score(
        true_labels, pred_labels, labels=list(range(6)), average=None, zero_division=0
    )

    precision = [float(p) for p in precision_arr]
    recall = [float(r) for r in recall_arr]
    f1 = [float(f) for f in f1_arr]
    support = [int(np.sum(true_labels == label_val)) for label_val in range(6)]

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(6)))

    return {
        "mse": mse,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
    }


def llm_metrics_to_core(
    predictions: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    test_predictions: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    model_name: str = "LLM Model",
) -> AggregatedModelData:
    """Convert LLM predictions and labels to core visualization data structure.

    Computes classification metrics for any provided split. Training metrics are
    only produced when both ``predictions`` and ``labels`` are supplied. Test
    metrics are only produced when both ``test_predictions`` and ``test_labels``
    are supplied.

    Args:
        predictions: LLM predicted ratings for the training split.
        labels: True labels for the training split.
        test_predictions: LLM predicted ratings on test data (optional).
        test_labels: True labels on test data (optional).
        model_name: Name of the model.

    Returns:
        AggregatedModelData with whichever splits were provided.
    """
    train_metrics = None
    if predictions is not None and labels is not None:
        predictions = np.asarray(predictions, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if predictions.size and labels.size:
            train_metrics_dict = compute_metrics_from_llm_data(predictions, labels)
            train_metrics = DatasetMetrics(
                mse=train_metrics_dict["mse"],
                accuracy=train_metrics_dict["accuracy"],
                precision=train_metrics_dict["precision"],
                recall=train_metrics_dict["recall"],
                f1=train_metrics_dict["f1"],
                support=train_metrics_dict["support"],
                confusion_matrix=train_metrics_dict["confusion_matrix"],
            )

    test_metrics = None
    if test_predictions is not None and test_labels is not None:
        test_predictions = np.asarray(test_predictions, dtype=float)
        test_labels = np.asarray(test_labels, dtype=float)
        if test_predictions.size and test_labels.size:
            test_metrics_dict = compute_metrics_from_llm_data(
                test_predictions, test_labels
            )
            test_metrics = DatasetMetrics(
                mse=test_metrics_dict["mse"],
                accuracy=test_metrics_dict["accuracy"],
                precision=test_metrics_dict["precision"],
                recall=test_metrics_dict["recall"],
                f1=test_metrics_dict["f1"],
                support=test_metrics_dict["support"],
                confusion_matrix=test_metrics_dict["confusion_matrix"],
            )

    return AggregatedModelData(
        model=model_name,
        train=train_metrics,
        val=None,
        test=test_metrics,
    )


def get_llm_model_groups(model_names: list[str]) -> tuple[dict[str, list[str]], bool]:
    """Get model grouping configuration for LLM models.

    Args:
        model_names: List of model names to group

    Returns:
        Tuple of (model_groups, show_large_variants)
        Each model gets its own group for LLM models.
    """
    model_groups = {name: [name] for name in model_names}
    return model_groups, False
