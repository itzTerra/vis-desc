"""Conversion functions for LLM model metrics to core visualization data structures."""

from __future__ import annotations

import numpy as np
from evaluation.core import AggregatedModelData, DatasetMetrics
from evaluation.llm.helpers import MODEL_NAME_MAP, compute_metrics_from_llm_data


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
    model_name = MODEL_NAME_MAP.get(
        model_name, model_name
    )  # Map to display name if available

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
