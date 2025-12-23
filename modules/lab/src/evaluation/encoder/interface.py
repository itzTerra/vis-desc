"""Conversion functions for encoder metrics to core visualization data structures."""

from __future__ import annotations

from typing import Any
import numpy as np

from evaluation.core import AggregatedModelData, DatasetMetrics


def _convert_dataset_metrics(
    metrics_dict: dict[str, Any] | None,
) -> DatasetMetrics | None:
    """Convert encoder metrics dict to DatasetMetrics."""
    if metrics_dict is None:
        return None

    mse = metrics_dict.get("mse", np.nan)
    accuracy = metrics_dict.get("accuracy", np.nan)
    precision = metrics_dict.get("precision", [])
    recall = metrics_dict.get("recall", [])
    f1 = metrics_dict.get("f1", [])
    support = metrics_dict.get("support", [])
    confusion_matrix = np.array(metrics_dict.get("confusion_matrix", []))

    folds = None
    if "folds" in metrics_dict:
        folds = [_convert_dataset_metrics(fold) for fold in metrics_dict["folds"]]

    return DatasetMetrics(
        mse=float(mse) if not np.isnan(mse) else np.nan,
        accuracy=float(accuracy) if not np.isnan(accuracy) else np.nan,
        precision=list(precision),
        recall=list(recall),
        f1=list(f1),
        support=list(support),
        confusion_matrix=confusion_matrix,
        folds=folds,
    )


def encoder_metrics_to_core(
    encoder_models: list[dict[str, Any]],
) -> list[AggregatedModelData]:
    """Convert encoder aggregated metrics to core visualization data structures.

    Args:
        encoder_models: List of aggregated model dicts from aggregate_metrics_by_model()

    Returns:
        List of AggregatedModelData objects ready for visualization
    """
    result = []
    for model_dict in encoder_models:
        model_name = model_dict.get("model", "unknown")
        train = _convert_dataset_metrics(model_dict.get("train"))
        val = _convert_dataset_metrics(model_dict.get("val"))
        test = _convert_dataset_metrics(model_dict.get("test"))

        result.append(
            AggregatedModelData(model=model_name, train=train, val=val, test=test)
        )

    return result


def get_encoder_model_groups() -> tuple[dict[str, list[str]], bool]:
    """Get model grouping configuration for encoder models.

    Returns:
        Tuple of (model_groups, show_large_variants)
    """
    model_groups = {
        "Random": ["random"],
        "Ridge": ["ridge_minilm", "ridge_mbert", "ridge_minilm_lg", "ridge_mbert_lg"],
        "RandomForest": ["rf_minilm", "rf_mbert", "rf_minilm_lg", "rf_mbert_lg"],
        "SVM": ["svm_minilm", "svm_mbert"],
        "CatBoost": [
            "catboost_minilm",
            "catboost_mbert",
            "catboost_minilm_lg",
            "catboost_mbert_lg",
        ],
        "FinetunedMBERT": ["finetuned-mbert"],
    }
    return model_groups, True
