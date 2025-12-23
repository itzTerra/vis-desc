"""Conversion functions for NLI model scores to core visualization data structures with calibration."""

from __future__ import annotations

from typing import Any, Callable
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
from evaluation.nli.helpers import (
    IsotonicCalibration,
    to_int_0_5,
    Calibrator,
)


def compute_metrics_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics from continuous scores and discrete labels.

    Args:
        scores: Predicted scores (continuous values, typically 0-5 range)
        labels: True labels (discrete values 0-5)

    Returns:
        Dictionary with mse, accuracy, precision, recall, f1, support, confusion_matrix
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)

    # Round scores to discrete labels
    pred_labels = to_int_0_5(scores)
    true_labels = to_int_0_5(labels)

    # Compute metrics
    mse = float(mean_squared_error(labels, scores))

    accuracy = float(accuracy_score(true_labels, pred_labels))

    # Compute per-label metrics for all labels at once
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


def nli_scores_to_core(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    test_scores: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    model_name: str = "NLI Model",
    calibration_method: Callable[[], Calibrator] | None = None,
    n_cv_splits: int = 5,
) -> AggregatedModelData:
    """Convert NLI raw scores to core visualization data structures with calibration.

    Applies calibration to map raw model scores to the 0-5 scale of true labels.
    Uses cross-validation to get per-fold metrics and averaged CV metrics.

    Args:
        train_scores: Raw model scores on training data
        y_train: True labels (0-5) on training data
        test_scores: Raw model scores on test data (optional)
        y_test: True labels (0-5) on test data (optional)
        model_name: Name of the model
        calibration_method: Calibration method factory (default: IsotonicCalibration)
        n_cv_splits: Number of cross-validation folds

    Returns:
        AggregatedModelData with train, val (CV), and test metrics
    """
    if calibration_method is None:
        calibration_method = IsotonicCalibration

    train_scores = np.asarray(train_scores, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    # Calibrate and compute CV metrics
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
    cv_fold_metrics = []
    cv_calibrated_scores = np.zeros_like(train_scores)

    for train_idx, val_idx in kfold.split(train_scores):
        X_train_fold = train_scores[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = train_scores[val_idx]
        y_val_fold = y_train[val_idx]

        cal_method = calibration_method()
        calibrator = cal_method.fit(X_train_fold, y_train_fold)
        val_pred = calibrator.predict(X_val_fold)

        cv_calibrated_scores[val_idx] = val_pred
        fold_metrics = compute_metrics_from_scores(val_pred, y_val_fold)
        cv_fold_metrics.append(fold_metrics)

    # Average CV metrics across folds
    val_metrics = DatasetMetrics(
        mse=float(np.mean([m["mse"] for m in cv_fold_metrics])),
        accuracy=float(np.mean([m["accuracy"] for m in cv_fold_metrics])),
        precision=[
            float(np.mean([m["precision"][i] for m in cv_fold_metrics]))
            for i in range(6)
        ],
        recall=[
            float(np.mean([m["recall"][i] for m in cv_fold_metrics])) for i in range(6)
        ],
        f1=[float(np.mean([m["f1"][i] for m in cv_fold_metrics])) for i in range(6)],
        support=[
            int(np.mean([m["support"][i] for m in cv_fold_metrics])) for i in range(6)
        ],
        confusion_matrix=np.mean(
            [m["confusion_matrix"] for m in cv_fold_metrics], axis=0
        ),
        folds=[
            DatasetMetrics(
                mse=m["mse"],
                accuracy=m["accuracy"],
                precision=m["precision"],
                recall=m["recall"],
                f1=m["f1"],
                support=m["support"],
                confusion_matrix=m["confusion_matrix"],
            )
            for m in cv_fold_metrics
        ],
    )

    # Train metrics: calibrate on full train set and evaluate
    cal_method = calibration_method()
    calibrator = cal_method.fit(train_scores, y_train)
    train_pred = calibrator.predict(train_scores)
    train_metrics_dict = compute_metrics_from_scores(train_pred, y_train)
    train_metrics = DatasetMetrics(
        mse=train_metrics_dict["mse"],
        accuracy=train_metrics_dict["accuracy"],
        precision=train_metrics_dict["precision"],
        recall=train_metrics_dict["recall"],
        f1=train_metrics_dict["f1"],
        support=train_metrics_dict["support"],
        confusion_matrix=train_metrics_dict["confusion_matrix"],
    )

    # Test metrics (if available): use calibrator trained on full training set
    test_metrics = None
    if test_scores is not None and y_test is not None:
        test_scores = np.asarray(test_scores, dtype=float)
        y_test = np.asarray(y_test, dtype=float)
        test_pred = calibrator.predict(test_scores)
        test_metrics_dict = compute_metrics_from_scores(test_pred, y_test)
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
        val=val_metrics,
        test=test_metrics,
    )


def get_nli_model_groups(model_names: list[str]) -> tuple[dict[str, list[str]], bool]:
    """Get model grouping configuration for NLI models.

    Args:
        model_names: List of model names to group

    Returns:
        Tuple of (model_groups, show_large_variants)
        Each model gets its own group for NLI models, without large variant rows.
    """
    model_groups = {name: [name] for name in model_names}
    return model_groups, False
