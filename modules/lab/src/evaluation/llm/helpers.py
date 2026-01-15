import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils import DATA_DIR


def load_metric_file(filepath: Path) -> dict:
    """Load a metric JSON file.

    Args:
        filepath: Path to metric JSON file

    Returns:
        Dictionary containing metric data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def load_dataset(dataset_name: str, data_dir: Path = None) -> pd.DataFrame:
    """Load a dataset by name (train or test).

    Args:
        dataset_name: Either 'train' or 'test'
        data_dir: Directory containing datasets. Defaults to standard location.

    Returns:
        DataFrame with columns: text, label, features, dataset
    """
    if data_dir is None:
        data_dir = DATA_DIR / "datasets" / "small"

    dataset_path = data_dir / f"{dataset_name}.parquet"
    return pd.read_parquet(dataset_path)


def calculate_correlation(
    predictions: list[int | str | None], true_labels: list[int], method: str = "pearson"
) -> Optional[float]:
    """Calculate correlation between model predictions and true labels.

    Args:
        predictions: List of predicted ratings (may contain None for errors)
        true_labels: List of true labels (0-5 scale)
        method: Correlation method ('pearson' or 'spearman')

    Returns:
        Correlation coefficient, or None if calculation fails
    """
    if len(predictions) != len(true_labels):
        raise ValueError(
            f"Predictions and labels must have same length: "
            f"{len(predictions)} vs {len(true_labels)}"
        )

    # Convert predictions to integers, filter out None values
    valid_pairs = [
        (int(pred), true)
        for pred, true in zip(predictions, true_labels)
        if pred is not None
    ]

    if len(valid_pairs) == 0:
        return None

    preds_arr = np.array([p[0] for p in valid_pairs])
    labels_arr = np.array([p[1] for p in valid_pairs])

    try:
        if method == "pearson":
            corr, _ = pearsonr(preds_arr, labels_arr)
        elif method == "spearman":
            corr, _ = spearmanr(preds_arr, labels_arr)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        return float(corr)
    except (ValueError, RuntimeError):
        return None


def calculate_error_rate(
    predictions: list[int | str | None], total_samples: Optional[int] = None
) -> float:
    """Calculate error rate (percentage of None/failed predictions).

    Args:
        predictions: List of predictions (may contain None)
        total_samples: Total number of samples. If None, uses len(predictions).

    Returns:
        Error rate as percentage (0-100)
    """
    if total_samples is None:
        total_samples = len(predictions)

    if total_samples == 0:
        return 0.0

    error_count = sum(1 for p in predictions if p is None)
    return (error_count / total_samples) * 100


def extract_model_metrics(
    metrics_file: Path, data_dir: Optional[Path] = None
) -> list[dict]:
    """Extract metrics for all models in a metrics file.

    For each model result in the file:
    - Load the corresponding dataset
    - Calculate correlation with true labels
    - Calculate error rate
    - Extract throughput

    Args:
        metrics_file: Path to metrics JSON file
        data_dir: Directory containing datasets

    Returns:
        List of dicts with keys:
        - model_name: Name of the model
        - prompt: Prompt ID/description
        - dataset: 'train' or 'test'
        - throughput: Samples per second
        - error_rate: Percentage of failed predictions
        - correlation: Pearson correlation with true labels
        - output_errors: Count of output parsing errors
        - num_samples: Total number of samples evaluated
        - latency_mean: Mean latency in ms
        - latency_std: Std dev of latency in ms
    """
    metrics = load_metric_file(metrics_file)
    prompt_id = metrics_file.stem
    prompt = metrics.get("prompt", "")

    results = []

    for model_result in metrics.get("models", []):
        model_name = model_result.get("model_name", "unknown")
        dataset_name = model_result.get("dataset", "test")
        predictions = model_result.get("outputs", [])
        output_errors = model_result.get("output_errors", 0)
        performance = model_result.get("performance", {})

        # Load true labels
        dataset_df = load_dataset(dataset_name, data_dir)
        true_labels = dataset_df["label"].tolist()

        # Calculate metrics
        correlation = calculate_correlation(predictions, true_labels, method="pearson")
        error_rate = calculate_error_rate(predictions, len(true_labels))
        throughput = performance.get("throughput", 0)
        latency_mean = performance.get("latency_mean", 0)
        latency_std = performance.get("latency_std", 0)

        results.append(
            {
                "model_name": model_name,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "dataset": dataset_name,
                "throughput": throughput,
                "error_rate": error_rate,
                "correlation": correlation,
                "output_errors": output_errors,
                "num_samples": len(true_labels),
                "latency_mean": latency_mean,
                "latency_std": latency_std,
            }
        )

    return results
