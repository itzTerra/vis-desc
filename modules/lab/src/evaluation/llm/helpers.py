import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass

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
                "prompt_token_count": metrics.get("prompt_token_count"),
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


def load_metric_files(files: list[Path]) -> list[dict]:
    """Load multiple metric JSON files.

    Args:
        files: List of paths to metric JSON files

    Returns:
        List of dictionaries from loaded files, with 'config_id' added
    """
    data = []
    for i, fp in enumerate(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["config_id"] = i
            data.append(obj)
        except Exception as e:
            print(f"Skip {fp}: {e}")
    return data


def prepare_llm_metrics_summary(items: list[dict]) -> pd.DataFrame:
    """Prepare a summary table of LLM metrics from loaded metric files.

    Args:
        items: List of metric dictionaries loaded from JSON files

    Returns:
        DataFrame with model metrics aggregated by model name
    """
    all_metrics = []

    for item in items:
        config_id = item.get("config_id", 0)
        prompt_id = item.get("prompt_id", "")

        for model_result in item.get("models", []):
            model_name = model_result.get("model_name", "unknown")
            dataset = model_result.get("dataset", "test")
            throughput = model_result.get("performance", {}).get("throughput", 0)
            output_errors = model_result.get("output_errors", 0)
            predictions = model_result.get("outputs", [])
            error_rate = calculate_error_rate(predictions)
            correlation = model_result.get("correlation", None)

            all_metrics.append(
                {
                    "config_id": config_id,
                    "model_name": model_name,
                    "prompt_id": prompt_id,
                    "dataset": dataset,
                    "throughput": throughput,
                    "error_rate": error_rate,
                    "correlation": correlation,
                    "output_errors": output_errors,
                }
            )

    return pd.DataFrame(all_metrics)


def to_llm_score_data(
    items: list[dict], df_train: pd.DataFrame, df_test: pd.DataFrame
) -> list:
    """Convert LLM metric items to score data format for evaluation.

    Properly handles train and test datasets by reading the 'dataset' field in each model result.
    Groups results by (config_id, model_name) and combines train/test splits.

    Args:
        items: List of metric items from load_metric_files
        df_train: Training dataset with 'label' column
        df_test: Test dataset with 'label' column

    Returns:
        List of ScoreData objects with train_scores and test_scores properly populated
    """

    @dataclass
    class ScoreData:
        model_name: str
        train_scores: np.ndarray
        y_train: np.ndarray
        test_scores: np.ndarray | None
        y_test: np.ndarray | None
        config_id: int

    y_train = df_train["label"].values.astype(float)
    y_test = df_test["label"].values.astype(float)

    model_data_map = {}

    for item in items:
        config_id = item.get("config_id", 0)

        for model_result in item.get("models", []):
            model_name = model_result.get("model_name", "")
            if not model_name:
                continue

            outputs = model_result.get("outputs", [])
            if not outputs:
                continue

            dataset_name = model_result.get("dataset", "test")
            key = (config_id, model_name)

            if key not in model_data_map:
                model_data_map[key] = {
                    "model_name": model_name,
                    "config_id": config_id,
                    "train_scores": None,
                    "test_scores": None,
                }

            outputs_array = np.asarray(outputs, dtype=float)

            if dataset_name == "train":
                model_data_map[key]["train_scores"] = outputs_array
            elif dataset_name == "test":
                model_data_map[key]["test_scores"] = outputs_array

    score_data_list = []
    for data in model_data_map.values():
        train_scores = (
            data["train_scores"]
            if data["train_scores"] is not None
            else np.array([], dtype=float)
        )
        test_scores = data["test_scores"]

        score_data_list.append(
            ScoreData(
                model_name=data["model_name"],
                train_scores=train_scores,
                y_train=y_train if train_scores.size > 0 else np.array([], dtype=float),
                test_scores=test_scores,
                y_test=y_test
                if test_scores is not None and test_scores.size > 0
                else None,
                config_id=data["config_id"],
            )
        )

    return score_data_list
