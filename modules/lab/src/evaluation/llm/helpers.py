import json
from pathlib import Path
from typing import Optional
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
from adjustText import adjust_text

from utils import DATA_DIR
from evaluation.llm.interface import compute_metrics_from_llm_data


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
    - Calculate RMSE and accuracy
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
        - rmse: Root mean squared error
        - accuracy: Classification accuracy
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

        # Calculate correlation
        correlation = calculate_correlation(predictions, true_labels, method="pearson")
        error_rate = calculate_error_rate(predictions, len(true_labels))

        # Calculate RMSE and accuracy from valid predictions
        rmse = None
        accuracy = None
        valid_pairs = [
            (pred, true)
            for pred, true in zip(predictions, true_labels)
            if pred is not None
        ]
        if valid_pairs:
            preds_arr = np.asarray([p for p, _ in valid_pairs], dtype=float)
            labels_arr = np.asarray([lab for _, lab in valid_pairs], dtype=float)
            computed_metrics = compute_metrics_from_llm_data(preds_arr, labels_arr)
            rmse = np.sqrt(computed_metrics["mse"])
            accuracy = computed_metrics["accuracy"]

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
                "rmse": rmse,
                "accuracy": accuracy,
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
        List of dictionaries from loaded files, with 'config_id' and 'prompt_id' added
    """
    data = []
    for i, fp in enumerate(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["config_id"] = i
            obj["prompt_id"] = fp.stem.rsplit("_", 1)[0]
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


def extract_model_size(model_name: str) -> int:
    """Extract numeric suffix from model name (e.g., 'gpt-4-70b' -> 70).

    Args:
        model_name: Name of the model

    Returns:
        Extracted model size, or 50 if no match found
    """
    match = re.search(r"-(\d+)(?:b)?$", model_name)
    return int(match.group(1)) if match else 50


def model_size_to_bubble_size(size: int) -> float:
    """Convert model size to bubble size for scatter plots.

    Args:
        size: Model size in billions of parameters

    Returns:
        Bubble size for matplotlib scatter plot
    """
    return size / 3 + 30


def map_prompt_to_complexity(token_count: int) -> tuple[int, str]:
    """Map prompt token count to complexity category and label.

    Args:
        token_count: Number of tokens in the prompt

    Returns:
        Tuple of (category_index, label_string)
    """
    if token_count <= 200:
        return 0, "Low (120)"
    elif token_count <= 900:
        return 1, "Medium (634)"
    else:
        return 2, "High (1851)"


def plot_model_metrics_scatter(
    df: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    min_threshold: float = 0.1,
    y_padding: float = 0.05,
    figsize: tuple[float, float] = (6, 8),
    shorten_model_names: bool = False,
) -> None:
    """Create a scatter plot of model metrics vs prompt complexity.

    Creates a bubble chart where:
    - X-axis: Prompt complexity (Low/Medium/High)
    - Y-axis: Specified metric (e.g., correlation or accuracy)
    - Bubble size: Model size (from model name)
    - Color: Different prompts

    Args:
        df: DataFrame with columns: model_name, prompt, prompt_token_count, and metric_column
        metric_column: Name of the metric column to plot (e.g., 'correlation', 'accuracy')
        ylabel: Label for the y-axis
        min_threshold: Minimum metric value to include in plot
        y_padding: Padding for y-axis limits as fraction of range
        figsize: Figure size as (width, height)
        shorten_model_names: Whether to shorten model names for display
    """
    df_filtered = df[df[metric_column] > min_threshold]

    if df_filtered.empty:
        print(
            f"Skip {metric_column} scatter plot: no data above threshold {min_threshold}."
        )
        return

    unique_prompts = df_filtered["prompt"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_prompts)))
    prompt_color_map = {prompt: colors[i] for i, prompt in enumerate(unique_prompts)}

    fig, ax = plt.subplots(figsize=figsize)

    offset_map = {}

    for prompt in unique_prompts:
        prompt_data = df_filtered[df_filtered["prompt"] == prompt].sort_values(
            metric_column
        )

        x_positions = []
        y_values = []
        bubble_sizes = []

        for idx, (row_idx, row) in enumerate(prompt_data.iterrows()):
            x_pos, _ = map_prompt_to_complexity(row["prompt_token_count"])
            model_size = extract_model_size(row["model_name"])

            offset = 0.08 if idx % 2 == 0 else -0.08
            offset_map[row_idx] = offset
            x_positions.append(x_pos + offset)
            y_values.append(row[metric_column])
            bubble_sizes.append(model_size_to_bubble_size(model_size))

        ax.scatter(
            x_positions,
            y_values,
            s=bubble_sizes,
            alpha=0.7,
            color=prompt_color_map[prompt],
            edgecolors="black",
            linewidth=1.5,
        )

    texts = []
    for row_idx, row in df_filtered.iterrows():
        x_pos, _ = map_prompt_to_complexity(row["prompt_token_count"])
        model_name = row["model_name"]

        if shorten_model_names:
            model_name = model_name.split("/")[-1][:20]

        offset = offset_map[row_idx]

        text = ax.text(
            x_pos + offset,
            row[metric_column],
            model_name,
            fontsize=8,
            fontweight="bold",
            alpha=0.85,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"
            ),
        )
        texts.append(text)

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5, alpha=0.5),
        expand_points=(2.5, 2.5),
        expand_text=(2.0, 2.0),
        force_points=(1.0, 1.0),
        force_text=(1.0, 1.0),
        ax=ax,
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low (120)", "Medium (634)", "High (1851)"])
    ax.set_xlabel("Prompt Complexity", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(
        [
            df_filtered[metric_column].min() - y_padding,
            df_filtered[metric_column].max() + y_padding,
        ]
    )

    plt.tight_layout()
    plt.show()


def build_prompt_configuration_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a table of prompt configurations with their evaluation metrics.

    Iterates through COMBINATION_PLANS in canonical order to define row order.
    For each configuration, generates a Prompt instance, uses its ID to look up
    the best correlation, RMSE, and accuracy in df_metrics, and builds the table row.

    Args:
        df_metrics: DataFrame with metric data including 'prompt_id', 'correlation', 'rmse', and 'accuracy' columns

    Returns:
        DataFrame with columns: #, Task Description, Examples, CoT, Best corr., Best RMSE, Best Acc.
        Best values in each metric column are formatted in bold for LaTeX output.
    """
    from models.llm.prompts import (
        COMBINATION_PLANS,
        GUIDELINE_CONFIGS,
        Prompt,
        select_examples,
        select_suffix,
        PROMPT_PARTS,
    )

    guideline_display = {"small": "Small", "medium": "Medium", "full": "Large"}

    table_data = []
    row_idx = 1

    # Track all metric values to find best
    all_corr_values = []
    all_rmse_values = []
    all_acc_values = []

    for plan in COMBINATION_PLANS:
        for guideline_key in plan["task_descriptions"]:
            guideline_config = GUIDELINE_CONFIGS[guideline_key]
            for cot_option in plan["cot_options"]:
                suffix_key, suffix_text = select_suffix(cot_option)
                prompt = Prompt(
                    system=PROMPT_PARTS["system"],
                    guideline=guideline_config["text"],
                    examples=select_examples(plan["include_examples"], cot_option),
                    suffix_key=suffix_key,
                    suffix=suffix_text,
                )

                prompt_id = prompt.get_id()
                config_metrics = df_metrics[
                    df_metrics["prompt_id"].str.startswith(prompt_id)
                ]

                best_corr = ""
                best_rmse = ""
                best_acc = ""

                if not config_metrics.empty:
                    corr_val = config_metrics["correlation"].max()
                    if pd.notna(corr_val):
                        best_corr = f"{corr_val:.3f}"
                        all_corr_values.append((row_idx, corr_val))

                    rmse_val = config_metrics["rmse"].min()
                    if pd.notna(rmse_val):
                        best_rmse = f"{rmse_val:.3f}"
                        all_rmse_values.append((row_idx, rmse_val))

                    acc_val = config_metrics["accuracy"].max()
                    if pd.notna(acc_val):
                        best_acc = f"{acc_val:.3f}"
                        all_acc_values.append((row_idx, acc_val))

                examples_val = "yes" if plan["include_examples"] else ""
                cot_val = cot_option if cot_option != "none" else ""

                table_data.append(
                    {
                        "#": row_idx,
                        "Task Description": guideline_display[guideline_key],
                        "Examples": examples_val,
                        "CoT": cot_val,
                        "Best corr. across models": best_corr,
                        "Best RMSE across models": best_rmse,
                        "Best Acc. across models": best_acc,
                    }
                )
                row_idx += 1

    df = pd.DataFrame(table_data)

    # Find best values across all configurations
    best_corr_row = (
        max(all_corr_values, key=lambda x: x[1])[0] if all_corr_values else None
    )
    best_rmse_row = (
        min(all_rmse_values, key=lambda x: x[1])[0] if all_rmse_values else None
    )
    best_acc_row = (
        max(all_acc_values, key=lambda x: x[1])[0] if all_acc_values else None
    )

    # Apply bold formatting to best values
    for idx, row in df.iterrows():
        row_num = row["#"]
        if row_num == best_corr_row and row["Best corr. across models"]:
            df.at[idx, "Best corr. across models"] = (
                r"\textbf{" + row["Best corr. across models"] + "}"
            )
        if row_num == best_rmse_row and row["Best RMSE across models"]:
            df.at[idx, "Best RMSE across models"] = (
                r"\textbf{" + row["Best RMSE across models"] + "}"
            )
        if row_num == best_acc_row and row["Best Acc. across models"]:
            df.at[idx, "Best Acc. across models"] = (
                r"\textbf{" + row["Best Acc. across models"] + "}"
            )

    # Apply bold to column headers (for LaTeX output)
    df.columns = [
        r"\textbf{" + col + "}"
        if "Best" in col
        or col == "#"
        or col == "Task Description"
        or col == "Examples"
        or col == "CoT"
        else col
        for col in df.columns
    ]

    return df
