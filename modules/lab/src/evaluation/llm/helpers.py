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
from evaluation.core import LABEL_FONT_SIZE
from utils import calculate_metrics

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

    If a model appears multiple times (different seeds), compute metrics separately
    for each seed and then average them.

    Args:
        metrics_file: Path to metrics JSON file
        data_dir: Directory containing datasets

    Returns:
        List of dicts with keys:
        - model_name: Name of the model
        - prompt: Prompt ID/description
        - dataset: 'train' or 'test'
        - throughput: Samples per second (averaged across seeds)
        - error_rate: Percentage of failed predictions (averaged across seeds)
        - correlation: Pearson correlation with true labels (averaged across seeds)
        - rmse: Root mean squared error (averaged across seeds)
        - accuracy: Classification accuracy (averaged across seeds)
        - output_errors: Count of output parsing errors (summed across seeds)
        - num_samples: Total number of samples evaluated
        - latency_mean: Mean latency in ms (averaged across seeds)
        - latency_std: Std dev of latency in ms (averaged across seeds)
    """
    metrics = load_metric_file(metrics_file)
    prompt_id = metrics_file.stem
    prompt = metrics.get("prompt", "")

    model_results_by_name = {}

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

        seed_metrics = {
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

        if model_name not in model_results_by_name:
            model_results_by_name[model_name] = []
        model_results_by_name[model_name].append(seed_metrics)

    results = []

    for model_name, seed_metrics_list in model_results_by_name.items():
        if len(seed_metrics_list) == 1:
            metric = seed_metrics_list[0]
            results.append(
                {
                    "model_name": model_name,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "prompt_token_count": metrics.get("prompt_token_count"),
                    "dataset": metric["dataset"],
                    "throughput": metric["throughput"],
                    "error_rate": metric["error_rate"],
                    "correlation": metric["correlation"],
                    "rmse": metric["rmse"],
                    "accuracy": metric["accuracy"],
                    "output_errors": metric["output_errors"],
                    "num_samples": metric["num_samples"],
                    "latency_mean": metric["latency_mean"],
                    "latency_std": metric["latency_std"],
                }
            )
        else:
            valid_metrics = [
                m
                for m in seed_metrics_list
                if m["correlation"] is not None or m["rmse"] is not None
            ]
            if valid_metrics:
                correlations = [
                    m["correlation"]
                    for m in valid_metrics
                    if m["correlation"] is not None
                ]
                rmses = [m["rmse"] for m in valid_metrics if m["rmse"] is not None]
                accuracies = [
                    m["accuracy"] for m in valid_metrics if m["accuracy"] is not None
                ]
                error_rates = [m["error_rate"] for m in seed_metrics_list]
                throughputs = [m["throughput"] for m in seed_metrics_list]
                latency_means = [m["latency_mean"] for m in seed_metrics_list]
                latency_stds = [m["latency_std"] for m in seed_metrics_list]

                avg_correlation = np.mean(correlations) if correlations else None
                avg_rmse = np.mean(rmses) if rmses else None
                avg_accuracy = np.mean(accuracies) if accuracies else None

                results.append(
                    {
                        "model_name": model_name,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_token_count": metrics.get("prompt_token_count"),
                        "dataset": seed_metrics_list[0]["dataset"],
                        "throughput": np.mean(throughputs) if throughputs else 0,
                        "error_rate": np.mean(error_rates) if error_rates else 0,
                        "correlation": avg_correlation,
                        "rmse": avg_rmse,
                        "accuracy": avg_accuracy,
                        "output_errors": sum(
                            m["output_errors"] for m in seed_metrics_list
                        ),
                        "num_samples": seed_metrics_list[0]["num_samples"],
                        "latency_mean": np.mean(latency_means) if latency_means else 0,
                        "latency_std": np.mean(latency_stds) if latency_stds else 0,
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
    return size / 2 + 60


def map_prompt_to_complexity(
    token_count: int, x_positions: list[float]
) -> tuple[int, str]:
    """Map prompt token count to complexity category and label.

    Args:
        token_count: Number of tokens in the prompt

    Returns:
        Tuple of (category_index, label_string)
    """
    if token_count <= 200:
        return x_positions[0], "Low (120)"
    elif token_count <= 900:
        return x_positions[1], "Medium (634)"
    else:
        return x_positions[2], "High (1851)"


def three_state_jitter(index: int, magnitude: float) -> float:
    remainder = index % 3
    if remainder == 0:
        return -magnitude
    if remainder == 1:
        return 0.0
    return magnitude


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
    X_POSITIONS = [0, 1, 2]  # Low, Medium, High complexity

    for prompt in unique_prompts:
        prompt_data = df_filtered[df_filtered["prompt"] == prompt].sort_values(
            metric_column
        )

        x_positions = []
        y_values = []
        bubble_sizes = []

        for idx, (row_idx, row) in enumerate(prompt_data.iterrows()):
            x_pos, _ = map_prompt_to_complexity(row["prompt_token_count"], X_POSITIONS)
            model_size = extract_model_size(row["model_name"])

            offset = three_state_jitter(idx, 0.12)
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
        x_pos, _ = map_prompt_to_complexity(row["prompt_token_count"], X_POSITIONS)
        model_name = row["model_name"]

        if shorten_model_names:
            model_name = model_name.split("/")[-1][:20]

        offset = offset_map[row_idx]

        text = ax.text(
            x_pos + offset,
            row[metric_column],
            model_name,
            fontsize=9,
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

    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(["Low (120)", "Medium (634)", "High (1851)"])
    ax.set_xlabel("Prompt Complexity", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, fontweight="bold")
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


def plot_model_metrics_combined_scatter(
    df: pd.DataFrame,
    min_threshold: float = 0.1,
    y_padding: float = 0.05,
    figsize: tuple[float, float] = (8, 10),
    shorten_model_names: bool = False,
    show_labels: bool = False,
) -> None:
    """Plot correlation, RMSE, and accuracy in a single scatter plot.

    Creates a combined bubble chart where:
    - X-axis: Prompt complexity (Low/Medium/High) with small metric-specific offsets
    - Y-axis: Metric value (raw values for correlation, RMSE, accuracy)
    - Bubble size: Model size inferred from name
    - Color: Prompt identifier
    - Shape: Metric type (circle=corr, triangle=rmse, square=acc) with legend

    Args:
        df: DataFrame with columns: model_name, prompt, prompt_token_count,
            correlation, rmse, accuracy
        min_threshold: Minimum value to include for correlation/accuracy
        y_padding: Padding around min/max for y-axis limits
        figsize: Figure size
        shorten_model_names: Whether to truncate long model names in labels
        show_labels: Whether to render model name text labels (can get crowded)
    """
    required_cols = {
        "model_name",
        "prompt",
        "prompt_token_count",
        "correlation",
        "rmse",
        "accuracy",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Skip combined scatter: missing columns {sorted(missing)}")
        return

    # Filter rows that have at least one metric present and meet thresholds where applicable
    df_nonnull = df.copy()
    df_nonnull = df_nonnull[
        (
            pd.notna(df_nonnull["correlation"])
            & (df_nonnull["correlation"] > min_threshold)
        )
        | (pd.notna(df_nonnull["rmse"]))
        | (pd.notna(df_nonnull["accuracy"]) & (df_nonnull["accuracy"] > min_threshold))
    ]

    if df_nonnull.empty:
        print("Skip combined scatter: no data to plot after filtering.")
        return

    unique_prompts = df_nonnull["prompt"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_prompts)))
    prompt_color_map = {prompt: colors[i] for i, prompt in enumerate(unique_prompts)}

    # Metric-specific marker styles and x-offsets to reduce overlap
    metric_styles = {
        "rmse": {"marker": "^", "label": "RMSE ↓", "offset": 0},
        "correlation": {"marker": "o", "label": "Correlation ↑", "offset": 0},
        "accuracy": {"marker": "s", "label": "Accuracy ↑", "offset": 0},
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Track y-limits across all plotted points
    all_y_values = []

    X_POSITIONS = [0.4, 1, 1.6]  # Low, Medium, High complexity

    # Plot for each prompt and metric
    for prompt in unique_prompts:
        prompt_df = df_nonnull[df_nonnull["prompt"] == prompt]

        for metric_name, style in metric_styles.items():
            series_df = prompt_df[pd.notna(prompt_df[metric_name])]
            if metric_name in {"correlation", "accuracy"}:
                series_df = series_df[series_df[metric_name] > min_threshold]
            if series_df.empty:
                continue

            x_positions = []
            y_values = []
            sizes = []

            # Alternate micro-jitter to reduce exact overlap within same metric/prompt
            for idx, (_, row) in enumerate(
                series_df.sort_values(metric_name).iterrows()
            ):
                x_pos, _ = map_prompt_to_complexity(
                    row["prompt_token_count"], X_POSITIONS
                )
                jitter = three_state_jitter(idx, 0.12)
                x_positions.append(x_pos + style["offset"] + jitter)
                y_values.append(row[metric_name])
                sizes.append(
                    model_size_to_bubble_size(extract_model_size(row["model_name"]))
                )
                all_y_values.append(row[metric_name])

            ax.scatter(
                x_positions,
                y_values,
                s=sizes,
                alpha=0.7,
                color=prompt_color_map[prompt],
                edgecolors="black",
                linewidth=1.2,
                marker=style["marker"],
            )

            if show_labels:
                # Place labels slightly above points, then adjust to reduce overlaps
                if "_texts" not in locals():
                    _texts = []
                for x, y, (_, row) in zip(
                    x_positions, y_values, series_df.sort_values(metric_name).iterrows()
                ):
                    name = row["model_name"]
                    if shorten_model_names:
                        name = name.rsplit("-", 1)[0]
                    txt = ax.text(
                        x,
                        y,
                        name,
                        fontsize=9,
                        fontweight="bold",
                        alpha=0.9,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            alpha=0.7,
                            edgecolor="gray",
                        ),
                    )
                    _texts.append(txt)

    # Axes formatting
    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(["Low (120)", "Medium (634)", "High (1851)"])
    ax.set_xlim([0, 2])
    ax.set_xlabel("Prompt Complexity", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if all_y_values:
        y_min = min(all_y_values) - y_padding
        y_max = max(all_y_values) + y_padding
        ax.set_ylim([y_min, y_max])

    # If labels were drawn, nudge them to avoid overlaps and add arrows
    if show_labels and "_texts" in locals() and _texts:
        adjust_text(
            _texts,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, alpha=0.6),
            force_static=(7, 7),
            force_text=(5.2, 5.2),
            force_explode=(2.1, 2.1),
            ax=ax,
        )

    # Legends: one for metrics (shapes), one for prompts (colors)
    from matplotlib.lines import Line2D

    metric_legend = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="grey",
            label=style["label"],
            markerfacecolor="grey",
            markersize=8,
            linestyle="None",
        )
        for style in metric_styles.values()
    ]

    leg1 = ax.legend(
        metric_legend,
        [s["label"] for s in metric_styles.values()],
        title="Metric",
        loc="upper left",
    )
    ax.add_artist(leg1)
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
        Prompt,
        select_examples,
        select_suffix,
        PROMPT_PARTS,
    )

    guideline_display = {"small": "Short", "medium": "Expanded", "full": "Full"}

    table_data = []
    row_idx = 1

    # Track all metric values to find best
    all_corr_values = []
    all_rmse_values = []
    all_acc_values = []

    for plan in COMBINATION_PLANS:
        for task_descr_key in plan["task_descriptions"]:
            for cot_option in plan["cot_options"]:
                suffix_key, suffix_text = select_suffix(cot_option)
                prompt = Prompt(
                    system=PROMPT_PARTS["system"],
                    guideline=PROMPT_PARTS["task_descriptions"][task_descr_key],
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
                        "Task Descr.": guideline_display[task_descr_key],
                        "Examples": examples_val,
                        "CoT": cot_val,
                        "CORR": best_corr,
                        "RMSE": best_rmse,
                        "ACC": best_acc,
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
        if row_num == best_corr_row and row["CORR"]:
            df.at[idx, "CORR"] = r"\textbf{" + row["CORR"] + "}"
        if row_num == best_rmse_row and row["RMSE"]:
            df.at[idx, "RMSE"] = r"\textbf{" + row["RMSE"] + "}"
        if row_num == best_acc_row and row["ACC"]:
            df.at[idx, "ACC"] = r"\textbf{" + row["ACC"] + "}"

    # Apply bold to column headers (for LaTeX output)
    df.columns = [
        r"\textbf{" + col + "}"
        if col.isupper()
        or col == "#"
        or col == "Task Description"
        or col == "Examples"
        or col == "CoT"
        else col
        for col in df.columns
    ]

    return df


def compute_optimization_comparison_table(
    df_current: pd.DataFrame,
    df_pre_optim: pd.DataFrame,
    selected_config_idx: int,
    config_names: list[str],
    selected_model_name: str,
) -> pd.DataFrame:
    """Compute optimization comparison table showing pre vs post optimization metrics.

    Creates a table with the selected model and average of other models, showing:
    - Current metrics (CORR, RMSE, ACC)
    - Delta metrics (Δ CORR, Δ RMSE, Δ ACC) as signed differences from pre-optimization

    Args:
        df_current: DataFrame with current metrics
        df_pre_optim: DataFrame with pre-optimization metrics
        selected_config_idx: Index of selected configuration
        config_names: List of configuration names
        selected_model_name: Name of the selected model

    Returns:
        DataFrame with columns: Model, CORR, Δ CORR, RMSE, Δ RMSE, ACC, Δ ACC
    """
    config_name = config_names[selected_config_idx]

    # Filter metrics for selected configuration
    pre_optim_config_data = df_pre_optim[
        df_pre_optim["prompt_id"].str.startswith(config_name)
    ].reset_index(drop=True)

    current_config_data = df_current[
        df_current["prompt_id"].str.startswith(config_name)
    ].reset_index(drop=True)

    # Get metrics for selected model
    selected_model_pre = pre_optim_config_data[
        pre_optim_config_data["model_name"] == selected_model_name
    ]
    selected_model_current = current_config_data[
        current_config_data["model_name"] == selected_model_name
    ]

    # Calculate averages of other models
    other_models_pre = pre_optim_config_data[
        pre_optim_config_data["model_name"] != selected_model_name
    ]
    other_models_current = current_config_data[
        current_config_data["model_name"] != selected_model_name
    ]

    comparison_data = []

    if not selected_model_current.empty and not selected_model_pre.empty:
        sel_curr = selected_model_current.iloc[0]
        sel_pre = selected_model_pre.iloc[0]

        row = {
            "Model": "Optimized",
            "CORR": sel_curr["correlation"],
            "Δ CORR": sel_curr["correlation"] - sel_pre["correlation"],
            "RMSE": sel_curr["rmse"],
            "Δ RMSE": sel_curr["rmse"] - sel_pre["rmse"],
            "ACC": sel_curr["accuracy"],
            "Δ ACC": sel_curr["accuracy"] - sel_pre["accuracy"],
        }
        comparison_data.append(row)

    if not other_models_current.empty and not other_models_pre.empty:
        avg_corr_curr = other_models_current["correlation"].mean()
        avg_corr_pre = other_models_pre["correlation"].mean()
        avg_rmse_curr = other_models_current["rmse"].mean()
        avg_rmse_pre = other_models_pre["rmse"].mean()
        avg_acc_curr = other_models_current["accuracy"].mean()
        avg_acc_pre = other_models_pre["accuracy"].mean()

        row = {
            "Model": "Avg-of-Others",
            "CORR": avg_corr_curr,
            "Δ CORR": avg_corr_curr - avg_corr_pre,
            "RMSE": avg_rmse_curr,
            "Δ RMSE": avg_rmse_curr - avg_rmse_pre,
            "ACC": avg_acc_curr,
            "Δ ACC": avg_acc_curr - avg_acc_pre,
        }
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def compute_scale_comparison_table(
    score_data, to_int_0_5_func, llm_metrics_to_core_func
):
    """Compute comparison table of metrics for Normal vs Relaxed scale on test set.

    Args:
        score_data: ScoreData object with test_scores and y_test
        to_int_0_5_func: Function to convert scores to 0-5 scale
        llm_metrics_to_core_func: Function to convert metrics to core format

    Returns:
        Tuple of (df_comparison: DataFrame, metrics_dict: dict with normal and relaxed metrics)
    """
    if (
        score_data.test_scores is None
        or score_data.test_scores.size == 0
        or score_data.y_test is None
    ):
        return None, None

    test_predictions = to_int_0_5_func(score_data.test_scores)
    test_labels = score_data.y_test

    normal_model = llm_metrics_to_core_func(
        predictions=None,
        labels=None,
        test_predictions=test_predictions,
        test_labels=test_labels,
        model_name=score_data.model_name,
    )

    normal_rmse = np.sqrt(normal_model.test.mse) if normal_model.test else None
    normal_acc = normal_model.test.accuracy if normal_model.test else None

    relaxed_rmse = None
    relaxed_acc = None

    if test_predictions is not None and test_labels is not None:
        test_preds_relaxed = np.array(test_predictions) // 2
        test_labels_relaxed = np.array(test_labels) // 2
        relaxed_metrics = calculate_metrics(test_preds_relaxed, test_labels_relaxed)
        relaxed_rmse = np.sqrt(relaxed_metrics["mse"])
        relaxed_acc = relaxed_metrics["accuracy"]

    comparison_data = {
        "Scale": ["Normal", "Relaxed"],
        "RMSE": [
            f"{normal_rmse:.4f}" if normal_rmse is not None else "N/A",
            f"{relaxed_rmse:.4f}" if relaxed_rmse is not None else "N/A",
        ],
        "Accuracy": [
            f"{normal_acc:.4f}" if normal_acc is not None else "N/A",
            f"{relaxed_acc:.4f}" if relaxed_acc is not None else "N/A",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)

    metrics_dict = {
        "normal_rmse": normal_rmse,
        "normal_acc": normal_acc,
        "relaxed_rmse": relaxed_rmse,
        "relaxed_acc": relaxed_acc,
    }

    return df_comparison, metrics_dict


def generate_scale_comparison_output(
    filtered_score_data: list,
    config_idx: int,
    to_int_0_5_func,
    llm_metrics_to_core_func,
) -> tuple[pd.DataFrame | None, dict | None, str]:
    """Generate scale comparison table and LaTeX output for selected configuration.

    Filters score data by configuration index, computes scale comparison metrics,
    and formats output with bold headers.

    Args:
        filtered_score_data: List of ScoreData objects to filter
        config_idx: Configuration index to select
        to_int_0_5_func: Function to convert scores to 0-5 scale
        llm_metrics_to_core_func: Function to convert metrics to core format

    Returns:
        Tuple of (df_comparison: DataFrame with bold headers, metrics_dict: dict, model_name: str)
        Returns (None, None, "") if no data found
    """
    filtered_score_data_comparison = [
        sd for sd in filtered_score_data if sd.config_id == config_idx
    ]

    if not filtered_score_data_comparison:
        return None, None, ""

    sd_comparison = filtered_score_data_comparison[0]

    df_comparison, metrics = compute_scale_comparison_table(
        sd_comparison, to_int_0_5_func, llm_metrics_to_core_func
    )

    if df_comparison is None:
        return None, None, sd_comparison.model_name

    df_comparison_bold = df_comparison.rename(columns=lambda col: f"\\textbf{{{col}}}")

    return df_comparison_bold, metrics, sd_comparison.model_name


def format_optimization_comparison_latex(df_comparison_table: pd.DataFrame) -> str:
    """Format optimization comparison table as LaTeX with colored delta columns.

    All numeric columns are formatted to 3 decimal places.
    Colors delta columns based on whether the metric improved:
    - Δ CORR: green if positive (improvement), red if negative (worse)
    - Δ RMSE: green if negative (improvement), red if positive (worse)
    - Δ ACC: green if positive (improvement), red if negative (worse)

    Args:
        df_comparison_table: DataFrame with comparison metrics and deltas

    Returns:
        LaTeX formatted table string
    """
    df_latex = df_comparison_table.copy()

    for idx, row in df_latex.iterrows():
        # Format regular metric columns to 3 decimal places
        if "CORR" in df_latex.columns:
            corr = row["CORR"]
            if pd.notna(corr):
                df_latex.at[idx, "CORR"] = f"{corr:.3f}"

        if "RMSE" in df_latex.columns:
            rmse = row["RMSE"]
            if pd.notna(rmse):
                df_latex.at[idx, "RMSE"] = f"{rmse:.3f}"

        if "ACC" in df_latex.columns:
            acc = row["ACC"]
            if pd.notna(acc):
                df_latex.at[idx, "ACC"] = f"{acc:.3f}"

        # Format and color delta columns
        if "Δ CORR" in df_latex.columns:
            delta_corr = row["Δ CORR"]
            if pd.notna(delta_corr) and delta_corr != 0:
                if delta_corr > 0:
                    df_latex.at[idx, "Δ CORR"] = (
                        f"\\textcolor{{green!70!black}}{{{delta_corr:+.3f}}}"
                    )
                else:
                    df_latex.at[idx, "Δ CORR"] = (
                        f"\\textcolor{{red}}{{{delta_corr:+.3f}}}"
                    )

        if "Δ RMSE" in df_latex.columns:
            delta_rmse = row["Δ RMSE"]
            if pd.notna(delta_rmse) and delta_rmse != 0:
                if delta_rmse < 0:
                    df_latex.at[idx, "Δ RMSE"] = (
                        f"\\textcolor{{green!70!black}}{{{delta_rmse:+.3f}}}"
                    )
                else:
                    df_latex.at[idx, "Δ RMSE"] = (
                        f"\\textcolor{{red}}{{{delta_rmse:+.3f}}}"
                    )

        if "Δ ACC" in df_latex.columns:
            delta_acc = row["Δ ACC"]
            if pd.notna(delta_acc) and delta_acc != 0:
                if delta_acc > 0:
                    df_latex.at[idx, "Δ ACC"] = (
                        f"\\textcolor{{green!70!black}}{{{delta_acc:+.3f}}}"
                    )
                else:
                    df_latex.at[idx, "Δ ACC"] = (
                        f"\\textcolor{{red}}{{{delta_acc:+.3f}}}"
                    )

    header_bold_table = df_latex.rename(columns=lambda col: f"\\textbf{{{col}}}")
    latex_table = header_bold_table.to_latex(index=False, escape=False, bold_rows=False)

    return latex_table


def plot_metrics_vs_prompt_token_count(
    df_metrics: pd.DataFrame,
    items: list[dict],
    figsize: tuple[float, float] = (12, 7),
    metrics_to_plot: list[str] | None = None,
    normalize_throughput: bool = True,
    y_label: str = "Metric Value",
    show_legend: bool = True,
) -> tuple:
    """Plot metrics (correlation, accuracy, RMSE, throughput) vs prompt token count.

    Creates a line plot showing how model performance metrics vary with prompt complexity
    (measured by token count). All metrics are plotted on the same scale for comparison.

    Args:
        df_metrics: DataFrame with columns: prompt_id, correlation, rmse, accuracy, throughput
        items: List of metric items with prompt_id and prompt_token_count
        figsize: Figure size as (width, height)
        metrics_to_plot: Optional list of metrics to plot. Supported values:
            ["correlation", "rmse", "accuracy", "throughput"].
            Throughput is plotted normalized to its max when normalize_throughput=True.
        normalize_throughput: Whether to normalize throughput to [0, 1] scale
        y_label: Label for the y-axis
        show_legend: Whether to display the legend

    Returns:
        Tuple of (fig, ax, df_grouped) where df_grouped contains the aggregated metrics
    """
    import matplotlib.colors as mcolors

    prompt_to_token_count = {}
    for item in items:
        prompt_id = item.get("prompt_id", "unknown")
        prompt_token_count = item.get("prompt_token_count", None)
        if prompt_id and prompt_token_count is not None:
            prompt_to_token_count[prompt_id] = int(prompt_token_count)

    def find_token_count(df_id):
        for item_id, token_count in prompt_to_token_count.items():
            if df_id.startswith(item_id):
                return token_count
        return None

    df_metrics["prompt_token_count"] = df_metrics["prompt_id"].apply(find_token_count)

    df_grouped = (
        df_metrics.dropna(subset=["prompt_token_count"])
        .groupby("prompt_token_count")[
            ["correlation", "rmse", "accuracy", "throughput"]
        ]
        .mean()
        .reset_index()
    )

    df_grouped = df_grouped.sort_values("prompt_token_count")

    fig, ax = plt.subplots(figsize=figsize)

    # Determine which metrics to plot
    allowed_metrics = {"correlation", "rmse", "accuracy", "throughput"}
    if metrics_to_plot is None:
        metrics_to_plot = ["correlation", "rmse", "accuracy", "throughput"]
    metrics_to_plot = [m for m in metrics_to_plot if m in allowed_metrics]

    styles = {
        "rmse": {"marker": "^", "label": "RMSE ↓"},
        "correlation": {"marker": "o", "label": "Correlation ↑"},
        "accuracy": {"marker": "s", "label": "Accuracy ↑"},
        # Label for throughput is set dynamically below based on normalize_throughput
        "throughput": {"marker": "D", "label": "Throughput ↑"},
    }

    for metric in metrics_to_plot:
        if metric == "throughput":
            if normalize_throughput:
                max_val = df_grouped["throughput"].max()
                if max_val <= 0:
                    continue
                y_values = df_grouped["throughput"] / max_val
                styles["throughput"]["label"] = "Throughput (normalized) ↑"
            else:
                y_values = df_grouped["throughput"]
                styles["throughput"]["label"] = "Throughput ↑"
        else:
            y_values = df_grouped[metric]

        line = ax.plot(
            df_grouped["prompt_token_count"],
            y_values,
            marker=styles[metric]["marker"],
            linewidth=1.5,
            linestyle="--",
            label=styles[metric]["label"],
            markersize=8,
        )[0]
        color = line.get_color()
        line.set_alpha(0.6)
        line.set_markerfacecolor(mcolors.to_rgba(color, alpha=1.0))
        line.set_markeredgecolor(mcolors.to_rgba(color, alpha=1.0))

    ax.set_xlabel("Prompt Token Count", fontsize=LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, fontweight="bold")
    if show_legend:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    return fig, ax, df_grouped
