import json
from pathlib import Path
from typing import Optional
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
from adjustText import adjust_text
from utils import calculate_metrics

from utils import DATA_DIR
from evaluation.plot_style import METRIC_DECIMAL_PLACES
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
        - weighted_f1: Weighted F1 score (averaged across seeds)
        - output_errors: Count of output parsing errors (summed across seeds)
        - num_samples: Total number of samples evaluated
        - latency_mean: Mean latency in ms (averaged across seeds)
        - latency_std: Std dev of latency in ms (averaged across seeds)
    """
    metrics = load_metric_file(metrics_file)
    prompt_id = metrics_file.stem
    prompt = metrics.get("prompt", "")

    model_results_by_key = {}

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
        weighted_f1 = None
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
            weighted_f1 = computed_metrics["weighted_f1"]

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
            "weighted_f1": weighted_f1,
            "output_errors": output_errors,
            "num_samples": len(true_labels),
            "latency_mean": latency_mean,
            "latency_std": latency_std,
        }

        key = (model_name, dataset_name)
        if key not in model_results_by_key:
            model_results_by_key[key] = []
        model_results_by_key[key].append(seed_metrics)

    results = []

    for (model_name, _dataset_name), seed_metrics_list in model_results_by_key.items():
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
                    "weighted_f1": metric["weighted_f1"],
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
                weighted_f1s = [
                    m["weighted_f1"]
                    for m in valid_metrics
                    if m["weighted_f1"] is not None
                ]
                error_rates = [m["error_rate"] for m in seed_metrics_list]
                throughputs = [m["throughput"] for m in seed_metrics_list]
                latency_means = [m["latency_mean"] for m in seed_metrics_list]
                latency_stds = [m["latency_std"] for m in seed_metrics_list]

                avg_correlation = np.mean(correlations) if correlations else None
                avg_rmse = np.mean(rmses) if rmses else None
                avg_accuracy = np.mean(accuracies) if accuracies else None
                avg_weighted_f1 = np.mean(weighted_f1s) if weighted_f1s else None

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
                        "weighted_f1": avg_weighted_f1,
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


def load_metrics_from_dir(
    metrics_dir: Path,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Load all LLM metric JSON files from a directory.

    Combines the two common loading patterns used across evaluation notebooks:
    extracting per-model metric rows into a DataFrame and loading raw items with
    config metadata.

    Args:
        metrics_dir: Directory containing metric JSON files

    Returns:
        Tuple of:
        - df_metrics: DataFrame with one row per model result and computed metrics
        - items: Raw metric dicts with config_id and prompt_id fields added
        - config_names: List of prompt IDs ordered by config_id
    """
    metric_files = sorted(metrics_dir.glob("*.json"))

    items = load_metric_files(metric_files)
    config_names = [
        item.get("prompt_id", f"config_{i}") for i, item in enumerate(items)
    ]

    all_metrics = []
    for metric_file in metric_files:
        try:
            model_metrics = extract_model_metrics(metric_file)
            all_metrics.extend(model_metrics)
        except Exception as e:
            print(f"✗ Error processing {metric_file.name}: {e}")

    df_metrics = pd.DataFrame(all_metrics)
    print(
        f"Loaded {len(metric_files)} metric file(s) → "
        f"{len(df_metrics)} model result(s), {len(config_names)} configuration(s)"
    )

    return df_metrics, items, config_names


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


def plot_model_metrics_combined_scatter(
    df: pd.DataFrame,
    min_threshold: float = 0.1,
    y_padding: float = 0.05,
    figsize: tuple[float, float] = (8, 10),
    shorten_model_names: bool = False,
    show_labels: bool = False,
) -> None:
    """Plot correlation, RMSE, and weighted F1 in a single scatter plot.

    Creates a combined bubble chart where:
    - X-axis: Prompt complexity (Low/Medium/High) with small metric-specific offsets
    - Y-axis: Metric value (raw values for correlation, RMSE, weighted F1)
    - Bubble size: Model size inferred from name
    - Color: Prompt identifier
    - Shape: Metric type (circle=corr, triangle=rmse, square=weighted_f1) with legend

    Args:
        df: DataFrame with columns: model_name, prompt, prompt_token_count,
            correlation, rmse, weighted_f1
        min_threshold: Minimum value to include for correlation/weighted_f1
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
        "weighted_f1",
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
        | (
            pd.notna(df_nonnull["weighted_f1"])
            & (df_nonnull["weighted_f1"] > min_threshold)
        )
    ]

    if df_nonnull.empty:
        print("Skip combined scatter: no data to plot after filtering.")
        return

    unique_prompts = df_nonnull["prompt"].unique()
    colors = sns.color_palette("tab10", len(unique_prompts))
    prompt_color_map = {prompt: colors[i] for i, prompt in enumerate(unique_prompts)}

    # Metric-specific marker styles and x-offsets to reduce overlap
    metric_styles = {
        "rmse": {"marker": "^", "label": "RMSE ↓", "offset": 0},
        "correlation": {"marker": "o", "label": "Correlation ↑", "offset": 0},
        "weighted_f1": {"marker": "s", "label": "Weighted F1 ↑", "offset": 0},
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
            if metric_name in {"correlation", "weighted_f1"}:
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

            sns.scatterplot(
                x=x_positions,
                y=y_values,
                s=sizes,
                alpha=0.7,
                color=prompt_color_map[prompt],
                edgecolors="black",
                linewidth=1.2,
                marker=style["marker"],
                ax=ax,
                legend=False,
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
    ax.set_xlabel("Prompt Complexity")
    ax.set_ylabel("Metric Value")
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
    fig.savefig(
        DATA_DIR / "figures" / "llm-metrics-to-complexity.pdf",
        bbox_inches="tight",
    )
    plt.show()


def build_prompt_configuration_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a table of prompt configurations with their evaluation metrics.

    Iterates through COMBINATION_PLANS in canonical order to define row order.
    For each configuration, generates a Prompt instance, uses its ID to look up
    the best correlation, RMSE, and weighted F1 in df_metrics, and builds the table row.

    Args:
        df_metrics: DataFrame with metric data including 'prompt_id', 'correlation', 'rmse', and 'weighted_f1' columns

    Returns:
        DataFrame with columns: #, Task Description, Examples, CoT, Best corr., Best RMSE, Best F1
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
    all_f1_values = []

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
                best_f1 = ""

                if not config_metrics.empty:
                    corr_val = config_metrics["correlation"].max()
                    if pd.notna(corr_val):
                        best_corr = f"{corr_val:.{METRIC_DECIMAL_PLACES}f}"
                        all_corr_values.append((row_idx, corr_val))

                    rmse_val = config_metrics["rmse"].min()
                    if pd.notna(rmse_val):
                        best_rmse = f"{rmse_val:.{METRIC_DECIMAL_PLACES}f}"
                        all_rmse_values.append((row_idx, rmse_val))

                    f1_val = config_metrics["weighted_f1"].max()
                    if pd.notna(f1_val):
                        best_f1 = f"{f1_val:.{METRIC_DECIMAL_PLACES}f}"
                        all_f1_values.append((row_idx, f1_val))

                examples_val = "✓" if plan["include_examples"] else ""
                cot_val = cot_option if cot_option != "none" else ""

                table_data.append(
                    {
                        "#": row_idx,
                        "Task Descr.": guideline_display[task_descr_key],
                        "Examples": examples_val,
                        "CoT": cot_val,
                        "CORR": best_corr,
                        "RMSE": best_rmse,
                        "F1w": best_f1,
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
    best_f1_row = max(all_f1_values, key=lambda x: x[1])[0] if all_f1_values else None

    # Apply bold formatting to best values
    for idx, row in df.iterrows():
        row_num = row["#"]
        if row_num == best_corr_row and row["CORR"]:
            df.at[idx, "CORR"] = r"\textbf{" + row["CORR"] + "}"
        if row_num == best_rmse_row and row["RMSE"]:
            df.at[idx, "RMSE"] = r"\textbf{" + row["RMSE"] + "}"
        if row_num == best_f1_row and row["F1w"]:
            df.at[idx, "F1w"] = r"\textbf{" + row["F1w"] + "}"

    # Apply bold to column headers and add direction arrows for metric columns
    _ARROW_UP = r" $\uparrow$"
    _ARROW_DOWN = r" $\downarrow$"

    def _col_header(col: str) -> str:
        should_bold = col.isupper() or col in ("#", "Task Descr.", "Examples", "CoT")
        if not should_bold:
            return col
        bold = r"\textbf{" + col + "}"
        if col == "RMSE":
            return bold + _ARROW_DOWN
        if col in ("CORR", "F1w"):
            return bold + _ARROW_UP
        return bold

    df.columns = [_col_header(col) for col in df.columns]

    return df


def compute_optimization_comparison_table(
    df_current: pd.DataFrame,
    df_pre_optim: pd.DataFrame,
    selected_model_name: str,
) -> pd.DataFrame:
    """Compute optimization comparison table showing pre vs post optimization metrics.

    Creates a table with the selected model and average of other models, showing:
    - Current metrics (CORR, RMSE, F1)
    - Delta metrics (Δ CORR, Δ RMSE, Δ F1) as signed differences from pre-optimization

    Args:
        df_current: Pre-filtered DataFrame with current metrics
        df_pre_optim: Pre-filtered DataFrame with pre-optimization metrics
        selected_model_name: Name of the selected model

    Returns:
        DataFrame with columns: Model, CORR, Δ CORR, RMSE, Δ RMSE, F1, Δ F1
    """
    # Get metrics for selected model
    selected_model_pre = df_pre_optim[df_pre_optim["model_name"] == selected_model_name]
    selected_model_current = df_current[df_current["model_name"] == selected_model_name]

    # Calculate averages of other models
    other_models_pre = df_pre_optim[df_pre_optim["model_name"] != selected_model_name]
    other_models_current = df_current[df_current["model_name"] != selected_model_name]

    comparison_data = []

    if not selected_model_current.empty:
        sel_curr = selected_model_current.iloc[0]
        if not selected_model_pre.empty:
            sel_pre = selected_model_pre.iloc[0]
            row = {
                "Model": "Optimized",
                "CORR": sel_curr["correlation"],
                "Δ CORR": sel_curr["correlation"] - sel_pre["correlation"],
                "RMSE": sel_curr["rmse"],
                "Δ RMSE": sel_curr["rmse"] - sel_pre["rmse"],
                "F1w": sel_curr["weighted_f1"],
                "Δ F1w": sel_curr["weighted_f1"] - sel_pre["weighted_f1"],
            }
        else:
            row = {
                "Model": "Optimized",
                "CORR": sel_curr["correlation"],
                "Δ CORR": None,
                "RMSE": sel_curr["rmse"],
                "Δ RMSE": None,
                "F1w": sel_curr["weighted_f1"],
                "Δ F1w": None,
            }
        comparison_data.append(row)
    elif not selected_model_pre.empty:
        sel_pre = selected_model_pre.iloc[0]
        row = {
            "Model": "Optimized",
            "CORR": sel_pre["correlation"],
            "Δ CORR": None,
            "RMSE": sel_pre["rmse"],
            "Δ RMSE": None,
            "F1w": sel_pre["weighted_f1"],
            "Δ F1w": None,
        }
        comparison_data.append(row)

    if not other_models_current.empty:
        avg_corr_curr = other_models_current["correlation"].mean()
        avg_rmse_curr = other_models_current["rmse"].mean()
        avg_f1_curr = other_models_current["weighted_f1"].mean()
        if not other_models_pre.empty:
            avg_corr_pre = other_models_pre["correlation"].mean()
            avg_rmse_pre = other_models_pre["rmse"].mean()
            avg_f1_pre = other_models_pre["weighted_f1"].mean()
            row = {
                "Model": "Other (AVG)",
                "CORR": avg_corr_curr,
                "Δ CORR": avg_corr_curr - avg_corr_pre,
                "RMSE": avg_rmse_curr,
                "Δ RMSE": avg_rmse_curr - avg_rmse_pre,
                "F1w": avg_f1_curr,
                "Δ F1w": avg_f1_curr - avg_f1_pre,
            }
        else:
            row = {
                "Model": "Other (AVG)",
                "CORR": avg_corr_curr,
                "Δ CORR": None,
                "RMSE": avg_rmse_curr,
                "Δ RMSE": None,
                "F1w": avg_f1_curr,
                "Δ F1w": None,
            }
        comparison_data.append(row)
    elif not other_models_pre.empty:
        avg_corr_pre = other_models_pre["correlation"].mean()
        avg_rmse_pre = other_models_pre["rmse"].mean()
        avg_f1_pre = other_models_pre["weighted_f1"].mean()
        row = {
            "Model": "Other (AVG)",
            "CORR": avg_corr_pre,
            "Δ CORR": None,
            "RMSE": avg_rmse_pre,
            "Δ RMSE": None,
            "F1w": avg_f1_pre,
            "Δ F1w": None,
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
    normal_f1 = None
    if normal_model.test and normal_model.test.f1 and normal_model.test.support:
        total = sum(normal_model.test.support)
        normal_f1 = (
            sum(f * s for f, s in zip(normal_model.test.f1, normal_model.test.support))
            / total
            if total > 0
            else 0.0
        )

    relaxed_rmse = None
    relaxed_f1 = None

    if test_predictions is not None and test_labels is not None:
        test_preds_relaxed = np.array(test_predictions) // 2
        test_labels_relaxed = np.array(test_labels) // 2
        relaxed_metrics = calculate_metrics(test_preds_relaxed, test_labels_relaxed)
        relaxed_rmse = np.sqrt(relaxed_metrics["mse"])
        relaxed_support = relaxed_metrics["support"]
        relaxed_total = sum(relaxed_support)
        relaxed_f1 = (
            sum(f * s for f, s in zip(relaxed_metrics["f1"], relaxed_support))
            / relaxed_total
            if relaxed_total > 0
            else 0.0
        )

    comparison_data = {
        "Scale": ["Normal", "Relaxed"],
        "RMSE": [
            f"{normal_rmse:.{METRIC_DECIMAL_PLACES}f}"
            if normal_rmse is not None
            else "N/A",
            f"{relaxed_rmse:.{METRIC_DECIMAL_PLACES}f}"
            if relaxed_rmse is not None
            else "N/A",
        ],
        "F1w": [
            f"{normal_f1:.{METRIC_DECIMAL_PLACES}f}"
            if normal_f1 is not None
            else "N/A",
            f"{relaxed_f1:.{METRIC_DECIMAL_PLACES}f}"
            if relaxed_f1 is not None
            else "N/A",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)

    metrics_dict = {
        "normal_rmse": normal_rmse,
        "normal_f1": normal_f1,
        "relaxed_rmse": relaxed_rmse,
        "relaxed_f1": relaxed_f1,
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
    - Δ F1: green if positive (improvement), red if negative (worse)

    Args:
        df_comparison_table: DataFrame with comparison metrics and deltas

    Returns:
        LaTeX formatted table string
    """
    df_latex = df_comparison_table.copy()

    # Pre-compute best-value row index per metric column (before string formatting).
    # Higher is better: CORR, F1, Δ CORR, Δ F1; lower is better: RMSE, Δ RMSE.
    _lower_is_better = {"RMSE", "Δ RMSE"}
    best_idx: dict[str, int] = {}
    for col in df_latex.columns:
        if col in ("CORR", "RMSE", "F1w", "Δ CORR", "Δ RMSE", "Δ F1w"):
            series = pd.to_numeric(df_latex[col], errors="coerce")
            if series.notna().any():
                best_idx[col] = int(
                    series.idxmin() if col in _lower_is_better else series.idxmax()
                )

    for idx, row in df_latex.iterrows():
        # Format regular metric columns to 3 decimal places
        if "CORR" in df_latex.columns:
            corr = row["CORR"]
            if pd.notna(corr):
                df_latex.at[idx, "CORR"] = f"{corr:.{METRIC_DECIMAL_PLACES}f}"

        if "RMSE" in df_latex.columns:
            rmse = row["RMSE"]
            if pd.notna(rmse):
                df_latex.at[idx, "RMSE"] = f"{rmse:.{METRIC_DECIMAL_PLACES}f}"

        if "F1w" in df_latex.columns:
            f1 = row["F1w"]
            if pd.notna(f1):
                df_latex.at[idx, "F1w"] = f"{f1:.{METRIC_DECIMAL_PLACES}f}"

        # Format and color delta columns
        if "Δ CORR" in df_latex.columns:
            delta_corr = row["Δ CORR"]
            if pd.notna(delta_corr) and delta_corr != 0:
                if delta_corr > 0:
                    df_latex.at[idx, "Δ CORR"] = (
                        f"\\textcolor{{green!70!black}}{{{delta_corr:+.{METRIC_DECIMAL_PLACES}f}}}"
                    )
                else:
                    df_latex.at[idx, "Δ CORR"] = (
                        f"\\textcolor{{red}}{{{delta_corr:+.{METRIC_DECIMAL_PLACES}f}}}"
                    )

        if "Δ RMSE" in df_latex.columns:
            delta_rmse = row["Δ RMSE"]
            if pd.notna(delta_rmse) and delta_rmse != 0:
                if delta_rmse < 0:
                    df_latex.at[idx, "Δ RMSE"] = (
                        f"\\textcolor{{green!70!black}}{{{delta_rmse:+.{METRIC_DECIMAL_PLACES}f}}}"
                    )
                else:
                    df_latex.at[idx, "Δ RMSE"] = (
                        f"\\textcolor{{red}}{{{delta_rmse:+.{METRIC_DECIMAL_PLACES}f}}}"
                    )

        if "Δ F1w" in df_latex.columns:
            delta_f1 = row["Δ F1w"]
            if pd.notna(delta_f1) and delta_f1 != 0:
                if delta_f1 > 0:
                    df_latex.at[idx, "Δ F1w"] = (
                        f"\\textcolor{{green!70!black}}{{{delta_f1:+.{METRIC_DECIMAL_PLACES}f}}}"
                    )
                else:
                    df_latex.at[idx, "Δ F1w"] = (
                        f"\\textcolor{{red}}{{{delta_f1:+.{METRIC_DECIMAL_PLACES}f}}}"
                    )

    # Bold the best-value cell in each metric column.
    for col, bidx in best_idx.items():
        val = df_latex.at[bidx, col]
        if val != "" and pd.notna(val):
            df_latex.at[bidx, col] = r"\textbf{" + str(val) + "}"

    def _header(col: str) -> str:
        bold = f"\\textbf{{{col}}}"
        if "RMSE" in col:
            return bold + r" $\downarrow$"
        if any(k in col for k in ("CORR", "F1")):
            return bold + r" $\uparrow$"
        return bold

    _right_cols = {"CORR", "RMSE", "F1w"}
    column_format = "".join(
        "r" if col in _right_cols else "l" for col in df_latex.columns
    )

    header_bold_table = df_latex.rename(columns=_header)
    latex_table = header_bold_table.to_latex(
        index=False, escape=False, bold_rows=False, column_format=column_format
    )

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

        sns.lineplot(
            x=df_grouped["prompt_token_count"],
            y=y_values,
            ax=ax,
            marker=styles[metric]["marker"],
            linewidth=1.5,
            linestyle="--",
            label=styles[metric]["label"],
            markersize=8,
        )
        line = ax.lines[-1]
        color = line.get_color()
        line.set_alpha(0.6)
        line.set_markerfacecolor(mcolors.to_rgba(color, alpha=1.0))
        line.set_markeredgecolor(mcolors.to_rgba(color, alpha=1.0))

    ax.set_xlabel("Prompt Token Count")
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    return fig, ax, df_grouped
