from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import DATA_DIR
from evaluation.plot_style import (  # noqa: F401 – re-exported for callers
    CMAP_QUALITATIVE_PRIMARY,
    CMAP_QUALITATIVE_SECONDARY,
    CMAP_SEQUENTIAL_PRIMARY,
    CMAP_SEQUENTIAL_SECONDARY,
    GRID_ALPHA,
    GRID_LINESTYLE,
    LABEL_FONT_SIZE,
    METRIC_DECIMAL_PLACES,
    TITLE_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE,
    ANNOT_FONT_SIZE,
    apply_plot_style,
)


MODEL_DISPLAY_NAMES: dict[str, str] = {
    "random": "Random",
    "ridge_minilm": "Ridge",
    "rf_minilm": "RandomForest",
    "svm_minilm": "SVM",
    "catboost_minilm": "CatBoost",
    "finetuned-mbert": "Finetuned ModernBERT",
}


@dataclass
class DatasetMetrics:
    mse: float
    accuracy: float
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    confusion_matrix: np.ndarray
    folds: Optional[list["DatasetMetrics"]] = None


@dataclass
class AggregatedModelData:
    model: str
    train: Optional[DatasetMetrics] = None
    val: Optional[DatasetMetrics] = None
    test: Optional[DatasetMetrics] = None


def _pad_cm_to_six(cm: np.ndarray) -> np.ndarray:
    if cm.shape[0] < 6:
        padding = np.zeros((6 - cm.shape[0], cm.shape[1]))
        cm = np.vstack([cm, padding])
    if cm.shape[1] < 6:
        padding = np.zeros((cm.shape[0], 6 - cm.shape[1]))
        cm = np.hstack([cm, padding])
    return cm[:6, :6]


def _collapse_cm_relaxed(cm6: np.ndarray) -> np.ndarray:
    groups = [(0, 1), (2, 3), (4, 5)]
    out = np.zeros((3, 3), dtype=cm6.dtype)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            out[i, j] = cm6[np.ix_(gi, gj)].sum()
    return out


def _neighbor_correct_cm(cm: np.ndarray) -> np.ndarray:
    """Return a copy of cm where ±1 off-diagonal predictions count as correct.

    For each row i and each column j with |i - j| == 1, the count cm[i, j]
    is moved to the diagonal cm[i, i].  Predictions off by ≥ 2 are untouched.
    The total sample count is preserved.
    """
    out = cm.copy()
    n = min(cm.shape[0], cm.shape[1])
    for i in range(n):
        for delta in (-1, 1):
            j = i + delta
            if 0 <= j < n:
                out[i, i] += cm[i, j]
                out[i, j] = 0
    return out


def _metrics_from_cm(
    cm: np.ndarray,
) -> tuple[list[float], list[float], list[float], list[int], float, float]:
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    true_pos = cm.sum(axis=1).astype(float)
    precision = [
        (tp[i] / pred_pos[i]) if pred_pos[i] > 0 else 0.0 for i in range(cm.shape[0])
    ]
    recall = [
        (tp[i] / true_pos[i]) if true_pos[i] > 0 else 0.0 for i in range(cm.shape[0])
    ]
    f1 = [
        (2 * precision[i] * recall[i] / (precision[i] + recall[i]))
        if (precision[i] + recall[i]) > 0
        else 0.0
        for i in range(cm.shape[0])
    ]
    support = [int(v) for v in true_pos]
    total = cm.sum()
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    idx = np.arange(cm.shape[0])
    diff2 = (idx[:, None] - idx[None, :]) ** 2
    mse = float((cm * diff2).sum() / total) if total > 0 else 0.0
    return precision, recall, f1, support, accuracy, mse


def collapse_dataset_metrics_relaxed(dm: DatasetMetrics) -> DatasetMetrics:
    cm6 = _pad_cm_to_six(dm.confusion_matrix)
    cm3 = _collapse_cm_relaxed(cm6)
    precision, recall, f1, support, accuracy, mse = _metrics_from_cm(cm3)
    folds = None
    if dm.folds:
        folds = [collapse_dataset_metrics_relaxed(f) for f in dm.folds]
    return DatasetMetrics(
        mse=mse,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion_matrix=cm3,
        folds=folds,
    )


def collapse_dataset_metrics_neighbor(dm: DatasetMetrics) -> DatasetMetrics:
    """Compute DatasetMetrics where predictions within ±1 count as correct.

    The confusion matrix shape stays 6×6; only the off-by-one entries are
    folded into the diagonal before recomputing precision/recall/F1/accuracy.
    MSE is taken from the original DatasetMetrics (distance-weighted, already
    meaningful on its own).
    """
    cm6 = _pad_cm_to_six(dm.confusion_matrix)
    cm_corrected = _neighbor_correct_cm(cm6)
    precision, recall, f1, support, accuracy, _ = _metrics_from_cm(cm_corrected)
    folds = None
    if dm.folds:
        folds = [collapse_dataset_metrics_neighbor(f) for f in dm.folds]
    return DatasetMetrics(
        mse=dm.mse,  # preserve original distance-based MSE
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion_matrix=cm6,  # intentional: keep raw matrix for display; metrics derived from cm_corrected
        folds=folds,
    )


def extract_per_label_metrics(
    metrics_dict: AggregatedModelData, dataset: str = "test"
) -> pd.DataFrame | None:
    """Extract precision, recall, and F1 scores for labels 0-5 from aggregated model data."""
    data = getattr(metrics_dict, dataset, None)
    if data is None:
        return None

    precision = data.precision + [0.0] * (6 - len(data.precision))
    recall = data.recall + [0.0] * (6 - len(data.recall))
    f1 = data.f1 + [0.0] * (6 - len(data.f1))
    support = data.support + [0] * (6 - len(data.support))

    return pd.DataFrame(
        {
            "Label": range(6),
            "Precision": precision[:6],
            "Recall": recall[:6],
            "F1": f1[:6],
            "Support": support[:6],
        }
    )


def get_confusion_matrix(
    metrics_dict: AggregatedModelData, dataset: str = "test", class_mode: str = "full"
) -> np.ndarray | None:
    """Extract confusion matrix from aggregated model data.

    class_mode:
      - "full": returns a 6x6 matrix, padding if needed
      - "relaxed": merges classes (0,1), (2,3), (4,5) into a 3x3 matrix
      - "neighbor": returns the raw 6x6 matrix (same as "full"); metrics are
        relaxed separately via collapse_dataset_metrics_neighbor
    """
    data = getattr(metrics_dict, dataset, None)
    if data is None:
        return None

    cm = data.confusion_matrix
    if cm.shape[0] < 6:
        padding = np.zeros((6 - cm.shape[0], cm.shape[1]))
        cm = np.vstack([cm, padding])
    if cm.shape[1] < 6:
        padding = np.zeros((cm.shape[0], 6 - cm.shape[1]))
        cm = np.hstack([cm, padding])

    cm = cm[:6, :6]

    if class_mode == "relaxed":
        groups = [(0, 1), (2, 3), (4, 5)]
        out = np.zeros((3, 3), dtype=cm.dtype)
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                out[i, j] = cm[np.ix_(gi, gj)].sum()
        return out

    return cm


def plot_confusion_matrix(
    metrics_dict: AggregatedModelData,
    dataset: str = "test",
    show_proportional: bool = True,
    show_title: bool = True,
    class_mode: str = "full",
) -> None:
    """Plot confusion matrix showing hits (diagonal) and misses (off-diagonal)."""
    cm = get_confusion_matrix(metrics_dict, dataset, class_mode=class_mode)
    if cm is None:
        print(f"Warning: {dataset} not available for this model")
        return

    if show_proportional:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 3))
        ax2 = None

    cmap = (
        CMAP_SEQUENTIAL_SECONDARY
        if class_mode in ("relaxed", "neighbor")
        else CMAP_SEQUENTIAL_PRIMARY
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        ax=ax1,
        linewidths=0.5,
        linecolor="gray",
        cbar=False,
    )
    if show_title:
        ax1.set_title(
            f"{metrics_dict.model} - Confusion Matrix ({dataset.replace('_', ' ').title()})\n(Raw Counts)",
            fontweight="bold",
        )
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    if class_mode == "relaxed":
        labels = ["0/1", "2/3", "4/5"]
    elif class_mode == "neighbor":
        labels = [f"{i}" for i in range(6)]
    else:
        labels = [f"{i}" for i in range(6)]
    ax1.set_xticklabels(labels[: cm.shape[1]])
    ax1.set_yticklabels(labels[: cm.shape[0]])

    if show_proportional and ax2 is not None:
        cm_normalized = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm_normalized / row_sums

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            ax=ax2,
            linewidths=0.5,
            linecolor="gray",
            cbar=False,
            vmin=0,
            vmax=1,
        )
        if show_title:
            ax2.set_title(
                f"{metrics_dict.model} - Normalized Confusion Matrix ({dataset.replace('_', ' ').title()})\n(Proportion of True Label)",
                fontweight="bold",
            )
        ax2.set_xlabel("Predicted Label")
        ax2.set_ylabel("True Label")
        ax2.set_xticklabels(labels[: cm.shape[1]])
        ax2.set_yticklabels(labels[: cm.shape[0]])

    plt.tight_layout()
    if dataset == "test":
        fig.savefig(
            DATA_DIR
            / "figures"
            / f"{metrics_dict.model}_conf_{dataset}_{class_mode}.pdf",
            bbox_inches="tight",
        )
    plt.show()

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0

    print("\nConfusion Matrix Statistics:")
    print(f"  Total predictions: {total}")
    print(f"  Correct predictions (diagonal): {correct}")
    print(f"  Incorrect predictions (off-diagonal): {total - correct}")
    print(f"  Accuracy: {accuracy:.{METRIC_DECIMAL_PLACES}f}")
    print("\nPer-label statistics:")
    for i in range(cm.shape[0]):
        true_count = cm[i, :].sum()
        correct_count = cm[i, i]
        if true_count > 0:
            per_label_acc = correct_count / true_count
            print(
                f"  Label {i}: {correct_count}/{true_count} correct ({per_label_acc:.2%})"
            )
        else:
            print(f"  Label {i}: No samples")


def print_per_label_metrics(
    metrics_dict: AggregatedModelData, dataset: str = "test"
) -> None:
    """Print per-label metrics table for a specific dataset."""
    df = extract_per_label_metrics(metrics_dict, dataset)
    if df is None:
        print(f"No {dataset} metrics available for this model")
        return

    print(f"\n{dataset.upper()} - Per-Label Metrics:")
    print("-" * 60)
    print(f"{'Label':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(
            f"{int(row['Label']):<8} {row['Precision']:<12.{METRIC_DECIMAL_PLACES}f} {row['Recall']:<12.{METRIC_DECIMAL_PLACES}f} {row['F1']:<12.{METRIC_DECIMAL_PLACES}f} {int(row['Support']):<10}"
        )
    print("-" * 60)

    data = getattr(metrics_dict, dataset)
    print(f"Accuracy: {data.accuracy:.{METRIC_DECIMAL_PLACES}f}")
    print(f"MSE: {data.mse:.{METRIC_DECIMAL_PLACES}f}")
    print(f"RMSE: {np.sqrt(data.mse):.{METRIC_DECIMAL_PLACES}f}")
    if data.f1 and data.support:
        total_support = sum(data.support)
        weighted_f1 = (
            sum(f * s for f, s in zip(data.f1, data.support)) / total_support
            if total_support > 0
            else 0.0
        )
        print(f"Weighted F1: {weighted_f1:.{METRIC_DECIMAL_PLACES}f}")


def vis_specific_model_tables(
    metrics_data: AggregatedModelData,
) -> None:
    """Visualize per-label metrics tables for train/val/test datasets."""

    def _styled_no_index(df):
        fmt = {
            "Precision": f"{{:.{METRIC_DECIMAL_PLACES}f}}",
            "Recall": f"{{:.{METRIC_DECIMAL_PLACES}f}}",
            "F1": f"{{:.{METRIC_DECIMAL_PLACES}f}}",
            "Support": "{:d}",
        }
        return df.style.format(fmt).set_table_styles(
            [{"selector": "th.row_heading, th.blank", "props": [("display", "none")]}]
        )

    train_df = extract_per_label_metrics(metrics_data, "train")
    if train_df is not None:
        print("=" * 60)
        print("TRAIN METRICS")
        print("=" * 60)
        display(_styled_no_index(train_df))
        print(f"Accuracy: {metrics_data.train.accuracy:.{METRIC_DECIMAL_PLACES}f}")
        print(f"MSE: {metrics_data.train.mse:.{METRIC_DECIMAL_PLACES}f}")

    cv_df = extract_per_label_metrics(metrics_data, "val")
    if cv_df is not None:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION METRICS (Averaged Across Folds)")
        print("=" * 60)
        display(_styled_no_index(cv_df))
        print(f"Accuracy: {metrics_data.val.accuracy:.{METRIC_DECIMAL_PLACES}f}")
        print(f"MSE: {metrics_data.val.mse:.{METRIC_DECIMAL_PLACES}f}")

    test_df = extract_per_label_metrics(metrics_data, "test")
    if test_df is not None:
        print("\n" + "=" * 60)
        print("TEST METRICS")
        print("=" * 60)
        display(_styled_no_index(test_df))
        print(f"Accuracy: {metrics_data.test.accuracy:.{METRIC_DECIMAL_PLACES}f}")
        print(f"MSE: {metrics_data.test.mse:.{METRIC_DECIMAL_PLACES}f}")

    if metrics_data.val and metrics_data.val.folds:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION METRICS - PER FOLD")
        print("=" * 60)

        for fold_idx, fold_metrics in enumerate(metrics_data.val.folds):
            print(f"\n--- Fold {fold_idx + 1}/{len(metrics_data.val.folds)} ---")

            fold_df = pd.DataFrame(
                {
                    "Label": range(6),
                    "Precision": fold_metrics.precision[:6],
                    "Recall": fold_metrics.recall[:6],
                    "F1": fold_metrics.f1[:6],
                    "Support": fold_metrics.support[:6],
                }
            )

            display(_styled_no_index(fold_df))
            print(f"Accuracy: {fold_metrics.accuracy:.{METRIC_DECIMAL_PLACES}f}")
            print(f"MSE: {fold_metrics.mse:.{METRIC_DECIMAL_PLACES}f}")


def vis_specific_model_conf_matrices(
    metrics_data: AggregatedModelData,
    show_proportional: bool = True,
    show_title: bool = True,
    class_mode: str = "full",
) -> None:
    """Visualize confusion matrices for train/val/test datasets."""
    if metrics_data.train is not None:
        print("=" * 80)
        print("TRAIN SET CONFUSION MATRIX")
        print("=" * 80)
        plot_confusion_matrix(
            metrics_data, "train", show_proportional, show_title, class_mode=class_mode
        )

    if metrics_data.val is not None:
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION CONFUSION MATRIX")
        print("=" * 80)
        plot_confusion_matrix(
            metrics_data, "val", show_proportional, show_title, class_mode=class_mode
        )

    if metrics_data.test is not None:
        print("\n" + "=" * 80)
        print("TEST SET CONFUSION MATRIX")
        print("=" * 80)
        plot_confusion_matrix(
            metrics_data, "test", show_proportional, show_title, class_mode=class_mode
        )


def vis_all_models_plots(
    models: list[AggregatedModelData],
    dataset: str = "test",
    class_mode: str = "full",
    file_prefix: str = "models",
    mode: str = "models",
    single_model: str | None = None,
) -> None:
    """Compare per-label metrics across multiple aggregated model dicts."""

    apply_plot_style()

    def _make_paired_x(
        n_labels: int, pair_inner_gap: float = 0.45, group_sep: float = 1.3
    ) -> np.ndarray:
        n_pairs = n_labels // 2
        group_centers = np.arange(n_pairs) * group_sep
        x = np.zeros(n_labels)
        for p in range(n_pairs):
            x[p * 2] = group_centers[p] - pair_inner_gap / 2
            x[p * 2 + 1] = group_centers[p] + pair_inner_gap / 2
        return x

    if mode == "train_test":
        target = (
            next((m for m in models if m.model == single_model), None)
            if single_model is not None
            else (models[0] if models else None)
        )
        if target is None:
            print("No model found for train/test mode")
            return

        splits: dict[str, DatasetMetrics] = {}
        splits_relaxed: dict[str, DatasetMetrics] = {}
        for split_name, attr in [("Train", "train"), ("Test", "test")]:
            md = getattr(target, attr, None)
            if md is not None:
                original_mse = md.mse
                if class_mode == "combined":
                    if split_name == "Test":
                        r_md = collapse_dataset_metrics_relaxed(md)
                        r_md.mse = original_mse
                        splits_relaxed[split_name] = r_md
                    md.mse = original_mse
                else:
                    md = (
                        collapse_dataset_metrics_relaxed(md)
                        if class_mode == "relaxed"
                        else md
                    )
                    md.mse = original_mse
                splits[split_name] = md

        if not splits:
            print(f"No train/test data available for model {target.model}")
            return

        split_names = list(splits.keys())
        n_splits = len(split_names)
        _split_colors = sns.color_palette(
            CMAP_QUALITATIVE_PRIMARY, n_splits + 1
        ).as_hex()
        split_colors = {name: _split_colors[i] for i, name in enumerate(split_names)}
        relaxed_split_colors = {name: _split_colors[n_splits] for name in split_names}

        model_display_name = MODEL_DISPLAY_NAMES.get(target.model, target.model)
        sample_md = next(iter(splits.values()))

        for metric_name in ["Precision", "Recall", "F1"]:
            metric_key = metric_name.lower()
            n_labels = len(sample_md.__dict__.get(metric_key, []))

            if class_mode == "combined":
                pair_inner_gap = 0.4
                group_sep = 0.95
                bar_width = 0.18
                x = _make_paired_x(n_labels, pair_inner_gap, group_sep)
                offsets = (
                    np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits)
                    * bar_width
                )
                n_pairs = n_labels // 2
                group_centers = np.arange(n_pairs) * group_sep
                relaxed_bar_width = (
                    pair_inner_gap + n_splits * bar_width + bar_width / 2
                )
            else:
                x = np.arange(n_labels)
                bar_width = 0.35
                offsets = (
                    np.linspace(-(n_splits - 1) / 2, (n_splits - 1) / 2, n_splits)
                    * bar_width
                )

            fig, ax = plt.subplots(figsize=(12, 6))

            if class_mode == "combined" and splits_relaxed:
                for split_name, r_md in splits_relaxed.items():
                    r_values = r_md.__dict__.get(metric_key, [])
                    ax.bar(
                        group_centers,
                        r_values[:n_pairs],
                        width=relaxed_bar_width,
                        color=relaxed_split_colors[split_name],
                        alpha=1.0,
                        edgecolor="none",
                        zorder=1,
                        label=f"{model_display_name} ({split_name}, relaxed)",
                    )

            for s_idx, (split_name, md) in enumerate(splits.items()):
                values = md.__dict__.get(metric_key, [])
                ax.bar(
                    x + offsets[s_idx],
                    values[:n_labels],
                    width=bar_width,
                    label=f"{model_display_name} ({split_name})",
                    color=split_colors[split_name],
                    alpha=1.0,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=2,
                )
            ax.set_xlabel("Label")
            ax.set_ylabel(metric_name)
            if class_mode == "relaxed":
                xticks = ["0/1", "2/3", "4/5"][:n_labels]
            elif class_mode == "combined":
                xticks = [str(i) for i in range(n_labels)]
            else:
                xticks = [f"Label {i}" for i in range(n_labels)]
            ax.set_xticks(x)
            ax.set_xticklabels(xticks)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
            ax.legend(ncol=n_splits)
            plt.tight_layout()
            fig.savefig(
                DATA_DIR
                / "figures"
                / f"{file_prefix}_{metric_name.lower()}_train_test_{class_mode}.pdf",
                bbox_inches="tight",
            )
            plt.show()
        return

    models_data = []
    lg_models_data = []
    for m in models:
        if getattr(m, dataset, None) is not None:
            if "_lg" in m.model:
                md = getattr(m, dataset)
                original_mse = md.mse
                md = (
                    collapse_dataset_metrics_relaxed(md)
                    if class_mode == "relaxed"
                    else md
                )
                md.mse = original_mse
                lg_models_data.append({"model": m.model, "metrics": md})
            else:
                md = getattr(m, dataset)
                original_mse = md.mse
                if class_mode == "combined":
                    r_md = collapse_dataset_metrics_relaxed(md)
                    r_md.mse = original_mse
                    md.mse = original_mse
                    models_data.append(
                        {"model": m.model, "metrics": md, "metrics_relaxed": r_md}
                    )
                else:
                    md = (
                        collapse_dataset_metrics_relaxed(md)
                        if class_mode == "relaxed"
                        else md
                    )
                    md.mse = original_mse
                    models_data.append({"model": m.model, "metrics": md})

    if not models_data:
        print(f"No models with {dataset} available")
        return

    lg_lookup = {}
    for lg_md in lg_models_data:
        base_name = lg_md["model"].replace("_lg", "")
        lg_lookup[base_name] = lg_md

    def parse_model(name: str):
        parts = name.split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return parts[0], None

    base_to_models = {}
    for md in models_data:
        base, emb = parse_model(md["model"])
        base_to_models.setdefault(base, []).append((emb, md))

    MODEL_ORDER = ["random", "ridge", "rf", "svm", "catboost", "finetuned-mbert"]
    EMBED_ORDER = ["minilm", "mbert"]

    ordered_bases = [b for b in MODEL_ORDER if b in base_to_models]
    ordered_bases += [b for b in base_to_models.keys() if b not in ordered_bases]

    for base in ordered_bases:
        embeds_present = [emb for emb, _ in base_to_models[base]]
        ordered_embeds = [e for e in EMBED_ORDER if e in embeds_present]
        ordered_embeds += [e for e in embeds_present if e not in ordered_embeds]
        md_map = {emb: md for emb, md in base_to_models[base]}
        base_to_models[base] = [(emb, md_map[emb]) for emb in ordered_embeds]

    palette = (
        CMAP_QUALITATIVE_PRIMARY
        if class_mode == "relaxed"
        else CMAP_QUALITATIVE_PRIMARY
    )
    _palette_colors = sns.color_palette(palette, 2 * len(ordered_bases)).as_hex()
    base_colors = {base: _palette_colors[i] for i, base in enumerate(ordered_bases)}
    relaxed_base_colors = {
        base: _palette_colors[len(ordered_bases) + i]
        for i, base in enumerate(ordered_bases)
    }

    def darken(hex_color, factor=0.75):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    total_cluster_width = 0.85
    n_bases = len(ordered_bases)
    base_slot_width = total_cluster_width / n_bases if n_bases else 0.0
    base_slot_margin_factor = 0.15

    max_variants = 0
    for base in ordered_bases:
        max_variants = max(max_variants, len(base_to_models[base]))
    if max_variants == 0:
        max_variants = 1

    combined_pair_inner_gap = 0.45
    combined_group_sep = 1.3

    for metric_name in ["Precision", "Recall", "F1"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for base_idx, base in enumerate(ordered_bases):
            variants = base_to_models[base]
            n_variants = len(variants)
            inner_width = base_slot_width * (1 - base_slot_margin_factor)
            bar_width = inner_width / max_variants

            base_center_offset = -total_cluster_width / 2 + base_slot_width * (
                base_idx + 0.5
            )

            if class_mode == "combined":
                sample_vals = variants[0][1]["metrics"].__dict__.get(
                    metric_name.lower(), []
                )
                n_labels = len(sample_vals)
                x_template = _make_paired_x(
                    n_labels, combined_pair_inner_gap, combined_group_sep
                )
                n_pairs = n_labels // 2
                group_centers_base = (
                    np.arange(n_pairs) * combined_group_sep + base_center_offset
                )
                relaxed_bar_width = (
                    combined_pair_inner_gap + n_variants * bar_width + bar_width
                )
                first_r_md = variants[0][1].get(
                    "metrics_relaxed", variants[0][1].get("metrics")
                )
                r_values = first_r_md.__dict__.get(metric_name.lower(), [])
                ax.bar(
                    group_centers_base,
                    r_values[:n_pairs],
                    width=relaxed_bar_width,
                    color=relaxed_base_colors[base],
                    alpha=1.0,
                    edgecolor="none",
                    zorder=1,
                )

            for v_idx, (emb, md) in enumerate(variants):
                metric_key = metric_name.lower()
                values = md["metrics"].__dict__.get(metric_key, [])
                n_labels = len(values)

                if class_mode == "combined":
                    base_positions = x_template + base_center_offset
                else:
                    base_positions = np.arange(n_labels) + base_center_offset

                variant_offset_start = (v_idx - (n_variants - 1) / 2) * bar_width

                x_positions = base_positions + variant_offset_start
                c = base_colors[base]
                if emb == "mbert":
                    c = darken(c)
                ax.bar(
                    x_positions,
                    values[:n_labels],
                    width=bar_width,
                    label=MODEL_DISPLAY_NAMES.get(md["model"], md["model"]),
                    alpha=1.0,
                    color=c,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=2,
                )

                lg_md = lg_lookup.get(md["model"])
                if lg_md is not None:
                    lg_values = lg_md["metrics"].__dict__.get(metric_key, [])
                    for label_idx in range(min(n_labels, len(lg_values))):
                        x_pos = x_positions[label_idx]
                        lg_val = lg_values[label_idx]

                        line_y = lg_val
                        line_left = x_pos - bar_width / 2
                        line_right = x_pos + bar_width / 2

                        ax.plot(
                            [line_left, line_right],
                            [line_y, line_y],
                            color="black",
                            linewidth=2,
                            zorder=10,
                        )

        ax.set_xlabel("Label")
        ax.set_ylabel(metric_name)
        # ax.set_title(
        #     f"{metric_name} Across Models ({dataset.replace('_', ' ').title()} Dataset)",
        #     fontsize=TITLE_FONT_SIZE,
        #     fontweight="bold",
        # )
        if models_data:
            sample_vals = models_data[0]["metrics"].__dict__.get(
                metric_name.lower(), []
            )
            n_labels = len(sample_vals)
        else:
            n_labels = 6
        if class_mode == "relaxed":
            xticks = ["0/1", "2/3", "4/5"][:n_labels]
            ax.set_xticks(np.arange(n_labels))
        elif class_mode == "combined":
            xticks = [str(i) for i in range(n_labels)]
            ax.set_xticks(
                _make_paired_x(n_labels, combined_pair_inner_gap, combined_group_sep)
            )
        else:
            xticks = [f"Label {i}" for i in range(n_labels)]
            ax.set_xticks(np.arange(n_labels))
        ax.set_xticklabels(xticks)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_handles, new_labels = [], []
        for h, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                new_handles.append(h)
                new_labels.append(label)
        ax.legend(new_handles, new_labels, ncol=2)
        plt.tight_layout()
        fig.savefig(
            DATA_DIR
            / "figures"
            / f"{file_prefix}_{metric_name.lower()}_{dataset}_{class_mode}.pdf",
            bbox_inches="tight",
        )
        plt.show()

    flat_models = []
    for base in ordered_bases:
        for emb, md in base_to_models[base]:
            flat_models.append((base, emb, md))

    def build_cluster_positions():
        positions = []
        total_width = 1.0
        n_bases = len(ordered_bases)
        slot_width = total_width / n_bases if n_bases else 1.0
        margin_factor = 0.15

        max_variants = max(len(base_to_models.get(base, [])) for base in ordered_bases)
        if max_variants == 0:
            max_variants = 1

        for base_idx, base in enumerate(ordered_bases):
            variants = base_to_models[base]
            n_variants = len(variants)
            inner_width = slot_width * (1 - margin_factor)
            bar_width = inner_width / max_variants

            base_center_offset = -total_width / 2 + slot_width * (base_idx + 0.5)
            for v_idx, (emb, md) in enumerate(variants):
                variant_offset_start = (v_idx - (n_variants - 1) / 2) * bar_width

                pos = base_center_offset + variant_offset_start
                positions.append((base, emb, md, pos, bar_width))
        return positions

    n_bases = len(ordered_bases)
    clustered_positions = build_cluster_positions()
    variant_positions = [pos for _, _, _, pos, _ in clustered_positions]
    variant_labels = []
    for base, emb, md, pos, bw in clustered_positions:
        if emb is None:
            variant_labels.append(f"{base}")
        else:
            variant_labels.append(f"{base}")
            # variant_labels.append(f"{base}\n({emb})")

    def plot_single_metric(metric_key, title, ylim=None, transform=None, y_label=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        for base, emb, md, pos, bw in clustered_positions:
            val = md["metrics"].__dict__.get(metric_key)
            if val is None:
                continue
            if transform:
                val = transform(val)
            c = base_colors[base]
            if emb == "mbert":
                c = darken(c)
            bar = ax.bar(
                [pos],
                [val],
                width=bw,
                color=c,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )[0]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.{METRIC_DECIMAL_PLACES}f}",
                ha="center",
                va="bottom",
                fontsize=ANNOT_FONT_SIZE,
            )

            lg_md = lg_lookup.get(md["model"])
            if lg_md is not None:
                lg_val = lg_md["metrics"].__dict__.get(metric_key)
                if lg_val is not None:
                    if transform:
                        lg_val = transform(lg_val)
                    line_left = pos - bw / 2
                    line_right = pos + bw / 2
                    ax.plot(
                        [line_left, line_right],
                        [lg_val, lg_val],
                        color="black",
                        linewidth=2,
                        zorder=10,
                    )

        # ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_ylabel(y_label if y_label else metric_key.title())
        ax.grid(axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(variant_positions)
        ax.set_xticklabels(variant_labels, fontsize=TICK_FONT_SIZE)
        plt.tight_layout()
        fig.savefig(
            DATA_DIR
            / "figures"
            / f"{file_prefix}_{metric_key}_{dataset}_{class_mode}.pdf",
            bbox_inches="tight",
        )
        plt.show()

    # plot_single_metric(
    #     "mse",
    #     f"RMSE Across Models ({dataset.replace('_', ' ').title()} Dataset)",
    #     transform=lambda x: np.sqrt(x),
    #     y_label="RMSE",
    # )
    # plot_single_metric(
    #     "accuracy",
    #     f"Accuracy Across Models ({dataset.replace('_', ' ').title()} Dataset)",
    #     ylim=(0, 1.0),
    # )

    # fig, ax = plt.subplots(figsize=(12, 6))
    # for base, emb, md, pos, bw in clustered_positions:
    #     f1_list = md["metrics"].f1
    #     support_list = md["metrics"].support
    #     if not f1_list or not support_list:
    #         continue
    #     total_support = sum(support_list)
    #     weighted_f1 = (
    #         sum(f * s for f, s in zip(f1_list, support_list)) / total_support
    #         if total_support > 0
    #         else 0.0
    #     )
    #     c = base_colors[base]
    #     if emb == "mbert":
    #         c = darken(c)
    #     bar = ax.bar(
    #         [pos],
    #         [weighted_f1],
    #         width=bw,
    #         color=c,
    #         alpha=0.9,
    #         edgecolor="white",
    #         linewidth=0.5,
    #     )[0]
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height(),
    #         f"{weighted_f1:.4f}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=ANNOT_FONT_SIZE,
    #     )

    #     lg_md = lg_lookup.get(md["model"])
    #     if lg_md is not None:
    #         lg_f1_list = lg_md["metrics"].f1
    #         lg_support_list = lg_md["metrics"].support
    #         if lg_f1_list and lg_support_list:
    #             lg_total_support = sum(lg_support_list)
    #             lg_weighted_f1 = (
    #                 sum(f * s for f, s in zip(lg_f1_list, lg_support_list))
    #                 / lg_total_support
    #                 if lg_total_support > 0
    #                 else 0.0
    #             )
    #             line_left = pos - bw / 2
    #             line_right = pos + bw / 2
    #             ax.plot(
    #                 [line_left, line_right],
    #                 [lg_weighted_f1, lg_weighted_f1],
    #                 color="black",
    #                 linewidth=2,
    #                 zorder=10,
    #             )

    # ax.set_title(
    #     f"Weighted F1 Across Models ({dataset.replace('_', ' ').title()} Dataset)",
    #     fontsize=TITLE_FONT_SIZE,
    #     fontweight="bold",
    # )
    # ax.set_ylabel("Weighted F1", fontsize=LABEL_FONT_SIZE)
    # ax.set_ylim(0, 1.0)
    # ax.grid(axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
    # ax.set_xticks(variant_positions)
    # ax.set_xticklabels(variant_labels, fontsize=TICK_FONT_SIZE)
    # plt.tight_layout()
    # fig.savefig(
    #     DATA_DIR / "figures" / f"{file_prefix}_f1_{dataset}_{class_mode}.pdf",
    #     bbox_inches="tight",
    # )
    # plt.show()


def vis_all_models_tables(
    aggregated_models: list[AggregatedModelData],
    metrics: list[str] | None = None,
    splits: list[str] | None = None,
    model_groups: dict[str, list[str]] | None = None,
    show_large_variants: bool = True,
    class_mode: str = "full",
) -> pd.DataFrame:
    """Create comparison table of key metrics across all models.

    Args:
        aggregated_models: List of AggregatedModelData objects
        metrics: List of metric names to display (default: ["RMSE", "Acc", "F1"])
        splits: List of dataset splits to display (default: ["Train", "CV", "Test"])
        model_groups: Dictionary mapping group names to lists of model names.
                     If None, creates a simple list of all models without grouping.
        show_large_variants: Whether to show "(Large)" variant rows for single-model groups

    Returns:
        DataFrame with comparison table
    """
    if metrics is None:
        metrics = ["RMSE", "Acc", "F1w"]
    if splits is None:
        splits = ["Train", "CV", "Test"]

    if model_groups is None:
        model_groups = {m.model: [m.model] for m in aggregated_models}

    all_metrics = []
    for m in aggregated_models:

        def _get_ds(dataset: str) -> Optional[DatasetMetrics]:
            d = getattr(m, dataset, None)
            if d is None:
                return None
            return collapse_dataset_metrics_relaxed(d) if class_mode == "relaxed" else d

        def _safe_val(d: Optional[DatasetMetrics], key: str):
            if d is None:
                return np.nan
            val = getattr(d, key, None)
            return val if val is not None else np.nan

        def _rmse(d: Optional[DatasetMetrics]):
            if d is None:
                return np.nan
            return np.sqrt(d.mse) if d.mse is not None else np.nan

        def _weighted_f1(d: Optional[DatasetMetrics]):
            if d and d.f1 and d.support:
                total = sum(d.support)
                return (
                    float(sum(f * s for f, s in zip(d.f1, d.support)) / total)
                    if total > 0
                    else np.nan
                )
            return np.nan

        train_d = _get_ds("train")
        val_d = _get_ds("val")
        test_d = _get_ds("test")

        all_metrics.append(
            {
                "Model": m.model,
                "Train RMSE": _rmse(getattr(m, "train", None)),
                "CV RMSE": _rmse(getattr(m, "val", None)),
                "Test RMSE": _rmse(getattr(m, "test", None)),
                "Train Acc": _safe_val(train_d, "accuracy"),
                "CV Acc": _safe_val(val_d, "accuracy"),
                "Test Acc": _safe_val(test_d, "accuracy"),
                "Train F1w": _weighted_f1(train_d),
                "CV F1w": _weighted_f1(val_d),
                "Test F1w": _weighted_f1(test_d),
            }
        )

    model_to_metrics = {m["Model"]: m for m in all_metrics}

    columns = [f"{split} {metric}" for metric in metrics for split in splits]

    prefixes = list(model_groups.keys())
    rows = []

    for prefix, models in model_groups.items():
        # Special handling for finetuned-mbert with three variants
        if (
            prefix == "Finet. MBERT"
            and len(models) == 3
            and all(
                m
                in ["finetuned-mbert_lg-only", "finetuned-mbert", "finetuned-mbert_lg"]
                for m in models
            )
        ):
            row = [prefix]
            m_dict = model_to_metrics.get("finetuned-mbert", {})
            for col in columns:
                val = m_dict.get(col, np.nan)
                row.append(val if not pd.isna(val) else "")
            rows.append(row)

            if show_large_variants:
                row = ["(+Large)"]
                m_dict = model_to_metrics.get("finetuned-mbert_lg", {})
                for col in columns:
                    val = m_dict.get(col, np.nan)
                    row.append(val if not pd.isna(val) else "")
                rows.append(row)

            row = ["(Large only)"]
            m_dict = model_to_metrics.get("finetuned-mbert_lg-only", {})
            for col in columns:
                val = m_dict.get(col, np.nan)
                row.append(val if not pd.isna(val) else "")
            rows.append(row)
        elif len(models) == 1 or len(models) == 2 and any("_lg" in m for m in models):
            model, model_lg = (
                next(model for model in models if "_lg" not in model),
                next(model for model in models if "_lg" in model)
                if any("_lg" in m for m in models)
                else None,
            )
            row = [prefix]
            m_dict = model_to_metrics.get(model, {})
            for col in columns:
                val = m_dict.get(col, np.nan)
                row.append(val if not pd.isna(val) else "")
            rows.append(row)

            if show_large_variants and model_lg:
                row = ["(+Large)"]
                m_dict = model_to_metrics.get(model_lg, {})
                for col in columns:
                    val = m_dict.get(col, np.nan)
                    row.append(val if not pd.isna(val) else "")
                rows.append(row)
        else:
            row = [prefix] + [""] * len(columns)
            rows.append(row)
            for model in models:
                if "_lg" in model:
                    base_emb = model.split("_")[-2]
                    emb = "MiniLM" if base_emb == "minilm" else "MBERT"
                    variant = f"{emb} (+Large)"
                else:
                    emb = model.split("_")[-1]
                    variant = "MiniLM" if emb == "minilm" else "MBERT"
                row = [f"{variant}"]
                m_dict = model_to_metrics.get(model, {})
                for col in columns:
                    val = m_dict.get(col, np.nan)
                    row.append(val if not pd.isna(val) else "")
                rows.append(row)

    df = pd.DataFrame(rows, columns=["Model"] + columns)

    def bold_rows(s):
        styles = [""] * len(s)
        if s["Model"] in prefixes:
            styles[0] = "font-weight: bold"
        return styles

    def bold_best(s):
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() == 0:
            return [""] * len(s)
        best_val = numeric.min() if "RMSE" in s.name else numeric.max()
        return [
            "font-weight: bold"
            if (not pd.isna(v) and np.isclose(float(v), float(best_val), atol=1e-6))
            else ""
            for v in numeric
        ]

    def format_value(x):
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.{METRIC_DECIMAL_PLACES}f}"
        except (ValueError, TypeError):
            return str(x)

    format_dict = {col: format_value for col in columns}

    display(
        df.style.apply(bold_rows, axis=1).apply(bold_best, axis=0).format(format_dict)
        # .set_caption("Model Metrics Comparison")
    )

    return df


def format_metrics_for_latex(df_metrics: pd.DataFrame) -> str:
    """
    Format metrics DataFrame for LaTeX output with best values bolded.

    Args:
        df_metrics: DataFrame with 'Model' column and numeric metric columns.

    Returns:
        LaTeX table string with bolded best values and model names.
    """
    models_to_bold = [
        "Random",
        "Ridge",
        "RandomForest",
        "SVM",
        "CatBoost",
        "Finet. MBERT",
        "RoBERTa",
        "DeBERTa-L",
        "MBERT-L",
        "Llama4-Scout",
    ]

    df_latex = df_metrics.copy()
    numeric_cols = df_latex.columns[1:]

    # Identify group header rows (all data columns are empty/nan)
    n_data_cols = len(numeric_cols)
    group_header_positions = {
        pos
        for pos in range(len(df_metrics))
        if all(
            str(df_metrics.iloc[pos, j + 1]) in ("", "nan") for j in range(n_data_cols)
        )
    }

    for col in numeric_cols:
        s = pd.to_numeric(df_latex[col], errors="coerce")

        if s.notna().any():
            best_val = s.min() if "RMSE" in col else s.max()
            is_best = np.isclose(s, best_val, atol=1e-6) & s.notna()

            for idx in df_latex.index:
                val = df_latex.loc[idx, col]
                if val == "" or pd.isna(val):
                    df_latex.loc[idx, col] = ""
                    continue

                try:
                    float_val = float(val)
                    df_latex.loc[idx, col] = (
                        f"\\textbf{{{float_val:.{METRIC_DECIMAL_PLACES}f}}}"
                        if is_best[idx]
                        else f"{float_val:.{METRIC_DECIMAL_PLACES}f}"
                    )
                except (ValueError, TypeError):
                    df_latex.loc[idx, col] = str(val)

    # Build two-row column header: top spans metric groups, bottom lists split names.
    # Columns (excluding Model) follow "{Split} {Metric}" naming.
    seen_metrics: list[str] = []
    metric_splits: dict[str, list[str]] = {}
    for col in numeric_cols:
        parts = col.split(" ", 1)
        split, metric = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        if metric not in metric_splits:
            seen_metrics.append(metric)
            metric_splits[metric] = []
        metric_splits[metric].append(split)

    def _arrow(metric: str) -> str:
        return "$\\downarrow$" if "RMSE" in metric else "$\\uparrow$"

    top_cells = ["\\textbf{Model}"] + [
        f"\\multicolumn{{{len(metric_splits[m])}}}{{c}}{{\\textbf{{{m}}} {_arrow(m)}}}"
        for m in seen_metrics
    ]
    bottom_cells = [""] + [s for m in seen_metrics for s in metric_splits[m]]
    header_top = " & ".join(top_cells) + " \\\\"
    header_bottom = " & ".join(bottom_cells) + " \\\\"

    df_latex.columns = ["Model"] + list(numeric_cols)

    model_col = df_latex.columns[0]

    def format_model_name(val: str) -> str:
        return f"\\textbf{{{val}}}" if val in models_to_bold else val

    df_latex[model_col] = df_latex[model_col].apply(format_model_name)

    n_total_cols = len(df_latex.columns)
    col_format = "l" + "r" * (n_total_cols - 1)
    latex = df_latex.to_latex(index=False, escape=False, column_format=col_format)

    # Post-process the generated LaTeX:
    # 1. Replace the auto-generated header line with the two-row metric/split header.
    # 2. Replace group-label rows with \midrule + \multicolumn span.
    lines = latex.split("\n")
    result = []
    data_row_idx = 0
    in_data_section = False
    prev_was_midrule = False
    header_replaced = False

    for line in lines:
        stripped = line.strip()

        if stripped == "\\toprule":
            in_data_section = False
            prev_was_midrule = False
            result.append(line)
            continue

        if stripped == "\\midrule" and not header_replaced:
            # This is the midrule after the column header row — replace header
            result.append(header_top)
            result.append(header_bottom)
            result.append(line)
            header_replaced = True
            in_data_section = True
            prev_was_midrule = True
            continue

        if stripped == "\\midrule":
            in_data_section = True
            prev_was_midrule = True
            result.append(line)
            continue

        if stripped == "\\bottomrule":
            in_data_section = False
            prev_was_midrule = False
            result.append(line)
            continue

        # Skip the original pandas-generated header line (before first \midrule)
        if (
            not header_replaced
            and in_data_section is False
            and stripped.endswith("\\\\")
        ):
            continue

        if in_data_section and stripped:
            if data_row_idx in group_header_positions:
                label = df_metrics.iloc[data_row_idx, 0]
                if not prev_was_midrule:
                    result.append("\\midrule")
                result.append(
                    f"\\multicolumn{{{n_total_cols}}}{{l}}{{\\textit{{{label}}}}} \\\\"
                )
                data_row_idx += 1
                prev_was_midrule = False
                continue
            data_row_idx += 1
            prev_was_midrule = False

        result.append(line)

    return "\n".join(result)


def latex_escape(text: str) -> str:
    text = text.replace("\\", r"\textbackslash{}")
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def format_number(val: Any, decimals: int = 3) -> str:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return str(val) if val is not None else ""
    return f"{num:.{decimals}f}"


def load_train_test_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    train_path = data_dir / "datasets" / "small" / "train.parquet"
    test_path = data_dir / "datasets" / "small" / "test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found at {train_path}")
    df_train = pd.read_parquet(train_path)
    if "label" not in df_train.columns:
        raise KeyError("Train dataset must contain a 'label' column")
    df_test = None
    if test_path.exists():
        df_test_candidate = pd.read_parquet(test_path)
        if "label" in df_test_candidate.columns:
            df_test = df_test_candidate
    return df_train, df_test
