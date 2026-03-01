from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from evaluation.plot_style import (  # noqa: F401 – re-exported for callers
    LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    TICK_FONT_SIZE,
    LEGEND_FONT_SIZE,
    ANNOT_FONT_SIZE,
    PALETTE,
    CMAP_PRIMARY,
    CMAP_SECONDARY,
)


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

    raw_cmap = CMAP_PRIMARY if class_mode != "relaxed" else CMAP_SECONDARY
    norm_cmap = CMAP_PRIMARY if class_mode != "relaxed" else CMAP_SECONDARY

    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap=raw_cmap,
        ax=ax1,
        linewidths=0.5,
        linecolor="gray",
        cbar=False,
    )
    if show_title:
        ax1.set_title(
            f"{metrics_dict.model} - Confusion Matrix ({dataset.replace('_', ' ').title()})\n(Raw Counts)",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
    ax1.set_xlabel("Predicted Label", fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel("True Label", fontsize=LABEL_FONT_SIZE)
    if class_mode == "relaxed":
        labels = ["0/1", "2/3", "4/5"]
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
            cmap=norm_cmap,
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
                fontsize=TITLE_FONT_SIZE,
                fontweight="bold",
            )
        ax2.set_xlabel("Predicted Label", fontsize=LABEL_FONT_SIZE)
        ax2.set_ylabel("True Label", fontsize=LABEL_FONT_SIZE)
        ax2.set_xticklabels(labels[: cm.shape[1]])
        ax2.set_yticklabels(labels[: cm.shape[0]])

    plt.tight_layout()
    plt.show()

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0

    print("\nConfusion Matrix Statistics:")
    print(f"  Total predictions: {total}")
    print(f"  Correct predictions (diagonal): {correct}")
    print(f"  Incorrect predictions (off-diagonal): {total - correct}")
    print(f"  Accuracy: {accuracy:.4f}")
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
            f"{int(row['Label']):<8} {row['Precision']:<12.4f} {row['Recall']:<12.4f} {row['F1']:<12.4f} {int(row['Support']):<10}"
        )
    print("-" * 60)

    data = getattr(metrics_dict, dataset)
    print(f"Accuracy: {data.accuracy:.4f}")
    print(f"MSE: {data.mse:.4f}")
    print(f"RMSE: {np.sqrt(data.mse):.4f}")
    if data.f1:
        print(f"Macro-F1: {np.mean(data.f1):.4f}")


def vis_specific_model_tables(
    metrics_data: AggregatedModelData,
) -> None:
    """Visualize per-label metrics tables for train/val/test datasets."""

    def _styled_no_index(df):
        fmt = {
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
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
        print(f"Accuracy: {metrics_data.train.accuracy:.4f}")
        print(f"MSE: {metrics_data.train.mse:.4f}")

    cv_df = extract_per_label_metrics(metrics_data, "val")
    if cv_df is not None:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION METRICS (Averaged Across Folds)")
        print("=" * 60)
        display(_styled_no_index(cv_df))
        print(f"Accuracy: {metrics_data.val.accuracy:.4f}")
        print(f"MSE: {metrics_data.val.mse:.4f}")

    test_df = extract_per_label_metrics(metrics_data, "test")
    if test_df is not None:
        print("\n" + "=" * 60)
        print("TEST METRICS")
        print("=" * 60)
        display(_styled_no_index(test_df))
        print(f"Accuracy: {metrics_data.test.accuracy:.4f}")
        print(f"MSE: {metrics_data.test.mse:.4f}")

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
            print(f"Accuracy: {fold_metrics.accuracy:.4f}")
            print(f"MSE: {fold_metrics.mse:.4f}")


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
    models: list[AggregatedModelData], dataset: str = "test", class_mode: str = "full"
) -> None:
    """Compare per-label metrics across multiple aggregated model dicts."""
    models_data = []
    lg_models_data = []
    for m in models:
        if getattr(m, dataset, None) is not None:
            if "_lg" in m.model:
                md = getattr(m, dataset)
                md = (
                    collapse_dataset_metrics_relaxed(md)
                    if class_mode == "relaxed"
                    else md
                )
                lg_models_data.append({"model": m.model, "metrics": md})
            else:
                md = getattr(m, dataset)
                md = (
                    collapse_dataset_metrics_relaxed(md)
                    if class_mode == "relaxed"
                    else md
                )
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

    for metric_name in ["Precision", "Recall", "F1"]:
        rows = []
        for base in ordered_bases:
            for emb, md in base_to_models[base]:
                values = md["metrics"].__dict__.get(metric_name.lower(), [])
                for label_idx, val in enumerate(values):
                    if class_mode == "relaxed":
                        label_tick = ("0/1", "2/3", "4/5")[label_idx]
                    else:
                        label_tick = f"Label {label_idx}"
                    rows.append(
                        {"Model": md["model"], "Label": label_tick, metric_name: val}
                    )

                lg_md = lg_lookup.get(md["model"])
                if lg_md is not None:
                    lg_values = lg_md["metrics"].__dict__.get(metric_name.lower(), [])
                    for label_idx, val in enumerate(lg_values[: len(values)]):
                        if class_mode == "relaxed":
                            label_tick = ("0/1", "2/3", "4/5")[label_idx]
                        else:
                            label_tick = f"Label {label_idx}"
                        rows.append(
                            {
                                "Model": lg_md["model"],
                                "Label": label_tick,
                                metric_name: val,
                            }
                        )

        if not rows:
            continue
        df_metric = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=df_metric, x="Label", y=metric_name, hue="Model", ax=ax, alpha=0.9
        )
        ax.set_xlabel("Label", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(metric_name, fontsize=LABEL_FONT_SIZE)
        ax.set_title(
            f"{metric_name} Across Models ({dataset.replace('_', ' ').title()} Dataset)",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, ncol=2)
        plt.tight_layout()
        plt.show()

    flat_models = []
    for base in ordered_bases:
        for emb, md in base_to_models[base]:
            flat_models.append((base, emb, md))

    def _model_label(base: str, emb: str | None, suffix: str = "") -> str:
        if emb is None:
            return f"{base}{suffix}"
        return f"{base}\n({emb}{suffix})"

    def plot_single_metric(metric_key, title, ylim=None, transform=None, y_label=None):
        rows = []
        for base, emb, md in flat_models:
            val = md["metrics"].__dict__.get(metric_key)
            if val is None:
                continue
            if transform:
                val = transform(val)
            rows.append({"Model": _model_label(base, emb), "Value": val})

            lg_md = lg_lookup.get(md["model"])
            if lg_md is not None:
                lg_val = lg_md["metrics"].__dict__.get(metric_key)
                if lg_val is not None:
                    if transform:
                        lg_val = transform(lg_val)
                    rows.append(
                        {"Model": _model_label(base, emb, "+lg"), "Value": lg_val}
                    )

        if not rows:
            return
        df_single = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=df_single, x="Model", y="Value", ax=ax, alpha=0.9, errorbar=None
        )

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOT_FONT_SIZE,
                )

        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold")
        ax.set_ylabel(
            y_label if y_label else metric_key.title(), fontsize=LABEL_FONT_SIZE
        )
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        plt.tight_layout()
        plt.show()

    plot_single_metric(
        "mse",
        f"RMSE Across Models ({dataset.replace('_', ' ').title()} Dataset)",
        transform=lambda x: np.sqrt(x),
        y_label="RMSE",
    )
    plot_single_metric(
        "accuracy",
        f"Accuracy Across Models ({dataset.replace('_', ' ').title()} Dataset)",
        ylim=(0, 1.0),
    )

    rows = []
    for base, emb, md in flat_models:
        f1_list = md["metrics"].f1
        if not f1_list:
            continue
        macro_f1 = float(np.mean(f1_list))
        rows.append({"Model": _model_label(base, emb), "Macro-F1": macro_f1})

        lg_md = lg_lookup.get(md["model"])
        if lg_md is not None:
            lg_f1_list = lg_md["metrics"].f1
            if lg_f1_list:
                lg_macro_f1 = float(np.mean(lg_f1_list))
                rows.append(
                    {"Model": _model_label(base, emb, "+lg"), "Macro-F1": lg_macro_f1}
                )

    if rows:
        df_macro = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=df_macro, x="Model", y="Macro-F1", ax=ax, alpha=0.9, errorbar=None
        )

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    height,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOT_FONT_SIZE,
                )

        ax.set_title(
            f"Macro-F1 Across Models ({dataset.replace('_', ' ').title()} Dataset)",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_ylabel("Macro-F1", fontsize=LABEL_FONT_SIZE)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        plt.tight_layout()
        plt.show()


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
        metrics: List of metric names to display (default: ["RMSE", "Acc", "Macro-F1"])
        splits: List of dataset splits to display (default: ["Train", "CV", "Test"])
        model_groups: Dictionary mapping group names to lists of model names.
                     If None, creates a simple list of all models without grouping.
        show_large_variants: Whether to show "(Large)" variant rows for single-model groups

    Returns:
        DataFrame with comparison table
    """
    if metrics is None:
        metrics = ["RMSE", "Acc", "Macro-F1"]
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

        def _macro_f1(d: Optional[DatasetMetrics]):
            if d and d.f1:
                return float(np.mean(d.f1))
            return np.nan

        train_d = _get_ds("train")
        val_d = _get_ds("val")
        test_d = _get_ds("test")

        all_metrics.append(
            {
                "Model": m.model,
                "Train RMSE": _rmse(train_d),
                "CV RMSE": _rmse(val_d),
                "Test RMSE": _rmse(test_d),
                "Train Acc": _safe_val(train_d, "accuracy"),
                "CV Acc": _safe_val(val_d, "accuracy"),
                "Test Acc": _safe_val(test_d, "accuracy"),
                "Train Macro-F1": _macro_f1(train_d),
                "CV Macro-F1": _macro_f1(val_d),
                "Test Macro-F1": _macro_f1(test_d),
            }
        )

    model_to_metrics = {m["Model"]: m for m in all_metrics}

    columns = [f"{split} {metric}" for metric in metrics for split in splits]

    prefixes = list(model_groups.keys())
    rows = []

    for prefix, models in model_groups.items():
        # Special handling for finetuned-mbert with three variants
        if (
            prefix == "FinetunedMBERT"
            and len(models) == 3
            and all(
                m
                in ["finetuned-mbert_lg-only", "finetuned-mbert", "finetuned-mbert_lg"]
                for m in models
            )
        ):
            row = [prefix] + [""] * len(columns)
            rows.append(row)

            # Add three sub-rows for different dataset configurations
            variants = [
                ("finetuned-mbert_lg-only", "Large only"),
                ("finetuned-mbert", "Small only"),
                ("finetuned-mbert_lg", "Small+Large"),
            ]
            for model_name, label in variants:
                row = [label]
                m_dict = model_to_metrics.get(model_name, {})
                for col in columns:
                    val = m_dict.get(col, np.nan)
                    row.append(val if not pd.isna(val) else "")
                rows.append(row)
        elif len(models) == 1:
            model = models[0]
            row = [prefix]
            m_dict = model_to_metrics.get(model, {})
            for col in columns:
                val = m_dict.get(col, np.nan)
                row.append(val if not pd.isna(val) else "")
            rows.append(row)

            if show_large_variants:
                model = f"{models[0]}_lg"
                row = ["(+Large)"]
                m_dict = model_to_metrics.get(model, {})
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
        return ["font-weight: bold" if s["Model"] in prefixes else "" for _ in s]

    def format_value(x):
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)

    format_dict = {col: format_value for col in columns}

    display(
        df.style.apply(bold_rows, axis=1)
        .format(format_dict)
        .set_caption("Model Metrics Comparison")
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
        "FinetunedMBERT",
    ]

    df_latex = df_metrics.copy()
    numeric_cols = df_latex.columns[1:]

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
                        f"\\textbf{{{float_val:.4f}}}"
                        if is_best[idx]
                        else f"{float_val:.4f}"
                    )
                except (ValueError, TypeError):
                    df_latex.loc[idx, col] = str(val)

    df_latex.columns = [f"\\textbf{{{col}}}" for col in df_latex.columns]

    model_col = df_latex.columns[0]

    def format_model_name(val: str) -> str:
        return f"\\textbf{{{val}}}" if val in models_to_bold else val

    df_latex[model_col] = df_latex[model_col].apply(format_model_name)

    col_format = "r" * len(df_latex.columns)
    return df_latex.to_latex(index=False, escape=False, column_format=col_format)


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


def format_number(val: Any, decimals: int = 4) -> str:
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
