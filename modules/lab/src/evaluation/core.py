from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


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
    metrics_dict: AggregatedModelData, dataset: str = "test"
) -> np.ndarray | None:
    """Extract confusion matrix from aggregated model data."""
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

    return cm[:6, :6]


def plot_confusion_matrix(
    metrics_dict: AggregatedModelData,
    dataset: str = "test",
    show_proportional: bool = True,
    show_title: bool = True,
) -> None:
    """Plot confusion matrix showing hits (diagonal) and misses (off-diagonal)."""
    cm = get_confusion_matrix(metrics_dict, dataset)
    if cm is None:
        print(f"Warning: {dataset} not available for this model")
        return

    if show_proportional:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None

    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax1,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Count"},
    )
    if show_title:
        ax1.set_title(
            f"{metrics_dict.model} - Confusion Matrix ({dataset.replace('_', ' ').title()})\n(Raw Counts)",
            fontsize=14,
            fontweight="bold",
        )
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_xticklabels([f"Label {i}" for i in range(6)])
    ax1.set_yticklabels([f"Label {i}" for i in range(6)])

    if show_proportional and ax2 is not None:
        cm_normalized = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm_normalized / row_sums

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            ax=ax2,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Proportion"},
            vmin=0,
            vmax=1,
        )
        if show_title:
            ax2.set_title(
                f"{metrics_dict.model} - Normalized Confusion Matrix ({dataset.replace('_', ' ').title()})\n(Proportion of True Label)",
                fontsize=14,
                fontweight="bold",
            )
        ax2.set_xlabel("Predicted Label", fontsize=12)
        ax2.set_ylabel("True Label", fontsize=12)
        ax2.set_xticklabels([f"Label {i}" for i in range(6)])
        ax2.set_yticklabels([f"Label {i}" for i in range(6)])

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
    for i in range(6):
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
) -> None:
    """Visualize confusion matrices for train/val/test datasets."""
    print("=" * 80)
    print("TRAIN SET CONFUSION MATRIX")
    print("=" * 80)
    plot_confusion_matrix(metrics_data, "train", show_proportional, show_title)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION CONFUSION MATRIX")
    print("=" * 80)
    plot_confusion_matrix(metrics_data, "val", show_proportional, show_title)

    print("\n" + "=" * 80)
    print("TEST SET CONFUSION MATRIX")
    print("=" * 80)
    plot_confusion_matrix(metrics_data, "test", show_proportional, show_title)


def vis_all_models_plots(
    models: list[AggregatedModelData], dataset: str = "test"
) -> None:
    """Compare per-label metrics across multiple aggregated model dicts."""
    models_data = []
    lg_models_data = []
    for m in models:
        if getattr(m, dataset, None) is not None:
            if "_lg" in m.model:
                lg_models_data.append(
                    {"model": m.model, "metrics": getattr(m, dataset)}
                )
            else:
                models_data.append({"model": m.model, "metrics": getattr(m, dataset)})

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

    base_colors = {
        "random": "#4C78A8",
        "ridge": "#F58518",
        "rf": "#54A24B",
        "svm": "#B79F00",
        "catboost": "#E45756",
        "finetuned": "#7F3C8D",
        "finetuned-mbert": "#7F3C8D",
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
            for v_idx, (emb, md) in enumerate(variants):
                metric_key = metric_name.lower()
                values = md["metrics"].__dict__.get(metric_key, [])
                values = values + [0.0] * (6 - len(values))
                base_positions = np.arange(6) + base_center_offset

                variant_offset_start = (v_idx - (n_variants - 1) / 2) * bar_width

                x_positions = base_positions + variant_offset_start
                c = base_colors.get(base, "#888888")
                if emb == "mbert":
                    c = darken(c)
                ax.bar(
                    x_positions,
                    values[:6],
                    width=bar_width,
                    label=md["model"],
                    alpha=0.9,
                    color=c,
                    edgecolor="white",
                    linewidth=0.5,
                )

                lg_md = lg_lookup.get(md["model"])
                if lg_md is not None:
                    lg_values = lg_md["metrics"].__dict__.get(metric_key, [])
                    lg_values = lg_values + [0.0] * (6 - len(lg_values))
                    for label_idx in range(6):
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

        ax.set_xlabel("Label", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(
            f"{metric_name} Across Models ({dataset.replace('_', ' ').title()} Dataset)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels([f"Label {i}" for i in range(6)])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_handles, new_labels = [], []
        for h, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                new_handles.append(h)
                new_labels.append(label)
        ax.legend(new_handles, new_labels, fontsize=10, ncol=2)
        plt.tight_layout()
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
            variant_labels.append(f"{base}\n({emb})")

    def plot_single_metric(metric_key, title, ylim=None, transform=None, y_label=None):
        fig, ax = plt.subplots(figsize=(12, 6))
        for base, emb, md, pos, bw in clustered_positions:
            val = md["metrics"].__dict__.get(metric_key)
            if val is None:
                continue
            if transform:
                val = transform(val)
            c = base_colors.get(base, "#888888")
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
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
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

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(y_label if y_label else metric_key.title(), fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(variant_positions)
        ax.set_xticklabels(variant_labels, fontsize=10)
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

    fig, ax = plt.subplots(figsize=(12, 6))
    for base, emb, md, pos, bw in clustered_positions:
        f1_list = md["metrics"].f1
        if not f1_list:
            continue
        macro_f1 = np.mean(f1_list)
        c = base_colors.get(base, "#888888")
        if emb == "mbert":
            c = darken(c)
        bar = ax.bar(
            [pos],
            [macro_f1],
            width=bw,
            color=c,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{macro_f1:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        lg_md = lg_lookup.get(md["model"])
        if lg_md is not None:
            lg_f1_list = lg_md["metrics"].f1
            if lg_f1_list:
                lg_macro_f1 = np.mean(lg_f1_list)
                line_left = pos - bw / 2
                line_right = pos + bw / 2
                ax.plot(
                    [line_left, line_right],
                    [lg_macro_f1, lg_macro_f1],
                    color="black",
                    linewidth=2,
                    zorder=10,
                )

    ax.set_title(
        f"Macro-F1 Across Models ({dataset.replace('_', ' ').title()} Dataset)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(variant_positions)
    ax.set_xticklabels(variant_labels, fontsize=10)
    plt.tight_layout()
    plt.show()


def vis_all_models_tables(
    aggregated_models: list[AggregatedModelData],
    metrics: list[str] | None = None,
    splits: list[str] | None = None,
    model_groups: dict[str, list[str]] | None = None,
    show_large_variants: bool = True,
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

        def _safe(dataset, key):
            d = getattr(m, dataset, None)
            if d is None:
                return np.nan
            val = getattr(d, key, None)
            return val if val is not None else np.nan

        def _rmse(dataset):
            mse = _safe(dataset, "mse")
            return np.sqrt(mse) if not pd.isna(mse) else np.nan

        def _macro_f1(dataset):
            d = getattr(m, dataset, None)
            if d and d.f1:
                return float(np.mean(d.f1))
            return np.nan

        all_metrics.append(
            {
                "Model": m.model,
                "Train RMSE": _rmse("train"),
                "CV RMSE": _rmse("val"),
                "Test RMSE": _rmse("test"),
                "Train Acc": _safe("train", "accuracy"),
                "CV Acc": _safe("val", "accuracy"),
                "Test Acc": _safe("test", "accuracy"),
                "Train Macro-F1": _macro_f1("train"),
                "CV Macro-F1": _macro_f1("val"),
                "Test Macro-F1": _macro_f1("test"),
            }
        )

    model_to_metrics = {m["Model"]: m for m in all_metrics}

    columns = [f"{split} {metric}" for metric in metrics for split in splits]

    prefixes = list(model_groups.keys())
    rows = []

    for prefix, models in model_groups.items():
        if len(models) == 1:
            model = models[0]
            row = [prefix]
            m_dict = model_to_metrics.get(model, {})
            for col in columns:
                val = m_dict.get(col, np.nan)
                row.append(val if not pd.isna(val) else "")
            rows.append(row)

            if show_large_variants:
                model = f"{models[0]}_lg"
                row = ["(Large)"]
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
                    variant = f"{emb} (Large)"
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


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
        "\\": r"\\textbackslash{}",
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
