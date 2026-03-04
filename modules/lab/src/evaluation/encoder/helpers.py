import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Literal, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.plot_style import (
    CMAP_QUALITATIVE_PRIMARY,
    GRID_ALPHA,
    GRID_LINESTYLE,
    apply_plot_style,
)
from evaluation.core import (
    get_confusion_matrix,
    print_per_label_metrics,
)
from utils import DATA_DIR, average_metrics
import re

METRICS_DIR = DATA_DIR / "metrics" / "encoder"

MODEL_ORDER = ["random", "ridge", "rf", "svm", "catboost", "finetuned-mbert"]
EMBED_ORDER = ["minilm", "mbert"]


def load_cv_history_files() -> list[Path]:
    """Load all validation (CV) metric files with training curves."""
    return sorted(list(METRICS_DIR.glob("finetuned*_val_*.json")))


def load_train_history_files() -> list[Path]:
    """Load all train metric files with training curves."""
    return sorted(list(METRICS_DIR.glob("finetuned*_train_*.json")))


def parse_metrics_filename(path: str | Path) -> dict[str, Any]:
    """
    Parse a metrics filename according to modules/lab/data/metrics/metric-docs.md.

    Supports both formats:
    - model_embedding_lg_type_seed_yyyy-MM-dd-hh-mm-ss.json
    - model_embedding_lg_val_tX_seed_yyyy-MM-dd-hh-mm-ss.json

    Returns a dict with keys:
    - model_name: str (e.g., "svm_minilm_lg" or "finetuned-mbert_lg")
    - type: Literal['train','val','test']
    - trial: Optional[int]
    - seed: int
    - timestamp: str (yyyy-MM-dd-hh-mm-ss)
    - stem: filename without extension
    - common_key: filename stem without seed and timestamp (used for grouping)
    """
    p = Path(path)
    stem = p.stem
    # First, try val with trial: model_*_val_tX_seed_timestamp
    m_val = re.match(
        r"^(?P<model>.+)_val_t(?P<trial>\d+)_(?P<seed>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$",
        stem,
    )
    if m_val:
        model_name = m_val.group("model")
        trial = int(m_val.group("trial"))
        seed = int(m_val.group("seed"))
        ts = m_val.group("ts")
        return {
            "model_name": model_name,
            "type": "val",
            "trial": trial,
            "seed": seed,
            "timestamp": ts,
            "stem": stem,
            "common_key": f"{model_name}_val_t{trial}",
        }

    # General case: model_*_type_seed_timestamp (type in {train,test,val})
    m = re.match(
        r"^(?P<model>.+)_(?P<type>train|test|val)_(?P<seed>\d+)_(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$",
        stem,
    )
    if not m:
        raise ValueError(
            f"Filename '{p.name}' does not match expected metrics patterns."
        )
    model_name = m.group("model")
    type_ = m.group("type")
    seed = int(m.group("seed"))
    ts = m.group("ts")
    return {
        "model_name": model_name,
        "type": type_,
        "trial": None,
        "seed": seed,
        "timestamp": ts,
        "stem": stem,
        "common_key": f"{model_name}_{type_}",
    }


def group_metric_files(paths: list[str | Path]) -> dict[str, list[str]]:
    """
    Group metric files by common identifier (model + embedding + lg + type [+ trial]).

    Insensitive to timestamp. Each seed is counted only once per group; if multiple
    files exist for the same common identifier and seed, only the latest (by timestamp
    in filename) is kept.

    Returns a dict mapping common_key -> list[str] of selected filenames.
    The common_key matches the filename stem without seed and timestamp, e.g.:
      - "svm_minilm_lg_train"
      - "svm_minilm_lg_val_t1"
      - "finetuned-mbert_lg_test"
    """
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path in paths:
        info = parse_metrics_filename(path)
        key = info["common_key"]
        seed = info["seed"]
        grouped.setdefault(key, {})
        existing = grouped[key].get(seed)
        if not existing:
            grouped[key][seed] = {"info": info, "path": str(Path(path))}
        else:
            # Keep the one with latest timestamp
            if info["timestamp"] > existing["info"]["timestamp"]:
                grouped[key][seed] = {"info": info, "path": str(Path(path))}

    # Convert to desired output: common_key -> [paths]
    result: dict[str, list[str]] = {}
    for key, seeds_map in grouped.items():
        # deterministic order by seed
        ordered_paths = [
            entry["path"] for _, entry in sorted(seeds_map.items(), key=lambda x: x[0])
        ]
        result[key] = ordered_paths
    return result


def average_metric_files(paths: list[str | Path]) -> dict[str, Any]:
    """
    Read metric files and average their contents into a single metric dict following
    the same schema as defined in metric-docs.md.

    All input files must be of the same format bucket:
      - Train files (type == 'train')
      - Test files (type == 'test')
      - Val files (type == 'val' with identical folds length and presence of per-fold keys)

    Raises ValueError if formats/types are mixed or incompatible.
    """
    infos = [parse_metrics_filename(p) for p in paths]
    # Ensure same common_key
    common_keys = {i["common_key"] for i in infos}
    if len(common_keys) != 1:
        raise ValueError(
            f"All files must share the same identifier. Found: {sorted(common_keys)}"
        )
    type_ = infos[0]["type"]

    # Load JSON contents
    contents: list[dict[str, Any]] = []
    for p in paths:
        with open(Path(p), "r") as fh:
            contents.append(json.load(fh))

    # Basic format validations
    types = {c.get("type") for c in contents}
    if len(types) != 1 or type_ not in types:
        raise ValueError("All files must have the same 'type' field.")

    # model field must be the same
    models = {c.get("model") for c in contents}
    if len(models) != 1:
        raise ValueError("All files must have the same 'model' field.")
    model_name = next(iter(models))

    # params field presence consistency (if present should be same structure)
    # If params absent in some and present in others, treat as mismatch.
    has_params = {"params" in c for c in contents}
    if len(has_params) != 1:
        raise ValueError("Mismatch in presence of 'params' across files.")
    if True in has_params:
        # Require identical params
        params_set = {json.dumps(c["params"], sort_keys=True) for c in contents}
        if len(params_set) != 1:
            raise ValueError("All files must have identical 'params'.")
        params = contents[0]["params"]
    else:
        params = {}

    if type_ in ("train", "test"):
        # Average basic metrics (including confusion matrix)
        metrics_list = [
            {
                "mse": float(c["mse"]),
                "accuracy": float(c["accuracy"]),
                "precision": c["precision"],
                "recall": c["recall"],
                "f1": c["f1"],
                "support": c["support"],
                "confusion_matrix": c["confusion_matrix"],
            }
            for c in contents
        ]
        averaged = average_metrics(metrics_list)
        result = {
            "model": model_name,
            "params": params,
            "type": type_,
            "seed": -1,  # not meaningful for averaged outputs
            "mse": averaged["mse"],
            "accuracy": averaged["accuracy"],
            "precision": averaged["precision"],
            "recall": averaged["recall"],
            "f1": averaged["f1"],
            "support": averaged["support"],
            "confusion_matrix": averaged["confusion_matrix"],
        }
        # Optionally include corr if present
        if all("corr" in c for c in contents):
            corr_values = [float(c["corr"]) for c in contents]
            result["corr"] = float(np.mean(corr_values))
        # For train type, optionally aggregate train_losses if present
        if type_ == "train" and all("train_losses" in c for c in contents):
            # Average per-index; lengths may differ → require equality
            lengths = {len(c["train_losses"]) for c in contents}
            if len(lengths) == 1:
                arr = np.array([c["train_losses"] for c in contents], dtype=float)
                result["train_losses"] = np.mean(arr, axis=0).tolist()
        return result

    if type_ == "val":
        # Validate folds shape and keys
        folds_lengths = {len(c.get("folds", [])) for c in contents}
        if len(folds_lengths) != 1:
            raise ValueError("Validation files must have the same number of folds.")
        n_folds = next(iter(folds_lengths))

        # trial consistency (val_tX vs val without trial should have same common_key already)
        trials = {c.get("trial") for c in contents}
        if len(trials) != 1:
            raise ValueError(
                "Validation files must have the same 'trial' value (both None or same integer)."
            )
        trial_val = next(iter(trials))

        # Average per-fold metrics
        averaged_folds = []
        for f_idx in range(n_folds):
            fold_metrics_list = []
            for c in contents:
                fold = c["folds"][f_idx]
                fold_metrics_list.append(
                    {
                        "mse": float(fold["mse"]),
                        "accuracy": float(fold["accuracy"]),
                        "precision": fold["precision"],
                        "recall": fold["recall"],
                        "f1": fold["f1"],
                        "support": fold["support"],
                        "confusion_matrix": fold["confusion_matrix"],
                    }
                )
            avg_fold = average_metrics(fold_metrics_list)

            # Average losses if present uniformly
            train_losses = None
            val_losses = None
            if all("train_losses" in c["folds"][f_idx] for c in contents):
                lengths = {len(c["folds"][f_idx]["train_losses"]) for c in contents}
                if len(lengths) == 1:
                    arr = np.array(
                        [c["folds"][f_idx]["train_losses"] for c in contents],
                        dtype=float,
                    )
                    train_losses = np.mean(arr, axis=0).tolist()
            if all("val_losses" in c["folds"][f_idx] for c in contents):
                lengths = {len(c["folds"][f_idx]["val_losses"]) for c in contents}
                if len(lengths) == 1:
                    arr = np.array(
                        [c["folds"][f_idx]["val_losses"] for c in contents], dtype=float
                    )
                    val_losses = np.mean(arr, axis=0).tolist()

            averaged_folds.append(
                {
                    "mse": avg_fold["mse"],
                    "accuracy": avg_fold["accuracy"],
                    "precision": avg_fold["precision"],
                    "recall": avg_fold["recall"],
                    "f1": avg_fold["f1"],
                    "support": avg_fold["support"],
                    "confusion_matrix": avg_fold["confusion_matrix"],
                    **(
                        {"train_losses": train_losses}
                        if train_losses is not None
                        else {}
                    ),
                    **({"val_losses": val_losses} if val_losses is not None else {}),
                }
            )

        return {
            "model": model_name,
            "params": params,
            "type": "val",
            "seed": -1,
            "folds": averaged_folds,
            **({"trial": trial_val} if trial_val is not None else {}),
        }

    raise ValueError(f"Unsupported metrics type: {type_}")


def load_all_metric_files() -> list[Path]:
    """Load all metric JSON files from the metrics directory."""
    return sorted(list(METRICS_DIR.glob("*.json")))


def aggregate_metrics_by_model() -> list[dict[str, Any]]:
    """
    Aggregate metrics across all seeds for each model/embedding/lg/type combination.

    For finetuned-mbert, handles three variants:
    - finetuned-mbert_lg-only: large dataset only
    - finetuned-mbert_lg: large dataset (combined)
    - finetuned-mbert: small dataset only

    Returns:
        List of dicts with structure:
        {
            'model': str,  # e.g., "svm_minilm_lg" or "finetuned-mbert_lg-only"
            'params': dict,
            'train': dict | None,  # Averaged metrics
            'val': dict | None,
            'test': dict | None,
        }
    """
    all_files = load_all_metric_files()
    if not all_files:
        return []

    # Group files by common identifier (model+embedding+lg+type)
    grouped = group_metric_files(all_files)

    # Map: model_name -> {type -> averaged_metrics}
    model_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for common_key, file_paths in grouped.items():
        if not file_paths:
            continue

        # Parse to get model name and type
        info = parse_metrics_filename(file_paths[0])
        model_name = info["model_name"]
        type_ = info["type"]

        # Average metrics across seeds
        try:
            averaged = average_metric_files(file_paths)
            model_map[model_name][type_] = averaged
        except Exception as e:
            print(f"Warning: Could not average {common_key}: {e}")
            continue

    # Convert to output format
    result = []
    for model_name, types_dict in model_map.items():
        # Get params from any available type
        params = {}
        for metrics in types_dict.values():
            if "params" in metrics:
                params = metrics["params"]
                break

        # For validation data with folds, compute averaged metrics and preserve folds
        val_data = types_dict.get("val")
        if val_data and "folds" in val_data:
            folds = val_data["folds"]
            averaged_val = average_metrics(folds)
            val_data.update(averaged_val)

        result.append(
            {
                "model": model_name,
                "params": params,
                "train": types_dict.get("train"),
                "val": types_dict.get("val"),
                "test": types_dict.get("test"),
            }
        )

    return result


def sort_aggregated_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort models by base model and embedding type."""

    def parse_model(name: str) -> tuple[str, str | None]:
        parts = name.split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return parts[0], None

    def sort_key(entry: dict[str, Any]) -> tuple[int, int, str]:
        name = entry["model"] or ""
        base, emb = parse_model(name)

        base_idx = MODEL_ORDER.index(base) if base in MODEL_ORDER else len(MODEL_ORDER)
        emb_idx = (
            EMBED_ORDER.index(emb) if emb and emb in EMBED_ORDER else len(EMBED_ORDER)
        )
        return (base_idx, emb_idx, name)

    return sorted(models, key=sort_key)


def get_metric_value(
    model_data: dict[str, Any], dataset: Literal["train", "val", "test"], metric: str
) -> Any:
    """Extract a specific metric value from aggregated model data."""
    data = model_data.get(dataset)
    if data is None:
        return None
    return data.get(metric)


def compute_macro_f1(f1_list: list[float]) -> float:
    """Compute macro F1 score from per-label F1 scores."""
    if not f1_list:
        return 0.0
    return float(np.mean(f1_list))


def compute_rmse(mse: float) -> float:
    """Compute RMSE from MSE."""
    return float(np.sqrt(mse))


def print_model_summary(models: list[dict[str, Any]]) -> None:
    """Print a summary of all aggregated models."""
    print("\n" + "=" * 80)
    print("AGGREGATED MODEL METRICS SUMMARY")
    print("=" * 80)

    for i, model in enumerate(models):
        print(f"\n[{i}] {model['model']}")
        print(f"    Parameters: {model['params']}")

        for dataset in ["train", "val", "test"]:
            data = model.get(dataset)
            if data is None:
                continue

            print(f"\n    {dataset.upper()}:")
            print(f"      MSE: {data.get('mse', 'N/A'):.4f}")
            print(f"      RMSE: {compute_rmse(data.get('mse', 0)):.4f}")
            print(f"      Accuracy: {data.get('accuracy', 'N/A'):.4f}")

            f1 = data.get("f1", [])
            if f1:
                macro_f1 = compute_macro_f1(f1)
                print(f"      Macro-F1: {macro_f1:.4f}")
                print(f"      Per-label F1: {[f'{v:.4f}' for v in f1]}")

            precision = data.get("precision", [])
            if precision:
                print(f"      Per-label Precision: {[f'{v:.4f}' for v in precision]}")

            recall = data.get("recall", [])
            if recall:
                print(f"      Per-label Recall: {[f'{v:.4f}' for v in recall]}")

            support = data.get("support", [])
            if support:
                print(f"      Per-label Support: {support}")


def print_training_curves_info() -> None:
    """Print information about available training curve files."""
    cv_files = load_cv_history_files()
    train_files = load_train_history_files()

    print("\n" + "=" * 80)
    print("TRAINING CURVES AVAILABILITY")
    print("=" * 80)

    print(f"\nCross-Validation files: {len(cv_files)}")
    for i, f in enumerate(cv_files[:5]):
        print(f"  [{i}] {f.name}")
    if len(cv_files) > 5:
        print(f"  ... and {len(cv_files) - 5} more")

    print(f"\nTraining files: {len(train_files)}")
    for i, f in enumerate(train_files[:5]):
        print(f"  [{i}] {f.name}")
    if len(train_files) > 5:
        print(f"  ... and {len(train_files) - 5} more")


def load_cv_data(file_path: Path) -> dict[str, Any] | None:
    """Load CV data from a file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if "folds" not in data or not data["folds"]:
            return None
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def best_val_epochs_for_file(file_path: Path) -> dict[str, Any] | None:
    """Compute best validation epoch and loss per-fold and averaged from a CV file.

    Expects CV format per metric-docs.md with folds containing 'val_losses' (list)
    for finetuned-mbert models, or 'mse' (scalar) for other models.

    Returns a dict with:
    {
      'file': str,
      'per_fold': [{'fold': int, 'best_epoch': int, 'best_loss': float}],
      'averaged': {'best_epoch': int, 'best_loss': float} | None
    }
    """
    data = load_cv_data(file_path)
    if not data:
        return None

    folds = data.get("folds", [])
    if not isinstance(folds, list) or not folds:
        return None

    per_fold = []
    epoch_series: list[list[float]] = []

    for idx, fold in enumerate(folds):
        if not isinstance(fold, dict):
            continue

        # Check for val_losses (finetuned-mbert) or mse (other models)
        val_losses = fold.get("val_losses")
        if val_losses and isinstance(val_losses, list):
            # Per-epoch validation losses
            best_epoch = int(np.argmin(val_losses))
            best_loss = float(val_losses[best_epoch])
            per_fold.append(
                {"fold": idx, "best_epoch": best_epoch, "best_loss": best_loss}
            )
            epoch_series.append(val_losses)
        elif "mse" in fold:
            # Single MSE value (no per-epoch data)
            mse = float(fold["mse"])
            per_fold.append({"fold": idx, "best_epoch": 0, "best_loss": mse})
            epoch_series.append([mse])

    averaged: dict[str, Any] | None = None
    if epoch_series and all(len(s) == len(epoch_series[0]) for s in epoch_series):
        # Compute mean curve across folds
        arr = np.array(epoch_series)
        mean_curve = arr.mean(axis=0)
        avg_best_epoch = int(np.argmin(mean_curve))
        avg_best_loss = float(mean_curve[avg_best_epoch])
        averaged = {"best_epoch": avg_best_epoch, "best_loss": avg_best_loss}
    elif per_fold:
        # Fallback: average the best losses without consistent epoch curves
        avg_best_loss = float(np.mean([pf["best_loss"] for pf in per_fold]))
        averaged = {"best_epoch": -1, "best_loss": avg_best_loss}

    return {"file": file_path.name, "per_fold": per_fold, "averaged": averaged}


def print_best_val_epochs(files: list[Path]) -> None:
    """Print best validation epoch/loss for given CV files (per-fold and averaged)."""
    if not files:
        print("No validation files found.")
        return

    print("\n" + "=" * 80)
    print("BEST VALIDATION EPOCHS")
    print("=" * 80)
    for fp in files:
        result = best_val_epochs_for_file(fp)
        if not result:
            print(f"{fp.name}: No usable validation data")
            continue

        print(f"\nFile: {result['file']}")
        print("Per-fold:")
        for pf in result["per_fold"]:
            print(
                f"  Fold {pf['fold']}: epoch={pf['best_epoch']} loss={pf['best_loss']:.6f}"
            )
        if result["averaged"]:
            avg = result["averaged"]
            epoch_str = str(avg["best_epoch"]) if avg["best_epoch"] != -1 else "n/a"
            print(f"Averaged: epoch={epoch_str} loss={avg['best_loss']:.6f}")


def _compute_epoch_x_values(
    y_data: List[float], epoch_batch_counts: Optional[List[int]]
) -> Tuple[List[float], str]:
    if not epoch_batch_counts:
        return list(range(1, len(y_data) + 1)), "Step"

    x_values = []
    current_epoch = 0
    steps_in_epoch = epoch_batch_counts[0] if epoch_batch_counts else 1
    steps_done = 0

    for _ in y_data:
        steps_done += 1
        x_values.append(current_epoch + steps_done / steps_in_epoch)
        if steps_done >= steps_in_epoch:
            current_epoch += 1
            steps_done = 0
            if current_epoch < len(epoch_batch_counts):
                steps_in_epoch = epoch_batch_counts[current_epoch]

    return x_values, "Epoch"


def plot_train_val_runs(
    runs: List[Dict[str, Any]],
    title: str,
    x_label: str = "Step",
    y_label: str = "Loss",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot one or more training/validation loss curves on a single axis.

    Each item in runs can contain:
      - label: name of the run
      - train: {"y": list[float], "x": list[float] | None, "mark_best": bool}
      - val: {"y": list[float], "x": list[float] | None, "mark_best": bool}
    """
    if not runs:
        print("No runs provided for plotting")
        return ax if ax is not None else None

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette(CMAP_QUALITATIVE_PRIMARY, 6)

    max_x = 0

    for idx, run in enumerate(runs):
        label = run.get("label", f"Run {idx + 1}")

        def _plot_series(series: Optional[Dict[str, Any]], kind: str) -> None:
            nonlocal max_x
            if not series:
                return
            y = series.get("y") or []
            if len(y) == 0:
                return
            x = series.get("x") or list(range(1, len(y) + 1))

            max_x = max(max_x, max(x) if x else 0)

            color = colors[idx + (0 if kind == "train" else 3)]
            marker = "o" if kind == "train" else "s"
            linestyle = "-" if kind == "train" else "--"
            alpha = 0.85 if kind == "train" else 0.7

            # For training data with many points, reduce marker clutter
            if kind == "train" and len(x) > 100:
                marker = None  # No markers for dense training curves

            sns.lineplot(
                x=x,
                y=y,
                ax=ax,
                color=color,
                linewidth=2,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=f"{kind.title()} - {'Small+Large' if label.endswith('_lg') else 'Small'}",
                alpha=alpha,
            )

            if series.get("mark_best", True):
                best_idx = int(np.argmin(y))
                best_x = x[best_idx] if best_idx < len(x) else best_idx + 1
                best_y = y[best_idx]
                ax.axvline(best_x, color=color, linestyle="--", alpha=0.2, linewidth=1)
                sns.scatterplot(
                    x=[best_x],
                    y=[best_y],
                    ax=ax,
                    color=color,
                    marker="*",
                    s=100,
                    zorder=5,
                    legend=False,
                )

        _plot_series(run.get("train"), "train")
        _plot_series(run.get("val"), "val")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)

    if x_label == "Epoch":
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    plt.tight_layout()
    if fig is not None:
        fig.savefig(
            DATA_DIR / "figures" / "finetuned-mbert_training-curves.pdf",
            bbox_inches="tight",
        )
    return ax


def plot_cv_folds_grid(
    cv_data_list: List[Dict[str, Any]],
    title: str = "Cross-validation runs",
    folds: Optional[List[int]] = None,
) -> None:
    """
    Visualize training/validation curves for each fold across one or more CV runs.

    Args:
        folds: 1-based fold indices to plot. If None, all folds are plotted.
    """
    if not cv_data_list:
        print("No CV metric files provided")
        return

    max_folds = max(len(cv.get("folds", [])) for cv in cv_data_list)
    if max_folds == 0:
        print("No folds found in provided CV data")
        return

    # Determine which fold indices (0-based) to plot
    all_fold_indices = list(range(max_folds))
    if folds is not None:
        fold_indices = [i for i in all_fold_indices if (i + 1) in folds]
        if not fold_indices:
            print(
                f"None of the requested folds {folds} found (available: 1–{max_folds})"
            )
            return
    else:
        fold_indices = all_fold_indices

    n_plots = len(fold_indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten() if n_plots > 1 else np.array([axes])

    for plot_pos, fold_idx in enumerate(fold_indices):
        ax = axes[plot_pos]
        runs: List[Dict[str, Any]] = []
        x_label = "Step"

        for cv_idx, cv_data in enumerate(cv_data_list):
            cv_folds = cv_data.get("folds", [])
            if fold_idx >= len(cv_folds):
                continue

            fold = cv_folds[fold_idx]
            run_label = cv_data.get("model", f"CV {cv_idx + 1}")
            train_losses = fold.get("train_losses") or []
            val_losses = fold.get("val_losses") or []
            epoch_batch_counts = fold.get("epoch_batch_counts") or []

            train_x, lbl = _compute_epoch_x_values(train_losses, epoch_batch_counts)
            if lbl == "Epoch":
                x_label = "Epoch"

            val_positions = list(range(1, len(val_losses) + 1))

            runs.append(
                {
                    "label": run_label,
                    "train": {"y": train_losses, "x": train_x, "mark_best": True},
                    "val": {"y": val_losses, "x": val_positions, "mark_best": True},
                }
            )

        if runs:
            plot_train_val_runs(
                runs,
                title=f"Fold {fold_idx + 1}",
                x_label=x_label,
                ax=ax,
            )
        else:
            ax.set_title(f"Fold {fold_idx + 1} (no data)", fontweight="bold")
            ax.axis("off")

    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    # fig.suptitle(
    #     title,
    #     fontsize=SUPTITLE_FONT_SIZE,
    #     fontweight="bold",
    #     y=1.02,
    # )
    plt.tight_layout()
    fig.savefig(
        DATA_DIR / "figures" / "finetuned-mbert_cv-curves.pdf",
        bbox_inches="tight",
    )
    plt.show()


def load_train_data(file_path: Path) -> dict[str, Any] | None:
    """Load training data from a file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if "train_losses" not in data:
            return None
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def visualize_metric_files(
    files: List[Path],
    folds: Optional[List[int]] = None,
) -> None:
    """
    Visualize 1..n metric files (train or val).

    - Train files: plot train + val curves on one figure.
    - Val files: plot per-fold train + val curves on one figure.
    - Mixed selection: produce both figures.

    Args:
        folds: 1-based fold indices to plot for CV files. If None, all folds are plotted.
    """
    if not files:
        print("No files selected")
        return

    apply_plot_style()

    train_runs: List[Dict[str, Any]] = []
    cv_runs: List[Dict[str, Any]] = []

    final_x_label = "Step"

    for file_path in files:
        cv_data = load_cv_data(file_path)
        if cv_data and cv_data.get("folds"):
            cv_runs.append({**cv_data, "source": file_path})
            continue

        train_data = load_train_data(file_path)
        if train_data:
            train_losses = train_data.get("train_losses") or []
            val_losses = train_data.get("val_losses") or []
            epoch_batch_counts = train_data.get("epoch_batch_counts") or []

            train_x, x_lbl = _compute_epoch_x_values(train_losses, epoch_batch_counts)
            if x_lbl == "Epoch":
                final_x_label = "Epoch"

            val_positions = list(range(1, len(val_losses) + 1))

            train_runs.append(
                {
                    "label": train_data.get("model", file_path.stem),
                    "train": {"y": train_losses, "x": train_x, "mark_best": True},
                    "val": {"y": val_losses, "x": val_positions, "mark_best": True},
                }
            )
        else:
            print(
                f"Warning: {file_path.name} was not recognized as train or val metrics"
            )

    if train_runs:
        print(f"Plotting {len(train_runs)} training file(s)...")
        plot_train_val_runs(
            train_runs,
            title="",
            x_label=final_x_label,
        )

    if cv_runs:
        print(f"Plotting {len(cv_runs)} validation file(s) across folds...")
        plot_cv_folds_grid(cv_runs, title="", folds=folds)

    if not train_runs and not cv_runs:
        print("Nothing to visualize after loading files")


def load_catboost_learning_curves(catboost_dir: Path) -> Dict[str, np.ndarray]:
    learn_path = catboost_dir / "learn_error.tsv"
    test_path = catboost_dir / "test_error.tsv"
    curves: Dict[str, np.ndarray] = {}
    if learn_path.exists():
        df_learn = pd.read_csv(learn_path, sep="\t", header=None)
        curves["learn_error"] = df_learn.iloc[:, -1].to_numpy(dtype=float)
    if test_path.exists():
        df_test = pd.read_csv(test_path, sep="\t", header=None)
        curves["test_error"] = df_test.iloc[:, -1].to_numpy(dtype=float)
    if not curves:
        print(f"No CatBoost curves found in {catboost_dir}")
    return curves


def print_comparison_table(models: list[dict[str, Any]]) -> None:
    """Print a comparison table of key metrics across all models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(
        f"{'Model':<30} {'Train RMSE':<12} {'CV RMSE':<12} {'Test RMSE':<12} "
        f"{'Train Acc':<10} {'CV Acc':<10} {'Test Acc':<10}"
    )
    print("-" * 110)

    for model in models:
        name = model["model"][:29]

        def safe_metric(dataset: str, metric: str, transform=None) -> str:
            val = get_metric_value(model, dataset, metric)  # type: ignore
            if val is None:
                return "N/A".rjust(10)
            if transform:
                val = transform(val)
            return f"{val:.4f}"

        train_rmse = safe_metric("train", "mse", compute_rmse)
        cv_rmse = safe_metric("val", "mse", compute_rmse)
        test_rmse = safe_metric("test", "mse", compute_rmse)
        train_acc = safe_metric("train", "accuracy")
        cv_acc = safe_metric("val", "accuracy")
        test_acc = safe_metric("test", "accuracy")

        print(
            f"{name:<30} {train_rmse:<12} {cv_rmse:<12} {test_rmse:<12} "
            f"{train_acc:<10} {cv_acc:<10} {test_acc:<10}"
        )


def print_confusion_matrix_stats(
    metrics_dict: dict[str, Any], dataset: str = "test"
) -> None:
    """Print confusion matrix statistics for a specific dataset."""
    cm = get_confusion_matrix(metrics_dict, dataset)
    if cm is None:
        print(f"No {dataset} confusion matrix available")
        return

    print(f"\n{dataset.upper()} - Confusion Matrix:")
    print("-" * 60)

    # Print matrix header
    print("     ", end="")
    for i in range(6):
        print(f"Pred{i:1d} ", end="")
    print()

    # Print matrix rows
    for i in range(6):
        print(f"True{i}: ", end="")
        for j in range(6):
            print(f"{cm[i, j]:5d} ", end="")
        print()

    print("-" * 60)

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0

    print(f"Total predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nPer-label accuracy:")
    for i in range(6):
        true_count = cm[i, :].sum()
        correct_count = cm[i, i]
        if true_count > 0:
            per_label_acc = correct_count / true_count
            print(f"  Label {i}: {correct_count}/{true_count} ({per_label_acc:.2%})")
        else:
            print(f"  Label {i}: No samples")


def plot_feature_importance_bars(df: "pd.DataFrame") -> None:
    """
    Plot horizontal bar charts showing the accuracy, RMSE, and Macro F1 impact
    of removing each feature group from the model.

    Args:
        df: DataFrame with columns including "Removed Feature Group", "Accuracy Δ",
            "RMSE Δ", and optionally "Macro F1 Δ".
    """
    plot_df = df.copy()
    plot_df["Feature"] = plot_df["Removed Feature Group"]

    def _interp_color(a, b, t):
        return tuple(float(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def _importance_colors(
        deltas,
        important_mask,
        base_gray=(0.92, 0.92, 0.92),
        green=(0.0, 0.5, 0.0),
        dark_gray=(0.35, 0.35, 0.35),
    ):
        abs_imp = np.abs(deltas.astype(float))
        max_imp = abs_imp.max()
        if max_imp <= 0:
            max_imp = 1.0
        colors = []
        for val, imp_flag, mag in zip(deltas, important_mask, abs_imp):
            t = mag / max_imp
            if imp_flag:
                colors.append(_interp_color(base_gray, green, t))
            else:
                colors.append(_interp_color(base_gray, dark_gray, t))
        return colors

    # acc_sorted = plot_df.sort_values("Accuracy Δ", ascending=True)
    # acc_deltas = acc_sorted["Accuracy Δ"]
    # acc_mask = acc_deltas < 0
    # colors_acc = _importance_colors(acc_deltas, acc_mask)
    # fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(acc_sorted))))
    # sns.barplot(
    #     data=acc_sorted,
    #     y="Feature",
    #     x="Accuracy Δ",
    #     palette=colors_acc,
    #     alpha=0.95,
    #     edgecolor="black",
    #     errorbar=None,
    #     orient="h",
    #     ax=ax,
    # )
    # ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    # ax.set_xlabel("Accuracy Δ from Baseline", fontsize=LABEL_FONT_SIZE)
    # ax.set_ylabel("Removed Feature Group", fontsize=LABEL_FONT_SIZE)
    # ax.set_title(
    #     "Accuracy Impact: green = important (negative Δ)", fontsize=TITLE_FONT_SIZE
    # )
    # ax.grid(axis="x", alpha=0.3)
    # plt.tight_layout()
    # fig.savefig(
    #     DATA_DIR / "figures" / "feature_importance_accuracy_delta.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    # plt.show()
    # print(
    #     "Saved accuracy delta plot -> data/figures/feature_importance_accuracy_delta.png"
    # )

    rmse_sorted = plot_df.sort_values("RMSE Δ", ascending=False)
    rmse_deltas = rmse_sorted["RMSE Δ"]
    rmse_mask = rmse_deltas > 0
    colors_rmse = _importance_colors(rmse_deltas, rmse_mask)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(rmse_sorted))))
    sns.barplot(
        data=rmse_sorted,
        y="Feature",
        x="RMSE Δ",
        palette=colors_rmse,
        alpha=0.95,
        edgecolor="black",
        errorbar=None,
        orient="h",
        ax=ax,
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("RMSE Δ from Baseline")
    ax.set_ylabel("Removed Feature Group")
    ax.set_title("RMSE Impact: green = important (positive Δ)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        DATA_DIR / "figures" / "feature_importance_rmse_delta.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    print("Saved RMSE delta plot -> data/figures/feature_importance_rmse_delta.pdf")

    if "Macro F1 Δ" not in plot_df.columns:
        print("Warning: 'Macro F1 Δ' column not found, skipping Macro F1 plot.")
        return
    macro_sorted = plot_df.sort_values("Macro F1 Δ")
    macro_deltas = macro_sorted["Macro F1 Δ"]
    macro_mask = macro_deltas < 0
    colors_macro = _importance_colors(macro_deltas, macro_mask)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(macro_sorted))))
    sns.barplot(
        data=macro_sorted,
        y="Feature",
        x="Macro F1 Δ",
        palette=colors_macro,
        alpha=0.95,
        edgecolor="black",
        errorbar=None,
        orient="h",
        ax=ax,
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Macro F1 Δ from Baseline")
    ax.set_ylabel("Removed Feature Group")
    ax.set_title("Macro F1 Impact: green = important (negative Δ)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        DATA_DIR / "figures" / "feature_importance_macro_f1_delta.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    print(
        "Saved Macro F1 delta plot -> data/figures/feature_importance_macro_f1_delta.pdf"
    )


def export_feature_importance_latex(
    df: "pd.DataFrame",
    output_path: "Path | None" = None,
) -> str:
    """Build a gradient-coloured LaTeX table for feature importance and save it.

    Gradient columns:
    - "RMSE Δ": RdYlGn colourmap (positive = worse, red; centred at 0)
    - "Accuracy Δ": RdYlGn_r colourmap (positive = better, red; centred at 0)

    Args:
        df: DataFrame produced by the feature-importance analysis cell.
            Must contain at least "Removed Feature Group", "RMSE Δ", "Accuracy Δ".
        output_path: Destination .tex file path. Defaults to
            DATA_DIR / "figures" / "feature_importance_table.tex".

    Returns:
        The generated LaTeX string (also written to *output_path*).
    """
    from pathlib import Path as _Path
    import matplotlib.pyplot as _plt

    if output_path is None:
        output_path = DATA_DIR / "figures" / "feature_importance_table.tex"

    cmap_rdylgn = _plt.get_cmap("RdYlGn")
    cmap_rdylgn_r = _plt.get_cmap("RdYlGn_r")

    def _color_for_value(value, col_name, vmin, vmax):
        if pd.isna(value) or vmin == vmax:
            return None
        norm_value = (value - vmin) / (vmax - vmin)
        if col_name == "RMSE Δ":
            rgba = cmap_rdylgn(norm_value)
        elif col_name == "Accuracy Δ":
            rgba = cmap_rdylgn_r(norm_value)
        else:
            return None
        return rgba[:3]

    latex_lines = []
    latex_lines.append(r"\begin{table}[hbtp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}")
    latex_lines.append(r"\toprule")

    def _col_header(col: str) -> str:
        col_tex = col.replace("Δ", r"$\Delta$")
        bold = r"\textbf{" + col_tex + "}"
        cu = col.upper()
        if "RMSE" in cu and "\Delta" not in col_tex:
            return bold + r" $\downarrow$"
        if "\Delta" not in col_tex and ("ACCURACY" in cu or "F1" in cu):
            return bold + r" $\uparrow$"
        return bold

    header = " & ".join([_col_header(col) for col in df.columns]) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")

    gradient_info = {}
    for col in ["RMSE Δ", "Accuracy Δ"]:
        if col in df.columns:
            abs_max = max(abs(df[col].min()), abs(df[col].max()))
            gradient_info[col] = {"vmin": -abs_max, "vmax": abs_max}

    # Pre-compute best row index per numeric metric column for bold highlighting.
    _best_row: dict[str, int] = {}
    for col in df.columns:
        if col == "Removed Feature Group":
            continue
        cu = col.upper()
        if "RMSE" in cu or "ACCURACY" in cu or "F1" in cu:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                _best_row[col] = int(
                    series.idxmin() if "RMSE" in cu else series.idxmax()
                )

    for row_label, row in df.iterrows():
        cells = []
        for col, val in row.items():
            if col == "Removed Feature Group":
                cell_str = str(val).replace("–", r"--")
                if "Baseline" in str(val):
                    cell_str = r"\textbf{" + cell_str + "}"
            elif isinstance(val, (int, float)) and not pd.isna(val):
                cell_str = f"{val:.4f}"
                if col in gradient_info:
                    rgb = _color_for_value(
                        val, col, gradient_info[col]["vmin"], gradient_info[col]["vmax"]
                    )
                    if rgb:
                        r, g, b = rgb
                        cell_str = (
                            r"\cellcolor[rgb]{"
                            + f"{r:.3f},{g:.3f},{b:.3f}"
                            + "}"
                            + cell_str
                        )
                if col in _best_row and row_label == _best_row[col]:
                    if cell_str.startswith(r"\cellcolor"):
                        end = cell_str.index("}") + 1
                        cell_str = cell_str[:end] + r"\textbf{" + cell_str[end:] + "}"
                    else:
                        cell_str = r"\textbf{" + cell_str + "}"
            else:
                cell_str = str(val)
            cells.append(cell_str)
        latex_lines.append(" & ".join(cells) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{Feature Importances}")
    latex_lines.append(r"\label{tab:feature-importances}")
    latex_lines.append(r"\end{table}")

    tex = "\n".join(latex_lines)
    _Path(output_path).write_text(tex)
    print(tex)
    print(f"\nSaved LaTeX table -> {output_path}")
    print("\nNote: Add these packages to your LaTeX preamble:")
    print(r"  \usepackage{booktabs}")
    print(r"  \usepackage[table]{xcolor}")
    return tex


def main():
    """CLI entrypoint for aggregating and inspecting encoder metrics."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="metrics-aggregator",
        description=(
            "Aggregate and inspect encoder metrics: summaries, comparison tables, "
            "per-label stats, confusion matrices, training curves, and best CV epochs."
        ),
    )

    sub = parser.add_subparsers(dest="command")

    # Default aggregate summary
    agg = sub.add_parser(
        "aggregate",
        help="Aggregate metrics across seeds and print summaries + comparison table",
    )
    agg.add_argument(
        "--show-details-for",
        type=str,
        default=None,
        help="Model name to show per-label metrics and confusion matrices for",
    )

    # Training curves listing
    sub.add_parser("curves", help="List available CV and train curve files")

    # Best validation epochs
    best = sub.add_parser(
        "best-val",
        help="Show best validation epoch/loss per-fold and averaged from CV files",
    )
    best.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring filter for CV filenames (e.g., 'svm_minilm')",
    )

    args = parser.parse_args()

    if args.command in (None, "aggregate"):
        print("Loading and aggregating metrics...")
        models = aggregate_metrics_by_model()
        if not models:
            print("No metrics found!")
            return
        models = sort_aggregated_models(models)
        print(f"Found {len(models)} model configurations")
        print_model_summary(models)
        print_comparison_table(models)

        # Optional detailed view for a specific model
        target_model = getattr(args, "show-details-for", None)
        chosen = None
        if target_model:
            for m in models:
                if m.get("model") == target_model:
                    chosen = m
                    break
        else:
            # Default to first model with test data
            test_models = [m for m in models if m.get("test") is not None]
            chosen = test_models[0] if test_models else None

        if chosen:
            print("\n" + "=" * 80)
            print(f"DETAILED METRICS FOR: {chosen['model']}")
            print("=" * 80)
            for dataset in ["train", "val", "test"]:
                if chosen.get(dataset) is not None:
                    print_per_label_metrics(chosen, dataset)
                    print_confusion_matrix_stats(chosen, dataset)

        print_training_curves_info()
        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        return

    if args.command == "curves":
        print_training_curves_info()
        return

    if args.command == "best-val":
        files = load_cv_history_files()
        if args.filter:
            files = [f for f in files if args.filter in f.name]
        print_best_val_epochs(files)
        return


if __name__ == "__main__":
    main()
