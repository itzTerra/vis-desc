import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Literal, Iterable
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils import DATA_DIR
from models.encoder.common import (
    parse_metrics_filename,
    group_metric_files,
    average_metric_files,
    average_metrics,
)

METRICS_DIR = DATA_DIR / "metrics" / "encoder"

MODEL_ORDER = ["random", "ridge", "rf", "svm", "catboost", "finetuned-mbert"]
EMBED_ORDER = ["minilm", "mbert"]


def load_all_metric_files() -> list[Path]:
    """Load all metric JSON files from the metrics directory."""
    return sorted(list(METRICS_DIR.glob("*.json")))


def aggregate_metrics_by_model() -> list[dict[str, Any]]:
    """
    Aggregate metrics across all seeds for each model/embedding/lg/type combination.

    Returns:
        List of dicts with structure:
        {
            'model': str,  # e.g., "svm_minilm_lg"
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


def extract_per_label_metrics(
    metrics_dict: dict[str, Any], dataset: str = "test"
) -> pd.DataFrame | None:
    """
    Extract precision, recall, and F1 scores for labels 0-5 from aggregated model data.

    Args:
        metrics_dict: Aggregated model metrics dict with keys 'train','val','test'.
        dataset: Which dataset to use ('train', 'val', or 'test').

    Returns:
        DataFrame with precision, recall, F1, support per label or None if missing.
    """
    if metrics_dict.get(dataset) is None:
        return None

    data = metrics_dict[dataset]

    precision = data.get("precision", []) + [0.0] * (6 - len(data.get("precision", [])))
    recall = data.get("recall", []) + [0.0] * (6 - len(data.get("recall", [])))
    f1 = data.get("f1", []) + [0.0] * (6 - len(data.get("f1", [])))
    support = data.get("support", []) + [0] * (6 - len(data.get("support", [])))

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
    metrics_dict: dict[str, Any], dataset: str = "test"
) -> np.ndarray | None:
    """
    Extract confusion matrix from aggregated model data.

    Args:
        metrics_dict: Aggregated model metrics dict.
        dataset: Which dataset to use ('train', 'val', or 'test').

    Returns:
        6x6 confusion matrix as numpy array or None if missing.
    """
    if metrics_dict.get(dataset) is None:
        return None

    data = metrics_dict[dataset]
    cm = np.array(data.get("confusion_matrix", []))

    # Pad to 6x6 if needed
    if cm.shape[0] < 6:
        padded_cm = np.zeros((6, 6), dtype=int)
        padded_cm[: cm.shape[0], : cm.shape[1]] = cm
        cm = padded_cm

    return cm


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


def load_cv_history_files() -> list[Path]:
    """Load all validation (CV) metric files with training curves."""
    return sorted(list(METRICS_DIR.glob("finetuned*_val_*.json")))


def load_train_history_files() -> list[Path]:
    """Load all train metric files with training curves."""
    return sorted(list(METRICS_DIR.glob("finetuned*_train_*.json")))


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


def print_per_label_metrics(
    metrics_dict: dict[str, Any], dataset: str = "test"
) -> None:
    """Print per-label metrics table for a specific dataset."""
    df = extract_per_label_metrics(metrics_dict, dataset)
    if df is None:
        print(f"No {dataset} data available")
        return

    print(f"\n{dataset.upper()} - Per-Label Metrics:")
    print("-" * 60)
    print(f"{'Label':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(
            f"{int(row['Label']):<8} {row['Precision']:<12.4f} {row['Recall']:<12.4f} "
            f"{row['F1']:<12.4f} {int(row['Support']):<10}"
        )
    print("-" * 60)

    data = metrics_dict[dataset]
    print(f"Accuracy: {data['accuracy']:.4f}")
    print(f"MSE: {data['mse']:.4f}")
    print(f"RMSE: {compute_rmse(data['mse']):.4f}")
    f1_scores = data.get("f1", [])
    if f1_scores:
        print(f"Macro-F1: {compute_macro_f1(f1_scores):.4f}")


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


def _find_first_list(d: dict[str, Any], keys: Iterable[str]) -> list[float] | None:
    """Return the first list of floats found under any of the provided keys."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return [float(x) for x in v]
    return None


def best_val_epochs_for_file(file_path: Path) -> dict[str, Any] | None:
    """Compute best validation epoch and loss per-fold and averaged from a CV file.

    The function tries common keys to locate validation losses within each fold:
    one of ['val_losses', 'val_loss', 'cv_losses', 'mse', 'losses']. If per-epoch
    series are available for all folds, it also computes an averaged per-epoch
    curve and its best epoch.

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
        # Each fold may be a dict or nested under 'history'
        fold_dict = fold if isinstance(fold, dict) else {}
        if "history" in fold_dict and isinstance(fold_dict["history"], dict):
            fold_dict = fold_dict["history"]

        losses = _find_first_list(
            fold_dict, ["val_losses", "val_loss", "cv_losses", "mse", "losses"]
        )
        if not losses:
            # If only a scalar loss is present, treat as single epoch
            scalar = fold_dict.get("val_mse") or fold_dict.get("mse")
            if isinstance(scalar, (int, float)):
                loss_list = [float(scalar)]
            else:
                loss_list = []
        else:
            loss_list = losses

        if loss_list:
            best_epoch = int(np.argmin(loss_list))
            best_loss = float(loss_list[best_epoch])
            per_fold.append(
                {"fold": idx, "best_epoch": best_epoch, "best_loss": best_loss}
            )
            epoch_series.append(loss_list)

    averaged: dict[str, Any] | None = None
    if epoch_series and all(len(s) == len(epoch_series[0]) for s in epoch_series):
        # Compute mean curve across folds
        arr = np.array(epoch_series)
        mean_curve = arr.mean(axis=0)
        avg_best_epoch = int(np.argmin(mean_curve))
        avg_best_loss = float(mean_curve[avg_best_epoch])
        averaged = {"best_epoch": avg_best_epoch, "best_loss": avg_best_loss}
    elif per_fold:
        # Fallback: average the best losses without an epoch curve
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
