from __future__ import annotations

from typing import List, Tuple, Sequence, Protocol, Any, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
import json

from evaluation.core import format_number, latex_escape

MODEL_NAME_MAP = {
    "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX": "RoBERTa",
    "onnx-community/ModernBERT-large-zeroshot-v2.0-ONNX": "MBERT-L",
    "richardr1126/deberta-v3-large-zeroshot-v2.0-ONNX": "DeBERTa-L",
}


@dataclass
class MetricEntry:
    model_name: str
    dataset_split: str  # e.g., 'train', 'val', 'test'
    metrics: Dict[str, float]
    extra: Optional[Dict[str, float]] = None


@dataclass
class ModelScoreData:
    config_id: int
    model_name: str
    train_scores: np.ndarray
    y_train: np.ndarray
    test_scores: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None


def load_metric_files(files: List[Path]) -> List[Dict[str, Any]]:
    data = []
    for i, fp in enumerate(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            obj["id"] = i + 1
            data.append(obj)
        except Exception as e:
            print(f"Skip {fp}: {e}")
    return data


def to_model_score_data(
    items: List[Dict[str, Any]],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
) -> List[ModelScoreData]:
    results: List[ModelScoreData] = []
    y_train = df_train["label"].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy(dtype=float) if df_test is not None else None
    for file_idx, item in enumerate(items):
        models = item.get("models", [])
        # Group entries by model_name and split by dataset where available
        grouped: Dict[str, Dict[str, Any]] = {}
        for m in models:
            name = m.get("model_name", "")
            if not name:
                continue
            ds = (m.get("dataset") or "train").lower()
            if name not in grouped:
                grouped[name] = {}
            grouped[name][ds] = m

        for model_name, splits in grouped.items():
            # Prefer in-file train/test
            train_entry = (
                splits.get("train")
                or splits.get("val")
                or splits.get("validation")
                or next(iter(splits.values()), None)
            )
            test_entry = splits.get("test")

            train_scores = np.asarray(
                (train_entry or {}).get("scores", []), dtype=float
            )
            test_scores = (
                np.asarray((test_entry or {}).get("scores", []), dtype=float)
                if test_entry is not None
                else None
            )

            results.append(
                ModelScoreData(
                    config_id=file_idx,
                    model_name=model_name,
                    train_scores=train_scores,
                    y_train=y_train,
                    test_scores=test_scores,
                    y_test=y_test,
                )
            )
    return results


class Calibrator(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class CalibrationMethod(Protocol):
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator: ...


class _Q2QCalibrator:
    def __init__(self, xs_sorted: np.ndarray, ys_arr: np.ndarray) -> None:
        self._xs_sorted = xs_sorted
        self._ys_arr = ys_arr
        self._n = xs_sorted.size

    def _ecdf(self, x: float) -> float:
        return np.searchsorted(self._xs_sorted, x, side="right") / self._n

    def predict(self, x_arr: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_arr, dtype=float)
        q = np.array([self._ecdf(x) for x in x_arr])
        return np.quantile(self._ys_arr, np.clip(q, 0.0, 1.0))


class Q2QCalibration:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        xs_sorted = np.sort(np.asarray(scores, dtype=float))
        ys_arr = np.asarray(labels, dtype=float)
        return _Q2QCalibrator(xs_sorted, ys_arr)


class _IsotonicCalibrator:
    def __init__(self, iso) -> None:
        self._iso = iso

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._iso.predict(x)


class IsotonicCalibration:
    def __init__(self, out_of_bounds: str = "clip") -> None:
        self._out_of_bounds = out_of_bounds

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds=self._out_of_bounds)
        iso.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
        return _IsotonicCalibrator(iso)


class _OrdinalLogisticCalibrator:
    def __init__(self, thresholds: np.ndarray, coef: np.ndarray) -> None:
        self._thresholds = thresholds
        self._coef = coef

    def predict(self, x: np.ndarray) -> np.ndarray:
        from scipy.special import expit

        x = np.asarray(x, dtype=float).reshape(-1, 1)
        z = x * self._coef[0]
        cumulative_probs = expit(self._thresholds - z)
        cumulative_probs = np.clip(cumulative_probs, 1e-10, 1 - 1e-10)

        class_probs = np.zeros((len(x), 6))
        class_probs[:, 0] = cumulative_probs[:, 0]
        for j in range(1, 5):
            class_probs[:, j] = cumulative_probs[:, j] - cumulative_probs[:, j - 1]
        class_probs[:, 5] = 1 - cumulative_probs[:, 4]

        predicted_values = np.sum(class_probs * np.arange(6), axis=1)
        return np.clip(predicted_values, 0, 5)


class OrdinalLogisticCalibration:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        from scipy.optimize import minimize
        from scipy.special import expit

        scores_array = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        n_classes = 6
        n_thresholds = n_classes - 1

        def loss_function(params):
            coef = params[0]
            thresholds = np.sort(params[1:])

            z = scores_array * coef
            cumulative_probs = expit(thresholds[:, np.newaxis] - z[np.newaxis, :])
            cumulative_probs = np.clip(cumulative_probs, 1e-10, 1 - 1e-10)

            loss = 0.0
            for i, label in enumerate(labels):
                label_int = int(np.clip(label, 0, n_classes - 1))
                if label_int == 0:
                    prob = cumulative_probs[0, i]
                elif label_int == n_classes - 1:
                    prob = 1 - cumulative_probs[-1, i]
                else:
                    prob = (
                        cumulative_probs[label_int, i]
                        - cumulative_probs[label_int - 1, i]
                    )
                loss -= np.log(prob + 1e-10)

            return loss

        initial_params = [1.0] + list(
            np.percentile(labels, np.linspace(0, 100, n_thresholds))
        )
        result = minimize(loss_function, initial_params, method="BFGS")

        coef = np.array([result.x[0]])
        thresholds = np.sort(result.x[1:])

        return _OrdinalLogisticCalibrator(thresholds, coef)


def to_int_0_5(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 5).astype(int)


def plot_correlation_matrix(
    score_vectors: Sequence[np.ndarray],
    names: Sequence[str],
    figsize: Tuple[int, int] = (8, 6),
    title: str | None = None,
) -> None:
    lengths = [np.asarray(vec).size for vec in score_vectors]
    if len(score_vectors) < 2:
        print("Need at least two models with score vectors to compute correlations")
        return
    min_len = min(lengths)
    if len(set(lengths)) > 1:
        score_vectors = [np.asarray(vec)[:min_len] for vec in score_vectors]
        print(
            f"Note: score vectors had different lengths; truncated to {min_len} to align."
        )
    scores_array = np.vstack([np.asarray(v, dtype=float) for v in score_vectors])
    corr_matrix = np.corrcoef(scores_array)
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation Coefficient"},
        xticklabels=names,
        yticklabels=names,
    )
    if title:
        plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()
    print("Correlation coefficients:")
    print(pd.DataFrame(corr_matrix, index=names, columns=names))


def correlation_vectors_from_score_data(
    score_data: Sequence[ModelScoreData],
    config_id: int,
    split: str = "train",
    name_map: Dict[str, str] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    name_map = name_map or MODEL_NAME_MAP
    split = split.lower()
    assert split in {"train", "test"}, "split must be 'train' or 'test'"
    vecs: list[np.ndarray] = []
    names: list[str] = []
    seen: set[str] = set()
    for sd in score_data:
        if sd.config_id != config_id:
            continue
        raw_name = sd.model_name
        if raw_name in seen:
            continue
        if split == "train":
            if sd.train_scores is None or len(sd.train_scores) == 0:
                continue
            vecs.append(np.asarray(sd.train_scores, dtype=float))
        else:
            if sd.test_scores is None or len(sd.test_scores) == 0:
                continue
            vecs.append(np.asarray(sd.test_scores, dtype=float))
        names.append(name_map.get(raw_name, raw_name))
        seen.add(raw_name)
    return vecs, names


def plot_train_test_correlation_from_score_data(
    score_data: Sequence[ModelScoreData],
    config_id: int,
    name_map: Dict[str, str] | None = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    name_map = name_map or MODEL_NAME_MAP
    train_vecs, train_names = correlation_vectors_from_score_data(
        score_data, config_id, split="train", name_map=name_map
    )
    if len(train_vecs) >= 2:
        plot_correlation_matrix(
            train_vecs, train_names, figsize=figsize, title="Correlation (Train)"
        )
    else:
        print("Not enough train vectors to compute correlation matrix.")

    test_vecs, test_names = correlation_vectors_from_score_data(
        score_data, config_id, split="test", name_map=name_map
    )
    if len(test_vecs) >= 2:
        plot_correlation_matrix(
            test_vecs, test_names, figsize=figsize, title="Correlation (Test)"
        )
    else:
        print(
            "Not enough test vectors to compute correlation matrix or no test scores available."
        )


def plot_calibration_comparison(
    train_scores_sorted: np.ndarray,
    y_train_sorted: np.ndarray,
    a_train_pred_sorted: np.ndarray,
    b_train_pred_sorted: np.ndarray,
    y_test_sorted: np.ndarray | None = None,
    a_test_pred_sorted: np.ndarray | None = None,
    b_test_pred_sorted: np.ndarray | None = None,
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].set_title(f"Train: scores and calibrated mappings{title_suffix}")
    axes[0].plot(
        y_train_sorted, label="True label (train)", color="black", linewidth=1.5
    )
    axes[0].plot(a_train_pred_sorted, label="Calibration A (train)")
    axes[0].plot(b_train_pred_sorted, label="Calibration B (train)")
    axes[0].set_ylabel("Score (0-5 scale)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    if (
        y_test_sorted is not None
        or a_test_pred_sorted is not None
        or b_test_pred_sorted is not None
    ):
        axes[1].set_title(
            f"Test: ground truth and calibrated predictions{title_suffix}"
        )
        if y_test_sorted is not None:
            axes[1].plot(
                y_test_sorted, label="True label (test)", color="black", linewidth=1.5
            )
        if a_test_pred_sorted is not None:
            axes[1].plot(a_test_pred_sorted, label="Calibration A (test)")
        if b_test_pred_sorted is not None:
            axes[1].plot(b_test_pred_sorted, label="Calibration B (test)")
        axes[1].set_ylabel("Score (0-5 scale)")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")
        print("No test metrics/labels available; showing train-only calibration plot.")
    axes[-1].set_xlabel("Sample index (sorted by true label)")
    plt.tight_layout()
    plt.show()


def run_cv_comparison(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    method_a: CalibrationMethod | None = None,
    method_b: CalibrationMethod | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, accuracy_score

    method_a = method_a or IsotonicCalibration()
    method_b = method_b or Q2QCalibration()
    a_rmses: List[float] = []
    a_accs: List[float] = []
    b_rmses: List[float] = []
    b_accs: List[float] = []
    min_len = min(len(train_scores), len(y_train))
    train_scores_cv = np.asarray(train_scores[:min_len], dtype=float)
    y_train_cv = np.asarray(y_train[:min_len], dtype=float)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kfold.split(train_scores_cv):
        X_train_fold = train_scores_cv[train_idx]
        y_train_fold = y_train_cv[train_idx]
        X_val_fold = train_scores_cv[val_idx]
        y_val_fold = y_train_cv[val_idx]
        a_cal = method_a.fit(X_train_fold, y_train_fold)
        b_cal = method_b.fit(X_train_fold, y_train_fold)
        a_pred = a_cal.predict(X_val_fold)
        b_pred = b_cal.predict(X_val_fold)
        a_rmse = np.sqrt(mean_squared_error(y_val_fold, a_pred))
        b_rmse = np.sqrt(mean_squared_error(y_val_fold, b_pred))
        a_acc = accuracy_score(to_int_0_5(y_val_fold), to_int_0_5(a_pred))
        b_acc = accuracy_score(to_int_0_5(y_val_fold), to_int_0_5(b_pred))
        a_rmses.append(a_rmse)
        a_accs.append(a_acc)
        b_rmses.append(b_rmse)
        b_accs.append(b_acc)
    results_df = pd.DataFrame(
        {
            "Method": ["Isotonic", "Q2Q"],
            "Mean RMSE": [np.mean(a_rmses), np.mean(b_rmses)],
            "Std RMSE": [np.std(a_rmses), np.std(b_rmses)],
            "Mean Accuracy": [np.mean(a_accs), np.mean(b_accs)],
            "Std Accuracy": [np.std(a_accs), np.std(b_accs)],
        }
    )
    return results_df


def plot_learning_curves(
    curves: dict[str, np.ndarray], title: str | None = None
) -> None:
    plt.figure(figsize=(10, 6))
    for name, series in curves.items():
        plt.plot(series, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Error / Loss")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _prettify_model_name(model_name: str) -> str:
    return MODEL_NAME_MAP.get(model_name, model_name)


def _compute_combined_correlation(
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
) -> float | None:
    if not scores_list or not labels_list:
        return None
    scores = np.concatenate([np.asarray(s, dtype=float) for s in scores_list])
    labels = np.concatenate([np.asarray(lab, dtype=float) for lab in labels_list])
    if len(scores) == 0 or len(labels) == 0 or len(scores) != len(labels):
        return None
    try:
        corr = np.corrcoef(scores, labels)[0, 1]
        return float(corr) if not np.isnan(corr) else None
    except Exception:
        return None


def _get_combined_corr_for_model_in_item(
    item: Dict[str, Any],
    model_name: str,
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> float | None:
    models = [m for m in item.get("models", []) if m.get("model_name") == model_name]
    if not models:
        return None

    scores_list = []
    labels_list = []

    for m in models:
        scores = m.get("scores")
        if not scores:
            continue
        dataset_name = (m.get("dataset") or "train").lower()

        if dataset_name in ("train", "val", "validation"):
            if df_train is not None and "label" in df_train.columns:
                labels = df_train["label"].to_numpy(dtype=float)
                if len(labels) == len(scores):
                    scores_list.append(scores)
                    labels_list.append(labels)
        elif dataset_name == "test":
            if df_test is not None and "label" in df_test.columns:
                labels = df_test["label"].to_numpy(dtype=float)
                if len(labels) == len(scores):
                    scores_list.append(scores)
                    labels_list.append(labels)

    return _compute_combined_correlation(scores_list, labels_list)


def build_table_a(
    items: List[Dict[str, Any]],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for it in items:
        models = it.get("models", [])
        best_corr = None

        seen_model_names = set()
        for m in models:
            model_name = m.get("model_name")
            if not model_name or model_name in seen_model_names:
                continue
            seen_model_names.add(model_name)

            if df_train is not None or df_test is not None:
                c = _get_combined_corr_for_model_in_item(
                    it, model_name, df_train, df_test
                )
            else:
                c = m.get("corr")

            if c is None:
                continue
            best_corr = c if best_corr is None else max(best_corr, c)

        rows.append(
            {
                "id": it.get("id"),
                "hypothesis_template": it.get("hypothesis_template", ""),
                "candidate_labels": ", ".join(it.get("candidate_labels", [])),
                "best_corr": best_corr,
            }
        )
    return pd.DataFrame(rows)


def build_table_b(
    items: List[Dict[str, Any]],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> pd.DataFrame:
    seen_names = set()
    for it in items:
        for m in it.get("models", []):
            name = m.get("model_name")
            if name:
                seen_names.add(name)
    ordered_model_names = [name for name in MODEL_NAME_MAP if name in seen_names]
    extra_model_names = [name for name in seen_names if name not in MODEL_NAME_MAP]
    model_names = ordered_model_names + sorted(extra_model_names)
    rows = []
    for it in items:
        row = {"id": it.get("id")}
        for name in model_names:
            if df_train is not None or df_test is not None:
                corr = _get_combined_corr_for_model_in_item(it, name, df_train, df_test)
            else:
                grouped: Dict[str, Dict[str, Any]] = {}
                for m in it.get("models", []):
                    n = m.get("model_name")
                    if not n:
                        continue
                    ds = (m.get("dataset") or "train").lower()
                    if n not in grouped:
                        grouped[n] = {}
                    grouped[n][ds] = m
                g = grouped.get(name, {})
                m = (
                    g.get("train")
                    or g.get("val")
                    or g.get("validation")
                    or (next(iter(g.values()), {}) if g else {})
                )
                corr = m.get("corr")

            row[f"{name} corr"] = corr

            grouped_perf: Dict[str, Dict[str, Any]] = {}
            for m in it.get("models", []):
                n = m.get("model_name")
                if not n:
                    continue
                ds = (m.get("dataset") or "train").lower()
                if n not in grouped_perf:
                    grouped_perf[n] = {}
                grouped_perf[n][ds] = m

            g_perf = grouped_perf.get(name, {})
            m_perf = (
                g_perf.get("train")
                or g_perf.get("val")
                or g_perf.get("validation")
                or (next(iter(g_perf.values()), {}) if g_perf else {})
            )
            perf = m_perf.get("performance") or {}
            row[f"{name} throughput"] = perf.get("throughput")
        rows.append(row)
    df = pd.DataFrame(rows)
    avg = df.drop(columns=["id"]).mean(numeric_only=True)
    avg_row = {k: (None if k == "id" else v) for k, v in avg.to_dict().items()}
    avg_row["id"] = "AVG"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df


def _best_corr(
    item: Dict[str, Any],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> float | None:
    best = None
    seen_model_names = set()
    for model in item.get("models", []):
        name = model.get("model_name")
        if not name or name in seen_model_names:
            continue
        seen_model_names.add(name)

        if df_train is not None or df_test is not None:
            corr = _get_combined_corr_for_model_in_item(item, name, df_train, df_test)
        else:
            grouped: Dict[str, Dict[str, Any]] = {}
            for m in item.get("models", []):
                n = m.get("model_name")
                if not n:
                    continue
                ds = (m.get("dataset") or "train").lower()
                if n not in grouped:
                    grouped[n] = {}
                grouped[n][ds] = m
            g = grouped.get(name, {})
            m = (
                g.get("train")
                or g.get("val")
                or g.get("validation")
                or (next(iter(g.values()), {}) if g else {})
            )
            corr = m.get("corr")

        if corr is None:
            continue
        best = corr if best is None else max(best, corr)

    return best


def _round_numeric(
    df: pd.DataFrame, decimals: int = 4, column_decimals: Dict[str, int] | None = None
) -> pd.DataFrame:
    column_decimals = column_decimals or {}
    for col in df.columns:
        if col == "Rank" or col == "#":
            continue
        col_decimals = column_decimals.get(col, decimals)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(col_decimals)
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            df[col] = coerced.round(col_decimals).where(~coerced.isna(), df[col])
    return df


def df_to_latex(
    df: pd.DataFrame,
    bold_columns: List[str] | None = None,
    column_decimals: Dict[str, int] | None = None,
) -> str:
    bold_columns = bold_columns or []
    column_decimals = column_decimals or {}
    id_col = next((c for c in ("#", "Rank", "id") if c in df.columns), df.columns[0])
    non_avg = df[df[id_col] != "AVG"] if "AVG" in df[id_col].values else df
    max_idx: Dict[str, int] = {}
    for col in bold_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(non_avg[col], errors="coerce")
        if series.empty or series.isna().all():
            continue
        max_idx[col] = int(series.idxmax())
    headers = [latex_escape(str(h)) for h in df.columns]
    colspec = "l" + ("r" * (len(headers) - 1))
    lines: List[str] = []
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for i, row in df.iterrows():
        cells: List[str] = []
        for col, val in row.items():
            if col == "#" or col == "Rank":
                s = str(val) if val is not None else ""
            elif pd.api.types.is_number(val):
                decimals = column_decimals.get(col, 4)
                s = format_number(val, decimals=decimals)
            else:
                s = str(val) if val is not None else ""
            should_bold = col in max_idx and (i == max_idx[col])
            s_esc = latex_escape(s)
            cells.append(("\\textbf{" + s_esc + "}") if should_bold else s_esc)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def df_to_latex_multirow_header(
    df: pd.DataFrame,
    bold_columns: List[str] | None = None,
    column_decimals: Dict[str, int] | None = None,
) -> str:
    bold_columns = bold_columns or []
    column_decimals = column_decimals or {}
    id_col = next((c for c in ("#", "Rank", "id") if c in df.columns), df.columns[0])
    non_avg = df[df[id_col] != "AVG"] if "AVG" in df[id_col].values else df
    max_idx: Dict[str, int] = {}
    for col in bold_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(non_avg[col], errors="coerce")
        if series.empty or series.isna().all():
            continue
        max_idx[col] = int(series.idxmax())

    corr_cols = [c for c in df.columns if c.endswith(" Corr")]
    thr_cols = [c for c in df.columns if c.endswith(" Throughput")]

    colspec = "l" + ("r" * len(corr_cols)) + ("r" * len(thr_cols))
    lines: List[str] = []
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\hline")

    # First header row with multicolumn spans
    id_col_display = df.columns[0] if len(df.columns) > 0 else "#"
    header_row1 = [f"\\multirow{{2}}{{*}}{{{latex_escape(id_col_display)}}}"]
    if corr_cols:
        header_row1.append(
            f"\\multicolumn{{{len(corr_cols)}}}{{c}}{{Pearson correlation coefficient}}"
        )
    if thr_cols:
        header_row1.append(
            f"\\multicolumn{{{len(thr_cols)}}}{{c}}{{Throughput (predictions/sec)}}"
        )
    lines.append(" & ".join(header_row1) + " \\\\")

    # Second header row with model names
    header_row2 = [""]
    for col in corr_cols:
        model_name = col.replace(" Corr", "")
        header_row2.append(latex_escape(model_name))
    for col in thr_cols:
        model_name = col.replace(" Throughput", "")
        header_row2.append(latex_escape(model_name))
    lines.append(" & ".join(header_row2) + " \\\\")
    lines.append("\\hline")

    for i, row in df.iterrows():
        cells: List[str] = []
        is_avg_row = row.get(id_col) == "AVG"
        for col, val in row.items():
            if col == "#" or col == "Rank":
                s = str(val) if val is not None else ""
            elif pd.api.types.is_number(val):
                decimals = column_decimals.get(col, 4)
                s = format_number(val, decimals=decimals)
            else:
                s = str(val) if val is not None else ""
            should_bold = col in max_idx and (i == max_idx[col])
            s_esc = latex_escape(s)
            if is_avg_row:
                cells.append("\\textit{" + s_esc + "}")
            elif should_bold:
                cells.append("\\textbf{" + s_esc + "}")
            else:
                cells.append(s_esc)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def get_table_b_styles() -> List[Dict[str, Any]]:
    return [
        {
            "selector": "th.col_heading",
            "props": [
                ("max-width", "16ch"),
                ("white-space", "normal"),
                ("word-wrap", "break-word"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("max-width", "16ch"),
                ("white-space", "nowrap"),
            ],
        },
        {
            "selector": "th.blank",
            "props": [("display", "none")],
        },
        {
            "selector": "th.col_heading.level0",
            "props": [("display", "table-cell")],
        },
    ]


def create_avg_row_combined(
    df_b: pd.DataFrame, main_rows: pd.DataFrame
) -> pd.DataFrame:
    id_col = "#" if "#" in df_b.columns else "Rank"
    avg_row = df_b[df_b[id_col] == "AVG"]
    return pd.concat([main_rows, avg_row], ignore_index=True)


def create_table_b_styler(df_b: pd.DataFrame, bold_cols: List[str]) -> Any:
    id_col = "#" if "#" in df_b.columns else "Rank"
    main_rows = df_b[df_b[id_col] != "AVG"]
    combined = create_avg_row_combined(df_b, main_rows)

    def _highlight_best(col: pd.Series) -> List[str]:
        if col.name not in bold_cols:
            return [""] * len(col)
        best_val = pd.to_numeric(main_rows[col.name], errors="coerce").max()
        styles: List[str] = []
        for val, rank in zip(col, combined[id_col]):
            if rank == "AVG":
                styles.append("")
                continue
            is_best = pd.to_numeric(val, errors="coerce") == best_val
            styles.append("font-weight:bold" if is_best else "")
        return styles

    def _italic_avg_row(col: pd.Series) -> List[str]:
        return [
            "font-style:italic" if rank == "AVG" else "" for rank in combined[id_col]
        ]

    table_styles = get_table_b_styles()
    return (
        combined.style.apply(_highlight_best, subset=bold_cols)
        .apply(_italic_avg_row)
        .hide(axis="index")
        .set_table_styles(table_styles)
    )


def _sort_items_by_best_corr(
    items: List[Dict[str, Any]],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> List[Dict[str, Any]]:
    items_with_scores = [(it, _best_corr(it, df_train, df_test)) for it in items]
    return [
        it
        for it, score in sorted(
            items_with_scores,
            key=lambda pair: pair[1] if pair[1] is not None else float("-inf"),
            reverse=True,
        )
    ]


def _prepare_table_a(
    sorted_items: List[Dict[str, Any]],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, str]:
    df_a = build_table_a(sorted_items, df_train, df_test)
    df_a["id"] = list(range(1, len(df_a) + 1))
    df_a = df_a.rename(
        columns={
            "id": "#",
            "hypothesis_template": "Hypothesis Template",
            "candidate_labels": "Candidate Labels",
            "best_corr": "Best Corr",
        }
    )[["#", "Hypothesis Template", "Candidate Labels", "Best Corr"]]
    column_decimals = {"Best Corr": 3}
    df_a = _round_numeric(df_a, column_decimals=column_decimals)
    latex_a = df_to_latex(df_a, bold_columns=[], column_decimals=column_decimals)
    return df_a, latex_a


def _build_model_rename_map(df_b: pd.DataFrame) -> Dict[str, str]:
    rename_map = {"id": "#"}
    for col in df_b.columns:
        if col.endswith(" corr"):
            raw_name = col[:-5]
            model_name = _prettify_model_name(raw_name)
            rename_map[col] = f"{model_name} Corr"
        elif col.endswith(" throughput"):
            raw_name = col[:-11]
            model_name = _prettify_model_name(raw_name)
            rename_map[col] = f"{model_name} Throughput"
    return rename_map


def _reorder_table_b_columns(df_b: pd.DataFrame) -> pd.DataFrame:
    corr_cols = [c for c in df_b.columns if c.endswith(" Corr")]
    thr_cols = [c for c in df_b.columns if c.endswith(" Throughput")]
    id_col = "#" if "#" in df_b.columns else "Rank"
    return df_b[[id_col] + corr_cols + thr_cols]


def _prepare_table_b(
    sorted_items: List[Dict[str, Any]],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, str, List[str]]:
    df_b = build_table_b(sorted_items, df_train, df_test)
    df_b.loc[df_b["id"] != "AVG", "id"] = list(range(1, len(df_b)))

    rename_map = _build_model_rename_map(df_b)
    df_b = df_b.rename(columns=rename_map)

    df_b = _reorder_table_b_columns(df_b)

    corr_cols = [c for c in df_b.columns if c.endswith(" Corr")]
    thr_cols = [c for c in df_b.columns if c.endswith(" Throughput")]
    bold_cols = corr_cols + thr_cols

    column_decimals = {col: 1 for col in thr_cols}
    column_decimals.update({col: 4 for col in corr_cols})
    df_b = _round_numeric(df_b, column_decimals=column_decimals)
    latex_b = df_to_latex_multirow_header(
        df_b, bold_columns=bold_cols, column_decimals=column_decimals
    )
    return df_b, latex_b, bold_cols


def prepare_nli_tables(
    items: List[Dict[str, Any]],
    df_train: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
):
    sorted_items = _sort_items_by_best_corr(items, df_train, df_test)
    df_a, latex_a = _prepare_table_a(sorted_items, df_train, df_test)
    df_b, latex_b, bold_cols = _prepare_table_b(sorted_items, df_train, df_test)
    return df_a, latex_a, df_b, latex_b, bold_cols
