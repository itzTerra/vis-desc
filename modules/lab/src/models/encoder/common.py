from abc import ABC, abstractmethod
import os
import random
import hashlib
import json
from datetime import datetime
from typing import Any, Literal
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna
from transformers import AutoTokenizer
from utils import DATA_DIR, PersistentDict
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from pathlib import Path

BERT_TOKENIZER_MAX_LENGTH = 160
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS_DIR = DATA_DIR / "metrics" / "encoder"
MODEL_DIR = DATA_DIR / "models" / "encoder"

for d in [METRICS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


class ModelNamer(ABC):
    @abstractmethod
    def _get_base_model_name(self) -> str:
        pass

    def get_model_name(self) -> str:
        if not hasattr(self, "include_large"):
            raise ValueError("ModelNamer subclass must have 'include_large' attribute.")
        embeddings = getattr(self, "embeddings", "")
        base_name = self._get_base_model_name()
        return f"{base_name}{f'_{embeddings}' if embeddings else ''}{'_lg' if self.include_large else ''}"


class RidgeNamer(ModelNamer):
    def _get_base_model_name(self) -> str:
        return "ridge"


class SVMNamer(ModelNamer):
    def _get_base_model_name(self) -> str:
        return "svm"


class RandomForestNamer(ModelNamer):
    def _get_base_model_name(self) -> str:
        return "rf"


class CatBoostNamer(ModelNamer):
    def _get_base_model_name(self) -> str:
        return "catboost"


class RandomBaselineNamer(ModelNamer):
    def _get_base_model_name(self) -> str:
        return "random"


class FinetunedBertNamer(ModelNamer):
    def _get_base_model_name(self) -> str:
        return "finetuned-mbert"


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for text, features, and labels.
    """

    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.features = dataframe.features
        self.labels = dataframe.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # Use position-based indexing to avoid KeyError when DataFrame indices
        # are not contiguous or start at non-zero values after splits/concats.
        text = str(self.text.iloc[index])
        features = torch.tensor(self.features.iloc[index], dtype=torch.float)
        label = torch.tensor(self.labels.iloc[index], dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=BERT_TOKENIZER_MAX_LENGTH,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "features": features,
            "labels": label,
        }


class CachedOptimizationContext:
    tokenizer = None
    sm_train_cache = {}
    lg_train_cache = {}

    def __init__(
        self,
        include_minilm_embeddings=False,
        include_modernbert_embeddings=False,
        include_large=False,
    ):
        if CachedOptimizationContext.tokenizer is None:
            CachedOptimizationContext.tokenizer = AutoTokenizer.from_pretrained(
                "answerdotai/ModernBERT-base"
            )
        self.tokenizer = CachedOptimizationContext.tokenizer

        # Create cache key from parameters
        sm_cache_key = (include_minilm_embeddings, include_modernbert_embeddings)

        if sm_cache_key not in CachedOptimizationContext.sm_train_cache:
            sm_train = pd.read_parquet(
                DATA_DIR / "datasets" / "small" / "train.parquet"
            )
            sm_train["weight"] = compute_sample_weight(
                class_weight="balanced", y=sm_train["label"]
            )
            if include_minilm_embeddings:
                minilm_embeddings = pd.read_parquet(
                    DATA_DIR / "datasets" / "small" / "minilm_embeddings.parquet"
                )
                sm_train = sm_train.merge(
                    minilm_embeddings[["cls"]], left_index=True, right_index=True
                )
            if include_modernbert_embeddings:
                modernbert_embeddings = pd.read_parquet(
                    DATA_DIR
                    / "datasets"
                    / "small"
                    / "modernbert_cls_embeddings.parquet"
                )
                sm_train = sm_train.merge(
                    modernbert_embeddings[["cls"]], left_index=True, right_index=True
                )
            CachedOptimizationContext.sm_train_cache[sm_cache_key] = (
                sm_train.reset_index(drop=True)
            )
        self.sm_train = CachedOptimizationContext.sm_train_cache[sm_cache_key]

        if include_large:
            lg_cache_key = (include_minilm_embeddings, include_modernbert_embeddings)

            if lg_cache_key not in CachedOptimizationContext.lg_train_cache:
                lg_train = pd.read_parquet(
                    DATA_DIR / "datasets" / "large" / "combined.parquet"
                )
                lg_train["weight"] = compute_sample_weight(
                    class_weight="balanced", y=lg_train["label"]
                )
                if include_minilm_embeddings:
                    minilm_embeddings_large = pd.read_parquet(
                        DATA_DIR / "datasets" / "large" / "minilm_embeddings.parquet"
                    )
                    lg_train = lg_train.merge(
                        minilm_embeddings_large[["cls"]],
                        left_index=True,
                        right_index=True,
                    )
                if include_modernbert_embeddings:
                    modernbert_embeddings_large = pd.read_parquet(
                        DATA_DIR
                        / "datasets"
                        / "large"
                        / "modernbert_cls_embeddings.parquet"
                    )
                    lg_train = lg_train.merge(
                        modernbert_embeddings_large[["cls"]],
                        left_index=True,
                        right_index=True,
                    )
                CachedOptimizationContext.lg_train_cache[lg_cache_key] = (
                    lg_train.reset_index(drop=True)
                )
            self.lg_train = CachedOptimizationContext.lg_train_cache[lg_cache_key]
        else:
            self.lg_train = None


def hash_tokenizer(tokenizer):
    """
    Return (sha256_hash, vocab_size, added_tokens_count) for diagnostics.
    """
    vocab = tokenizer.get_vocab()  # dict: token -> id
    items = sorted(vocab.items())
    sha = hashlib.sha256()
    for k, v in items:
        sha.update(f"{k}:{v}".encode())
    added = getattr(tokenizer, "added_tokens_encoder", {})
    sha.update(json.dumps(sorted(added.items())).encode())
    return sha.hexdigest(), len(vocab), len(added)


def run_cross_validation(
    trial,
    train_and_eval_func,
    n_splits=5,
    include_minilm_embeddings=False,
    include_modernbert_embeddings=False,
    include_large=False,
    small_dataset_weight_multiplier=None,
    two_stage_training=False,
):
    if (
        include_large
        and small_dataset_weight_multiplier is None
        and not two_stage_training
    ):
        raise ValueError(
            "small_dataset_weight_multiplier must be set when include_large is True "
            "unless two_stage_training is enabled."
        )

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_scores = []
    optimization_context = CachedOptimizationContext(
        include_minilm_embeddings=include_minilm_embeddings,
        include_modernbert_embeddings=include_modernbert_embeddings,
        include_large=include_large,
    )
    tokenizer = optimization_context.tokenizer
    sm_train = optimization_context.sm_train
    lg_train = optimization_context.lg_train

    for fold, (train_index, val_index) in enumerate(
        kf.split(sm_train, sm_train["label"])
    ):
        train_fold_df = sm_train.loc[train_index].copy()
        val_fold_df = sm_train.loc[val_index].copy()

        lg_size = None
        if lg_train is not None:
            lg_size = len(lg_train)
            if two_stage_training:
                # For two-stage training, concatenate without weight adjustment
                # The training function will handle stages separately
                # Pass lg_size as metadata so the training function knows where to split
                train_fold_df = pd.concat([lg_train, train_fold_df], ignore_index=True)
            else:
                # Traditional approach: upscale small dataset weights to balance with large dataset
                train_fold_df["weight"] *= small_dataset_weight_multiplier
                train_fold_df = pd.concat([lg_train, train_fold_df], ignore_index=True)

        score = train_and_eval_func(
            train_fold_df.reset_index(drop=True),
            val_fold_df.reset_index(drop=True),
            tokenizer,
            lg_size,
            fold,
        )

        fold_scores.append(score)

        print(f"Fold {fold + 1}/{n_splits}: {fold_scores[-1]:.4f}")

        # Optuna pruning
        # trial.report(score, fold)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    mean_mse = np.mean(fold_scores)
    return mean_mse


def run_study(objective_func, study_name, search_space=None, n_trials=None):
    """
    Creates and runs an Optuna study.
    """
    TEST_RUN = os.environ.get("TEST_RUN", "false").lower() in ("true", "1", "t")

    set_seed()

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{(DATA_DIR / 'optuna' / 'optuna_db.sqlite3').as_posix()}",
            direction="minimize",
            # pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.GridSampler(search_space) if search_space else None,
            load_if_exists=True,
        )
        study.optimize(objective_func, n_trials=n_trials if not TEST_RUN else 1)
    except Exception as e:
        print(f"Failed to create study '{study_name}': {e}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name_with_timestamp = f"{study_name}_{timestamp}"
        print(f"Retrying with timestamped name: {study_name_with_timestamp}")
        study = optuna.create_study(
            study_name=study_name_with_timestamp,
            storage=f"sqlite:///{(DATA_DIR / 'optuna' / 'optuna_db.sqlite3').as_posix()}",
            direction="minimize",
            # pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.GridSampler(search_space) if search_space else None,
            load_if_exists=True,
        )
        study.optimize(objective_func, n_trials=n_trials if not TEST_RUN else 1)

    print("\n--- Optuna Study Summary ---")
    print(f"Study: {study_name}")
    print(f"Best trial MSE: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive metrics for predictions.

    Returns:
        dict with keys:
        - mse: Mean squared error
        - accuracy: Accuracy of rounded scores
        - precision: Precision for each class (0-5)
        - recall: Recall for each class (0-5)
        - f1: F1-score for each class (0-5)
        - confusion_matrix: Confusion matrix for rounded scores
    """
    if np.any(np.isnan(y_pred)):
        return {
            "mse": float("nan"),
            "accuracy": float("nan"),
            "precision": [float("nan")] * 6,
            "recall": [float("nan")] * 6,
            "f1": [float("nan")] * 6,
            "support": [0] * 6,
            "confusion_matrix": [[0] * 6 for _ in range(6)],
        }
    # Round predictions and true values to nearest integer (0-5)
    y_true_rounded = np.clip(np.round(y_true), 0, 5).astype(int)
    y_pred_rounded = np.clip(np.round(y_pred), 0, 5).astype(int)

    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true_rounded, y_pred_rounded)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_rounded, y_pred_rounded, labels=[0, 1, 2, 3, 4, 5], zero_division=0
    )
    cm = confusion_matrix(y_true_rounded, y_pred_rounded, labels=[0, 1, 2, 3, 4, 5])

    return {
        "mse": float(mse),
        "accuracy": float(accuracy),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def average_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {
            "mse": float("nan"),
            "accuracy": float("nan"),
            "precision": [float("nan")] * 6,
            "recall": [float("nan")] * 6,
            "f1": [float("nan")] * 6,
            "support": [0] * 6,
            "confusion_matrix": [[0] * 6 for _ in range(6)],
        }

    cms = np.array([m["confusion_matrix"] for m in metrics], dtype=float)
    cm_avg = np.nanmean(cms, axis=0)
    cm_avg = np.nan_to_num(cm_avg, nan=0.0)

    return {
        "mse": float(np.nanmean([m["mse"] for m in metrics])),
        "accuracy": float(np.nanmean([m["accuracy"] for m in metrics])),
        "precision": np.nanmean([m["precision"] for m in metrics], axis=0).tolist(),
        "recall": np.nanmean([m["recall"] for m in metrics], axis=0).tolist(),
        "f1": np.nanmean([m["f1"] for m in metrics], axis=0).tolist(),
        "support": np.sum([m["support"] for m in metrics], axis=0).tolist(),
        "confusion_matrix": np.rint(cm_avg).astype(int).tolist(),
    }


def get_model_filename(
    model_name: str, seed: int, datetime: datetime, ext: str = ".onnx"
) -> Path:
    filename = f"{model_name}_{seed}_{datetime.strftime('%Y-%m-%d-%H-%M-%S')}{ext}"
    return MODEL_DIR / filename


def get_metrics_filename(
    model_name: str,
    type: Literal["train", "val", "test"],
    seed: int,
    datetime: datetime,
    trial: int | None = None,
) -> Path:
    """
    Generate a standardized filename for saving metrics based on data/metrics/metrics-docs.md.

    Args:
        model_name: Name of the model.
        type: Type of dataset ('train', 'val', 'test').
        seed: Random seed used.
        datetime: Datetime object for timestamping.
    Returns:
        Path object for the metrics file.
    """
    filename = f"{model_name}_{type}{f'_t{trial}' if trial is not None else ''}_{seed}_{datetime.strftime('%Y-%m-%d-%H-%M-%S')}.json"
    return METRICS_DIR / filename


class PersistentMetrics(PersistentDict):
    @classmethod
    def from_parts(
        cls,
        model_name: str,
        type: Literal["train", "val", "test"],
        seed: int,
        datetime: datetime,
        trial: int | None = None,
    ) -> "PersistentMetrics":
        filename = get_metrics_filename(model_name, type, seed, datetime, trial)
        metrics = PersistentMetrics(filename)

        metrics.setdefault("model", model_name)
        metrics.setdefault("type", type)
        metrics.setdefault("seed", seed)
        if type in ("train", "val"):
            metrics.setdefault("train_losses", [])
            metrics.setdefault("val_losses", [])
        if trial is not None:
            metrics.setdefault("trial", trial)

        return metrics

    @staticmethod
    def dummy() -> "PersistentMetrics":
        """Create a dummy PersistentMetrics that does not persist to disk."""

        class DummyPersistentMetrics(PersistentMetrics):
            def __init__(self):
                self.setdefault("train_losses", [])
                self.setdefault("val_losses", [])

            def __setitem__(self, key, val):
                dict.__setitem__(self, key, val)

            def update(self, *args, **kwargs):
                for k, v in dict(*args, **kwargs).items():
                    self[k] = v

        return DummyPersistentMetrics()
