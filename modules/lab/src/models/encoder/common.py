import os
import random
import hashlib
import json
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna
from transformers import AutoTokenizer
from utils import DATA_DIR
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
TRAINING_HISTORY_DIR = METRICS_DIR / "training_history"

for d in [METRICS_DIR, MODEL_DIR, TRAINING_HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)


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
        text = str(self.text[index])
        features = torch.tensor(self.features[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)

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
    import json

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
    fold_best_epochs = []
    optimization_context = CachedOptimizationContext(
        include_minilm_embeddings=include_minilm_embeddings,
        include_modernbert_embeddings=include_modernbert_embeddings,
        include_large=include_large,
    )
    tokenizer = optimization_context.tokenizer
    if diagnostics_enabled():
        tok_hash, vocab_size, added_size = hash_tokenizer(tokenizer)
        print(
            f"[DIAG] Tokenizer hash={tok_hash} vocab_size={vocab_size} added_tokens={added_size}"
        )
    sm_train = optimization_context.sm_train
    lg_train = optimization_context.lg_train

    for fold, (train_index, val_index) in enumerate(
        kf.split(sm_train, sm_train["label"])
    ):
        train_fold_df = sm_train.loc[train_index].copy()
        val_fold_df = sm_train.loc[val_index].copy()

        if diagnostics_enabled():
            train_summary = summarize_dataframe(train_fold_df, "train")
            val_summary = summarize_dataframe(val_fold_df, "val")
            print(
                f"[DIAG] Trial {getattr(trial, 'number', '?')} Fold {fold} | Train size={train_summary['size']} Val size={val_summary['size']}"
            )
            print(
                f"[DIAG] Train label dist: {train_summary['label_distribution']} | Val label dist: {val_summary['label_distribution']}"
            )
            cache_files = list(Path(os.environ.get("HF_HOME")).glob("**/*"))
            cache_sizes = sum(f.stat().st_size for f in cache_files if f.is_file())
            print(f"[DIAG] Cache size before fold {fold}: {cache_sizes / 1e9:.2f} GB")

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

        if isinstance(score, tuple):
            mse, best_epoch = score
            fold_scores.append(mse)
            fold_best_epochs.append(best_epoch)
        else:
            fold_scores.append(score)

        print(f"Fold {fold + 1}/{n_splits}: {fold_scores[-1]:.4f}")

        # Optuna pruning
        # trial.report(score, fold)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
        if diagnostics_enabled():
            tok_hash_after, vocab_size_after, added_size_after = hash_tokenizer(
                tokenizer
            )
            print(
                f"[DIAG] Tokenizer post-fold hash={tok_hash_after} "
                f"vocab_size={vocab_size_after} added_tokens={added_size_after} "
                f"(unchanged={tok_hash_after == tok_hash})"
            )

    mean_mse = np.mean(fold_scores)
    if fold_best_epochs:
        return {"mse": mean_mse, "best_epochs": fold_best_epochs}
    else:
        return mean_mse


# ---------------------- Diagnostics Utilities ----------------------


def diagnostics_enabled():
    return os.environ.get("ENCODER_DIAG", "0") in ("1", "true", "True")


def hash_model_parameters(model):
    sha = hashlib.sha256()
    for p in model.parameters():
        sha.update(p.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def summarize_dataframe(df: pd.DataFrame, name: str):
    labels = df["label"].values
    unique, counts = np.unique(labels, return_counts=True)
    dist_map = {int(k): int(v) for k, v in zip(unique, counts)}
    return {
        "name": name,
        "size": len(df),
        "label_distribution": dist_map,
    }


def run_study(objective_func, study_name, search_space=None, n_trials=None):
    """
    Creates and runs an Optuna study.
    """
    from datetime import datetime

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
    if study.best_trial.user_attrs:
        print("Best trial user attributes:")
        for key, value in study.best_trial.user_attrs.items():
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


def save_metrics_separate(
    model_name: str,
    params: dict,
    train: dict | None,
    val: dict | None,
    test: dict | None,
    extra_train: dict | None = None,
    timestamp: str | None = None,
) -> None:
    """Persist metrics for train/val/test splits into separate JSON files.

    This is a standalone utility extracted from the trainer class so it can also
    be reused by scripts that compute aggregated or ensemble metrics (e.g.
    feature importance / ablation studies).

    Each JSON contains:
      - model: the model name (includes any run label suffixes)
      - params: parameter dict used for the run
      - dataset: one of train | val | test
      - metric keys (mse, accuracy, precision, recall, f1, support, confusion_matrix, ...)
      - any optional extra_train data only in the train file
    """

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _write(split_name: str, data: dict | None, extra: dict | None = None):
        payload = {
            "model": model_name,
            "params": params,
            "dataset": split_name,
        }
        if data is not None:
            payload.update(data)
        if extra and split_name == "train":
            for k, v in extra.items():
                payload[k] = v
        out_path = METRICS_DIR / f"{model_name}_{split_name}_{timestamp}.json"
        try:
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved {split_name} metrics to {out_path}")
        except Exception as e:
            print(f"Failed to save {split_name} metrics: {e}")

    if train is not None:
        _write("train", train, extra_train)
    if val is not None:
        _write("val", val)
    if test is not None:
        _write("test", test)


def _average_metrics(seed_metrics_list):
    """Average metrics structure returned by trainer.run_full_training across seeds.

    Each metrics dict may contain train_metrics, cv_metrics, test_metrics with sub keys.
    Arrays (precision, recall, f1, support, confusion_matrix) are averaged element-wise.
    Scalar entries (mse, accuracy) are averaged directly.
    """

    def _avg_split(split_key):
        splits = [
            m.get(split_key) for m in seed_metrics_list if m.get(split_key) is not None
        ]
        if not splits:
            return None
        out = {}
        for k in ["mse", "accuracy"]:
            if k in splits[0]:
                out[k] = float(np.mean([s[k] for s in splits]))
        for k in ["precision", "recall", "f1", "support", "confusion_matrix"]:
            if k in splits[0]:
                arr = np.array([s[k] for s in splits], dtype=float)
                out[k] = arr.mean(axis=0).tolist()
        return out

    return {
        "train_metrics": _avg_split("train_metrics"),
        "cv_metrics": _avg_split("cv_metrics"),
        "test_metrics": _avg_split("test_metrics"),
    }
