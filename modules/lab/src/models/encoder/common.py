import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
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

BERT_TOKENIZER_MAX_LENGTH = 160
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    sm_train = None
    lg_train = None

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

        if CachedOptimizationContext.sm_train is None:
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
            CachedOptimizationContext.sm_train = sm_train.reset_index(drop=True)
        self.sm_train = CachedOptimizationContext.sm_train

        if include_large and CachedOptimizationContext.lg_train is None:
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
            CachedOptimizationContext.lg_train = lg_train.reset_index(drop=True)
        self.lg_train = CachedOptimizationContext.lg_train


def run_cross_validation(
    trial,
    train_and_eval_func,
    n_splits=5,
    include_minilm_embeddings=False,
    include_modernbert_embeddings=False,
    include_large=False,
    small_dataset_weight_multiplier=None,
):
    if include_large and small_dataset_weight_multiplier is None:
        raise ValueError(
            "small_dataset_weight_multiplier must be set when include_large is True."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_scores = []
    optimization_context = CachedOptimizationContext(
        include_minilm_embeddings=include_minilm_embeddings,
        include_modernbert_embeddings=include_modernbert_embeddings,
        include_large=include_large,
    )
    tokenizer = optimization_context.tokenizer
    sm_train = optimization_context.sm_train
    lg_train = optimization_context.lg_train

    for fold, (train_index, val_index) in enumerate(kf.split(sm_train)):
        train_fold_df = sm_train.loc[train_index].copy()
        val_fold_df = sm_train.loc[val_index].copy()

        if lg_train is not None:
            # Upscale small dataset weights to balance with large dataset
            train_fold_df["weight"] *= small_dataset_weight_multiplier
            train_fold_df = pd.concat([lg_train, train_fold_df], ignore_index=True)

        score = train_and_eval_func(
            train_fold_df.reset_index(drop=True),
            val_fold_df.reset_index(drop=True),
            tokenizer,
        )

        fold_scores.append(score)
        print(f"Fold {fold + 1}/{n_splits}: {score:.4f}")

        # Optuna pruning
        # trial.report(score, fold)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    return np.mean(fold_scores)


def run_study(objective_func, study_name, search_space=None, n_trials=None):
    """
    Creates and runs an Optuna study.
    """
    TEST_RUN = os.environ.get("TEST_RUN", "false").lower() in ("true", "1", "t")

    study = optuna.create_study(
        study_name=study_name,
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
        - precision: Precision for each class (1-5)
        - recall: Recall for each class (1-5)
        - f1: F1-score for each class (1-5)
        - confusion_matrix: Confusion matrix for rounded scores
    """
    # Round predictions and true values to nearest integer (1-5)
    y_true_rounded = np.clip(np.round(y_true), 1, 5).astype(int)
    y_pred_rounded = np.clip(np.round(y_pred), 1, 5).astype(int)

    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true_rounded, y_pred_rounded)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_rounded, y_pred_rounded, labels=[1, 2, 3, 4, 5], zero_division=0
    )
    cm = confusion_matrix(y_true_rounded, y_pred_rounded, labels=[1, 2, 3, 4, 5])

    return {
        "mse": float(mse),
        "accuracy": float(accuracy),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
