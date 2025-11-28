#!/usr/bin/env python3

from typing import Literal
import argparse
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models.encoder.common import (
    TRAINING_HISTORY_DIR,
    run_study,
    SEED,
    run_cross_validation,
    set_seed,
)
import json
import multiprocessing as mp


def run_modernbert_trial(
    trial,
    trial_params: dict,
    include_large: bool,
):
    """Top-level function executed in an isolated subprocess for ModernBERT fine-tuning.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object.
    trial_params : dict
        Hyperparameters for this trial.
    include_large : bool
        Whether to include the large (noisy) dataset in stage 1 pretraining.
    """
    from models.encoder.modernbert_finetune_nn import train_finetuned_mbert

    set_seed()

    def train_and_evaluate_fold(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        tokenizer,
        lg_size=None,
        fold_num=0,
    ):
        if include_large and lg_size is not None:
            lg_train_df = train_df.iloc[:lg_size].copy()
            sm_train_df = train_df.iloc[lg_size:].copy()
        else:
            lg_train_df = None
            sm_train_df = train_df.copy()

        history_path = (
            TRAINING_HISTORY_DIR
            / f"finetuned-mbert_trial_{trial.number}_fold_{fold_num}.json"
        )

        result = train_finetuned_mbert(
            train_df=sm_train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            params=trial_params,
            seed=SEED,
            train_lg_df=lg_train_df,
            save_history=True,
            history_path=history_path,
        )

        # Add trial and fold metadata to saved history
        if history_path.exists():
            with open(history_path, "r") as f:
                history_data = json.load(f)
            history_data["trial_number"] = trial.number
            history_data["fold_num"] = fold_num
            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2)
            best_epoch = history_data.get("best_epoch")
        else:
            best_epoch = None

        return result["final_metrics"]["mse"], best_epoch

    return run_cross_validation(
        trial=trial,
        train_and_eval_func=train_and_evaluate_fold,
        include_minilm_embeddings=False,
        include_modernbert_embeddings=False,
        include_large=include_large,
        n_splits=5,
        two_stage_training=True,
    )


class ObjectiveProvider:
    def __init__(self, embeddings: Literal["minilm", "mbert"], include_large: bool):
        self.embeddings = embeddings
        self.include_minilm_embeddings = embeddings == "minilm"
        self.include_modernbert_embeddings = embeddings == "mbert"
        self.include_large = include_large

    def get_objective_fn(self) -> callable:
        raise NotImplementedError

    def get_study_name(self) -> str:
        raise NotImplementedError

    def get_n_trials(self) -> int:
        return 100

    def get_search_space(self) -> dict | None:
        return None


class RidgeObjectiveProvider(ObjectiveProvider):
    def get_n_trials(self):
        return 200

    def get_objective_fn(self) -> callable:
        def objective(trial):
            ridge_alpha = trial.suggest_float("ridge_alpha", 0.0001, 1000.0, log=True)
            # ridge_alpha = trial.suggest_categorical("ridge_alpha", [0.01])
            small_dataset_weight_multiplier = (
                trial.suggest_float(
                    "small_dataset_weight_multiplier", 10.0, 10000.0, log=True
                )
                if self.include_large
                else None
            )

            def train_and_evaluate_fold(
                train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                tokenizer,
                lg_size=None,
                fold_num=0,
            ):
                train_features = np.hstack(
                    (
                        np.vstack(train_df["cls"].values),
                        np.vstack(train_df["features"].values),
                    )
                )
                val_features = np.hstack(
                    (
                        np.vstack(val_df["cls"].values),
                        np.vstack(val_df["features"].values),
                    )
                )

                train_labels = train_df["label"].values
                y_true_fold = val_df["label"].values

                regressor = make_pipeline(
                    MinMaxScaler(), Ridge(alpha=ridge_alpha, random_state=SEED)
                )
                regressor.fit(
                    train_features,
                    train_labels,
                    ridge__sample_weight=train_df["weight"].values,
                )

                y_pred_fold = regressor.predict(val_features)
                mse = mean_squared_error(y_true_fold, y_pred_fold)
                return mse

            return run_cross_validation(
                trial=trial,
                train_and_eval_func=train_and_evaluate_fold,
                include_minilm_embeddings=self.include_minilm_embeddings,
                include_modernbert_embeddings=self.include_modernbert_embeddings,
                include_large=self.include_large,
                small_dataset_weight_multiplier=small_dataset_weight_multiplier,
            )

        return objective

    def get_study_name(self) -> str:
        return f"{self.embeddings}_ridge{'_lg' if self.include_large else ''}"


class SVMObjectiveProvider(ObjectiveProvider):
    def get_n_trials(self):
        return 100

    def get_objective_fn(self) -> callable:
        def objective(trial):
            c = trial.suggest_float("svr_c", 0.0001, 10.0, log=True)
            epsilon = trial.suggest_float("svr_epsilon", 0.0, 1.0)
            small_dataset_weight_multiplier = (
                trial.suggest_float(
                    "small_dataset_weight_multiplier", 10.0, 10000.0, log=True
                )
                if self.include_large
                else None
            )

            def train_and_evaluate_fold(
                train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                tokenizer,
                lg_size=None,
                fold_num=0,
            ):
                train_features = np.hstack(
                    (
                        np.vstack(train_df["cls"].values),
                        np.vstack(train_df["features"].values),
                    )
                )
                val_features = np.hstack(
                    (
                        np.vstack(val_df["cls"].values),
                        np.vstack(val_df["features"].values),
                    )
                )

                train_labels = train_df["label"].values
                y_true_fold = val_df["label"].values

                regressor = make_pipeline(
                    MinMaxScaler(), SVR(kernel="rbf", C=c, epsilon=epsilon)
                )
                regressor.fit(
                    train_features,
                    train_labels,
                    svr__sample_weight=train_df["weight"].values,
                )

                y_pred_fold = regressor.predict(val_features)
                mse = mean_squared_error(y_true_fold, y_pred_fold)
                return mse

            return run_cross_validation(
                trial=trial,
                train_and_eval_func=train_and_evaluate_fold,
                include_minilm_embeddings=self.include_minilm_embeddings,
                include_modernbert_embeddings=self.include_modernbert_embeddings,
                include_large=self.include_large,
                small_dataset_weight_multiplier=small_dataset_weight_multiplier,
            )

        return objective

    def get_study_name(self) -> str:
        return f"{self.embeddings}_svm{'_lg' if self.include_large else ''}"


class RandomForestObjectiveProvider(ObjectiveProvider):
    def get_n_trials(self):
        return 150

    def get_objective_fn(self) -> callable:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 4, 32),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }

            def train_and_evaluate_fold(
                train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                tokenizer,
                lg_size=None,
                fold_num=0,
            ):
                train_features = np.hstack(
                    (
                        np.vstack(train_df["cls"].values),
                        np.vstack(train_df["features"].values),
                    )
                )
                val_features = np.hstack(
                    (
                        np.vstack(val_df["cls"].values),
                        np.vstack(val_df["features"].values),
                    )
                )

                train_labels = train_df["label"].values
                y_true_fold = val_df["label"].values

                regressor = RandomForestRegressor(
                    **params, random_state=SEED, n_jobs=-1
                )
                regressor.fit(
                    train_features,
                    train_labels,
                    sample_weight=train_df["weight"].values,
                )

                y_pred_fold = regressor.predict(val_features)
                mse = mean_squared_error(y_true_fold, y_pred_fold)
                return mse

            return run_cross_validation(
                trial=trial,
                train_and_eval_func=train_and_evaluate_fold,
                include_minilm_embeddings=self.include_minilm_embeddings,
                include_modernbert_embeddings=self.include_modernbert_embeddings,
                include_large=self.include_large,
            )

        return objective

    def get_study_name(self) -> str:
        return f"{self.embeddings}_rf{'_lg' if self.include_large else ''}"


class CatBoostObjectiveProvider(ObjectiveProvider):
    def get_n_trials(self):
        return 200

    def get_objective_fn(self) -> callable:
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 500, 2000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "depth": trial.suggest_int("depth", 1, 5),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1e-10, 100.0, log=True
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 1.0, 30.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-10, 10.0, log=True
                ),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }

            def train_and_evaluate_fold(
                train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                tokenizer,
                lg_size=None,
                fold_num=0,
            ):
                train_features = np.hstack(
                    (
                        np.vstack(train_df["cls"].values),
                        np.vstack(train_df["features"].values),
                    )
                )
                val_features = np.hstack(
                    (
                        np.vstack(val_df["cls"].values),
                        np.vstack(val_df["features"].values),
                    )
                )

                train_labels = train_df["label"].values
                y_true_fold = val_df["label"].values

                regressor = CatBoostRegressor(
                    **params,
                    random_seed=SEED,
                    verbose=0,
                    early_stopping_rounds=50,
                    thread_count=-1,
                )
                regressor.fit(
                    train_features,
                    train_labels,
                    sample_weight=train_df["weight"].values,
                    eval_set=(val_features, y_true_fold),
                )

                y_pred_fold = regressor.predict(val_features)
                mse = mean_squared_error(y_true_fold, y_pred_fold)
                return mse

            return run_cross_validation(
                trial=trial,
                train_and_eval_func=train_and_evaluate_fold,
                include_minilm_embeddings=self.include_minilm_embeddings,
                include_modernbert_embeddings=self.include_modernbert_embeddings,
                include_large=self.include_large,
            )

        return objective

    def get_study_name(self) -> str:
        return f"{self.embeddings}_catboost{'_lg' if self.include_large else ''}"


class ModernBertFinetuneObjectiveProvider(ObjectiveProvider):
    SEARCH_SPACE = {
        "stage1_epochs": [2, 5, 10],
        "lr_bert": [1e-5, 3e-5],
        "lr_custom": [1e-5, 3e-5, 5e-5, 8e-5],
        "dropout_rate": [0.1, 0.2, 0.3],
        "weight_decay": [0.01, 0.05],
        "optimizer_warmup": [0.1, 0.2],
        "feature_hidden_size": [512, 768, 1024],
        "frozen_bert_epochs": [1, 2, 3, 4, 5],
    }

    def get_n_trials(self) -> int:
        return 20

    def get_search_space(self) -> dict | None:
        return self.SEARCH_SPACE

    def get_objective_fn(self) -> callable:
        def objective(trial):
            params = {
                "stage1_epochs": (
                    trial.suggest_categorical(
                        "stage1_epochs", self.SEARCH_SPACE["stage1_epochs"]
                    )
                    if self.include_large
                    else 0
                ),
                "lr_bert": trial.suggest_categorical(
                    "lr_bert", self.SEARCH_SPACE["lr_bert"]
                ),
                "lr_custom": trial.suggest_categorical(
                    "lr_custom", self.SEARCH_SPACE["lr_custom"]
                ),
                "dropout_rate": trial.suggest_categorical(
                    "dropout_rate", self.SEARCH_SPACE["dropout_rate"]
                ),
                "weight_decay": trial.suggest_categorical(
                    "weight_decay", self.SEARCH_SPACE["weight_decay"]
                ),
                "optimizer_warmup": trial.suggest_categorical(
                    "optimizer_warmup", self.SEARCH_SPACE["optimizer_warmup"]
                ),
                "feature_hidden_size": trial.suggest_categorical(
                    "feature_hidden_size", self.SEARCH_SPACE["feature_hidden_size"]
                ),
                "frozen_bert_epochs": trial.suggest_categorical(
                    "frozen_bert_epochs", self.SEARCH_SPACE["frozen_bert_epochs"]
                ),
            }

            ctx = mp.get_context("spawn")
            with ctx.Pool(1) as pool:
                result = pool.apply(
                    run_modernbert_trial,
                    (
                        trial,
                        params,
                        self.include_large,
                    ),
                )
            if isinstance(result, dict):
                mse = result["mse"]
                best_epochs = result["best_epochs"]
                trial.set_user_attr("mean_best_epoch", float(np.mean(best_epochs)))
                trial.set_user_attr("median_best_epoch", float(np.median(best_epochs)))
            else:
                mse = result
            return mse

        return objective

    def get_study_name(self) -> str:
        return f"finetuned-mbert{'_lg' if self.include_large else ''}"


PROVIDER_MAPPING = {
    "ridge": RidgeObjectiveProvider,
    "svm": SVMObjectiveProvider,
    "rf": RandomForestObjectiveProvider,
    "catboost": CatBoostObjectiveProvider,
    "finetuned-mbert": ModernBertFinetuneObjectiveProvider,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model training with Optuna optimization"
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        required=True,
        choices=["minilm", "mbert"],
        help="Embedding type to use: 'minilm' or 'mbert'",
    )
    parser.add_argument(
        "--lg",
        action="store_true",
        help="Include large dataset in training",
    )
    parser.add_argument(
        "--max-n-trials",
        type=int,
        default=None,
        help="Maximum number of trials to run (caps the provider's default n_trials)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Custom study name to use instead of the default generated name",
    )
    parser.add_argument(
        "providers",
        nargs="*",
        choices=["ridge", "svm", "rf", "catboost", "finetuned-mbert"],
        help="Objective providers to run. If empty, runs all defined providers. Valid options: ridge, svm, rf, catboost, modernbert",
    )
    args = parser.parse_args()

    include_large = args.lg
    embeddings = args.embeddings
    providers_to_run = (
        args.providers if args.providers else list(PROVIDER_MAPPING.keys())
    )

    for provider_name in providers_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running study for provider: {provider_name}")
        print(f"{'=' * 60}\n")

        provider_class = PROVIDER_MAPPING[provider_name]
        provider = provider_class(embeddings=embeddings, include_large=include_large)

        try:
            objective_fn = provider.get_objective_fn()
            study_name = (
                args.study_name if args.study_name else provider.get_study_name()
            )
            n_trials = provider.get_n_trials()

            if args.max_n_trials is not None:
                n_trials = min(n_trials, args.max_n_trials)

            run_study(
                objective_func=objective_fn,
                study_name=study_name,
                search_space=None,
                n_trials=n_trials,
            )
        except NotImplementedError as e:
            print(f"Skipping {provider_name}: {e}")
            continue
