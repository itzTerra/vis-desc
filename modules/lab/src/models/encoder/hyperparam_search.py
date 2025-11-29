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

HYPERPARAM_SEARCH_DIR = TRAINING_HISTORY_DIR / "hyperparam_search"


def save_trial_overview(trial_number: int, params: dict, seed_data: list[dict]):
    """Save an overview JSON with MSEs, accuracies, and best epochs for a trial.

    Parameters
    ----------
    trial_number : int
        The trial number.
    params : dict
        Hyperparameters for this trial.
    seed_data : list[dict]
        List of dictionaries, one per seed, each containing:
        - seed: int
        - mse: float (average across folds)
        - folds: list of dicts with fold_num, mse, accuracy, best_epoch
    """
    overview = {
        "trial_number": trial_number,
        "hyperparameters": params,
        "seeds": seed_data,
        "summary": {
            "mean_mse_across_seeds": float(np.mean([s["mse"] for s in seed_data])),
            "mean_accuracy_across_seeds": float(
                np.mean(
                    [np.mean([f["accuracy"] for f in s["folds"]]) for s in seed_data]
                )
            ),
            "mean_best_epoch_across_all": float(
                np.mean(
                    [
                        f["best_epoch"]
                        for s in seed_data
                        for f in s["folds"]
                        if f["best_epoch"] is not None
                    ]
                )
            )
            if any(f["best_epoch"] is not None for s in seed_data for f in s["folds"])
            else None,
        },
    }

    overview_path = HYPERPARAM_SEARCH_DIR / f"trial_{trial_number}_overview.json"
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=2)
    print(f"Saved trial overview to {overview_path}")


def run_modernbert_trial_single_seed(
    trial,
    trial_params: dict,
    include_large: bool,
    seed: int,
):
    """Execute a single ModernBERT fine-tuning trial with one seed in an isolated subprocess.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object.
    trial_params : dict
        Hyperparameters for this trial.
    include_large : bool
        Whether to include the large (noisy) dataset in stage 1 pretraining.
    seed : int
        Random seed for this run.

    Returns
    -------
    dict
        Dictionary containing 'mse' and 'best_epochs' for this seed.
    """
    from models.encoder.modernbert_finetune_nn import train_finetuned_mbert

    set_seed(seed)
    fold_details = []

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
            / f"finetuned-mbert_trial_{trial.number}_seed_{seed}_fold_{fold_num}.json"
        )

        result = train_finetuned_mbert(
            train_df=sm_train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            params=trial_params,
            seed=seed,
            train_lg_df=lg_train_df,
            save_history=True,
            history_path=history_path,
        )

        if history_path.exists():
            with open(history_path, "r") as f:
                history_data = json.load(f)
            history_data["trial_number"] = trial.number
            history_data["seed"] = seed
            history_data["fold_num"] = fold_num
            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2)
            best_epoch = history_data.get("best_epoch")
        else:
            best_epoch = None

        final_metrics = result["final_metrics"]
        fold_details.append(
            {
                "fold_num": fold_num,
                "mse": float(final_metrics["mse"]),
                "accuracy": float(final_metrics["accuracy"]),
                "best_epoch": best_epoch,
            }
        )

        return final_metrics["mse"], best_epoch

    seed_result = run_cross_validation(
        trial=trial,
        train_and_eval_func=train_and_evaluate_fold,
        include_minilm_embeddings=False,
        include_modernbert_embeddings=False,
        include_large=include_large,
        n_splits=5,
        two_stage_training=True,
    )

    if isinstance(seed_result, dict):
        seed_result["folds"] = fold_details
    else:
        seed_result = {"mse": seed_result, "best_epochs": [], "folds": fold_details}

    return seed_result


def run_modernbert_trial_multi_seed(
    trial,
    trial_params: dict,
    include_large: bool,
    seeds: list[int],
):
    """Run ModernBERT trial across multiple seeds using separate subprocesses.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object.
    trial_params : dict
        Hyperparameters for this trial.
    include_large : bool
        Whether to include the large (noisy) dataset in stage 1 pretraining.
    seeds : list[int]
        List of seeds to run in parallel subprocesses.

    Returns
    -------
    dict
        Dictionary containing averaged 'mse' and collected 'best_epochs'.
    """
    all_seed_results = []
    seed_data = []
    ctx = mp.get_context("spawn")

    for seed_idx, seed in enumerate(seeds):
        print(f"  Running seed {seed} (#{seed_idx + 1}/{len(seeds)})...")
        with ctx.Pool(1) as pool:
            seed_result = pool.apply(
                run_modernbert_trial_single_seed,
                (trial, trial_params, include_large, seed),
            )
        all_seed_results.append(seed_result)
        print(
            f"  Seed {seed} completed: MSE = {seed_result['mse'] if isinstance(seed_result, dict) else seed_result:.4f}"
        )

        if isinstance(seed_result, dict) and "folds" in seed_result:
            seed_data.append(
                {
                    "seed": seed,
                    "mse": seed_result["mse"],
                    "folds": seed_result["folds"],
                }
            )

    if seed_data:
        save_trial_overview(trial.number, trial_params, seed_data)

    if isinstance(all_seed_results[0], dict):
        avg_mse = np.mean([r["mse"] for r in all_seed_results])
        all_best_epochs = [
            epoch for r in all_seed_results for epoch in r["best_epochs"]
        ]
        return {"mse": avg_mse, "best_epochs": all_best_epochs}
    else:
        return {"mse": np.mean(all_seed_results), "best_epochs": []}


class ObjectiveProvider:
    def __init__(
        self,
        embeddings: Literal["minilm", "mbert"],
        include_large: bool,
        seeds: list[int] | None = None,
    ):
        self.embeddings = embeddings
        self.include_minilm_embeddings = embeddings == "minilm"
        self.include_modernbert_embeddings = embeddings == "mbert"
        self.include_large = include_large
        self.seeds = seeds

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
        "stage1_epochs": [3, 6],
        "lr_bert": [1e-5],
        "lr_custom": [5e-5, 1e-4],
        "dropout_rate": [0.1, 0.3],
        "weight_decay": [0.01],
        "optimizer_warmup": [0.2],
        "feature_hidden_size": [768],
        "stage1_frozen_bert_epochs": [1],
        "stage2_frozen_bert_epochs": [0, 5],
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
                "stage1_frozen_bert_epochs": (
                    trial.suggest_categorical(
                        "stage1_frozen_bert_epochs",
                        self.SEARCH_SPACE["stage1_frozen_bert_epochs"],
                    )
                    if self.include_large
                    else 0
                ),
                "stage2_frozen_bert_epochs": trial.suggest_categorical(
                    "stage2_frozen_bert_epochs",
                    self.SEARCH_SPACE["stage2_frozen_bert_epochs"],
                ),
            }

            seeds = self.seeds if self.seeds else [SEED]

            if len(seeds) == 1:
                ctx = mp.get_context("spawn")
                with ctx.Pool(1) as pool:
                    result = pool.apply(
                        run_modernbert_trial_single_seed,
                        (trial, params, self.include_large, seeds[0]),
                    )
                if isinstance(result, dict) and "folds" in result:
                    seed_data = [
                        {
                            "seed": seeds[0],
                            "mse": result["mse"],
                            "folds": result["folds"],
                        }
                    ]
                    save_trial_overview(trial.number, params, seed_data)
            else:
                result = run_modernbert_trial_multi_seed(
                    trial, params, self.include_large, seeds
                )

            if isinstance(result, dict):
                mse = result["mse"]
                best_epochs = result["best_epochs"]
                if best_epochs:
                    trial.set_user_attr("mean_best_epoch", float(np.mean(best_epochs)))
                    trial.set_user_attr(
                        "median_best_epoch", float(np.median(best_epochs))
                    )
                if len(seeds) > 1:
                    trial.set_user_attr("num_seeds", len(seeds))
                    trial.set_user_attr("seeds_used", seeds)
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
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to use for finetuned-mbert trials (runs multiple times and averages results). Default uses single seed (42).",
    )
    parser.add_argument(
        "providers",
        nargs="*",
        choices=["ridge", "svm", "rf", "catboost", "finetuned-mbert"],
        help="Objective providers to run. If empty, runs all defined providers. Valid options: ridge, svm, rf, catboost, modernbert",
    )
    args = parser.parse_args()

    HYPERPARAM_SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    include_large = args.lg
    embeddings = args.embeddings
    seeds = args.seeds
    providers_to_run = (
        args.providers if args.providers else list(PROVIDER_MAPPING.keys())
    )

    for provider_name in providers_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running study for provider: {provider_name}")
        if seeds and provider_name == "finetuned-mbert":
            print(f"Using seeds: {seeds}")
        print(f"{'=' * 60}\n")

        provider_class = PROVIDER_MAPPING[provider_name]
        # Only pass seeds to finetuned-mbert provider
        if provider_name == "finetuned-mbert":
            provider = provider_class(
                embeddings=embeddings, include_large=include_large, seeds=seeds
            )
        else:
            provider = provider_class(
                embeddings=embeddings, include_large=include_large
            )

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
