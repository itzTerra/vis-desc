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
from models.encoder.modernbert_finetune_nn import (
    check_gradient_flow,
)
import json


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
        "stage1_epochs": [2, 5, 10],  # Epochs for large dataset pretraining
        "lr_bert": [5e-6],
        "lr_custom": [8e-5],
        "dropout_rate": [0.1],
        "weight_decay": [0.01],
        "optimizer_warmup": [0.1],
        "feature_hidden_size": [512],
        # "regressor_hidden_size": [256, 512],
    }
    BATCH_SIZE = 64
    STAGE2_MAX_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    def get_n_trials(self) -> int:
        return 20

    def get_search_space(self) -> dict | None:
        return self.SEARCH_SPACE

    def get_objective_fn(self) -> callable:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
        from tqdm.auto import tqdm
        from text2features import FeatureExtractorPipeline
        from models.encoder.common import device, CustomDataset
        from models.encoder.modernbert_finetune_nn import (
            ModernBertWithFeaturesTrainable,
        )

        def objective(trial):
            stage1_epochs = (
                trial.suggest_categorical(
                    "stage1_epochs", self.SEARCH_SPACE["stage1_epochs"]
                )
                if self.include_large
                else 0
            )
            lr_bert = trial.suggest_categorical("lr_bert", self.SEARCH_SPACE["lr_bert"])
            lr_custom = trial.suggest_categorical(
                "lr_custom", self.SEARCH_SPACE["lr_custom"]
            )
            dropout_rate = trial.suggest_categorical(
                "dropout_rate", self.SEARCH_SPACE["dropout_rate"]
            )
            weight_decay = trial.suggest_categorical(
                "weight_decay", self.SEARCH_SPACE["weight_decay"]
            )
            optimizer_warmup = trial.suggest_categorical(
                "optimizer_warmup", self.SEARCH_SPACE["optimizer_warmup"]
            )
            feature_hidden_size = trial.suggest_categorical(
                "feature_hidden_size", self.SEARCH_SPACE["feature_hidden_size"]
            )

            def train_and_evaluate_fold(
                train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                tokenizer,
                lg_size=None,
                fold_num=0,
            ):
                # Separate small and large datasets if large is included
                if self.include_large and lg_size is not None:
                    # The train_df contains both large and small datasets concatenated
                    # Large dataset comes first, small dataset at the end (see run_cross_validation)
                    lg_train_df = train_df.iloc[:lg_size].copy()
                    sm_train_df = train_df.iloc[lg_size:].copy()
                else:
                    lg_train_df = None
                    sm_train_df = train_df.copy()

                # Scale features for all datasets
                scaler = MinMaxScaler()

                if lg_train_df is not None and not lg_train_df.empty:
                    lg_features_scaled = scaler.fit_transform(
                        np.vstack(lg_train_df["features"].values)
                    )
                    lg_train_df["features"] = [
                        f for f in np.nan_to_num(lg_features_scaled)
                    ]

                    if not sm_train_df.empty:
                        sm_features_scaled = scaler.transform(
                            np.vstack(sm_train_df["features"].values)
                        )
                        sm_train_df["features"] = [
                            f for f in np.nan_to_num(sm_features_scaled)
                        ]
                else:
                    sm_features_scaled = scaler.fit_transform(
                        np.vstack(sm_train_df["features"].values)
                    )
                    sm_train_df["features"] = [
                        f for f in np.nan_to_num(sm_features_scaled)
                    ]

                val_features_scaled = scaler.transform(
                    np.vstack(val_df["features"].values)
                )
                val_df["features"] = [f for f in np.nan_to_num(val_features_scaled)]

                g = torch.Generator()
                g.manual_seed(SEED)

                val_dataset = CustomDataset(val_df, tokenizer)
                val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE)

                set_seed()

                model = ModernBertWithFeaturesTrainable.from_pretrained(
                    "answerdotai/ModernBERT-base",
                    feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
                    dropout_rate=dropout_rate,
                    feature_hidden_size=feature_hidden_size,
                )
                model.to(device)

                # Stage 1: Train on large noisy dataset (if available)
                if (
                    self.include_large
                    and lg_train_df is not None
                    and not lg_train_df.empty
                ):
                    print(
                        f"Stage 1: Training on large dataset ({len(lg_train_df)} samples) for {stage1_epochs} epochs"
                    )
                    lg_train_dataset = CustomDataset(lg_train_df, tokenizer)
                    lg_train_loader = DataLoader(
                        lg_train_dataset,
                        batch_size=self.BATCH_SIZE,
                        shuffle=True,
                        generator=g,
                    )

                    optimizer = AdamW(
                        [
                            {"params": model.model.parameters(), "lr": lr_bert},
                            {"params": model.feature_ff.parameters(), "lr": lr_custom},
                            {"params": model.regressor.parameters(), "lr": lr_custom},
                        ],
                        weight_decay=weight_decay,
                    )

                    total_steps = len(lg_train_loader) * stage1_epochs
                    warmup_steps = int(total_steps * optimizer_warmup)
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps,
                    )

                    model.train()
                    for epoch in range(stage1_epochs):
                        progress_bar = tqdm(
                            lg_train_loader,
                            desc=f"Stage 1 - Epoch {epoch + 1}/{stage1_epochs}",
                        )
                        for batch in progress_bar:
                            optimizer.zero_grad()

                            outputs = model(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                features=batch["features"].to(device),
                                labels=batch["labels"].to(device),
                            )

                            loss = outputs.loss
                            if torch.isnan(loss):
                                print("Loss is NaN, skipping backward pass.")
                                continue

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            progress_bar.set_postfix({"loss": loss.item()})

                            del loss, outputs
                            if device.type == "cuda":
                                torch.cuda.empty_cache()

                # Stage 2: Fine-tune on small clean dataset with early stopping
                if not sm_train_df.empty:
                    print(
                        f"Stage 2: Fine-tuning on small dataset ({len(sm_train_df)} samples) with early stopping (max {self.STAGE2_MAX_EPOCHS} epochs)"
                    )
                    sm_train_dataset = CustomDataset(sm_train_df, tokenizer)
                    sm_train_loader = DataLoader(
                        sm_train_dataset,
                        batch_size=self.BATCH_SIZE,
                        shuffle=True,
                        generator=g,
                    )

                    optimizer = AdamW(
                        [
                            {"params": model.model.parameters(), "lr": lr_bert},
                            {"params": model.feature_ff.parameters(), "lr": lr_custom},
                            {"params": model.regressor.parameters(), "lr": lr_custom},
                        ],
                        weight_decay=weight_decay,
                    )

                    total_steps = len(sm_train_loader) * self.STAGE2_MAX_EPOCHS
                    warmup_steps = int(total_steps * optimizer_warmup)
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=total_steps,
                    )

                    best_val_loss = float("inf")
                    patience_counter = 0
                    train_loss_history = []  # [(batch_number, loss), ...]
                    val_loss_history = []  # [(batch_number, loss), ...]
                    total_batches = 0
                    batches_per_epoch = len(sm_train_loader)

                    history_path = (
                        TRAINING_HISTORY_DIR
                        / f"finetuned-mbert_trial_{trial.number}_fold_{fold_num}.json"
                    )

                    model.train()
                    for epoch in range(self.STAGE2_MAX_EPOCHS):
                        # Training phase
                        epoch_train_losses = []
                        progress_bar = tqdm(
                            sm_train_loader,
                            desc=f"Stage 2 - Epoch {epoch + 1}/{self.STAGE2_MAX_EPOCHS}",
                        )
                        for step, batch in enumerate(progress_bar):
                            optimizer.zero_grad()

                            outputs = model(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                features=batch["features"].to(device),
                                labels=batch["labels"].to(device),
                            )

                            loss = outputs.loss

                            if torch.isnan(loss) or torch.isinf(loss):
                                print(
                                    f"Invalid loss detected: {loss.item()}, skipping batch"
                                )
                                continue
                            if loss.item() > 1000:
                                print(
                                    f"Warning: Very high loss {loss.item():.2f}, clipping"
                                )
                                loss = torch.clamp(loss, max=100)

                            current_batch = total_batches + step
                            train_loss_history.append((current_batch, loss.item()))
                            epoch_train_losses.append(loss.item())

                            torch.autograd.set_detect_anomaly(True)
                            loss.backward()

                            if step == 0 or (step + 1) % 10 == 0:
                                check_gradient_flow(model, step + 1, epoch + 1)

                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            torch.autograd.set_detect_anomaly(False)
                            scheduler.step()
                            progress_bar.set_postfix({"loss": loss.item()})

                            del loss, outputs
                            if device.type == "cuda":
                                torch.cuda.empty_cache()

                        total_batches += batches_per_epoch

                        # Validation phase
                        model.eval()
                        epoch_val_losses = []
                        with torch.no_grad():
                            for batch in tqdm(
                                val_loader, desc="Validating", leave=False
                            ):
                                outputs = model(
                                    input_ids=batch["input_ids"].to(device),
                                    attention_mask=batch["attention_mask"].to(device),
                                    features=batch["features"].to(device),
                                    labels=batch["labels"].to(device),
                                )
                                epoch_val_losses.append(outputs.loss.item())

                        avg_train_loss = np.mean(epoch_train_losses)
                        avg_val_loss = np.mean(epoch_val_losses)
                        val_loss_history.append((total_batches - 1, avg_val_loss))
                        print(
                            f"\nEpoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
                        )

                        # Early stopping check
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            print(f"✓ New best validation loss: {best_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            print(
                                f"✗ No improvement. Patience: {patience_counter}/{self.EARLY_STOPPING_PATIENCE}"
                            )

                            if patience_counter >= self.EARLY_STOPPING_PATIENCE:
                                print(f"Early stopping triggered at epoch {epoch + 1}")
                                break

                        model.train()

                    # Save training history after all epochs
                    history_data = {
                        "trial_number": trial.number,
                        "fold_num": fold_num,
                        "batches_per_epoch": batches_per_epoch,
                        "current_epoch": epoch + 1,
                        "train_loss_history": train_loss_history,  # [(batch, loss), ...]
                        "val_loss_history": val_loss_history,  # [(batch, loss), ...]
                        "hyperparameters": {
                            "stage1_epochs": stage1_epochs if self.include_large else 0,
                            "lr_bert": lr_bert,
                            "lr_custom": lr_custom,
                            "dropout_rate": dropout_rate,
                            "weight_decay": weight_decay,
                            "optimizer_warmup": optimizer_warmup,
                            "feature_hidden_size": feature_hidden_size,
                        },
                    }
                    with open(history_path, "w") as f:
                        json.dump(history_data, f, indent=2)

                # Validate
                model.eval()
                y_true_fold, y_pred_fold = [], []
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Evaluating"):
                        outputs = model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            features=batch["features"].to(device),
                            labels=batch["labels"].to(device),
                        )
                        predictions = outputs.logits.squeeze()

                        y_true_fold.extend(batch["labels"].cpu().numpy())
                        y_pred_fold.extend(predictions.cpu().numpy())

                mse = mean_squared_error(y_true_fold, y_pred_fold)
                return mse

            return run_cross_validation(
                trial=trial,
                train_and_eval_func=train_and_evaluate_fold,
                include_minilm_embeddings=False,
                include_modernbert_embeddings=False,
                include_large=self.include_large,
                n_splits=5,
                two_stage_training=True,
            )

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
