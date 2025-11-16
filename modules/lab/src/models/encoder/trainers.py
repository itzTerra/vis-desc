from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from models.encoder.common import (
    SEED,
    calculate_metrics,
    device,
    CustomDataset,
    CachedOptimizationContext,
)
from models.encoder.modernbert_finetune_nn import ModernBertWithFeaturesTrainable
from text2features import FeatureExtractorPipeline


class BaseTrainer(ABC):
    """Base class for all trainers."""

    def __init__(
        self,
        params: Dict[str, Any],
        embeddings: Optional[Literal["minilm", "mbert"]] = None,
        include_large: bool = False,
        enable_cv: bool = False,
        enable_test: bool = False,
    ):
        self.params = params
        self.embeddings = embeddings
        self.include_large = include_large
        self.enable_cv = enable_cv
        self.enable_test = enable_test
        self.model = None
        self.model_name = self._get_model_name()

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name for file saving."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the model on the full training set."""
        pass

    @abstractmethod
    def cross_validate(self, n_splits: int = 5) -> Dict[str, Any]:
        """Perform cross-validation and return average metrics."""
        pass

    @abstractmethod
    def export(self, model_path: Path) -> None:
        """Export the trained model to disk."""
        pass

    def save_metrics(
        self,
        train_metrics: Dict,
        cv_metrics: Optional[Dict],
        test_metrics: Optional[Dict],
        output_dir: Path,
        **extra_data,
    ):
        """Save training, CV, and test metrics to JSON."""
        metrics = {
            "model": self.model_name,
            "params": self.params,
            "train_metrics": train_metrics,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            **extra_data,
        }
        metrics_path = output_dir / f"{self.model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def run_full_training(self, output_dir: Path) -> Dict[str, Any]:
        """Execute full training pipeline: train, evaluate, cross-validate, export, and save metrics."""
        print(f"\n{'=' * 60}")
        print(f"Training {self.model_name}")
        print(f"{'=' * 60}\n")

        # Train
        self.train()

        # Evaluate on training set
        train_metrics = self.evaluate_train()

        # Cross-validate (optional)
        cv_metrics = None
        if self.enable_cv:
            cv_metrics = self.cross_validate()

        # Export model
        model_path = output_dir / f"{self.model_name}.onnx"
        self.export(model_path)

        # Evaluate on test set (optional)
        test_metrics = None
        if self.enable_test:
            test_metrics = self.evaluate_test(model_path)

        # Save metrics
        extra_data = self._get_extra_metrics_data()
        self.save_metrics(
            train_metrics, cv_metrics, test_metrics, output_dir, **extra_data
        )

        print(f"\nTrain MSE: {train_metrics['mse']:.4f}")
        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
        if cv_metrics:
            print(f"CV MSE: {cv_metrics['mse']:.4f}")
            print(f"CV Accuracy: {cv_metrics['accuracy']:.4f}")
        if test_metrics:
            print(f"Test MSE: {test_metrics['mse']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        return {
            "train_metrics": train_metrics,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }

    @abstractmethod
    def evaluate_train(self) -> Dict[str, Any]:
        """Evaluate model on training set."""
        pass

    @abstractmethod
    def evaluate_test(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate model on test set using saved ONNX model."""
        pass

    def _get_extra_metrics_data(self) -> Dict[str, Any]:
        """Override to add extra data to metrics file."""
        return {}


class BaseSklearnTrainer(BaseTrainer):
    """Base class for sklearn-like models."""

    def __init__(
        self,
        params: Dict[str, Any],
        embeddings: Literal["minilm", "mbert"],
        include_large: bool = False,
        enable_cv: bool = False,
        enable_test: bool = False,
    ):
        super().__init__(params, embeddings, include_large, enable_cv, enable_test)
        self.include_minilm = embeddings == "minilm"
        self.include_mbert = embeddings == "mbert"
        self.X_train = None
        self.y_train = None
        self.weights = None
        self._load_data()

    def _get_model_name(self) -> str:
        base_name = self._get_base_model_name()
        return f"{base_name}_{self.embeddings}{'_lg' if self.include_large else ''}"

    @abstractmethod
    def _get_base_model_name(self) -> str:
        """Return the base model name (e.g., 'ridge', 'svm')."""
        pass

    @abstractmethod
    def _create_model(self):
        """Create and return the sklearn model instance."""
        pass

    def _load_data(self):
        """Load and prepare training data."""
        context = CachedOptimizationContext(
            include_minilm_embeddings=self.include_minilm,
            include_modernbert_embeddings=self.include_mbert,
            include_large=self.include_large,
        )
        train_df = context.sm_train.copy()

        if self.include_large and context.lg_train is not None:
            weight_mult = self.params.get("small_dataset_weight_multiplier", 1.0)
            train_df["weight"] *= weight_mult
            train_df = pd.concat([context.lg_train, train_df], ignore_index=True)

        self.X_train = np.hstack(
            (
                np.vstack(train_df["cls"].values),
                np.vstack(train_df["features"].values),
            )
        )
        self.y_train = train_df["label"].values
        self.weights = train_df["weight"].values

    def train(self) -> None:
        """Train the sklearn model."""
        self.model = self._create_model()
        fit_params = self._get_fit_params()
        self.model.fit(self.X_train, self.y_train, **fit_params)

    @abstractmethod
    def _get_fit_params(self) -> Dict[str, Any]:
        """Return fit parameters including sample weights."""
        pass

    def evaluate_train(self) -> Dict[str, Any]:
        """Evaluate model on training set."""
        y_pred_train = self.model.predict(self.X_train)
        return calculate_metrics(self.y_train, y_pred_train)

    def cross_validate(self, n_splits: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on small dataset only."""
        context = CachedOptimizationContext(
            include_minilm_embeddings=self.include_minilm,
            include_modernbert_embeddings=self.include_mbert,
            include_large=False,
        )
        sm_train = context.sm_train

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        fold_metrics = []

        for fold, (train_index, val_index) in enumerate(kf.split(sm_train)):
            train_fold_df = sm_train.loc[train_index].copy()
            val_fold_df = sm_train.loc[val_index].copy()

            X_train_fold = np.hstack(
                (
                    np.vstack(train_fold_df["cls"].values),
                    np.vstack(train_fold_df["features"].values),
                )
            )
            X_val_fold = np.hstack(
                (
                    np.vstack(val_fold_df["cls"].values),
                    np.vstack(val_fold_df["features"].values),
                )
            )
            y_train_fold = train_fold_df["label"].values
            y_val_fold = val_fold_df["label"].values
            weights_fold = train_fold_df["weight"].values

            # Train fold model
            fold_model = self._create_model()
            fit_params = self._get_fit_params_for_weights(weights_fold)
            fold_model.fit(X_train_fold, y_train_fold, **fit_params)

            # Evaluate
            y_pred = fold_model.predict(X_val_fold)
            metrics = calculate_metrics(y_val_fold, y_pred)
            fold_metrics.append(metrics)

            print(
                f"Fold {fold + 1}/{n_splits}: MSE={metrics['mse']:.4f}, Acc={metrics['accuracy']:.4f}"
            )

        # Average metrics across folds
        return self._average_fold_metrics(fold_metrics)

    def _get_fit_params_for_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        """Get fit params for a specific weight array (used in CV)."""
        # Default implementation - override if needed
        return self._get_fit_params()

    def _average_fold_metrics(self, fold_metrics: list) -> Dict[str, Any]:
        """Average metrics across folds."""
        return {
            "mse": np.mean([m["mse"] for m in fold_metrics]),
            "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "precision": np.mean(
                [m["precision"] for m in fold_metrics], axis=0
            ).tolist(),
            "recall": np.mean([m["recall"] for m in fold_metrics], axis=0).tolist(),
            "f1": np.mean([m["f1"] for m in fold_metrics], axis=0).tolist(),
            "support": np.sum([m["support"] for m in fold_metrics], axis=0).tolist(),
            "confusion_matrix": np.sum(
                [m["confusion_matrix"] for m in fold_metrics], axis=0
            ).tolist(),
        }

    def evaluate_test(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate model on test set using saved ONNX model."""
        import onnxruntime as rt
        from utils import DATA_DIR

        # Load test data
        test_df = pd.read_parquet(DATA_DIR / "datasets" / "small" / "test.parquet")

        # Load embeddings
        if self.include_minilm:
            minilm_embeddings = pd.read_parquet(
                DATA_DIR / "datasets" / "small" / "minilm_embeddings_test.parquet"
            )
            test_df = test_df.merge(
                minilm_embeddings[["cls"]], left_index=True, right_index=True
            )
        elif self.include_mbert:
            modernbert_embeddings = pd.read_parquet(
                DATA_DIR
                / "datasets"
                / "small"
                / "modernbert_cls_embeddings_test.parquet"
            )
            test_df = test_df.merge(
                modernbert_embeddings[["cls"]], left_index=True, right_index=True
            )

        # Prepare test features
        X_test = np.hstack(
            (
                np.vstack(test_df["cls"].values),
                np.vstack(test_df["features"].values),
            )
        )
        y_test = test_df["label"].values

        # Load ONNX model and predict
        sess = rt.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        y_pred = sess.run([label_name], {input_name: X_test.astype(np.float32)})[
            0
        ].flatten()

        return calculate_metrics(y_test, y_pred)

    def export(self, model_path: Path) -> None:
        """Export sklearn model to ONNX format."""
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnxruntime as rt

        n_features = self.X_train.shape[1]
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(self.model, initial_types=initial_type, target_opset=15)

        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())

        print(f"ONNX model saved to {model_path}")

        sess = rt.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        sample_X = self.X_train.astype(np.float32)
        pred_onx = sess.run([label_name], {input_name: sample_X})[0]
        pred_sklearn = self.model.predict(sample_X)

        diff = np.abs(pred_sklearn - pred_onx.flatten())
        if not np.all(diff < 1e-5):
            print(
                "Warning: Insufficient accuracy between sklearn and ONNX predictions."
            )
            print(f"Max difference: {np.max(diff)}")


class RidgeTrainer(BaseSklearnTrainer):
    """Trainer for Ridge regression."""

    def _get_base_model_name(self) -> str:
        return "ridge"

    def _create_model(self):
        return make_pipeline(
            MinMaxScaler(),
            Ridge(alpha=self.params["ridge_alpha"], random_state=SEED),
        )

    def _get_fit_params(self) -> Dict[str, Any]:
        return {"ridge__sample_weight": self.weights}

    def _get_fit_params_for_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        return {"ridge__sample_weight": weights}


class SVMTrainer(BaseSklearnTrainer):
    """Trainer for SVM regression."""

    def _get_base_model_name(self) -> str:
        return "svm"

    def _create_model(self):
        return make_pipeline(
            MinMaxScaler(),
            SVR(
                kernel="rbf",
                C=self.params["svr_c"],
                epsilon=self.params["svr_epsilon"],
            ),
        )

    def _get_fit_params(self) -> Dict[str, Any]:
        return {"svr__sample_weight": self.weights}

    def _get_fit_params_for_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        return {"svr__sample_weight": weights}


class RandomForestTrainer(BaseSklearnTrainer):
    """Trainer for Random Forest regression."""

    def _get_base_model_name(self) -> str:
        return "rf"

    def _create_model(self):
        # Extract RF-specific params
        rf_params = {
            k: v
            for k, v in self.params.items()
            if k not in ["small_dataset_weight_multiplier"]  # Exclude non-RF params
        }
        return RandomForestRegressor(**rf_params, random_state=SEED, n_jobs=-1)

    def _get_fit_params(self) -> Dict[str, Any]:
        return {"sample_weight": self.weights}

    def _get_fit_params_for_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        return {"sample_weight": weights}


class CatBoostTrainer(BaseSklearnTrainer):
    """Trainer for CatBoost regression."""

    def _get_base_model_name(self) -> str:
        return "catboost"

    def _create_model(self):
        # Extract CatBoost-specific params
        cb_params = {
            k: v
            for k, v in self.params.items()
            if k not in ["small_dataset_weight_multiplier"]  # Exclude non-CB params
        }
        return CatBoostRegressor(
            **cb_params, random_seed=SEED, verbose=100, thread_count=-1
        )

    def _get_fit_params(self) -> Dict[str, Any]:
        return {"sample_weight": self.weights}

    def _get_fit_params_for_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        return {"sample_weight": weights}


class ModernBertTrainer(BaseTrainer):
    """Trainer for fine-tuned ModernBERT model."""

    def __init__(
        self,
        params: Dict[str, Any],
        include_large: bool = False,
        enable_cv: bool = False,
        enable_test: bool = False,
    ):
        super().__init__(
            params,
            embeddings=None,
            include_large=include_large,
            enable_cv=enable_cv,
            enable_test=enable_test,
        )
        self.tokenizer = None
        self.train_df = None
        self.batch_losses = []
        self._load_data()

    def _get_model_name(self) -> str:
        return f"finetuned_mbert{'_lg' if self.include_large else ''}"

    def _load_data(self):
        """Load and prepare training data."""
        context = CachedOptimizationContext(
            include_minilm_embeddings=False,
            include_modernbert_embeddings=False,
            include_large=self.include_large,
        )
        self.tokenizer = context.tokenizer
        train_df = context.sm_train.copy()

        if self.include_large and context.lg_train is not None:
            train_df = pd.concat([context.lg_train, train_df], ignore_index=True)

        scaler = MinMaxScaler()
        train_features_scaled = scaler.fit_transform(
            np.vstack(train_df["features"].values)
        )
        train_df["features"] = [f for f in np.nan_to_num(train_features_scaled)]

        self.train_df = train_df

    def train(self) -> None:
        """Train the ModernBERT model."""
        train_dataset = CustomDataset(self.train_df, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"])

        self.model = ModernBertWithFeaturesTrainable.from_pretrained(
            "answerdotai/ModernBERT-base",
            feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
            dropout_rate=self.params["dropout_rate"],
            feature_hidden_size=self.params["feature_hidden_size"],
        )
        self.model.feature_ff.apply(self.model._init_custom_weights)
        self.model.regressor.apply(self.model._init_custom_weights)
        self.model.to(device)

        optimizer = AdamW(
            [
                {"params": self.model.model.parameters(), "lr": self.params["lr_bert"]},
                {
                    "params": self.model.feature_ff.parameters(),
                    "lr": self.params["lr_custom"],
                },
                {
                    "params": self.model.regressor.parameters(),
                    "lr": self.params["lr_custom"],
                },
            ],
            weight_decay=self.params["weight_decay"],
        )
        total_steps = len(train_loader) * self.params["num_epochs"]
        warmup_steps = int(total_steps * self.params["optimizer_warmup"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.batch_losses = []
        self.model.train()
        for epoch in range(self.params["num_epochs"]):
            epoch_losses = []
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.params['num_epochs']}"
            )
            for batch in progress_bar:
                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    features=batch["features"].to(device),
                    labels=batch["labels"].to(device),
                )

                loss = outputs.loss
                if torch.isnan(loss):
                    print("Loss is NaN, skipping backward pass.")
                    continue

                loss_value = loss.item()
                epoch_losses.append(loss_value)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix({"loss": loss_value})

                del loss, outputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            self.batch_losses.append(epoch_losses)
            print(f"Epoch {epoch + 1} avg loss: {np.mean(epoch_losses):.4f}")

    def evaluate_train(self) -> Dict[str, Any]:
        """Evaluate model on training set."""
        train_dataset = CustomDataset(self.train_df, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"])

        self.model.eval()
        y_true_train, y_pred_train = [], []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Evaluating training set"):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    features=batch["features"].to(device),
                    labels=batch["labels"].to(device),
                )
                predictions = outputs.logits.squeeze()
                y_true_train.extend(batch["labels"].cpu().numpy())
                y_pred_train.extend(predictions.cpu().numpy())

        return calculate_metrics(np.array(y_true_train), np.array(y_pred_train))

    def cross_validate(self, n_splits: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on small dataset only."""
        context = CachedOptimizationContext(
            include_minilm_embeddings=False,
            include_modernbert_embeddings=False,
            include_large=False,
        )
        sm_train = context.sm_train

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        fold_metrics = []
        fold_batch_losses = []

        for fold, (train_index, val_index) in enumerate(kf.split(sm_train)):
            print(f"\nFold {fold + 1}/{n_splits}")
            train_fold_df = sm_train.loc[train_index].copy().reset_index(drop=True)
            val_fold_df = sm_train.loc[val_index].copy().reset_index(drop=True)

            # Scale features
            scaler = MinMaxScaler()
            train_features_scaled = scaler.fit_transform(
                np.vstack(train_fold_df["features"].values)
            )
            val_features_scaled = scaler.transform(
                np.vstack(val_fold_df["features"].values)
            )
            train_fold_df["features"] = [
                f for f in np.nan_to_num(train_features_scaled)
            ]
            val_fold_df["features"] = [f for f in np.nan_to_num(val_features_scaled)]

            # Create datasets
            train_dataset = CustomDataset(train_fold_df, self.tokenizer)
            val_dataset = CustomDataset(val_fold_df, self.tokenizer)
            train_loader = DataLoader(
                train_dataset, batch_size=self.params["batch_size"]
            )
            val_loader = DataLoader(val_dataset, batch_size=self.params["batch_size"])

            # Initialize model
            fold_model = ModernBertWithFeaturesTrainable.from_pretrained(
                "answerdotai/ModernBERT-base",
                feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
                dropout_rate=self.params["dropout_rate"],
                feature_hidden_size=self.params["feature_hidden_size"],
            )
            fold_model.feature_ff.apply(fold_model._init_custom_weights)
            fold_model.regressor.apply(fold_model._init_custom_weights)
            fold_model.to(device)

            # Setup optimizer and scheduler
            optimizer = AdamW(
                [
                    {
                        "params": fold_model.model.parameters(),
                        "lr": self.params["lr_bert"],
                    },
                    {
                        "params": fold_model.feature_ff.parameters(),
                        "lr": self.params["lr_custom"],
                    },
                    {
                        "params": fold_model.regressor.parameters(),
                        "lr": self.params["lr_custom"],
                    },
                ],
                weight_decay=self.params["weight_decay"],
            )

            total_steps = len(train_loader) * self.params["num_epochs"]
            warmup_steps = int(total_steps * self.params["optimizer_warmup"])
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

            # Train
            batch_losses = []
            fold_model.train()
            for epoch in range(self.params["num_epochs"]):
                epoch_losses = []
                for batch in train_loader:
                    optimizer.zero_grad()

                    outputs = fold_model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        features=batch["features"].to(device),
                        labels=batch["labels"].to(device),
                    )

                    loss = outputs.loss
                    if torch.isnan(loss):
                        continue

                    epoch_losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    del loss, outputs
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                batch_losses.append(epoch_losses)

            # Evaluate
            fold_model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = fold_model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        features=batch["features"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    predictions = outputs.logits.squeeze()
                    y_true.extend(batch["labels"].cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())

            metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
            fold_metrics.append(metrics)
            fold_batch_losses.append(batch_losses)

            print(
                f"Fold {fold + 1} MSE: {metrics['mse']:.4f}, Acc: {metrics['accuracy']:.4f}"
            )

        # Average metrics across folds
        avg_metrics = {
            "mse": np.mean([m["mse"] for m in fold_metrics]),
            "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
            "precision": np.mean(
                [m["precision"] for m in fold_metrics], axis=0
            ).tolist(),
            "recall": np.mean([m["recall"] for m in fold_metrics], axis=0).tolist(),
            "f1": np.mean([m["f1"] for m in fold_metrics], axis=0).tolist(),
            "support": np.sum([m["support"] for m in fold_metrics], axis=0).tolist(),
            "confusion_matrix": np.sum(
                [m["confusion_matrix"] for m in fold_metrics], axis=0
            ).tolist(),
            "batch_losses": fold_batch_losses,
        }

        return avg_metrics

    def evaluate_test(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate model on test set using saved ONNX model."""
        import onnxruntime as rt
        from utils import DATA_DIR

        # Load test data
        test_df = pd.read_parquet(DATA_DIR / "datasets" / "small" / "test.parquet")

        # Scale features
        scaler = MinMaxScaler()
        # Fit scaler on training data features
        train_features = np.vstack(self.train_df["features"].values)
        scaler.fit(train_features)

        # Transform test features
        test_features_scaled = scaler.transform(np.vstack(test_df["features"].values))
        test_df["features"] = [f for f in np.nan_to_num(test_features_scaled)]

        # Create test dataset
        test_dataset = CustomDataset(test_df, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=self.params["batch_size"])

        # Load ONNX model
        sess = rt.InferenceSession(str(model_path))

        # Run inference
        y_true_test, y_pred_test = [], []
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            input_ids = batch["input_ids"].numpy()
            attention_mask = batch["attention_mask"].numpy()
            features = batch["features"].numpy()
            labels = batch["labels"].numpy()

            # Run ONNX inference
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "features": features,
            }
            outputs = sess.run(None, onnx_inputs)
            predictions = outputs[0].squeeze()

            y_true_test.extend(labels)
            y_pred_test.extend(predictions)

        return calculate_metrics(np.array(y_true_test), np.array(y_pred_test))

    def export(self, model_path: Path) -> None:
        """Export ModernBERT model to ONNX format."""
        self.model.eval()
        self.model.to("cpu")

        dummy_input_ids = torch.randint(
            0, self.model.config.vocab_size, (1, 160), dtype=torch.long
        )
        dummy_attention_mask = torch.ones(1, 160, dtype=torch.long)
        dummy_features = torch.randn(
            1, FeatureExtractorPipeline.FEATURE_COUNT, dtype=torch.float
        )

        input_names = ["input_ids", "attention_mask", "features"]
        output_names = ["logits"]

        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask, dummy_features),
            model_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "features": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )
        print(f"ONNX model saved to {model_path}")

    def _get_extra_metrics_data(self) -> Dict[str, Any]:
        """Add batch losses to metrics."""
        return {"batch_losses": self.batch_losses}
