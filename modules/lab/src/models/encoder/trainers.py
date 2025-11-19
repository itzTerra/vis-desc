from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
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
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import onnxruntime as rt
from utils import DATA_DIR

from models.encoder.common import (
    METRICS_DIR,
    MODEL_DIR,
    SEED,
    calculate_metrics,
    device,
    CustomDataset,
    CachedOptimizationContext,
)
from models.encoder.common import TRAINING_HISTORY_DIR
from models.encoder.modernbert_finetune_nn import ModernBertWithFeaturesTrainable
from text2features import FeatureExtractorPipeline


class BaseTrainer(ABC):
    """Base class for all trainers."""

    def __init__(
        self,
        params: Dict[str, Any],
        embeddings: Optional[Literal["minilm", "mbert"]] = None,
        include_large: bool = False,
        enable_train: bool = True,
        enable_cv: bool = False,
        enable_test: bool = False,
        save_model: bool = True,
    ):
        self.params = params
        self.embeddings = embeddings
        self.include_large = include_large
        self.enable_train = enable_train
        self.enable_cv = enable_cv
        self.enable_test = enable_test
        self.save_model = save_model
        self.model = None
        self.model_name = self._get_model_name()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name for file saving."""
        pass

    def _get_model_extension(self) -> str:
        """Return the model file extension (default: .onnx)."""
        return ".onnx"

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
        metrics_path = METRICS_DIR / f"{self.model_name}_{self.timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def run_full_training(self) -> Dict[str, Any]:
        """Execute training pipeline: train, evaluate, cross-validate, export, and save metrics based on enabled flags."""
        print(f"\n{'=' * 60}")
        print(f"Processing {self.model_name}")
        print(f"{'=' * 60}\n")

        model_path = (
            MODEL_DIR
            / f"{self.model_name}_{self.timestamp}{self._get_model_extension()}"
        )
        train_metrics = None
        cv_metrics = None
        test_metrics = None

        # Train (optional)
        if self.enable_train:
            print("Training model...")
            self.train()

            # Evaluate on training set
            train_metrics = self.evaluate_train()
            print(f"Train MSE: {train_metrics['mse']:.4f}")
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")

            # Export model (optional)
            if self.save_model:
                self.export(model_path)
                print(f"Model exported to {model_path}")
            else:
                print("Model export skipped (save_model is False)")

        # Cross-validate (optional, independent)
        if self.enable_cv:
            print("\nPerforming cross-validation...")
            cv_metrics = self.cross_validate()
            print(f"CV MSE: {cv_metrics['mse']:.4f}")
            print(f"CV Accuracy: {cv_metrics['accuracy']:.4f}")

        # Evaluate on test set (optional, requires exported model)
        if self.enable_test:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate_test(model_path)
            print(f"Test MSE: {test_metrics['mse']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        # Save metrics (only if at least one metric was computed)
        if train_metrics or cv_metrics or test_metrics:
            extra_data = self._get_extra_metrics_data()
            self.save_metrics(train_metrics, cv_metrics, test_metrics, **extra_data)

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
        enable_train: bool = True,
        enable_cv: bool = False,
        enable_test: bool = False,
        save_model: bool = True,
    ):
        super().__init__(
            params,
            embeddings,
            include_large,
            enable_train,
            enable_cv,
            enable_test,
            save_model,
        )
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
                DATA_DIR / "datasets" / "small" / "minilm_embeddings.parquet"
            )
            test_df = test_df.merge(
                minilm_embeddings[["cls"]], left_index=True, right_index=True
            )
        elif self.include_mbert:
            modernbert_embeddings = pd.read_parquet(
                DATA_DIR / "datasets" / "small" / "modernbert_cls_embeddings.parquet"
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


class WeightedRandomSampler(BaseEstimator, RegressorMixin):
    """
    A regressor that randomly samples predictions from the training label distribution.
    Uses sample weights to weight the distribution.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Store the training labels and their weights for sampling."""
        self.y_train_ = np.asarray(y)
        if sample_weight is not None:
            self.weights_ = np.asarray(sample_weight)
            # Normalize weights to sum to 1
            self.weights_ = self.weights_ / self.weights_.sum()
        else:
            # Uniform weights if not provided
            self.weights_ = np.ones(len(y)) / len(y)

        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X):
        """Randomly sample from the training distribution for each prediction."""
        n_samples = len(X)
        # Sample indices from training set according to weights
        sampled_indices = self.rng_.choice(
            len(self.y_train_), size=n_samples, p=self.weights_
        )
        return self.y_train_[sampled_indices]


class WeightedRandomBaselineTrainer(BaseTrainer):
    """Trainer for weighted random baseline that samples from training distribution.

    This model doesn't use embeddings - it just samples from the training label distribution.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        include_large: bool = False,
        enable_train: bool = True,
        enable_cv: bool = False,
        enable_test: bool = False,
        save_model: bool = True,
    ):
        super().__init__(
            params,
            embeddings=None,
            include_large=include_large,
            enable_train=enable_train,
            enable_cv=enable_cv,
            enable_test=enable_test,
            save_model=save_model,
        )
        self.y_train = None
        self.weights = None
        self.train_df = None
        self._load_data()

    def _get_model_name(self) -> str:
        return f"random{'_lg' if self.include_large else ''}"

    def _get_model_extension(self) -> str:
        """Random baseline uses JSON format instead of ONNX."""
        return ".json"

    def _load_data(self):
        """Load training data (only labels needed - no weights used)."""
        context = CachedOptimizationContext(
            include_minilm_embeddings=False,
            include_modernbert_embeddings=False,
            include_large=self.include_large,
        )
        train_df = context.sm_train.copy()

        if self.include_large and context.lg_train is not None:
            train_df = pd.concat([context.lg_train, train_df], ignore_index=True)

        self.train_df = train_df
        self.y_train = train_df["label"].values

    def train(self) -> None:
        """Train the random sampler using uniform distribution over training labels."""
        self.model = WeightedRandomSampler(random_state=SEED)
        # We need dummy X data since the sampler expects it (but doesn't use it)
        X_dummy = np.zeros((len(self.y_train), 1))
        # Pass no sample_weight to use uniform distribution
        self.model.fit(X_dummy, self.y_train, sample_weight=None)

    def evaluate_train(self) -> Dict[str, Any]:
        """Evaluate model on training set."""
        X_dummy = np.zeros((len(self.y_train), 1))
        y_pred_train = self.model.predict(X_dummy)
        return calculate_metrics(self.y_train, y_pred_train)

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

        for fold, (train_index, val_index) in enumerate(kf.split(sm_train)):
            train_fold_df = sm_train.loc[train_index].copy()
            val_fold_df = sm_train.loc[val_index].copy()

            y_train_fold = train_fold_df["label"].values
            y_val_fold = val_fold_df["label"].values

            # Train fold model with uniform distribution
            fold_model = WeightedRandomSampler(random_state=SEED)
            X_train_dummy = np.zeros((len(y_train_fold), 1))
            fold_model.fit(X_train_dummy, y_train_fold, sample_weight=None)

            # Evaluate
            X_val_dummy = np.zeros((len(y_val_fold), 1))
            y_pred = fold_model.predict(X_val_dummy)
            metrics = calculate_metrics(y_val_fold, y_pred)
            fold_metrics.append(metrics)

            print(
                f"Fold {fold + 1}/{n_splits}: MSE={metrics['mse']:.4f}, Acc={metrics['accuracy']:.4f}"
            )

        # Average metrics across folds
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
        """Evaluate model on test set using saved distribution."""
        from utils import DATA_DIR

        # Load test data
        test_df = pd.read_parquet(DATA_DIR / "datasets" / "small" / "test.parquet")

        # model_path already has .json extension from _get_model_extension()
        if not model_path.exists():
            raise FileNotFoundError(f"Distribution file not found: {model_path}")

        with open(model_path, "r") as f:
            dist_data = json.load(f)

        # Recreate the model from saved distribution
        model = WeightedRandomSampler(random_state=SEED)
        model.y_train_ = np.array(dist_data["y_train"])
        model.weights_ = np.array(dist_data["weights"])
        model.rng_ = np.random.RandomState(SEED)

        # Generate predictions
        X_test_dummy = np.zeros((len(test_df), 1))
        y_pred_test = model.predict(X_test_dummy)

        y_true_test = test_df["label"].values
        return calculate_metrics(y_true_test, y_pred_test)

    def export(self, model_path: Path) -> None:
        """Save model distribution to JSON file instead of ONNX."""
        # model_path already has .json extension from _get_model_extension()
        dist_data = {
            "y_train": self.model.y_train_.tolist(),
            "weights": self.model.weights_.tolist(),
            "random_state": SEED,
            "model_type": "WeightedRandomSampler",
        }

        with open(model_path, "w") as f:
            json.dump(dist_data, f, indent=2)

        print(f"Distribution saved to {model_path}")

    def _get_extra_metrics_data(self) -> Dict[str, Any]:
        """Add training distribution statistics to metrics."""
        # Since we're using uniform weights, just compute simple mean/std
        train_mean = np.mean(self.model.y_train_)
        train_std = np.std(self.model.y_train_)

        # Add label distribution
        unique_labels, label_counts = np.unique(self.model.y_train_, return_counts=True)
        label_distribution = {
            int(label): int(count) for label, count in zip(unique_labels, label_counts)
        }

        return {
            "train_mean": float(train_mean),
            "train_std": float(train_std),
            "train_size": len(self.model.y_train_),
            "label_distribution": label_distribution,
        }


class ModernBertTrainer(BaseTrainer):
    """Trainer for fine-tuned ModernBERT model."""

    def __init__(
        self,
        params: Dict[str, Any],
        include_large: bool = False,
        enable_train: bool = True,
        enable_cv: bool = False,
        enable_test: bool = False,
        save_model: bool = True,
    ):
        super().__init__(
            params,
            embeddings=None,
            include_large=include_large,
            enable_train=enable_train,
            enable_cv=enable_cv,
            enable_test=enable_test,
            save_model=save_model,
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
        g = torch.Generator()
        g.manual_seed(SEED)

        train_dataset = CustomDataset(self.train_df, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            generator=g,
        )

        self.model = ModernBertWithFeaturesTrainable.from_pretrained(
            "answerdotai/ModernBERT-base",
            feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
            dropout_rate=self.params["dropout_rate"],
            feature_hidden_size=self.params["feature_hidden_size"],
        )
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

        early_stopping_patience = int(self.params.get("early_stopping_patience", 3))

        self.batch_losses = []
        train_losses_per_epoch = []
        best_train_loss = float("inf")
        epochs_without_improve = 0

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
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss detected: {loss.item()}, skipping batch")
                    continue
                if loss.item() > 1000:
                    print(f"Warning: Very high loss {loss.item():.2f}, clipping")
                    loss = torch.clamp(loss, max=100)

                loss_value = loss.item()
                epoch_losses.append(loss_value)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                progress_bar.set_postfix({"loss": loss_value})

                # del loss, outputs
                # if device.type == "cuda":
                #     torch.cuda.empty_cache()

            avg_train_loss = (
                float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            )
            self.batch_losses.append(epoch_losses)
            train_losses_per_epoch.append(avg_train_loss)

            if not np.isnan(avg_train_loss) and avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            print(f"Epoch {epoch + 1} avg loss: {avg_train_loss:.4f}")

            if epochs_without_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1} (patience={early_stopping_patience})."
                )
                break

        training_history = {
            "model": self.model_name,
            "train_losses": train_losses_per_epoch,
            "best_train_loss": float(best_train_loss)
            if best_train_loss != float("inf")
            else None,
            "best_epoch": int(np.argmin(train_losses_per_epoch)) + 1
            if len(train_losses_per_epoch) > 0
            else None,
            "batch_losses": self.batch_losses,
        }
        save_path = (
            TRAINING_HISTORY_DIR / f"{self.model_name}_train_{self.timestamp}.json"
        )
        try:
            with open(save_path, "w") as f:
                json.dump(training_history, f, indent=2)
            print(f"Saved training history to {save_path}")
        except Exception as e:
            print(f"Failed to save training history: {e}")

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
        fold_histories = []

        # Early stopping params (can be overridden via self.params)
        early_stopping_patience = int(self.params.get("early_stopping_patience", 3))

        for fold, (train_index, val_index) in enumerate(kf.split(sm_train)):
            print(f"\nFold {fold + 1}/{n_splits}")
            train_fold_df = sm_train.loc[train_index].copy().reset_index(drop=True)
            val_fold_df = sm_train.loc[val_index].copy().reset_index(drop=True)

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

            g = torch.Generator()
            g.manual_seed(SEED)

            train_dataset = CustomDataset(train_fold_df, self.tokenizer)
            val_dataset = CustomDataset(val_fold_df, self.tokenizer)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.params["batch_size"],
                shuffle=True,
                generator=g,
            )
            val_loader = DataLoader(val_dataset, batch_size=self.params["batch_size"])

            fold_model = ModernBertWithFeaturesTrainable.from_pretrained(
                "answerdotai/ModernBERT-base",
                feature_input_size=FeatureExtractorPipeline.FEATURE_COUNT,
                dropout_rate=self.params["dropout_rate"],
                feature_hidden_size=self.params["feature_hidden_size"],
            )
            fold_model.to(device)

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

            # Train with per-epoch validation and early stopping
            train_losses_per_epoch = []
            val_losses_per_epoch = []
            best_val_loss = float("inf")
            epochs_without_improve = 0

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

                # End of epoch training
                avg_train_loss = (
                    float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                )
                train_losses_per_epoch.append(avg_train_loss)

                # Compute validation loss for this epoch
                fold_model.eval()
                val_epoch_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = fold_model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            features=batch["features"].to(device),
                            labels=batch["labels"].to(device),
                        )
                        loss = outputs.loss
                        if not torch.isnan(loss):
                            val_epoch_losses.append(loss.item())

                avg_val_loss = (
                    float(np.mean(val_epoch_losses))
                    if val_epoch_losses
                    else float("nan")
                )
                val_losses_per_epoch.append(avg_val_loss)

                # Early stopping check
                if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                print(
                    f"Fold {fold + 1} Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
                )

                # Resume training mode for next epoch
                fold_model.train()

                if epochs_without_improve >= early_stopping_patience:
                    print(
                        f"Early stopping triggered for fold {fold + 1} at epoch {epoch + 1} (patience={early_stopping_patience})."
                    )
                    break

            # Final evaluation on validation set using the current model
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

            # Record per-fold history
            fold_history = {
                "train_losses": train_losses_per_epoch,
                "val_losses": val_losses_per_epoch,
                "best_val_loss": float(best_val_loss)
                if best_val_loss != float("inf")
                else None,
                "best_epoch": int(np.argmin(val_losses_per_epoch)) + 1
                if len(val_losses_per_epoch) > 0
                else None,
                "final_metrics": metrics,
            }
            fold_histories.append(fold_history)

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
            "training_history_saved": str(
                TRAINING_HISTORY_DIR / f"{self.model_name}_cv_{self.timestamp}.json"
            ),
            "fold_histories": fold_histories,
        }

        # Save training history to file
        training_history = {
            "model": self.model_name,
            "n_splits": n_splits,
            "folds": fold_histories,
        }
        save_path = TRAINING_HISTORY_DIR / f"{self.model_name}_cv_{self.timestamp}.json"
        try:
            with open(save_path, "w") as f:
                json.dump(training_history, f, indent=2)
            print(f"Saved training history to {save_path}")
            avg_metrics["training_history_path"] = str(save_path)
        except Exception as e:
            print(f"Failed to save training history: {e}")

        return avg_metrics

    def evaluate_test(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate model on test set using saved ONNX model."""

        test_df = pd.read_parquet(DATA_DIR / "datasets" / "small" / "test.parquet")

        scaler = MinMaxScaler()
        train_features = np.vstack(self.train_df["features"].values)
        scaler.fit(train_features)

        test_features_scaled = scaler.transform(np.vstack(test_df["features"].values))
        test_df["features"] = [f for f in np.nan_to_num(test_features_scaled)]

        test_dataset = CustomDataset(test_df, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=self.params["batch_size"])

        sess = rt.InferenceSession(str(model_path))

        y_true_test, y_pred_test = [], []
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            input_ids = batch["input_ids"].numpy()
            attention_mask = batch["attention_mask"].numpy()
            features = batch["features"].numpy()
            labels = batch["labels"].numpy()

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
