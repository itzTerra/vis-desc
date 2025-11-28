from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor
from tqdm.auto import tqdm
import onnxruntime as rt
from models.encoder.modernbert_finetune_nn import BATCH_SIZE
from utils import DATA_DIR
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from catboost.utils import convert_to_onnx_object


from models.encoder.common import (
    MODEL_DIR,
    SEED,
    calculate_metrics,
    device,
    CustomDataset,
    CachedOptimizationContext,
    set_seed,
    save_metrics_separate as save_metrics_separate_fn,
)
from models.encoder.common import TRAINING_HISTORY_DIR
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
        label: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.params = params
        self.embeddings = embeddings
        self.include_large = include_large
        self.enable_train = enable_train
        self.enable_cv = enable_cv
        self.enable_test = enable_test
        self.save_model = save_model
        # Optional user-provided label to distinguish runs (e.g., feature group ablations)
        self.label = label
        self.model = None
        self.model_name = self._get_model_name()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.seed = seed if seed is not None else SEED

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

    def save_metrics_separate(
        self,
        train: Optional[Dict],
        val: Optional[Dict],
        test: Optional[Dict],
        extra_train: Optional[Dict] = None,
    ) -> None:
        """Delegate metrics saving to shared utility (see common.save_metrics_separate)."""
        save_metrics_separate_fn(
            model_name=self.model_name,
            params=self.params,
            train=train,
            val=val,
            test=test,
            extra_train=extra_train,
            timestamp=self.timestamp,
        )

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

        if self.enable_train:
            print("Training model...")

            set_seed(self.seed)

            self.train()

            train_metrics = self.evaluate_train()
            print(f"Train MSE: {train_metrics['mse']:.4f}")
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")

            if self.save_model:
                self.export(model_path)
                print(f"Model exported to {model_path}")
            else:
                print("Model export skipped (save_model is False)")

        if self.enable_test:
            print("\nEvaluating on test set...")

            set_seed(self.seed)

            test_metrics = self.evaluate_test(model_path)
            print(f"Test MSE: {test_metrics['mse']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        if self.enable_cv:
            print("\nPerforming cross-validation...")

            set_seed(self.seed)

            cv_metrics = self.cross_validate()
            print(f"CV MSE: {cv_metrics['mse']:.4f}")
            print(f"CV Accuracy: {cv_metrics['accuracy']:.4f}")

        # Save metrics (only if at least one metric was computed)
        if train_metrics or cv_metrics or test_metrics:
            extra_train = self._get_extra_metrics_data()
            self.save_metrics_separate(
                train_metrics, cv_metrics, test_metrics, extra_train=extra_train
            )

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
        feature_mask: Optional[np.ndarray] = None,
        minilm_mask: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            params,
            embeddings,
            include_large,
            enable_train,
            enable_cv,
            enable_test,
            save_model,
            label=label,
            seed=seed,
        )
        self.include_minilm = embeddings == "minilm"
        self.include_mbert = embeddings == "mbert"
        self.X_train = None
        self.y_train = None
        self.weights = None
        self.feature_mask = feature_mask
        self.minilm_mask = minilm_mask
        self._load_data()

    def _get_model_name(self) -> str:
        base_name = self._get_base_model_name()
        label_str = f"_{self.label}" if self.label else ""
        return f"{base_name}_{self.embeddings}{'_lg' if self.include_large else ''}{label_str}"

    @abstractmethod
    def _get_base_model_name(self) -> str:
        """Return the base model name (e.g., 'ridge', 'svm')."""
        pass

    @abstractmethod
    def _create_model(self):
        """Create and return the sklearn model instance."""
        pass

    def _load_data(self):
        """Load and prepare training data, applying feature/minilm masks if provided."""
        context = CachedOptimizationContext(
            include_minilm_embeddings=self.include_minilm,
            include_modernbert_embeddings=self.include_mbert,
            include_large=self.include_large,
        )
        train_df = context.sm_train.copy()

        if self.include_large and context.lg_train is not None:
            weight_mult = self.params["small_dataset_weight_multiplier"]
            train_df["weight"] *= weight_mult
            train_df = pd.concat([context.lg_train, train_df], ignore_index=True)

        # Apply masks
        cls_arr = np.vstack(train_df["cls"].values)
        features_arr = np.vstack(train_df["features"].values)
        if self.minilm_mask is not None and self.include_minilm:
            cls_arr = cls_arr[:, self.minilm_mask]
        if self.feature_mask is not None:
            features_arr = features_arr[:, self.feature_mask]
        self.X_train = np.hstack((cls_arr, features_arr))
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
        """Perform cross-validation.

        Always stratifies and validates on the small dataset folds. If
        ``self.include_large`` is True, the full large dataset is appended to
        every training fold (never to the validation fold) and the small fold
        weights are optionally upscaled using ``small_dataset_weight_multiplier``.
        This mirrors the logic in ``run_cross_validation`` from ``common.py``.
        """
        context = CachedOptimizationContext(
            include_minilm_embeddings=self.include_minilm,
            include_modernbert_embeddings=self.include_mbert,
            include_large=self.include_large,
        )
        sm_train = context.sm_train
        lg_train = context.lg_train  # may be None

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        fold_metrics = []

        for fold, (train_index, val_index) in enumerate(
            kf.split(sm_train, sm_train["label"])
        ):
            train_fold_df = sm_train.loc[train_index].copy()
            val_fold_df = sm_train.loc[val_index].copy()

            # If large dataset is used, concatenate it to the training fold and
            # upscale the small dataset weights to balance influence.
            if self.include_large and lg_train is not None:
                weight_mult = self.params["small_dataset_weight_multiplier"]
                train_fold_df["weight"] *= weight_mult
                train_fold_df = pd.concat([lg_train, train_fold_df], ignore_index=True)

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
        }

    def evaluate_test(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate model on test set using saved ONNX model, applying feature/minilm masks."""
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
        cls_arr = np.vstack(test_df["cls"].values)
        features_arr = np.vstack(test_df["features"].values)
        if self.minilm_mask is not None and self.include_minilm:
            cls_arr = cls_arr[:, self.minilm_mask]
        if self.feature_mask is not None:
            features_arr = features_arr[:, self.feature_mask]
        X_test = np.hstack((cls_arr, features_arr))
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

        n_features = self.X_train.shape[1]
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(self.model, initial_types=initial_type, target_opset=15)

        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())

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
            Ridge(alpha=self.params["ridge_alpha"], random_state=self.seed),
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
        if self.include_large:
            return make_pipeline(
                MinMaxScaler(),
                Nystroem(
                    kernel="rbf",
                    n_components=500,
                    random_state=self.seed,
                ),
                SGDRegressor(
                    loss="squared_error",
                    alpha=1.0 / self.params["svr_c"],
                    max_iter=100000,
                    random_state=self.seed,
                ),
            )
        else:
            return make_pipeline(
                MinMaxScaler(),
                SVR(
                    kernel="rbf",
                    C=self.params["svr_c"],
                    epsilon=self.params["svr_epsilon"],
                ),
            )

    def _get_fit_params(self) -> Dict[str, Any]:
        if self.include_large:
            return {"sgdregressor__sample_weight": self.weights}
        else:
            return {"svr__sample_weight": self.weights}


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
        return RandomForestRegressor(**rf_params, random_state=self.seed, n_jobs=-1)

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
            **cb_params, random_seed=self.seed, verbose=100, thread_count=-1
        )

    def _get_fit_params(self) -> Dict[str, Any]:
        return {"sample_weight": self.weights}

    def _get_fit_params_for_weights(self, weights: np.ndarray) -> Dict[str, Any]:
        return {"sample_weight": weights}

    def export(self, model_path: Path) -> None:
        """Export CatBoost model to ONNX format using CatBoost's built-in converter."""

        # Use CatBoost's native ONNX conversion
        onx = convert_to_onnx_object(self.model)

        # Save the ONNX model
        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())

        # Verify the conversion
        sess = rt.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        sample_X = self.X_train.astype(np.float32)
        pred_onx = sess.run([output_name], {input_name: sample_X})[0]
        pred_catboost = self.model.predict(sample_X)

        diff = np.abs(pred_catboost - pred_onx.flatten())
        if not np.all(diff < 1e-5):
            print(
                "Warning: Insufficient accuracy between CatBoost and ONNX predictions."
            )
            print(f"Max difference: {np.max(diff)}")
        else:
            print(f"ONNX export verified successfully (max diff: {np.max(diff):.2e})")


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
        label: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            params,
            embeddings=None,
            include_large=include_large,
            enable_train=enable_train,
            enable_cv=enable_cv,
            enable_test=enable_test,
            save_model=save_model,
            label=label,
            seed=seed,
        )
        self.y_train = None
        self.weights = None
        self.train_df = None
        self._load_data()

    def _get_model_name(self) -> str:
        label_str = f"_{self.label}" if self.label else ""
        return f"random{'_lg' if self.include_large else ''}{label_str}"

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
        self.model = WeightedRandomSampler(random_state=self.seed)
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

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        fold_metrics = []

        for fold, (train_index, val_index) in enumerate(
            kf.split(sm_train, sm_train["label"])
        ):
            train_fold_df = sm_train.loc[train_index].copy()
            val_fold_df = sm_train.loc[val_index].copy()

            y_train_fold = train_fold_df["label"].values
            y_val_fold = val_fold_df["label"].values

            # Train fold model with uniform distribution
            fold_model = WeightedRandomSampler(random_state=self.seed)
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
        # Load test data
        test_df = pd.read_parquet(DATA_DIR / "datasets" / "small" / "test.parquet")

        # model_path already has .json extension from _get_model_extension()
        if not model_path.exists():
            raise FileNotFoundError(f"Distribution file not found: {model_path}")

        with open(model_path, "r") as f:
            dist_data = json.load(f)

        # Recreate the model from saved distribution
        model = WeightedRandomSampler(random_state=self.seed)
        model.y_train_ = np.array(dist_data["y_train"])
        model.weights_ = np.array(dist_data["weights"])
        model.rng_ = np.random.RandomState(self.seed)

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
            "random_state": self.seed,
            "model_type": "WeightedRandomSampler",
        }

        with open(model_path, "w") as f:
            json.dump(dist_data, f, indent=2)

        print(f"Distribution saved to {model_path}")

    def _get_extra_metrics_data(self) -> Dict[str, Any]:
        return {}


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
        label: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            params,
            embeddings=None,
            include_large=include_large,
            enable_train=enable_train,
            enable_cv=enable_cv,
            enable_test=enable_test,
            save_model=save_model,
            label=label,
            seed=seed,
        )
        self.tokenizer = None
        self.train_df = None
        self.batch_losses = []

    def _get_model_name(self) -> str:
        label_str = f"_{self.label}" if self.label else ""
        return f"finetuned_mbert{'_lg' if self.include_large else ''}{label_str}"

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
        from models.encoder.modernbert_finetune_nn import train_finetuned_mbert

        if self.tokenizer is None or self.train_df is None:
            self._load_data()

        history_path = (
            TRAINING_HISTORY_DIR / f"{self.model_name}_train_{self.timestamp}.json"
        )

        result = train_finetuned_mbert(
            train_df=self.train_df,
            val_df=None,
            tokenizer=self.tokenizer,
            params=self.params,
            seed=self.seed,
            train_lg_df=None,
            save_history=True,
            history_path=history_path,
        )
        self.model = result["model"]

    def evaluate_train(self) -> Dict[str, Any]:
        """Evaluate model on training set."""
        train_dataset = CustomDataset(self.train_df, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

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

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        fold_metrics = []
        fold_histories = []

        for fold, (train_index, val_index) in enumerate(
            kf.split(sm_train, sm_train["label"])
        ):
            from models.encoder.modernbert_finetune_nn import train_finetuned_mbert

            print(f"\nFold {fold + 1}/{n_splits}")
            train_fold_df = sm_train.loc[train_index].copy().reset_index(drop=True)
            val_fold_df = sm_train.loc[val_index].copy().reset_index(drop=True)

            training_params = {
                "lr_bert": self.params["lr_bert"],
                "lr_custom": self.params["lr_custom"],
                "dropout_rate": self.params["dropout_rate"],
                "weight_decay": self.params["weight_decay"],
                "optimizer_warmup": self.params["optimizer_warmup"],
                "feature_hidden_size": self.params["feature_hidden_size"],
                "stage1_epochs": self.params["stage1_epochs"],
                "frozen_bert_epochs": self.params["frozen_bert_epochs"],
            }

            result = train_finetuned_mbert(
                train_df=train_fold_df,
                val_df=val_fold_df,
                tokenizer=context.tokenizer,
                params=training_params,
                seed=self.seed,
                train_lg_df=None,
                save_history=False,
                history_path=None,
            )

            metrics = result["final_metrics"]
            fold_metrics.append(metrics)

            fold_history = {
                "train_losses": result["train_losses"],
                "val_losses": result["val_losses"],
                "best_val_loss": result["best_val_loss"],
                "best_epoch": int(np.argmin(result["val_losses"])) + 1
                if result["val_losses"]
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
        test_df = test_df.reset_index(drop=True)

        scaler = MinMaxScaler()
        train_features = np.vstack(self.train_df["features"].values)
        scaler.fit(train_features)

        test_features_scaled = scaler.transform(np.vstack(test_df["features"].values))
        test_df["features"] = [f for f in np.nan_to_num(test_features_scaled)]

        test_dataset = CustomDataset(test_df, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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
            opset_version=18,
            dynamo=False,
        )
        print(f"ONNX model saved to {model_path}")

    def _get_extra_metrics_data(self) -> Dict[str, Any]:
        """Add batch losses to metrics."""
        return {"batch_losses": self.batch_losses}
