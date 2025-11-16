#!/usr/bin/env python3

import argparse
from pathlib import Path

from models.encoder.trainers import (
    RidgeTrainer,
    SVMTrainer,
    RandomForestTrainer,
    CatBoostTrainer,
    ModernBertTrainer,
)
from utils import DATA_DIR

MODEL_PARAMS = {
    "ridge": {
        "minilm": {"ridge_alpha": 0.01},
        "minilm_lg": {"ridge_alpha": 0.01, "small_dataset_weight_multiplier": 100.0},
        "mbert": {"ridge_alpha": 0.01},
        "mbert_lg": {"ridge_alpha": 0.01, "small_dataset_weight_multiplier": 100.0},
    },
    "svm": {
        "minilm": {"svr_c": 1.0, "svr_epsilon": 0.1},
        "minilm_lg": {
            "svr_c": 1.0,
            "svr_epsilon": 0.1,
            "small_dataset_weight_multiplier": 100.0,
        },
        "mbert": {"svr_c": 1.0, "svr_epsilon": 0.1},
        "mbert_lg": {
            "svr_c": 1.0,
            "svr_epsilon": 0.1,
            "small_dataset_weight_multiplier": 100.0,
        },
    },
    "rf": {
        "minilm": {
            "n_estimators": 500,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        "minilm_lg": {
            "n_estimators": 500,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        "mbert": {
            "n_estimators": 500,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        "mbert_lg": {
            "n_estimators": 500,
            "max_depth": 16,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
        },
    },
    "catboost": {
        "minilm": {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "border_count": 128,
        },
        "minilm_lg": {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "border_count": 128,
        },
        "mbert": {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "border_count": 128,
        },
        "mbert_lg": {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "border_count": 128,
        },
    },
    "finetuned-mbert": {
        "no_lg": {
            "num_epochs": 20,
            "lr_bert": 5e-5,
            "lr_custom": 1e-4,
            "dropout_rate": 0.1,
            "weight_decay": 1e-2,
            "optimizer_warmup": 0.1,
            "feature_hidden_size": 768,
            "batch_size": 16,
        },
        "lg": {
            "num_epochs": 20,
            "lr_bert": 5e-5,
            "lr_custom": 1e-4,
            "dropout_rate": 0.1,
            "weight_decay": 1e-2,
            "optimizer_warmup": 0.1,
            "feature_hidden_size": 768,
            "batch_size": 16,
        },
    },
}

TRAINER_CLASSES = {
    "ridge": RidgeTrainer,
    "svm": SVMTrainer,
    "rf": RandomForestTrainer,
    "catboost": CatBoostTrainer,
    "finetuned-mbert": ModernBertTrainer,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final training of all models with best hyperparameters"
    )
    parser.add_argument(
        "models",
        nargs="*",
        choices=[
            "ridge",
            "svm",
            "rf",
            "catboost",
            "finetuned-mbert",
            "all",
        ],
        default=["all"],
        help="Models to train. If 'all' or empty, trains all models.",
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        nargs="+",
        choices=["minilm", "mbert"],
        required=True,
        help="Embedding types to use for non-finetuned models",
    )
    parser.add_argument(
        "--lg",
        action="store_true",
        help="Include large dataset in training",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models and metrics (default: DATA_DIR/models)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable cross-validation (disabled by default)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test set evaluation (disabled by default)",
    )

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = args.models
    if "all" in models_to_train or not models_to_train:
        models_to_train = ["ridge", "svm", "rf", "catboost", "finetuned-mbert"]

    embeddings_list = args.embeddings
    include_large = args.lg
    enable_cv = args.cv
    enable_test = args.test

    for model_name in models_to_train:
        if model_name == "finetuned-mbert":
            # Fine-tuned ModernBERT doesn't use pre-computed embeddings
            key = "lg" if include_large else "no_lg"
            params = MODEL_PARAMS["finetuned-mbert"][key]
            trainer = ModernBertTrainer(
                params=params,
                include_large=include_large,
                enable_cv=enable_cv,
                enable_test=enable_test,
            )
            trainer.run_full_training(output_dir)
        else:
            # Train for each embedding type
            for embeddings in embeddings_list:
                key = f"{embeddings}{'_lg' if include_large else ''}"
                params = MODEL_PARAMS[model_name][key]

                trainer_class = TRAINER_CLASSES[model_name]
                trainer = trainer_class(
                    params=params,
                    embeddings=embeddings,
                    include_large=include_large,
                    enable_cv=enable_cv,
                    enable_test=enable_test,
                )
                trainer.run_full_training(output_dir)

    print("\n" + "=" * 60)
    print("All training completed!")
    print(f"Models and metrics saved to: {output_dir}")
    print("=" * 60)
