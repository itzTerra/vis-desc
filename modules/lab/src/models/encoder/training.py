#!/usr/bin/env python3

import argparse

from models.encoder.common import set_seed
from models.encoder.trainers import (
    RidgeTrainer,
    SVMTrainer,
    RandomForestTrainer,
    CatBoostTrainer,
    ModernBertTrainer,
    WeightedRandomBaselineTrainer,
)

MODEL_PARAMS = {
    "random": {
        "no_lg": {},
        "lg": {"small_dataset_weight_multiplier": 100.0},
    },
    "ridge": {
        "minilm": {"ridge_alpha": 0.01},
        "minilm_lg": {"ridge_alpha": 0.01, "small_dataset_weight_multiplier": 100.0},
        "mbert": {"ridge_alpha": 0.01},
        "mbert_lg": {"ridge_alpha": 0.01, "small_dataset_weight_multiplier": 100.0},
    },
    "svm": {
        "minilm": {"svr_c": 2.5276, "svr_epsilon": 0.0018},
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
            "n_estimators": 507,
            "max_depth": 30,
            "min_samples_split": 7,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": False,
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
            "iterations": 1233,
            "learning_rate": 0.0671,
            "depth": 2,
            "l2_leaf_reg": 3.3785e-07,
            "bagging_temperature": 15.6715,
            "random_strength": 9.8879e-08,
            "border_count": 63,
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
    "random": WeightedRandomBaselineTrainer,
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
            "random",
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
        default=None,
        help="Embedding types to use for non-finetuned models (if not provided, only finetuned-mbert will be trained)",
    )
    parser.add_argument(
        "--lg",
        action="store_true",
        help="Include large dataset in training",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training (disabled by default unless no other mode specified)",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Enable cross-validation (disabled by default)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test set evaluation (disabled by default)",
    )

    args = parser.parse_args()

    # If no mode is specified, default to training only
    if not args.train and not args.val and not args.test:
        args.train = True

    models_to_train = args.models
    if "all" in models_to_train or not models_to_train:
        models_to_train = [
            "random",
            "ridge",
            "svm",
            "rf",
            "catboost",
            "finetuned-mbert",
        ]

    embeddings_list = args.embeddings
    include_large = args.lg
    enable_train = args.train
    enable_cv = args.val
    enable_test = args.test

    set_seed()

    for model_name in models_to_train:
        if model_name == "finetuned-mbert":
            # Fine-tuned ModernBERT doesn't use pre-computed embeddings
            key = "lg" if include_large else "no_lg"
            params = MODEL_PARAMS["finetuned-mbert"][key]
            trainer = ModernBertTrainer(
                params=params,
                include_large=include_large,
                enable_train=enable_train,
                enable_cv=enable_cv,
                enable_test=enable_test,
            )
            trainer.run_full_training()
        elif model_name == "random":
            # Weighted random baseline doesn't use embeddings at all
            key = "lg" if include_large else "no_lg"
            params = MODEL_PARAMS[model_name].get(key, {})

            trainer_class = TRAINER_CLASSES[model_name]
            trainer = trainer_class(
                params=params,
                include_large=include_large,
                enable_train=enable_train,
                enable_cv=enable_cv,
                enable_test=enable_test,
            )
            trainer.run_full_training()
        else:
            # Skip other models if no embeddings specified
            if embeddings_list is None:
                print(
                    f"Skipping {model_name} (no embeddings specified, only finetuned-mbert and random are valid)"
                )
                continue

            # Train for each embedding type
            for embeddings in embeddings_list:
                key = f"{embeddings}{'_lg' if include_large else ''}"
                params = MODEL_PARAMS[model_name][key]

                trainer_class = TRAINER_CLASSES[model_name]
                trainer = trainer_class(
                    params=params,
                    embeddings=embeddings,
                    include_large=include_large,
                    enable_train=enable_train,
                    enable_cv=enable_cv,
                    enable_test=enable_test,
                )
                trainer.run_full_training()

    print("\n" + "=" * 60)
    phases = []
    if enable_train:
        phases.append("training")
    if enable_cv:
        phases.append("cross-validation")
    if enable_test:
        phases.append("testing")

    print(f"All {' and '.join(phases)} completed!")
    print("=" * 60)
