#!/usr/bin/env python3

import argparse

from models.encoder.common import set_seed
from models.encoder.trainers import (
    BaseTrainer,
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
        "lg": {"small_dataset_weight_multiplier": 200.0},
    },
    "ridge": {
        "minilm": {"ridge_alpha": 67.8330},
        "minilm_lg": {"ridge_alpha": 67.8330, "small_dataset_weight_multiplier": 200.0},
        "mbert": {"ridge_alpha": 64.4018},
        "mbert_lg": {"ridge_alpha": 64.4018, "small_dataset_weight_multiplier": 200.0},
    },
    "svm": {
        "minilm": {"svr_c": 2.5276, "svr_epsilon": 0.0018},
        "minilm_lg": {
            "svr_c": 2.5276,
            "svr_epsilon": 0.0018,
            "small_dataset_weight_multiplier": 200.0,
        },
        "mbert": {"svr_c": 2.4737, "svr_epsilon": 0.0032},
        "mbert_lg": {
            "svr_c": 2.4737,
            "svr_epsilon": 0.0032,
            "small_dataset_weight_multiplier": 200.0,
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
            "n_estimators": 507,
            "max_depth": 30,
            "min_samples_split": 7,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": False,
            "small_dataset_weight_multiplier": 200.0,
        },
        "mbert": {
            "n_estimators": 726,
            "max_depth": 17,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
        },
        "mbert_lg": {
            "n_estimators": 726,
            "max_depth": 17,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
            "small_dataset_weight_multiplier": 200.0,
        },
    },
    "catboost": {
        "minilm": {
            "iterations": 1510,
            "learning_rate": 0.014286,
            "depth": 3,
            "l2_leaf_reg": 0.0000034047,
            "bagging_temperature": 2.5656,
            "random_strength": 4.1266,
            "border_count": 207,
        },
        "minilm_lg": {
            "iterations": 1510,
            "learning_rate": 0.014286,
            "depth": 3,
            "l2_leaf_reg": 0.0000034047,
            "bagging_temperature": 2.5656,
            "random_strength": 4.1266,
            "border_count": 207,
            "small_dataset_weight_multiplier": 200.0,
        },
        "mbert": {
            "iterations": 1603,
            "learning_rate": 0.018419,
            "depth": 3,
            "l2_leaf_reg": 4.2173,
            "bagging_temperature": 15.7630,
            "random_strength": 6.6764,
            "border_count": 136,
        },
        "mbert_lg": {
            "iterations": 1603,
            "learning_rate": 0.018419,
            "depth": 3,
            "l2_leaf_reg": 4.2173,
            "bagging_temperature": 15.7630,
            "random_strength": 6.6764,
            "border_count": 136,
            "small_dataset_weight_multiplier": 200.0,
        },
    },
    "finetuned-mbert": {
        "no_lg": {
            "stage1_epochs": 5,
            "stage2_epochs": 19,
            "lr_bert": 3e-5,
            "lr_custom": 8e-5,
            "dropout_rate": 0.05,
            "weight_decay": 0.01,
            "optimizer_warmup": 0.2,
            "feature_hidden_size": 768,
            "frozen_bert_epochs": 5,
        },
        "lg": {
            "stage1_epochs": 5,
            "stage2_epochs": 20,
            "lr_bert": 3e-5,
            "lr_custom": 8e-5,
            "dropout_rate": 0.1,
            "weight_decay": 0.01,
            "optimizer_warmup": 0.2,
            "feature_hidden_size": 768,
            "frozen_bert_epochs": 4,
        },
    },
}

TRAINER_CLASSES: dict[str, BaseTrainer] = {
    "random": WeightedRandomBaselineTrainer,
    "ridge": RidgeTrainer,
    "svm": SVMTrainer,
    "rf": RandomForestTrainer,
    "catboost": CatBoostTrainer,
    "finetuned-mbert": ModernBertTrainer,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final training of all models with best hyperparameters (optionally across multiple seeds)"
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
    parser.add_argument(
        "--direct-test",
        action="store_true",
        help="Use trained model directly for test evaluation instead of ONNX export/load",
    )

    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Do not save/export the trained model(s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Specific seed to use for the run.",
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
    use_direct_test = args.direct_test
    seed = args.seed

    set_seed(seed)

    save_model = not args.no_save_model

    for model_name in models_to_train:
        if model_name == "finetuned-mbert":
            key = "lg" if include_large else "no_lg"
            params = MODEL_PARAMS["finetuned-mbert"][key]
            trainer = ModernBertTrainer(
                params=params,
                include_large=include_large,
                enable_train=enable_train,
                enable_cv=enable_cv,
                enable_test=enable_test,
                save_model=save_model,
                seed=seed,
                label=None,
                use_direct_test=use_direct_test,
            )
            trainer.run_full_training()
        elif model_name == "random":
            key = "lg" if include_large else "no_lg"
            params = MODEL_PARAMS[model_name].get(key, {})
            trainer_class = TRAINER_CLASSES[model_name]
            trainer = trainer_class(
                params=params,
                include_large=include_large,
                enable_train=enable_train,
                enable_cv=enable_cv,
                enable_test=enable_test,
                save_model=save_model,
                seed=seed,
                label=None,
                use_direct_test=use_direct_test,
            )
            trainer.run_full_training()
        else:
            if embeddings_list is None:
                print(
                    f"Skipping {model_name} (no embeddings specified, only finetuned-mbert and random are valid)"
                )
                continue
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
                    save_model=save_model,
                    seed=seed,
                    label=None,
                    use_direct_test=use_direct_test,
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
