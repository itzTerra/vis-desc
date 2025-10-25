"""
Train and optimize a ModernBERT model with a Ridge regression head.
"""

import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models.encoder.common import (
    run_study,
    run_cross_validation,
)

LARGE = False


def objective(trial):
    c = trial.suggest_float("svr_c", 0.0001, 10.0, log=True)
    epsilon = trial.suggest_float("svr_epsilon", 0.0, 1.0)
    small_dataset_weight_multiplier = (
        trial.suggest_float("small_dataset_weight_multiplier", 500.0, 1500.0, log=True)
        if LARGE
        else None
    )

    def train_and_evaluate_fold(
        train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer
    ):
        train_features = np.hstack(
            (np.vstack(train_df["cls"].values), np.vstack(train_df["features"].values))
        )
        val_features = np.hstack(
            (np.vstack(val_df["cls"].values), np.vstack(val_df["features"].values))
        )

        train_labels = train_df["label"].values
        y_true_fold = val_df["label"].values

        regressor = make_pipeline(
            MinMaxScaler(), SVR(kernel="rbf", C=c, epsilon=epsilon)
        )
        regressor.fit(
            train_features, train_labels, svr__sample_weight=train_df["weight"].values
        )

        y_pred_fold = regressor.predict(val_features)
        mse = mean_squared_error(y_true_fold, y_pred_fold)
        return mse

    return run_cross_validation(
        trial=trial,
        train_and_eval_func=train_and_evaluate_fold,
        include_minilm_embeddings=True,
        include_large=LARGE,
        small_dataset_weight_multiplier=small_dataset_weight_multiplier,
    )


if __name__ == "__main__":
    run_study(
        objective_func=objective,
        study_name=f"minilm_svm{'_lg' if LARGE else ''}",
        search_space=None,
        n_trials=100,
    )
