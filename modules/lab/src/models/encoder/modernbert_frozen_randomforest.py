import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from models.encoder.common import (
    run_study,
    SEED,
    run_cross_validation,
)

LARGE = False


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

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

        regressor = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
        regressor.fit(
            train_features, train_labels, sample_weight=train_df["weight"].values
        )

        y_pred_fold = regressor.predict(val_features)
        mse = mean_squared_error(y_true_fold, y_pred_fold)
        return mse

    return run_cross_validation(
        trial=trial,
        train_and_eval_func=train_and_evaluate_fold,
        include_modernbert_embeddings=True,
        include_large=LARGE,
    )


if __name__ == "__main__":
    run_study(
        objective_func=objective,
        study_name=f"modernbert_frozen_randomforest{'_lg' if LARGE else ''}",
        search_space=None,
        n_trials=100,
    )
