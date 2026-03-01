# Plots

All plotting code found in `modules/lab/src`.

## Python Files

### `evaluation/core.py`

| #   | Plot Name                                                          | Function                           |
| --- | ------------------------------------------------------------------ | ---------------------------------- |
| 1   | Confusion Matrix (raw counts + normalized proportions)             | `plot_confusion_matrix`            |
| 2   | Training / Validation Loss Curves                                  | `plot_train_val_runs`              |
| 3   | Cross-Validation Folds Grid                                        | `plot_cv_folds_grid`               |
| 4   | Per-Label Metrics Comparison (Precision, Recall, F1) across models | `vis_all_models_plots`             |
| 5   | Model Accuracy / RMSE / Macro-F1 Bar Charts                        | `vis_all_models_plots`             |
| 6   | Per-Label Metrics Tables (Train / CV / Test)                       | `vis_specific_model_tables`        |
| 7   | Confusion Matrices for Selected Model                              | `vis_specific_model_conf_matrices` |

### `evaluation/encoder/helpers.py`

| #   | Plot Name                                              | Function                       |
| --- | ------------------------------------------------------ | ------------------------------ |
| 8   | Confusion Matrix (raw counts + normalized proportions) | `plot_confusion_matrix`        |
| 9   | Training / Validation Loss Curves                      | `plot_train_val_runs`          |
| 10  | Cross-Validation Folds Grid                            | `plot_cv_folds_grid`           |
| 11  | Training / Validation Loss Curves (from metric files)  | `visualize_metric_files`       |
| 12  | Feature Importance – Accuracy Impact                   | `plot_feature_importance_bars` |
| 13  | Feature Importance – RMSE Impact                       | `plot_feature_importance_bars` |
| 14  | Feature Importance – Macro F1 Impact                   | `plot_feature_importance_bars` |
| 15  | Score / Label Distribution Bar Chart                   | `plot_score_distribution`      |
| 16  | Document Character Length Distribution                 | `plot_length_distributions`    |
| 17  | Document Token Count Distribution                      | `plot_length_distributions`    |
| 18  | Document Sentence Count Distribution                   | `plot_length_distributions`    |
| 19  | Summary Statistics Text Annotation                     | `plot_length_distributions`    |

### `evaluation/llm/helpers.py`

| #   | Plot Name                                                   | Function                              |
| --- | ----------------------------------------------------------- | ------------------------------------- |
| 20  | Model Metrics vs Prompt Complexity Scatter Plot             | `plot_model_metrics_scatter`          |
| 21  | Combined Metrics Bubble Chart (Correlation, RMSE, Accuracy) | `plot_model_metrics_combined_scatter` |
| 22  | Metrics vs Prompt Token Count Line Plot                     | `plot_metrics_vs_prompt_token_count`  |

### `evaluation/nli/helpers.py`

| #   | Plot Name                                                    | Function                                      |
| --- | ------------------------------------------------------------ | --------------------------------------------- |
| 23  | Correlation Matrix Heatmap                                   | `plot_correlation_matrix`                     |
| 24  | Train / Test Correlation Matrices                            | `plot_train_test_correlation_from_score_data` |
| 25  | Calibration Comparison Line Plots (Isotonic vs Q2Q)          | `plot_calibration_comparison`                 |
| 26  | Learning Curves                                              | `plot_learning_curves`                        |
| 27  | Calibration Comparison – both sort orders (Isotonic and Q2Q) | `plot_calibration_comparison_for_score_data`  |

### `dataset_small/book_collection.py`

| #   | Plot Name                    | Function                  |
| --- | ---------------------------- | ------------------------- |
| 28  | Genre Distribution Pie Chart | `plot_genre_distribution` |
