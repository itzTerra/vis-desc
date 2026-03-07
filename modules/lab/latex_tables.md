# LaTeX Tables

All LaTeX table generation code found in `modules/lab/src`.

## Python Files

### `evaluation/core.py`

| #   | Table Name                                                                        | Function                   |
| --- | --------------------------------------------------------------------------------- | -------------------------- |
| 1   | Model Metrics Comparison (RMSE, Accuracy, Weighted F1 (F1w) across Train/CV/Test) | `format_metrics_for_latex` |

Method: `pd.to_latex(escape=False)` with bolded best values per column.

---

### `evaluation/llm/helpers.py`

| #   | Table Name                                                           | Function                               |
| --- | -------------------------------------------------------------------- | -------------------------------------- |
| 2   | Model Metrics Comparison (Correlation, RMSE, Accuracy across splits) | `format_metrics_for_latex`             |
| 3   | Pre/Post-Optimization Comparison with colored delta columns          | `format_optimization_comparison_latex` |

- Table 2: `pd.to_latex()` with `\textbf{}` markup for best values.
- Table 3: Manual LaTeX with `\textcolor{green!70!black}{}` / `\textcolor{red}{}` for improvement/degradation deltas, then `df.to_latex(escape=False, column_format=...)` with right-aligned columns for `CORR`, `RMSE`, `F1w`. Columns are `F1w` / `Δ F1w` (weighted F1).

---

### `evaluation/nli/helpers.py`

| #   | Table Name                                                            | Function                      |
| --- | --------------------------------------------------------------------- | ----------------------------- |
| 4   | NLI Hypothesis Templates / Candidate Labels with Best Correlation     | `df_to_latex`                 |
| 5   | NLI Grouped Header Table (Pearson correlation + throughput, by model) | `df_to_latex_multirow_header` |

- Table 4: Manual construction with `\begin{tabular}`, `\hline`, supports bolding best values.
- Table 5: Manual construction with `\multirow`, `\multicolumn`, `\toprule / \midrule / \bottomrule`.

---

### `evaluation/encoder/interface.py`

| #   | Table Name                                                                                                                               | Function                                                     |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 6   | Encoder Model Metrics Comparison (RMSE, Accuracy – Random / Ridge / RF / SVM / CatBoost / FinetunedMBERT with MiniLM & MBERT embeddings) | `format_encoder_metrics_latex`                               |
| 7   | Encoder Relaxed-Class Metrics (same models, labels 0/1, 2/3, 4/5 merged)                                                                 | `format_encoder_metrics_latex` (with `class_mode="relaxed"`) |

- Both tables: calls `get_encoder_model_groups` → `vis_all_models_tables` → `format_metrics_for_latex` (booktabs style).

---

### `evaluation/encoder/helpers.py`

| #   | Table Name                                                                                       | Function                          |
| --- | ------------------------------------------------------------------------------------------------ | --------------------------------- |
| 8   | Feature Importance Analysis (RMSE, RMSE Δ, Accuracy Δ per feature group, gradient-colored cells) | `export_feature_importance_latex` |

- Table 8: Manual LaTeX with `\cellcolor[rgb]{R,G,B}` and `\textbf{}` for baseline row, saved to `data/figures/feature_importance_table.tex`.
- Requires `\usepackage{booktabs}` and `\usepackage[table]{xcolor}`.

---

## Summary

| File                              | Tables | Generation Method                                                           |
| --------------------------------- | ------ | --------------------------------------------------------------------------- |
| `evaluation/core.py`              | 1      | `pd.to_latex()`                                                             |
| `evaluation/llm/helpers.py`       | 2      | `pd.to_latex()` + manual LaTeX with `\textcolor{}`                          |
| `evaluation/nli/helpers.py`       | 2      | Manual LaTeX with `\begin{tabular}`, `\multirow`, `\multicolumn`            |
| `evaluation/encoder/interface.py` | 2      | `format_encoder_metrics_latex()` (booktabs)                                 |
| `evaluation/encoder/helpers.py`   | 1      | `export_feature_importance_latex()` with `\cellcolor[rgb]{}` gradient cells |
