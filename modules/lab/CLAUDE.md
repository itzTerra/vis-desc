# Lab module — notes for LLMs

## Plot style

All plots in this module use a unified style defined in
`src/evaluation/plot_style.py`. Follow the rules below whenever you
add or modify plotting code.

### Setup — call once per notebook

```python
from evaluation.plot_style import apply_plot_style
apply_plot_style()
```

Place this call near the top of the notebook, after imports. It sets
`matplotlib.rcParams` globally so individual plotting functions do not
need to repeat font or grid settings.

### Constants to import

```python
from evaluation.plot_style import (
    SUPTITLE_FONT_SIZE,  # 16 — figure suptitle (e.g. CV grid)
    TITLE_FONT_SIZE,     # 14 — per-axes title
    LABEL_FONT_SIZE,     # 11 — x/y axis labels
    TICK_FONT_SIZE,      # 10 — tick labels, in-axis text labels
    LEGEND_FONT_SIZE,    # 9  — legend text
    ANNOT_FONT_SIZE,     # 9  — bar value labels, in-plot annotations
    PALETTE,             # list[str] — 10 hex colours, fixed order
    CM_RAW,              # "Blues"    — raw-count heatmap/confusion matrix
    CM_NORM,             # "RdYlGn"  — row-normalised confusion matrix
    CM_CORR,             # "coolwarm"— correlation / similarity heatmap
    GRID_ALPHA,          # 0.3
    GRID_LINESTYLE,      # "--"
)
```

Import only what you use. `LABEL_FONT_SIZE` is re-exported from
`evaluation.core` for backwards compatibility, but prefer importing
from `plot_style` in new code.

### Rules for plotting functions

- **Never hard-code font sizes.** Use the named constants above.
- **Titles** — pass `fontsize=TITLE_FONT_SIZE` (and optionally
  `fontweight="bold"`) to `ax.set_title()` / `plt.title()`.
- **Figure suptitles** — `fontsize=SUPTITLE_FONT_SIZE`.
- **Axis labels** — `fontsize=LABEL_FONT_SIZE`. Do not add
  `fontweight="bold"` to axis labels.
- **Tick labels** — when setting them explicitly (e.g.
  `ax.set_xticklabels(...)`), pass `fontsize=TICK_FONT_SIZE`.
- **Legends** — `ax.legend(fontsize=LEGEND_FONT_SIZE)`.
- **Bar/annotation text** — `fontsize=ANNOT_FONT_SIZE`.
- **Colours** — use `PALETTE[i % len(PALETTE)]` for categorical series.
  Do not use `"tab10"`, `"Dark2"`, or `"tab20"` as dynamic palette names.
- **Heatmaps** — use `CM_RAW`, `CM_NORM`, or `CM_CORR` instead of
  bare strings like `"Blues"` or `"coolwarm"`.
- **Grid** — `apply_plot_style()` enables the grid globally. Do not
  call `ax.grid(...)` unless you need a non-default axis (`axis="x"`
  or `axis="y"`). When you must call it, use `alpha=GRID_ALPHA` and
  `linestyle=GRID_LINESTYLE`.

### Exception — large-format export plots

`dataset_small/book_collection.py::plot_genre_distribution` is a
22 × 13 inch export figure that intentionally uses oversized fonts
(38–48 pt). It saves and restores `rcParams` around its own update, so
it must **not** have `apply_plot_style()` called inside it. Do not
change its font sizes to match the standard constants.

### File map

| File                                   | What it plots                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| `src/evaluation/plot_style.py`         | Style constants + `apply_plot_style()`                                               |
| `src/evaluation/core.py`               | Confusion matrices, per-metric bar charts, macro-F1 bars                             |
| `src/evaluation/encoder/helpers.py`    | CV fold grids, train/val curves, feature-importance bars, score/length distributions |
| `src/evaluation/nli/helpers.py`        | Correlation heatmap, calibration comparison, learning curves                         |
| `src/evaluation/llm/helpers.py`        | Model-metrics scatter, combined scatter, metrics vs prompt token count               |
| `src/dataset_small/book_collection.py` | Genre distribution pie chart (large-format export)                                   |

## LaTeX tables

All LaTeX table generation functions must produce **bold header row(s)**. Apply
`\textbf{}` to every column header cell, including:

- Simple single-row headers — wrap each cell with `\textbf{...}`.
- `\multicolumn` headers — place `\textbf{...}` inside the content argument, e.g.
  `\multicolumn{N}{c}{\textbf{...}}`.
- `\multirow` headers — place `\textbf{...}` inside the content argument, e.g.
  `\multirow{2}{*}{\textbf{...}}`.
- Second-row model-name headers in multi-row layouts — wrap each entry with
  `\textbf{...}`.
