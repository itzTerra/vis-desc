from __future__ import annotations

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from IPython.utils.capture import capture_output

from evaluation.core import (
    AggregatedModelData,
    get_confusion_matrix,
    vis_all_models_tables,
)
from evaluation.encoder.interface import get_encoder_model_groups
from evaluation.llm.interface import get_llm_model_groups
from evaluation.nli.interface import get_nli_model_groups
from evaluation.plot_style import (
    ANNOT_FONT_SIZE,
    CMAP_SEQUENTIAL_PRIMARY,
    CMAP_SEQUENTIAL_SECONDARY,
    LABEL_FONT_SIZE,
    METRIC_DECIMAL_PLACES,
    TICK_FONT_SIZE,
)
from utils import DATA_DIR


def vis_combined_table(
    encoder_models: list[AggregatedModelData],
    nli_models: list[AggregatedModelData],
    llm_models: list[AggregatedModelData],
    class_mode: str = "full",
) -> pd.DataFrame:
    """Build and display a combined comparison table across all three methods.

    Separator rows labelled with each method name are inserted between the three
    groups of model rows.  Within each numeric column the globally best value is
    bolded (min for RMSE, max for F1w).
    """
    metrics = ["RMSE", "F1w"]
    splits = ["Train", "Test"]
    columns = [f"{split} {metric}" for metric in metrics for split in splits]

    encoder_groups, enc_show_large = get_encoder_model_groups()
    nli_groups, _ = get_nli_model_groups([m.model for m in nli_models])
    llm_groups, _ = get_llm_model_groups([m.model for m in llm_models])

    with capture_output():
        df_enc = vis_all_models_tables(
            encoder_models,
            metrics,
            splits,
            encoder_groups,
            enc_show_large,
            class_mode=class_mode,
        )
        df_nli = vis_all_models_tables(
            nli_models,
            metrics,
            splits,
            nli_groups,
            False,
            class_mode=class_mode,
        )
        df_llm = vis_all_models_tables(
            llm_models,
            metrics,
            splits,
            llm_groups,
            False,
            class_mode=class_mode,
        )

    METHOD_LABELS = ["Feature-based", "NLI", "LLM"]
    all_group_headers = (
        list(encoder_groups.keys()) + list(nli_groups.keys()) + list(llm_groups.keys())
    )

    def _sep(label: str) -> pd.DataFrame:
        return pd.DataFrame([{"Model": label, **{c: "" for c in columns}}])

    random_rows = df_enc[df_enc["Model"] == "Random"].copy()
    df_enc_remaining = df_enc[df_enc["Model"] != "Random"].reset_index(drop=True)

    combined = pd.concat(
        [
            random_rows,
            _sep("Feature-based"),
            df_enc_remaining,
            _sep("NLI"),
            df_nli,
            _sep("LLM"),
            df_llm,
        ],
        ignore_index=True,
    )

    def _style_rows(row):
        if row["Model"] in METHOD_LABELS:
            return ["background-color: #e8e8e8; font-weight: bold"] * len(row)
        if row["Model"] in all_group_headers:
            result = [""] * len(row)
            result[0] = "font-weight: bold"
            return result
        return [""] * len(row)

    def _bold_best(s):
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() == 0:
            return [""] * len(s)
        best_val = numeric.min() if "RMSE" in s.name else numeric.max()
        return [
            "font-weight: bold"
            if (not pd.isna(v) and np.isclose(float(v), float(best_val), atol=1e-6))
            else ""
            for v in numeric
        ]

    def _fmt(x):
        if pd.isna(x) or x == "":
            return ""
        try:
            return f"{float(x):.{METRIC_DECIMAL_PLACES}f}"
        except (ValueError, TypeError):
            return str(x)

    display(
        combined.style.apply(_style_rows, axis=1)
        .apply(_bold_best, axis=0)
        .format({col: _fmt for col in columns})
        .set_table_styles(
            [{"selector": "th.row_heading, th.blank", "props": [("display", "none")]}]
        )
    )
    return combined


def vis_combined_conf_matrices(
    encoder_models: list[AggregatedModelData],
    nli_models: list[AggregatedModelData],
    llm_models: list[AggregatedModelData],
    encoder_model_name: str = "catboost_minilm",
    encoder_display_name: str = "CatBoost",
    nli_model_name: str = "RoBERTa",
) -> None:
    """Plot a 2×3 grid of confusion matrices (test set) for the best model per method.

    Rows correspond to class modes (relaxed 3-class, normal 6-class); columns to
    methods (Feature-based encoder, NLI, LLM).  The figure is saved to
    DATA_DIR/figures/conf_matrices_overall.pdf.
    """
    enc_model = copy(next(m for m in encoder_models if m.model == encoder_model_name))
    enc_model.model = encoder_display_name
    nli_model = next(m for m in nli_models if m.model == nli_model_name)
    llm_model = llm_models[0]

    methods = [enc_model, nli_model, llm_model]
    col_headers = ["Feature-based", "NLI", "LLM"]
    col_titles = [f"{h}: {m.model}" for h, m in zip(col_headers, methods)]
    # modes = ["relaxed", "full"]
    # row_labels = ["Relaxed", "Normal"]
    modes = ["full"]
    row_labels = ["Normal"]
    cmaps = [CMAP_SEQUENTIAL_PRIMARY, CMAP_SEQUENTIAL_SECONDARY]

    fig, axes = plt.subplots(
        len(modes),
        3,
        figsize=(6.8, 2.6 * len(modes)),
        gridspec_kw={"hspace": 0.02, "wspace": 0.02},
        squeeze=False,
    )

    for row in range(len(modes)):
        mode = modes[row]
        cmap = cmaps[row]
        labels = (
            ["0/1", "2/3", "4/5"] if mode == "relaxed" else [str(i) for i in range(6)]
        )

        for col in range(3):
            ax = axes[row, col]
            cm = get_confusion_matrix(methods[col], "test", class_mode=mode)

            sns.heatmap(
                cm,
                annot=True,
                fmt=".0f",
                cmap=cmap,
                ax=ax,
                linewidths=0,
                cbar=False,
                xticklabels=labels if row == len(modes) - 1 else False,
                yticklabels=labels if col == 0 else False,
                annot_kws={"size": ANNOT_FONT_SIZE},
            )

            ax.grid(False)
            for collection in ax.collections:
                collection.set_edgecolor("face")

            n_classes = cm.shape[0]
            for idx, text in enumerate(ax.texts):
                if idx // n_classes == idx % n_classes:
                    text.set_fontweight("bold")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=LABEL_FONT_SIZE, pad=4)

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, length=2, pad=2)

            if col == 0:
                ax.set_ylabel(
                    f"True Label{f'\\n({row_labels[row]})' if len(row_labels) > 1 else ''}",
                    fontsize=LABEL_FONT_SIZE,
                    labelpad=6,
                )

    axes[len(modes) - 1, 1].set_xlabel(
        "Predicted Label", fontsize=LABEL_FONT_SIZE, labelpad=4
    )

    fig.subplots_adjust(left=0.15, bottom=0.09, right=0.99, top=0.95)
    fig.savefig(DATA_DIR / "figures" / "conf_matrices_overall.pdf", bbox_inches="tight")
    plt.show()
