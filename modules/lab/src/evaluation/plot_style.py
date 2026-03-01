import matplotlib as mpl
import seaborn as sns

SUPTITLE_FONT_SIZE = 16  # figure-level suptitle (e.g. CV grid)
TITLE_FONT_SIZE = 14  # per-axes title
LABEL_FONT_SIZE = 11  # x/y axis labels
TICK_FONT_SIZE = 10  # tick labels and x-tick text labels
LEGEND_FONT_SIZE = 9  # legend text
ANNOT_FONT_SIZE = 9  # in-plot numeric annotations / bar value labels


def get_deep_colormap_colors(n: int = 10) -> list[str]:
    """
    Get a list of colors from the 'deep' seaborn color palette.

    Parameters:
        n (int): Number of colors to retrieve (default: 10).
    Returns:
        List[str]: List of hex color codes from the 'deep' seaborn palette.
    """

    return sns.color_palette("deep", n).as_hex()


PALETTE = get_deep_colormap_colors()  # default color palette for plots

# Colormaps
CMAP_PRIMARY = "deep"  # raw-count confusion matrix
CMAP_SECONDARY = "Pastel1"  # raw-count confusion matrix (relaxed mode)

GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"


def apply_plot_style() -> None:
    """Apply the unified plot style via rcParams.

    Call once per notebook before any plotting. Individual plotting
    functions do not need to set font sizes or grid style themselves.
    """
    import matplotlib.style as style

    style.use("seaborn-v0_8-colorblind")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Source Sans 3"],
            "font.size": TICK_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.titlesize": SUPTITLE_FONT_SIZE,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LINESTYLE,
        }
    )
