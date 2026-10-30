"""Shared manuscript figure style: English labels, 8 pt Times New Roman."""

import matplotlib as mpl
from matplotlib import font_manager

FONT_FAMILY = "Times New Roman"
FONT_SIZE_PT = 8
FIGURE_SIZE_INCHES = (7.16, 4.6)
COLORS = {"random": "#2463A6", "procedural": "#CF651E"}
LINESTYLES = {"random": "-", "procedural": "--"}
LABELS = {"random": "Random initialization", "procedural": "Procedural initialization"}


def apply_paper_style():
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=FONT_FAMILY),
        fallback_to_default=False,
    )
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_PT,
        "font.weight": "normal",
        "axes.titlesize": FONT_SIZE_PT,
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_SIZE_PT,
        "axes.labelweight": "normal",
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "legend.title_fontsize": FONT_SIZE_PT,
        "figure.titlesize": FONT_SIZE_PT,
        "figure.labelsize": FONT_SIZE_PT,
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": False,
        "text.usetex": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.05,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        # Outline the verified Times New Roman glyphs for portable SVG output.
        "svg.fonttype": "path",
        "svg.hashsalt": "paper01-figure-style-v1",
        "savefig.dpi": 600,
    })
    return font_path
