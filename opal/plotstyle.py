"""Figure style matching Tech Science Press requirements: Arial/Helvetica labels 8-9 pt, black text,
maximum width 16.51 cm, vector PDF plus 900 dpi TIFF for submission."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CM = 1 / 2.54
FULL_W = 16.5 * CM
HALF_W = 8.0 * CM
# colour-blind-safe categorical order (validated reference palette, slots 1-8)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#6b6b6b"


def apply():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4, "lines.markersize": 4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e3",
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": "black",
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        # mathematical symbols in the same typeface as the labels
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans",
    })


def panel_label(ax, s):
    ax.text(-0.14, 1.04, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".tif", dpi=900, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path + ".png", dpi=200)
