"""Shared publication style + paths for the paper figures. Style only -- no physics, no data."""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUTDIR = Path(os.environ.get("ADONIS_PAPER_OUT", "output/paper"))
ALTGEN = Path(os.environ.get("ADONIS_ALTGEN", "output/altgen"))   # persisted physfit npz

# ADoNIS / ACHILLES / data colours, used identically in every section.
C_ADONIS = "#1f77b4"
C_ACHILLES = "#d62728"
C_DATA = "k"

RC = {
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "legend.frameon": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3,
    "mathtext.default": "regular",
}


def use():
    mpl.rcParams.update(RC)


def save(fig, name):
    """Write <name>.png + <name>.pdf into OUTDIR and return the png path."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / f"{name}.png"
    fig.savefig(png)
    fig.savefig(OUTDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"[out] {png}")
    return png
