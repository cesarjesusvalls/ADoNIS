"""FIGURE G -- where the uncertainty on an unfolded spectrum comes from.

The systematic blocks are switched on one at a time and the template error re-evaluated.  Decomposing
this way rather than by fitting each block alone is what keeps the parts adding up to the whole -- these
parameters are correlated with each other and with the templates, and a block fitted in isolation would
understate what it costs in the presence of the others.

The bars are NESTED, not stacked.  Errors add in quadrature, so a stack of contributions would rise to
sqrt-of-a-sum drawn as a sum, and the true total would float somewhere inside it -- which is exactly what
the first version of this figure showed.  Here each bar's TOP is the actual sigma once that block is
switched on, so the visible band is the increment and the tallest bar is the total.

A single panel: the per-truth-cell nested budget.

Usage:  python -m analysis.paper.sec5_unfold.fig_budget [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

BLOCKS = [("stat", "statistical", "#b9c6de"),
          ("xsec", "cross section (11)", "#1f4b9c"),
          ("flux", "flux (10, correlated)", "#e08214"),
          ("det", "detector (60, 5%)", "#1a9e57")]


def main(label="sec5"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    e = {k: np.asarray(z[f"budget_{k}"]) for k, _l, _c in BLOCKS}
    nt = len(e["stat"])
    x = np.arange(nt)

    total = e["det"]

    # SINGLE PANEL.  The cumulative panel (b) carried one number per block -- the same four numbers the
    # nested bars already show as their tops -- so it restated the left panel in a second geometry
    # rather than adding information.  With 30 truth cells the per-cell panel needs the width far more
    # than the summary needed the space.
    # VERTICAL.  With 58 truth cells a horizontal bar chart gives each cell a readable label and a full
    # figure height to separate them; horizontally they collide.  The nesting is unchanged -- each bar's
    # LENGTH is the actual sigma once that block is switched on, so the visible increment is what the
    # block costs and the longest bar is the total.
    fig = plt.figure(figsize=(4.2, max(3.5, 0.135 * nt + 1.1)))
    ax = fig.add_subplot(1, 1, 1)
    for k, lab, col in reversed(BLOCKS):          # longest first, each drawn over the last
        ax.barh(x, e[k], color=col, height=0.74, label=lab, zorder=2)
    ax.set_ylabel("linearised truth bin index")
    ax.set_xlabel(r"$\sigma(c_j)$ with block enabled")
    # ALWAYS LABEL THE LAST BIN.  Thinning to every second index drops the final one whenever the
    # count is even (58 cells -> ticks at 0,2,...,56, and bin 57 is drawn but unlabelled), which reads
    # as a missing bin rather than a missing tick.
    tk = list(range(0, nt, 2)) if nt > 20 else list(range(nt))
    if tk[-1] != nt - 1:
        tk.append(nt - 1)
    ax.set_yticks(tk)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.invert_yaxis()                              # cell 0 at the top, reading order
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=7, ncol=1, loc="lower right")
    ax.set_xlim(0, 1.32 * total.max())
    style.save(fig, f"{label}_figG")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
