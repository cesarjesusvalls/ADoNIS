"""FIGURE G -- where the uncertainty on an unfolded spectrum comes from.

The systematic blocks are switched on one at a time and the template error re-evaluated.  Decomposing
this way rather than by fitting each block alone is what keeps the parts adding up to the whole -- these
parameters are correlated with each other and with the templates, and a block fitted in isolation would
understate what it costs in the presence of the others.

The bars are NESTED, not stacked.  Errors add in quadrature, so a stack of contributions would rise to
sqrt-of-a-sum drawn as a sum, and the true total would float somewhere inside it -- which is exactly what
the first version of this figure showed.  Here each bar's TOP is the actual sigma once that block is
switched on, so the visible band is the increment and the tallest bar is the total.

(a) per truth cell, nested.  (b) the means, as multiples of the statistical error.

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

    fig = plt.figure(figsize=(9.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for k, lab, col in reversed(BLOCKS):          # tallest first, each drawn over the last
        ax.bar(x, e[k], color=col, width=0.74, label=lab, zorder=2)
    ax.set_xlabel("truth cell"); ax.set_ylabel(r"$\sigma(c_j)$ with block enabled")
    ax.set_xticks(x)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylim(0, 1.32 * total.max())
    ax.set_title("(a)  per-cell error budget (nested)", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    means = [e[k].mean() for k, _l, _c in BLOCKS]
    ax.plot(range(len(BLOCKS)), np.array(means) / means[0], "o-", color="#1f4b9c", ms=5, lw=1.4)
    for i, (m, (k, lab, _c)) in enumerate(zip(means, BLOCKS)):
        ax.annotate(f"{m:.3f}", (i, m / means[0]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5)
    ax.set_xticks(range(len(BLOCKS)))
    ax.set_xticklabels(["stat", "+xsec", "+flux", "+det"], fontsize=8)
    ax.set_ylabel(r"mean $\sigma(c_j)$ / statistical")
    ax.set_ylim(0.9, 1.05 * means[-1] / means[0])
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("(b)  cumulative", fontsize=9.5, loc="left")

    style.save(fig, f"{label}_figG")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
