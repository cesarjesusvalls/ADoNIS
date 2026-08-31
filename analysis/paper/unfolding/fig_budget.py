"""FIGURE G -- where the uncertainty on an unfolded spectrum comes from: systematic blocks switched on
one at a time and the template error re-evaluated.

Bars are NESTED, not stacked: each bar's top is the actual sigma once that block is enabled, so the
increment is the visible band and the tallest bar is the total (stacking would draw sqrt-of-a-sum as a
sum).

Usage:  python -m analysis.paper.unfolding.fig_budget [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis._cli import fig_cfg
from analysis.paper import style

ASPECT = 0.5
TICK_FS = 8
BLOCKS = [("stat", "statistical", "#b9c6de"),
          ("xsec", "cross section", style.C_QE),
          ("flux", "flux", style.C_RES),
          ("det", "detector", "#1a9e57")]


def main(label="t2k_cc0pi"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    e = {k: np.asarray(z[f"budget_{k}"]) for k, _l, _c in BLOCKS}
    nt = len(e["stat"])
    x = np.arange(nt)

    total = e["det"]

    height = max(3.2, float(fig_cfg(z).get("aspect", ASPECT)) * (0.135 * nt + 1.1))
    fig = plt.figure(figsize=(4.2, height))
    ax = fig.add_subplot(1, 1, 1)
    for k, lab, col in reversed(BLOCKS):
        ax.barh(x, e[k], color=col, height=0.74, label=lab, zorder=2)
    ax.set_ylabel("linearised truth bin index")
    ax.set_xlabel(r"$\sigma(c_j)$ with block enabled")
    ax.set_yticks([0, nt - 1])
    ax.invert_yaxis()

    ax.tick_params(axis="x", top=False, bottom=True, labelsize=TICK_FS)
    ax.tick_params(axis="y", length=0, labelsize=TICK_FS)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=TICK_FS, ncol=1, loc="upper right")
    ax.set_xlim(0, 1.32 * total.max())
    style.save(fig, "unfolding_uncertainty_budget")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
