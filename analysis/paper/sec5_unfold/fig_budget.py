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

# NAMES ONLY.  These labels once carried counts -- "cross section (11)", "detector (60, 5%)" -- which
# were literals, went stale across a change of signal definition, truth binning and reco grid, and ended
# up contradicting the caption on the same page.  Deriving them from the npz fixed the staleness but
# kept the duplication: the counts belong in the caption, stated once, where they can be read alongside
# everything else about the fit.  A legend only has to say which bar is which.
ASPECT = 0.5          # fallback; the config's figures.aspect wins.  1.0 = one row per bin
# ONE tick-label size for BOTH axes.  The y ticks were overridden to 6.5 to fit 29 labels at full
# height while x inherited style's 8, so the two axes of the same figure were set in different sizes.
# With the pitch now adapting to the height there is room for the shared value.
TICK_FS = 8
BLOCKS = [("stat", "statistical", "#b9c6de"),
          ("xsec", "cross section", style.C_QE),
          ("flux", "flux", style.C_RES),
          ("det", "detector", "#1a9e57")]


def _fig_cfg(z):
    """The `figures:` block of the config that produced `z`.

    Read from the npz, NOT from the yaml: the figure scripts are pure consumers of a persisted fit --
    that is what stops a plot disagreeing with the numbers it claims to show -- and a yaml can change
    after the fit has run.  Older files carry no cfg_json, so the module defaults stand in.
    """
    import json
    if "cfg_json" in z.files:
        return json.loads(str(z["cfg_json"])).get("figures", {}) or {}
    return {}


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
    # NEAR-SQUARE.  One row per truth bin at a readable pitch gives ~8.9 in for 58 bins, which is a
    # column-and-a-half of page for a figure whose message -- which block dominates, and where -- is
    # legible at half that.  ASPECT sets the trade: the bars get thinner, so the tick pitch has to
    # thin out with them or the labels collide.
    height = max(3.2, float(_fig_cfg(z).get("aspect", ASPECT)) * (0.135 * nt + 1.1))
    fig = plt.figure(figsize=(4.2, height))
    ax = fig.add_subplot(1, 1, 1)
    for k, lab, col in reversed(BLOCKS):          # longest first, each drawn over the last
        ax.barh(x, e[k], color=col, height=0.74, label=lab, zorder=2)
    ax.set_ylabel("linearised truth bin index")
    ax.set_xlabel(r"$\sigma(c_j)$ with block enabled")
    # ENDPOINTS ONLY.  The figure does not print the binning, so a truth bin index maps to nothing the
    # reader can look up -- labelling 20 of them implied a correspondence that is not on the page.  The
    # first and last say how many bins there are, which is the only thing the index axis has to carry
    # here; the point of the figure is which block dominates, not which cell is which.
    ax.set_yticks([0, nt - 1])
    ax.invert_yaxis()                              # cell 0 at the top, reading order

    # X KEEPS ITS TICK MARKS, Y DOES NOT.  On x they mark a continuous scale and the reader uses them
    # to place a bar's length between labels.  On y there is no scale to subdivide -- just two
    # endpoints on a categorical index -- so a tick mark there points at nothing.  No grid: the spines
    # already give the reference, and gridlines behind 58 bars add lines faster than readability.
    # BOTTOM ONLY.  style sets xtick.top=True, so axis="x" puts marks on the top spine too -- a second
    # copy of the scale along an edge nothing is measured from.
    ax.tick_params(axis="x", top=False, bottom=True, labelsize=TICK_FS)
    ax.tick_params(axis="y", length=0, labelsize=TICK_FS)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=TICK_FS, ncol=1, loc="upper right")
    ax.set_xlim(0, 1.32 * total.max())
    style.save(fig, "unfolding_uncertainty_budget")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
