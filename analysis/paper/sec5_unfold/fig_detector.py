"""FIGURE E -- the detector: what the smearing does, and what survives it.

Three statements, because an unfolding result is only readable if you know what was unfolded:

  (a) the RESPONSE, column-normalised.  Each column is one truth cell; the colour is the probability that
      such an event is reconstructed in each reco bin.  A diagonal band means the detector preserves the
      variable; the width of that band is the whole reason templates cannot be made arbitrarily fine.
  (b) EFFICIENCY per truth cell -- how much of the signal survives reco selection.
  (c) PURITY per reco bin -- how much of what is reconstructed there is signal at all.

Usage:  python -m analysis.paper.sec5_unfold.fig_detector [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

# Series colours are section 4's, so the two halves of the paper that share a fit read alike.
C_EFF, C_PUR = "#1f4b9c", "#e08214"


def main(label="sec5"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    A = np.asarray(z["A"]).sum(axis=2)              # sum the flux axis: the response as reco x truth
    eff, pur = np.asarray(z["eff"]), np.asarray(z["purity"])
    ntrue, nreco = A.shape[1], A.shape[0]

    # Column-normalised: P(reco bin | truth cell).  Raw counts would just show the flux spectrum.
    col = A.sum(axis=0)
    P = np.divide(A, col[None, :], out=np.zeros_like(A), where=col[None, :] > 0)

    # Heatmaps go sans-serif and spineless, as in sections 2 and 3 -- the matrix IS the frame there,
    # and a serif tick label beside a dense grid reads as clutter.
    ctx = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
           "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    with plt.rc_context(ctx):
        im = ax.imshow(P, aspect="auto", origin="lower", cmap=style.CMAP_CONSTRAINT,
                       interpolation="nearest")
    # Both axes are FLATTENED 2-D grids, so without separators the block structure reads as noise.
    # reco is 10 dpt x 6 dat (dpt slowest) and truth is 3 x 3, matching binning.Grid2D.
    for r in range(6, 60, 6):
        ax.axhline(r - 0.5, color="w", lw=0.45, alpha=0.55)
    for c in range(3, 9, 3):
        ax.axvline(c - 0.5, color="w", lw=0.8, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel(r"truth cell   (blocks: $\delta p_T$)"); ax.set_ylabel(r"reco bin   (blocks: $\delta p_T$)")
    ax.set_xticks(range(ntrue)); ax.set_yticks([0, 15, 30, 45, 59])
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$P(\mathrm{reco}\,|\,\mathrm{truth})$", fontsize=8)
    ax.set_title("(a)  response", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(np.arange(ntrue), eff, color=C_EFF, width=0.72)
    ax.axhline(np.nansum(eff * col) / col.sum(), color="k", ls="--", lw=0.9,
               label=f"mean {np.nansum(eff*col)/col.sum():.2f}")
    ax.set_xlabel("truth cell"); ax.set_ylabel("efficiency"); ax.set_ylim(0.5, 0.85)
    ax.set_xticks(range(ntrue)); ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b)  efficiency", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[0, 2])
    ax.step(np.arange(nreco), pur, where="mid", color=C_PUR, lw=1.3)
    ax.axhline(np.nanmean(pur), color="k", ls="--", lw=0.9, label=f"mean {np.nanmean(pur):.2f}")
    ax.set_xlabel("reco bin"); ax.set_ylabel("purity"); ax.set_ylim(0.70, 0.95)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(c)  purity", fontsize=9.5, loc="left")

    style.save(fig, f"{label}_figE")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
