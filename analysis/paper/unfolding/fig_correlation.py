"""FIGURE I -- the post-fit correlation matrix of all fit parameters, in blocks:

    [ c templates | f flux | theta cross section | d detector ]

The blocks are fitted together, so what the templates cost in uncertainty is set by how strongly they
correlate with flux, cross-section, and detector dials -- this is the matrix that shows it.

Usage:  python -m analysis.paper.unfolding.fig_correlation [label] [study]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

BLOCK_LABEL = {"template": r"$\vec{c}$  templates", "flux": r"$\vec{f}$  flux",
               "xsec": r"$\vec{x}$  cross section", "detector": r"$\vec{d}$  detector"}
BLOCK_FS = 7.5
BLOCK_SHORT = {"template": r"$\vec{c}$", "flux": r"$\vec{f}$",
               "xsec": r"$\vec{x}$", "detector": r"$\vec{d}$"}


def main(label="t2k_cc0pi", study="asimov"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    C = np.asarray(z[f"{study}_cov"])
    blocks = [str(b) for b in z["param_block"]]
    d = np.sqrt(np.abs(np.diag(C)))
    R = C / np.outer(np.maximum(d, 1e-300), np.maximum(d, 1e-300))

    order, bounds, seen = [], [], []
    for b in blocks:
        if b not in seen:
            seen.append(b)
    start = 0
    for b in seen:
        n = blocks.count(b)
        bounds.append((b, start, start + n))
        start += n

    ctx = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
           "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}
    fig = plt.figure(figsize=(5.6, 4.9))
    gs = fig.add_gridspec(1, 1)

    with plt.rc_context(ctx):
     for k, (M, bnd, ttl) in enumerate([
            (R, bounds, f"(a) all {len(blocks)} parameters")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(M, cmap=style.CMAP_GRAD_DIV, origin="upper", interpolation="nearest",
                       norm=SymLogNorm(linthresh=0.01, vmin=-1.0, vmax=1.0, base=10))
        for _b, st_, _e in bnd[1:]:
            ax.axhline(st_ - 0.5, color="k", lw=0.8)
            ax.axvline(st_ - 0.5, color="k", lw=0.8)
        ticks = [0.5 * (st_ + e_) - 0.5 for _b, st_, e_ in bnd]
        ax.set_xticks(ticks)
        ax.set_xticklabels([BLOCK_SHORT[b] for b, _s, _e in bnd], fontsize=BLOCK_FS, rotation=0,
                           ha="center", va="baseline")
        ax.tick_params(axis="x", pad=13)

        ax.set_yticks([])
        tr = ax.get_yaxis_transform()
        for b, st_, e_ in bnd:
            y0, y1 = st_ - 0.45, e_ - 0.55
            ax.plot([-0.012, -0.012], [y0, y1], transform=tr, color="0.35", lw=0.9,
                    solid_capstyle="butt", clip_on=False, zorder=8)
            for yy in (y0, y1):
                ax.plot([-0.012, -0.004], [yy, yy], transform=tr, color="0.35", lw=0.9,
                        clip_on=False, zorder=8)
            ax.annotate(BLOCK_LABEL[b], xy=(-0.022, 0.5 * (y0 + y1)), xycoords=tr,
                        xytext=(0, -2.6), textcoords="offset points", ha="right", va="baseline",
                        fontsize=BLOCK_FS, annotation_clip=False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if True:
            cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.88,
                              ticks=[-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
            cb.set_label("correlation (symlog)", fontsize=8)
            cb.ax.set_yticklabels(["$-1$", "$-0.1$", "$-0.01$", "$0$", "$0.01$", "$0.1$", "$1$"],
                                  fontsize=7)

    style.save(fig, "unfolding_param_correlations")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or []))
