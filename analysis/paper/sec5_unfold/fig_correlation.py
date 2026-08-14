"""FIGURE I -- the post-fit correlation matrix of all 90 parameters.

The four blocks are fitted together and cannot be read apart: what the templates cost in uncertainty is
decided by how strongly they correlate with the flux, the cross-section knobs and the detector dials.
This is the object that decides it.

    [ c templates (9) | f flux (10) | theta cross section (11) | d detector (60) ]

Read it for three things:
  * the template-flux block, which is why an injected flux distortion lands on the templates (figure H);
  * the template-template block, which is why a projection cannot simply add its cells in quadrature;
  * the detector block, which is nearly diagonal -- 60 independent 5% dials stay independent, because
    nothing in the data prefers one arrangement of them over another.

Usage:  python -m analysis.paper.sec5_unfold.fig_correlation [label] [study]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

BLOCK_LABEL = {"template": r"$c$  templates", "flux": r"$f$  flux",
               "xsec": r"$\theta$  cross sec.", "detector": r"$d$  detector"}


def main(label="sec5", study="asimov"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    C = np.asarray(z[f"{study}_cov"])
    blocks = [str(b) for b in z["param_block"]]
    d = np.sqrt(np.abs(np.diag(C)))
    R = C / np.outer(np.maximum(d, 1e-300), np.maximum(d, 1e-300))

    # block boundaries, in the order the fit packs them
    order, bounds, seen = [], [], []
    for b in blocks:
        if b not in seen:
            seen.append(b)
    start = 0
    for b in seen:
        n = blocks.count(b)
        bounds.append((b, start, start + n))
        start += n

    # Two views.  The detector block is 60 of the 90 parameters and is almost exactly diagonal, so on
    # its own the full matrix devotes two thirds of its area to showing that nothing happens there.  The
    # zoom carries the physics; the full matrix is what proves the zoom is not hiding anything.
    nz = [e for b, _s, e in bounds if b == "xsec"][0]        # templates + flux + xsec
    fig = plt.figure(figsize=(10.4, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.06], wspace=0.24)

    for k, (M, bnd, ttl) in enumerate([
            (R, bounds, f"(a) all {len(blocks)} parameters"),
            (R[:nz, :nz], [b for b in bounds if b[0] != "detector"],
             "(b) templates, flux and cross section")]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper", interpolation="nearest")
        for _b, st_, _e in bnd[1:]:
            ax.axhline(st_ - 0.5, color="k", lw=0.8)
            ax.axvline(st_ - 0.5, color="k", lw=0.8)
        ticks = [0.5 * (st_ + e_) - 0.5 for _b, st_, e_ in bnd]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([BLOCK_LABEL[b] for b, _s, _e in bnd], fontsize=7.5, rotation=22, ha="right")
        ax.set_yticklabels([BLOCK_LABEL[b] for b, _s, _e in bnd], fontsize=7.5, rotation=68, va="center")
        ax.tick_params(length=0)
        ax.set_title(ttl, fontsize=8.5, loc="left")
        if k == 1:
            fig.colorbar(im, ax=ax, pad=0.02, shrink=0.88).set_label("correlation", fontsize=8)
    ax = fig.axes[1]

    # the number the section turns on: how hard the templates pull against the flux
    ct = [(s, e) for b, s, e in bounds if b == "template"][0]
    fl = [(s, e) for b, s, e in bounds if b == "flux"][0]
    cf = R[ct[0]:ct[1], fl[0]:fl[1]]
    fig.suptitle(rf"post-fit correlation ({study})   —   template$\,\leftrightarrow\,$flux:"
                 rf" mean $|\rho|={np.abs(cf).mean():.2f}$, max $={np.abs(cf).max():.2f}$",
                 fontsize=9, y=0.99)

    style.save(fig, f"{label}_figI")


if __name__ == "__main__":
    main(*(sys.argv[1:3] or []))
