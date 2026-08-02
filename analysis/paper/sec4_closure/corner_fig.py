"""Sec 4.4 -- the parameter corner: 2-D correlations, Gaussian ellipse vs exact non-Gaussian contour.

Reads the exact grid corner (<label>_corner.npz: per-pair Delta-chi2 grids) and V (Gaussian).  Lower
triangle: each pair's exact 68%/95% contour (filled) with the Gaussian covariance ellipse (dashed)
overlaid, in sigma_post units, dials ordered by physics block.  Diagonal: the 1-D Gaussian marginal.
The thing 1-D intervals cannot show -- which dials are correlated/degenerate, and where the true contour
departs from the ellipse.

Usage:  python -m analysis.paper.sec4_closure.corner_fig [label]     (default sec4_closure_r16_noprior)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

LEV = [2.30, 6.17]                                                # 2-D 68% / 95% (2 dof)


def _ellipse(Vp, sa, sb, xx, yy):
    """Gaussian Delta-chi2 field on the (a,b) grid (in sigma units) from the 2x2 marginal submatrix."""
    Pi = np.linalg.inv(Vp)
    X, Y = np.meshgrid(xx * sa, yy * sb, indexing="ij")
    return Pi[0, 0] * X**2 + 2 * Pi[0, 1] * X * Y + Pi[1, 1] * Y**2


def main(label="sec4_closure_r16_noprior", dials=None, taylor=False):
    """dials: comma-sep knob names for a REDUCED (main-body) corner; None = all 16 (appendix).
    taylor: overlay the analytic 2nd-order-model Taylor contour (folds the old Sec-5.2 validation here)."""
    style.use()
    z = np.load(style.ALTGEN / f"{label}_corner.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    ax_sig = np.asarray(z["axis_sigma"]); dchi2 = np.asarray(z["dchi2"]); V = np.asarray(z["V"])
    pairs = {tuple(p): i for i, p in enumerate(z["pairs"])}
    spost = np.sqrt(np.abs(np.diag(V)))

    if dials:
        want = [d.strip() for d in dials.split(",")]
        order = sorted((sub.index(pn.index(d)) for d in want), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    else:
        order = sorted(range(len(sub)), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    n = len(order)
    Jb = Bb = W = None
    if taylor:
        from analysis.paper.sec4_closure.corner_taylor import profile_pair
        zd = np.load(style.ALTGEN / f"{label}_derivs.npz", allow_pickle=True)
        Jb, Bb, W = np.asarray(zd["Jb"]), np.asarray(zd["Bb"]), np.asarray(zd["W"])

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.5}):
        sz = max(6, 2.3 * n)
        fig, axes = plt.subplots(n, n, figsize=(sz, sz), squeeze=False)
        for ri in range(n):
            for ci in range(n):
                ax = axes[ri, ci]
                a, b = order[ci], order[ri]                       # column = x-dial, row = y-dial
                if ci > ri:
                    ax.axis("off"); continue
                if ci == ri:                                      # diagonal: 1-D Gaussian marginal
                    ax.plot(ax_sig, np.exp(-0.5 * ax_sig**2), color=style.KNOB_GROUP_COLOR[style.knob_group(pn[sub[a]])])
                    ax.set_yticks([])
                else:
                    key = (a, b) if a < b else (b, a)
                    D = dchi2[pairs[key]]
                    if a > b:                                     # stored as (b,a): transpose so x=a,y=b
                        D = D.T
                    X, Y = np.meshgrid(ax_sig, ax_sig, indexing="ij")
                    ax.contourf(X, Y, D, levels=[0] + LEV, colors=["#c9dcfb", "#eaf1fd"], alpha=0.9)
                    ax.contour(X, Y, D, levels=LEV, colors="#1f4b9c", linewidths=0.7)
                    GZ = _ellipse(V[np.ix_([a, b], [a, b])], spost[a], spost[b], ax_sig, ax_sig)
                    ax.contour(X, Y, GZ, levels=LEV, colors=style.FIT_EC, linewidths=0.7, linestyles="--")
                    if taylor:                                    # analytic autodiff-Taylor contour (folds 5.2)
                        T = profile_pair(Jb, Bb, W, a, b, ax_sig * spost[a], ax_sig * spost[b])
                        ax.contour(X, Y, T, levels=LEV, colors="#2e7d32", linewidths=0.7, linestyles=":")
                    ax.axhline(0, color="0.85", lw=0.3); ax.axvline(0, color="0.85", lw=0.3)
                ax.set_xlim(ax_sig.min(), ax_sig.max())
                if ci != ri:
                    ax.set_ylim(ax_sig.min(), ax_sig.max())
                ax.set_xticks([]); ax.set_yticks([])
                if ci == 0 and ri > 0:
                    ax.set_ylabel(style.plab(pn[sub[order[ri]]]), fontsize=7, rotation=0, ha="right", va="center")
                if ri == n - 1:
                    ax.set_xlabel(style.plab(pn[sub[order[ci]]]), fontsize=7, rotation=90)
        # legend (top-right off-diagonal cell)
        lg = axes[0, n - 1]; lg.axis("off")
        lg.plot([], [], color="#1f4b9c", lw=1.4, label="exact 68/95%")
        lg.plot([], [], color=style.FIT_EC, lw=1.4, ls="--", label="Gaussian")
        if taylor:
            lg.plot([], [], color="#2e7d32", lw=1.4, ls=":", label="autodiff Taylor")
        lg.legend(fontsize=9 if n > 8 else 11, loc="center")
        ttl = "Sec 4.4  parameter corner — exact (blue) vs Gaussian (orange)" + \
              (" vs autodiff-Taylor (green)" if taylor else "") + (f"   ({n} dials)" if dials else "")
        fig.suptitle(ttl, fontsize=12, y=0.925)
        fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.07, right=0.98, top=0.9, bottom=0.07)
        style.save(fig, "sec4_fig44_corner_reduced" if dials else "sec4_fig44_corner")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    dials = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--dials=")), None)
    taylor = "--taylor" in sys.argv[1:]
    main(*(pos[:1] or ["sec4_closure_r16_noprior"]), dials=dials, taylor=taylor)
