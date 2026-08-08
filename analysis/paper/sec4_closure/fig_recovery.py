"""Sec 4.1 — recovery + 1-D uncertainty: Gaussian sigma vs 68% BAYESIAN credible interval, per dial.

Reads the closure profile npz (<label>_profile.npz).  For each of the 16 dials: the best-fit (= profile
minimum, which after the Newton-decrement fit coincides with the GN best-fit) with TWO error bars overlaid
-- the symmetric Gaussian sigma (grey) and the asymmetric 68% credible interval (blue) -- against the
injected truth (star).  The credible interval is the highest-density 68.27% region of the marginal
posterior, approximated by exp(-Delta chi2_profile / 2) (profile-as-marginal / Laplace).  They coincide
where the posterior is Gaussian and separate where it is not (Eb_shift, at its E_b >= 0 wall).  This single
figure folds "does the fit recover truth" and "what do the error bars mean".

Usage:  python -m analysis.paper.sec4_closure.fig_recovery [label]     (default sec4_closure_random16)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from scipy.interpolate import CubicSpline
from analysis.paper.physical_fit import PRIOR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs

C_FIT, C_GAUSS = "#1f4b9c", "0.55"


def _credible(grid, prof, mass=0.6827):
    """68% BAYESIAN credible interval from a profiled Delta-chi2 curve: the marginal posterior is
    approximated by exp(-Delta chi2 / 2) (profile-as-marginal / Laplace); return (mode, lo, hi) in
    sigma_post units as the highest-density interval containing `mass`."""
    spl = CubicSpline(grid, prof)
    xf = np.linspace(grid.min(), grid.max(), 2001)
    dens = np.exp(-0.5 * np.maximum(spl(xf), 0.0)); dens /= np.trapezoid(dens, xf)
    o = np.argsort(dens)[::-1]
    cum = np.cumsum(dens[o]) * (xf[1] - xf[0])
    thr = dens[o][min(np.searchsorted(cum, mass), len(o) - 1)]
    sel = dens >= thr
    return float(xf[np.argmax(dens)]), float(xf[sel].min()), float(xf[sel].max())


def main(label="sec4_closure_random16"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_profile.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    grid = np.asarray(z["grid_sigma"]); prof = np.asarray(z["prof_dobj"])
    bfp = np.asarray(z["bfp"]); spost = np.asarray(z["sigma_post"]); truth = np.asarray(z["truth"])
    nom = np.asarray(theta_nominal(nominal_knobs())); prior = np.asarray(PRIOR)

    order = sorted(range(len(sub)), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        # SINGLE-COLUMN figure: sized for one journal column (~3.4 in), so fonts are set small here rather
        # than shrunk by the layout at insertion time.
        fig, ax = plt.subplots(figsize=(3.5, 5.0))
        yy = np.arange(len(order)); gid = [style.knob_group(pn[sub[c]]) for c in order]
        for i, c in enumerate(order):
            k = sub[c]; p = max(prior[k], 1e-12)
            mode, clo, chi = _credible(grids[c] if grids is not None else grid, prof[c])                      # 68% credible (sigma units)
            xhat = (bfp[k] + mode * spost[c] - nom[k]) / p
            # BOTH intervals on the SAME line (no vertical offset).  They are told apart by cap geometry,
            # not by position: the Gaussian gets TALL THIN caps drawn behind, the credible interval a THICK
            # SHORT bar on top.  Wherever the two differ, the tall grey cap stands clear of the blue bar
            # (and vice versa), so each endpoint stays readable even though the lines coincide.
            ax.errorbar([xhat], [yy[i]], xerr=[[spost[c] / p], [spost[c] / p]], fmt="none",
                        ecolor=C_GAUSS, capsize=5.5, capthick=1.0, elinewidth=1.0, zorder=2)
            ax.errorbar([xhat], [yy[i]], xerr=[[(mode - clo) * spost[c] / p], [(chi - mode) * spost[c] / p]],
                        fmt="none", ecolor=C_FIT, capsize=2.4, capthick=1.7, elinewidth=2.4, zorder=4)
            ax.plot(xhat, yy[i], "D", ms=5.5, color=C_FIT, mec="white", mew=0.6, zorder=6)
            ax.plot((truth[k] - nom[k]) / p, yy[i], marker="*", ms=13, color="k", zorder=5, ls="none")
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_yticks(yy); ax.set_yticklabels([style.plab(pn[sub[c]]) for c in order], fontsize=7.5)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel(r"$(\theta-\theta_{\rm nom})\,/\,\sigma_{\rm ref}$", fontsize=9)
        # sigma_ref is the Gate-I prior WIDTH, used purely as a per-dial length unit so 16 dials with
        # different units share one axis.  It is NOT a prior on the fit -- this figure is MLE
        # (data-only).  It is 20% of nominal for 15 of the 16 dials; Eb_shift, an additive offset
        # with nominal ~0, uses its 4 MeV width instead (a fractional axis would give 299 there).
        # Physics blocks as faint SHADED BANDS with the block name inside, top-right of its own band --
        # replaces the colour-tab column left of the axis, which cost horizontal room a single-column
        # figure cannot spare.
        bounds = [i for i in range(1, len(gid)) if gid[i] != gid[i - 1]]
        seg = [-.5] + [b - .5 for b in bounds] + [len(gid) - .5]
        yt = ax.get_yaxis_transform()                      # x in axes fraction, y in data
        for a, b in zip(seg[:-1], seg[1:]):
            g = gid[int((a + b) / 2 + .5)]
            ax.axhspan(a, b, facecolor=style.KNOB_GROUP_COLOR[g], alpha=0.10, zorder=0, lw=0)
            ax.text(0.982, a + 0.12, style.KNOB_GROUP_NAME[g].replace("\n", " "), transform=yt,
                    ha="right", va="top", fontsize=6.5, color="0.25", fontweight="bold", zorder=7)
        for b in bounds:                                   # thin separators between blocks
            ax.axhline(b - .5, color="0.35", lw=0.7, zorder=1)
        # full box: keep all four spines, but no ticks on top/right (style.py sets xtick.top/ytick.right)
        ax.tick_params(which="both", top=False, right=False, labelsize=7.5)
        ax.plot([], [], "*", color="k", ms=12, label="injected truth")
        ax.plot([], [], "D", color=C_FIT, mec="white", mew=0.6, ms=7, label="best-fit")
        ax.plot([], [], "-", color=C_FIT, lw=2.4, label="68% credible (Bayesian)")
        ax.plot([], [], "-", color=C_GAUSS, lw=1.0, label=r"Gaussian $\sigma$")
        ax.legend(fontsize=6.2, loc="lower left", framealpha=0.92, borderpad=0.35,
                  handletextpad=0.5, labelspacing=0.3, handlelength=1.4, borderaxespad=0.4)
        # no title -- the caption carries it
        fig.tight_layout(pad=0.4)
        style.save(fig, "sec4_fig41_recovery")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
