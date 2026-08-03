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
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        yy = np.arange(len(order)); gid = [style.knob_group(pn[sub[c]]) for c in order]
        for i, c in enumerate(order):
            k = sub[c]; p = max(prior[k], 1e-12)
            mode, clo, chi = _credible(grid, prof[c])                      # 68% credible (sigma units)
            xhat = (bfp[k] + mode * spost[c] - nom[k]) / p
            # symmetric Gaussian sigma (grey, offset up) for comparison
            ax.errorbar([xhat], [yy[i] + 0.2], xerr=[[spost[c] / p], [spost[c] / p]], fmt="none",
                        ecolor=C_GAUSS, capsize=2.5, lw=1.3, zorder=3)
            # asymmetric 68% credible interval (blue)
            ax.errorbar([xhat], [yy[i]], xerr=[[(mode - clo) * spost[c] / p], [(chi - mode) * spost[c] / p]],
                        fmt="none", ecolor=C_FIT, capsize=3, lw=1.6, zorder=4)
            ax.plot(xhat, yy[i], "D", ms=5.5, color=C_FIT, mec="white", mew=0.6, zorder=6)
            ax.plot((truth[k] - nom[k]) / p, yy[i], marker="*", ms=13, color="k", zorder=5, ls="none")
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_yticks(yy); ax.set_yticklabels([style.plab(pn[sub[c]]) for c in order], fontsize=9)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel(r"value $-$ nominal  (prior $\sigma$ units)", fontsize=10)
        style.knob_group_tabs(ax, gid, tabx=-0.185, tabw=0.02, labx=-0.235, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.plot([], [], "*", color="k", ms=12, label="injected truth")
        ax.plot([], [], "D", color=C_FIT, mec="white", mew=0.6, ms=7, label="best-fit")
        ax.plot([], [], "-", color=C_FIT, lw=1.6, label="68% credible (Bayesian)")
        ax.plot([], [], "-", color=C_GAUSS, lw=1.3, label=r"Gaussian $\sigma$")
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        ax.set_title("Sec 4.1  recovery + 1-D uncertainty: Gaussian $\\sigma$ vs 68% credible", fontsize=11.5, loc="left")
        fig.tight_layout()
        style.save(fig, "sec4_fig41_recovery")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
