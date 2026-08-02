"""Sec 4.1b non-Gaussian headline: the E_b posterior at its physical wall (shown as a DENSITY, not chi2).

E_b >= 0 is physical, so the (profile) likelihood for E_b, exp(-1/2 Delta chi2), is a TRUNCATED density that
piles against the wall.  The Gaussian approximation N(eb_hat, sigma) is symmetric and spills a large fraction
of its probability into UNPHYSICAL E_b < 0.  We show both densities, the posterior 68% credible interval
(shortest / highest-density), and the Gaussian's unphysical mass.

Reads output/altgen/sec4_ebwall.npz (fine 1-D E_b profile with a small injected value).

Usage:  python -m analysis.paper.sec4_closure.fig_ebwall
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_POST, C_GAUSS = "#1f4b9c", "#c44"


def _hpd(x, p, mass=0.6827):
    """Shortest interval containing `mass` of the density p(x) (x uniform-ish); returns (lo, hi)."""
    cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))]); cdf /= cdf[-1]
    best = (x[0], x[-1], np.inf)
    lo_targets = np.linspace(0, 1 - mass, 400)
    for a in lo_targets:
        xa = np.interp(a, cdf, x); xb = np.interp(a + mass, cdf, x)
        if xb - xa < best[2]:
            best = (xa, xb, xb - xa)
    return best[0], best[1]


def main():
    style.use()
    z = np.load(style.ALTGEN / "sec4_ebwall.npz", allow_pickle=True)
    eb = np.asarray(z["eb_grid"]); d = np.asarray(z["dchi2"])
    bfp = float(z["eb_bfp"]); sig = float(z["eb_sig"])

    # posterior density = exp(-1/2 Delta chi2), TRUNCATED at the wall (E_b >= 0); normalise on E_b >= 0.
    spl = CubicSpline(eb, d)
    xf = np.linspace(-0.6, eb.max(), 600)
    post = np.where(xf >= 0, np.exp(-0.5 * np.maximum(spl(xf), 0)), 0.0)
    xp = xf[xf >= 0]; pp = post[xf >= 0]
    norm = np.trapezoid(pp, xp); pp = pp / norm; post = post / norm
    gauss = stats.norm.pdf(xf, bfp, sig)
    unphys = stats.norm.cdf(0.0, bfp, sig)                        # Gaussian mass in E_b < 0
    lo, hi = _hpd(xp, pp)                                         # posterior 68% credible interval

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        ax.axvspan(xf.min(), 0.0, color="0.92", zorder=0)
        ax.text(xf.min() + 0.03, ax.get_ylim()[1], "unphysical\n$E_b<0$", fontsize=8.5, color="0.45", va="top")
        ax.axvline(0, color="0.5", lw=1.0)
        # Gaussian: fill the unphysical mass
        ax.plot(xf, gauss, "--", color=C_GAUSS, lw=1.7, label="Gaussian $N(\\hat E_b,\\sigma)$")
        ax.fill_between(xf, gauss, where=(xf < 0), color=C_GAUSS, alpha=0.25)
        # posterior density (truncated) + 68% credible band
        ax.plot(xf, post, "-", color=C_POST, lw=2.0, label="posterior $\\propto e^{-\\Delta\\chi^2/2}$ (walled)")
        m = (xp >= lo) & (xp <= hi)
        ax.fill_between(xp[m], pp[m], color=C_POST, alpha=0.22)
        ax.annotate(f"Gaussian: {unphys:.0%} of the\nprobability is unphysical",
                    (bfp - 1.6 * sig, gauss.max() * 0.35), fontsize=8.5, color=C_GAUSS, ha="center")
        ax.text(0.5 * (lo + hi), pp[m].min() * 0.4 if m.any() else 0, f"posterior 68%\n[{lo:.2f}, {hi:.2f}]",
                fontsize=8.5, color=C_POST, ha="center", va="bottom")
        ax.set_xlim(xf.min(), xf.max()); ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$E_b$ shift  [MeV]", fontsize=10); ax.set_ylabel("probability density", fontsize=10)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
        ax.set_title(r"$E_b$ at the wall: the posterior is truncated, the Gaussian spills unphysical",
                     fontsize=11, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        style.save(fig, "sec4_ebwall")
    print(f"  posterior 68% credible [{lo:.2f}, {hi:.2f}] MeV | Gaussian unphysical mass {unphys:.0%}")


if __name__ == "__main__":
    main()
