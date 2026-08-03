"""Sec 4 companion -- marginal (NUTS) vs profile vs Gaussian, per dial, on the SAME Taylor posterior.

Reads <label>_nuts.npz (blackjax NUTS samples of the Taylor-surrogate posterior + that model's 1-D profile).
Ridgeline in standardized units u = (theta - BFP)/sigma_post, so every dial is ~unit width and the three
curves are directly comparable:
  * MARGINAL (blue fill)  -- KDE of the NUTS samples: integrate over the other 15 dials (true Bayesian),
  * PROFILE  (orange line) -- exp(-1/2 Delta chi2_profile): maximize over the other 15 (what 4.1 plots),
  * GAUSSIAN (grey dashed)  -- N(0,1).
Where marginal and profile separate you are seeing the nuisance-volume (Occam) factor that profiling drops.

Usage:  python -m analysis.paper.sec4_closure.fig_nuts_compare [label]   (default sec4_closure_r16_noprior)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_MARG, C_PROF, C_GAUSS = "#1f4b9c", "#fe6100", "0.45"
ROWSCALE = 0.90
NX = 500


def main(label="sec4_closure_r16_noprior"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_nuts.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    spost = np.asarray(z["spost"]); grid = np.asarray(z["grid_sigma"]); prof = np.asarray(z["prof_dchi2"])
    U = np.asarray(z["samples_D"]) / spost[None, :]              # standardized (theta-BFP)/sigma_post
    ndiv = int(z["n_div"]); nsamp = int(z["n_samples"])
    n = len(sub)

    order = sorted(range(n), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    xc = np.linspace(-4.0, 4.0, NX)
    dens_m, dens_p, dens_g, gid = [], [], [], []
    for c in order:
        dm = gaussian_kde(U[:, c])(xc); dm /= np.trapezoid(dm, xc)         # NUTS marginal
        spl = CubicSpline(grid, prof[c]); dp = np.exp(-0.5 * np.maximum(spl(xc), 0.0))
        dp /= np.trapezoid(dp, xc)                                          # profile
        dg = np.exp(-0.5 * xc**2); dg /= np.trapezoid(dg, xc)               # Gaussian N(0,1)
        m = max(dm.max(), dp.max(), dg.max(), 1e-300)
        dens_m.append(dm / m); dens_p.append(dp / m); dens_g.append(dg / m)
        gid.append(style.knob_group(pn[sub[c]]))

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(4.6, 7.4))
        for i, c in enumerate(order):
            y0 = n - 1 - i
            ax.axhline(y0, color="0.85", lw=0.6, zorder=y0 * 3)
            ax.plot(xc, y0 + ROWSCALE * dens_g[i], color=C_GAUSS, lw=1.0, ls=(0, (3, 2)), zorder=y0 * 3 + 1)
            ax.fill_between(xc, y0, y0 + ROWSCALE * dens_m[i], color=C_MARG, alpha=0.32,
                            lw=1.1, edgecolor=style.darker(C_MARG), zorder=y0 * 3 + 2)
            ax.plot(xc, y0 + ROWSCALE * dens_p[i], color=C_PROF, lw=1.3, zorder=y0 * 3 + 3)
            ax.text(xc[0] + 0.15, y0 + ROWSCALE, style.plab(pn[sub[c]]), ha="left", va="top",
                    fontsize=8.5, zorder=y0 * 3 + 5)
        # group tabs
        for g in sorted(set(gid)):
            rows = [n - 1 - i for i in range(n) if gid[i] == g]
            lo, hi = min(rows) - 0.35, max(rows) + 0.75
            ax.add_patch(plt.Rectangle((xc[0] - 0.35, lo), 0.10, hi - lo, transform=ax.transData,
                                       facecolor=style.KNOB_GROUP_COLOR[g], edgecolor="none",
                                       clip_on=False, zorder=6))
            ax.text(xc[0] - 0.62, (lo + hi) / 2, style.KNOB_GROUP_NAME[g], ha="center", va="center",
                    rotation=90, fontsize=8, color=style.darker(style.KNOB_GROUP_COLOR[g]))
        ax.set_xlim(xc[0] - 0.05, xc[-1] + 0.05); ax.set_ylim(-0.5, n - 1 + ROWSCALE + 0.35)
        ax.set_yticks([]); ax.set_xlabel(r"$(\theta - \hat\theta)\,/\,\sigma_{\rm post}$", fontsize=10)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.fill_between([], [], color=C_MARG, alpha=0.32, lw=1.1, edgecolor=style.darker(C_MARG),
                        label="marginal (NUTS)")
        ax.plot([], [], color=C_PROF, lw=1.4, label="profile")
        ax.plot([], [], color=C_GAUSS, lw=1.0, ls=(0, (3, 2)), label="Gaussian")
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.002), ncol=3,
                  frameon=False, columnspacing=1.1, handlelength=1.4, handletextpad=0.5)
        ax.set_title(f"Sec 4  marginal vs profile  (Taylor surrogate, {nsamp} NUTS samp, {ndiv} div)",
                     fontsize=9.5, loc="left", pad=30)
        fig.tight_layout()
        style.save(fig, "sec4_nuts_compare")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
