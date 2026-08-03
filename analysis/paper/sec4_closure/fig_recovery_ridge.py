"""Sec 4.1 (ridgeline variant) -- the posterior of every dial as a joyplot: exact vs Gaussian, per row.

One row per Gate-I dial, x-axis = value - nominal in prior-sigma units (COMMON across rows, so a tightly
constrained dial is a narrow spike and a loose one is a broad bump -- the width itself shows the constraint).
Each row overlays the EXACT non-Gaussian marginal posterior (filled, physics-group colour, from
exp(-Delta chi2_profile / 2)) and the GAUSSIAN posterior (grey line, N(best-fit, sigma_post)); a black
vertical tick marks the injected truth.  Where the two curves coincide the posterior is Gaussian; where they
peel apart (Eb_shift at its wall, the loosely-constrained cross-section dials) it is not.

This is an ALTERNATIVE to fig_recovery.py (the error-bar version) -- kept as a separate entry point so we can
pick whichever reads better.  Usage:  python -m analysis.paper.sec4_closure.fig_recovery_ridge [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.physical_fit import PRIOR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs

C_EXACT = "#1f4b9c"     # exact non-Gaussian posterior  (blue -- matches the 4.4 corner)
C_GAUSS = "#fe6100"     # Gaussian posterior             (orange -- matches the 4.4 corner)
ROWSCALE = 0.90         # peak height in row units (<1 -> every curve stays within its own row, uniform height)
NX = 600


def _density(grid_sig, prof, x_row, xc):
    """Exact posterior density exp(-Delta chi2/2) on the common axis xc.  Beyond the profiled grid the
    Delta chi2 is continued LINEARLY from the boundary slope (a monotone, conservative extrapolation), so
    the density tapers smoothly to zero instead of dropping off a cliff at the grid edge."""
    spl = CubicSpline(x_row, prof)
    lo, hi = x_row.min(), x_row.max()
    slo = float(spl(lo, 1)); shi = float(spl(hi, 1))                # boundary slopes of the profile
    pc = np.empty_like(xc)
    inside = (xc >= lo) & (xc <= hi)
    pc[inside] = spl(xc[inside])
    pc[xc < lo] = spl(lo) + slo * (xc[xc < lo] - lo)               # slope is <=0 here -> profile rises left
    pc[xc > hi] = spl(hi) + shi * (xc[xc > hi] - hi)               # slope is >=0 here -> profile rises right
    return np.exp(-0.5 * np.maximum(pc, 0.0))


def main(label="sec4_closure_r16_noprior"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_profile.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    grid = np.asarray(z["grid_sigma"]); prof = np.asarray(z["prof_dobj"])
    bfp = np.asarray(z["bfp"]); spost = np.asarray(z["sigma_post"]); truth = np.asarray(z["truth"])
    nom = np.asarray(theta_nominal(nominal_knobs())); prior = np.asarray(PRIOR)

    order = sorted(range(len(sub)), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    n = len(order)

    # per-row x in prior-sigma units, then a common axis spanning them all
    xrows, dens_e, dens_g, tx, gid = [], [], [], [], []
    for c in order:
        k = sub[c]; p = max(prior[k], 1e-12)
        xrows.append((bfp[k] + grid * spost[c] - nom[k]) / p)
    xlo = min(xr.min() for xr in xrows) - 0.15
    xhi = max(xr.max() for xr in xrows) + 0.15
    xc = np.linspace(xlo, xhi, NX)
    for i, c in enumerate(order):
        k = sub[c]; p = max(prior[k], 1e-12)
        pe = _density(grid, prof[c], xrows[i], xc)                # exact posterior, unnormalised
        pe = pe / np.trapezoid(pe, xc)                            # -> unit-area probability density
        xbf = (bfp[k] - nom[k]) / p; sg = spost[c] / p
        pg = np.exp(-0.5 * ((xc - xbf) / sg) ** 2); pg = pg / np.trapezoid(pg, xc)   # Gaussian density
        m = max(pe.max(), pg.max(), 1e-300)                       # PER-KNOB norm: taller of the two sets it
        dens_e.append(pe / m); dens_g.append(pg / m)              # so the sharper curve stands taller in-row
        tx.append((truth[k] - nom[k]) / p); gid.append(style.knob_group(pn[sub[c]]))

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(9.8, 8.8))
        baselines = []
        for i in range(n):
            y0 = n - 1 - i                                        # row 0 (first dial) on top
            baselines.append(y0)
            ax.axhline(y0, color="0.85", lw=0.6, zorder=y0 * 3)   # baseline separator
            # Gaussian (orange, faint fill + line) then exact (blue fill) -- two colours, method-coded
            ax.fill_between(xc, y0, y0 + ROWSCALE * dens_g[i], color=C_GAUSS, alpha=0.10, zorder=y0 * 3 + 1)
            ax.plot(xc, y0 + ROWSCALE * dens_g[i], color=C_GAUSS, lw=1.2, zorder=y0 * 3 + 2)
            ax.fill_between(xc, y0, y0 + ROWSCALE * dens_e[i], color=C_EXACT, alpha=0.38,
                            lw=1.0, edgecolor=style.darker(C_EXACT), zorder=y0 * 3 + 3)
            # injected-truth tick (black vertical line up to the exact density there)
            ht = ROWSCALE * float(np.interp(tx[i], xc, dens_e[i]))
            ax.plot([tx[i], tx[i]], [y0, y0 + max(ht, 0.40)], color="k", lw=1.1, zorder=y0 * 3 + 3)

        # physics-group colour tabs + labels down the left edge
        for g in sorted(set(gid)):
            rows = [n - 1 - i for i in range(n) if gid[i] == g]
            lo, hi = min(rows) - 0.35, max(rows) + 0.75
            ax.add_patch(plt.Rectangle((xlo - 0.30, lo), 0.06, hi - lo, transform=ax.transData,
                                       facecolor=style.KNOB_GROUP_COLOR[g], edgecolor="none",
                                       clip_on=False, zorder=6))
            ax.text(xlo - 0.46, (lo + hi) / 2, style.KNOB_GROUP_NAME[g], ha="center", va="center",
                    rotation=90, fontsize=8, color=style.darker(style.KNOB_GROUP_COLOR[g]))

        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(xlo - 0.05, xhi + 0.05); ax.set_ylim(-0.5, n - 1 + ROWSCALE + 0.35)
        # dial labels as right-edge tick labels (clear of the densities)
        ax.set_yticks(baselines); ax.set_yticklabels([style.plab(pn[sub[c]]) for c in order], fontsize=9)
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(r"value $-$ nominal   (prior $\sigma$ units)", fontsize=10)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        # legend above the plot (horizontal), clear of the densities
        ax.fill_between([], [], color=C_EXACT, alpha=0.38, label="exact posterior")
        ax.plot([], [], color=C_GAUSS, lw=1.4, label=r"Gaussian $\mathcal{N}(\hat\theta,\sigma_{\rm post})$")
        ax.plot([], [], color="k", lw=1.1, label="injected truth")
        ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
                  frameon=False, columnspacing=1.6, handlelength=1.6)
        ax.set_title("Sec 4.1  per-dial posterior: exact (blue) vs Gaussian (orange)",
                     fontsize=11.5, loc="left", pad=22)
        fig.tight_layout()
        style.save(fig, "sec4_fig41_recovery_ridge")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
