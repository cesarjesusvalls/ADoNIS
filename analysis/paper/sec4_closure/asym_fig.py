"""Section 4 asymmetric-error figure: MINOS-style intervals from the EXACT profile, vs the injected truth.

The closure quotes a symmetric Gauss-Newton sigma.  This instead reads the exact profiled objective
(<label>_profile.npz) and, per dial, extracts WITHOUT symmetrising:
  * theta_hat  -- the true profile minimum (may sit slightly off the GN BFP if the fit under-shot),
  * sigma_lo / sigma_hi -- the Delta(objective)=1 crossings on each side (asymmetric 68% interval).
Then it compares the injected truth theta* to that interval: the asymmetric pull is (theta* - theta_hat)
divided by sigma_hi (if theta* above the min) or sigma_lo (below).  The point of the closure is that the
truth we injected lands inside the honestly-shaped interval -- so this is the un-symmetrised recovery test.

  (a) recovery: theta_hat with asymmetric error bars (blue) vs the GN symmetric sigma (grey) vs truth (*).
  (b) asymmetric pull (theta* - theta_hat)/sigma_+/- , band +/-1.

Usage:  python -m analysis.paper.sec4_closure.asym_fig [label]     (default sec4_closure_random16)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.physical_fit import PRIOR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs

C_ASY, C_SYM = "#1f4b9c", "0.6"


def profile_interval(grid, y):
    """Refined minimum + Delta=1 crossings of a profiled Delta-objective curve (all in sigma_post units).
    Returns (x_min, sig_lo, sig_hi) with sig_lo/hi the one-sided 68% half-widths (NaN if not bracketed)."""
    spl = CubicSpline(grid, y)
    xd = np.linspace(grid.min(), grid.max(), 4001); yd = spl(xd)
    imin = int(np.argmin(yd)); xmin = xd[imin]; ymin = yd[imin]
    d = yd - ymin - 1.0                                        # zero-crossings are the +/-1sigma points

    def cross(side):
        sel = xd < xmin if side < 0 else xd > xmin
        xs, ds = xd[sel], d[sel]
        s = np.where(np.diff(np.signbit(ds)))[0]              # sign changes of (yd-ymin-1)
        if len(s) == 0:
            return np.nan
        i = s[-1] if side < 0 else s[0]                       # crossing nearest the minimum
        x0, x1, y0, y1 = xs[i], xs[i + 1], ds[i], ds[i + 1]
        xc = x0 - y0 * (x1 - x0) / (y1 - y0)
        return abs(xc - xmin)
    return xmin, cross(-1), cross(+1)


def main(label="sec4_closure_random16"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_profile.npz", allow_pickle=True)
    subset = [int(k) for k in z["subset"]]; pnames = [str(x) for x in z["pnames"]]
    grid = np.asarray(z["grid_sigma"]); prof = np.asarray(z["prof_dobj"])
    bfp = np.asarray(z["bfp"]); spost = np.asarray(z["sigma_post"]); truth = np.asarray(z["truth"])
    nom = np.asarray(theta_nominal(nominal_knobs())); prior = np.asarray(PRIOR)

    order = sorted(range(len(subset)), key=lambda c: (style.knob_group(pnames[subset[c]]), subset[c]))
    rows = []
    for c in order:
        k = subset[c]
        xmin, slo, shi = profile_interval(grid, prof[c])
        th_hat = bfp[k] + xmin * spost[c]                     # refined (un-symmetrised) minimum, native
        elo, ehi = slo * spost[c], shi * spost[c]            # asymmetric errors, native units
        d = truth[k] - th_hat
        apull = d / ehi if d >= 0 else d / elo               # asymmetric pull
        rows.append(dict(k=k, name=pnames[k], g=style.knob_group(pnames[k]), th_hat=th_hat, bfp=bfp[k],
                         elo=elo, ehi=ehi, sym=spost[c], truth=truth[k], nom=nom[k], pr=prior[k],
                         xmin=xmin, apull=apull, spull=(truth[k] - bfp[k]) / spost[c]))

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(13.0, 7.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.28,
                              left=0.15, right=0.975, top=0.9, bottom=0.09)
        yy = np.arange(len(rows))
        labels = [style.plab(r["name"]) for r in rows]
        gid = [r["g"] for r in rows]

        # (a) recovery: asymmetric profile interval + symmetric GN sigma + injected truth ---------- #
        ax = fig.add_subplot(gs[0, 0])
        for i, r in enumerate(rows):
            p = max(r["pr"], 1e-12)
            xhat = (r["th_hat"] - r["nom"]) / p                            # profile-minimum center
            xbfp = (r["bfp"] - r["nom"]) / p                               # GN best-fit center (DIFFERENT)
            # asymmetric interval about the PROFILE MINIMUM (blue)
            ax.errorbar([xhat], [yy[i]], xerr=[[r["elo"] / p], [r["ehi"] / p]], fmt="none",
                        ecolor=C_ASY, capsize=3, lw=1.5, zorder=4)
            ax.plot(xhat, yy[i], marker="D", ms=6, color=C_ASY, mec="white", mew=0.6, zorder=6, ls="none")
            # symmetric GN sigma about the GN BEST-FIT (grey, offset up) -- its own, possibly different, centre
            ax.errorbar([xbfp], [yy[i] + 0.26], xerr=[[r["sym"] / p], [r["sym"] / p]], fmt="none",
                        ecolor=C_SYM, capsize=2.5, lw=1.3, zorder=3)
            ax.plot(xbfp, yy[i] + 0.26, marker="s", ms=4.5, color=C_SYM, mec="white", mew=0.5, zorder=3,
                    ls="none")
            ax.plot((r["truth"] - r["nom"]) / p, yy[i], marker="*", ms=13, color="k", zorder=5, ls="none")
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9); ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlabel(r"value $-$ nominal  (prior $\sigma$ units)", fontsize=9.5)
        style.knob_group_tabs(ax, gid, tabx=-0.16, tabw=0.016, labx=-0.205, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.plot([], [], "*", color="k", ms=11, label="injected truth")
        ax.plot([], [], "D", color=C_ASY, mec="white", mew=0.6, ms=6,
                label=r"profile min $\hat\theta$ $\pm\,\sigma_{+/-}$ ($\Delta=1$)")
        ax.plot([], [], "s", color=C_SYM, mec="white", mew=0.5, ms=5, label=r"GN best-fit $\pm\,\sigma_{\rm sym}$")
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
        ax.set_title("(a)  asymmetric recovery of the 16 dials", fontsize=10.5, loc="left")

        # (b) asymmetric pull ----------------------------------------------------------------------- #
        ax = fig.add_subplot(gs[0, 1])
        ax.axvspan(-1, 1, color="0.9", zorder=0); ax.axvline(0, color="k", lw=0.9)
        for i, r in enumerate(rows):
            gcol = style.KNOB_GROUP_COLOR[r["g"]]
            ax.plot(r["apull"], yy[i], "o", color=gcol, ms=5.5, zorder=3)
            ax.plot(r["spull"], yy[i], "x", color="0.6", ms=5, zorder=2)   # symmetric pull for reference
        ax.set_yticks(yy); ax.set_yticklabels([]); ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(-3, 3); ax.set_xlabel(r"pull to injected truth", fontsize=9.5)
        ax.plot([], [], "o", color="0.3", ms=6, label=r"asymmetric $(\theta^\star-\hat\theta)/\sigma_\pm$")
        ax.plot([], [], "x", color="0.6", ms=6, label="symmetric (GN)")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.set_title("(b)  pull (band $\\pm1$)", fontsize=10.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        inside = sum(abs(r["apull"]) <= 1 for r in rows)
        fig.suptitle(f"S4 asymmetric closure — profile-likelihood intervals vs injected truth "
                     f"({inside}/{len(rows)} within $1\\sigma$)   ·   {label}", fontsize=11, x=0.15, ha="left")
        style.save(fig, f"{label}_asym")

    # short text report
    print(f"{'dial':>16} {'th_hat':>8} {'-sig':>7} {'+sig':>7} {'GNsig':>7} {'truth':>8} "
          f"{'min_off':>7} {'apull':>6}")
    for r in rows:
        print(f"{r['name']:>16} {r['th_hat']:8.3f} {r['elo']:7.3f} {r['ehi']:7.3f} {r['sym']:7.3f} "
              f"{r['truth']:8.3f} {r['xmin']:7.2f} {r['apull']:6.2f}")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
