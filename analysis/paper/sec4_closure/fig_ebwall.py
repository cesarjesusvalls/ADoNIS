"""Sec 4 non-Gaussian headline: E_b at its physical wall -- Gaussian sigma vs profile.

Reads output/altgen/sec4_ebwall.npz (a fine 1-D profile of E_b with a small injected value).  E_b >= 0 is
physical (the model flattens below 0), so the profile likelihood is one-sided:
  * the GAUSSIAN sigma gives a symmetric interval that leaks into UNPHYSICAL E_b < 0,
  * the PROFILE (Delta chi2 = 1) interval stops at the wall.
The single panel makes the case that symmetric errors mislead near a boundary and the likelihood-ratio
interval is the correct one -- the anchor for the Sec-4 non-Gaussian discussion.

Usage:  python -m analysis.paper.sec4_closure.fig_ebwall
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_PROF, C_GAUSS = "#1f4b9c", "#c44"


def _cross(x, y, level, side, x0):
    """x where y crosses `level`, on the given side of x0 (nearest); NaN if not bracketed."""
    m = x < x0 if side < 0 else x > x0
    xs, ys = x[m], y[m] - level
    s = np.where(np.diff(np.signbit(ys)))[0]
    if len(s) == 0:
        return np.nan
    i = s[-1] if side < 0 else s[0]
    return xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])


def main():
    style.use()
    z = np.load(style.ALTGEN / "sec4_ebwall.npz", allow_pickle=True)
    eb = np.asarray(z["eb_grid"]); d = np.asarray(z["dchi2"])
    bfp = float(z["eb_bfp"]); sig = float(z["eb_sig"])

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.axvspan(eb.min() - 0.2, 0.0, color="0.9", zorder=0)                       # unphysical E_b<0
        ax.text(eb.min() + 0.05, 8.4, "unphysical\n$E_b<0$", fontsize=8.5, color="0.45", va="top")
        ax.axvline(0, color="0.5", lw=1.0)

        xx = np.linspace(eb.min(), eb.max(), 400)
        ax.plot(xx, ((xx - bfp) / sig)**2, "--", color=C_GAUSS, lw=1.6, label="Gaussian $\\sigma$")
        ax.plot(eb, d, "o-", color=C_PROF, ms=3.5, lw=1.5, label="profile ($\\Delta\\chi^2$)")
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.text(eb.max(), 1.06, r"$\Delta\chi^2=1$", fontsize=8, color="0.4", ha="right", va="bottom")

        # intervals at Delta=1
        g_lo, g_hi = bfp - sig, bfp + sig                                            # Gaussian (symmetric)
        p_lo = _cross(eb, d, 1.0, -1, bfp); p_hi = _cross(eb, d, 1.0, +1, bfp)       # profile crossings
        ax.plot([g_lo, g_hi], [-0.45, -0.45], color=C_GAUSS, lw=2.4, solid_capstyle="butt")
        ax.plot([max(p_lo, 0.0) if np.isnan(p_lo) else p_lo, p_hi], [-0.9, -0.9], color=C_PROF, lw=2.4,
                solid_capstyle="butt")
        ax.plot(bfp, -0.45, "|", color=C_GAUSS, ms=9); ax.plot(bfp, -0.9, "|", color=C_PROF, ms=9)
        ax.annotate("leaks below 0", (g_lo, -0.45), (g_lo - 0.05, -0.45), fontsize=7.5, color=C_GAUSS,
                    ha="right", va="center")

        ax.set_xlim(eb.min() - 0.2, eb.max()); ax.set_ylim(-1.2, 9)
        ax.set_xlabel(r"$E_b$ shift  [MeV]", fontsize=10); ax.set_ylabel(r"$\Delta\chi^2$", fontsize=10)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
        ax.set_title(r"$E_b$ at the physical wall: Gaussian $\sigma$ vs profile", fontsize=11.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        style.save(fig, "sec4_ebwall")
    print(f"  Gaussian: [{g_lo:.2f}, {g_hi:.2f}] MeV  (lower {'UNPHYSICAL' if g_lo<0 else 'ok'})")
    print(f"  profile : [{p_lo:.2f}, {p_hi:.2f}] MeV")


if __name__ == "__main__":
    main()
