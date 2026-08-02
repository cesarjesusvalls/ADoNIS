"""Sec 4.3b -- coverage at the E_b wall: Gaussian interval UNDER-covers, profile interval holds.

Reads output/altgen/sec4_ebcov.npz (per-toy Gaussian [th_hat +/- sigma] and profile [Delta chi2 = 1]
intervals for E_b, injected near its wall).  Left: the per-toy intervals against the near-wall truth and
the E_b=0 wall -- the Gaussian bars slide below 0 and miss the truth low; the profile bars stop at the wall.
Right: empirical 68% coverage, Gaussian vs profile, against the nominal 0.68.

Usage:  python -m analysis.paper.sec4_closure.fig_ebcov
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_GAUSS, C_PROF = "#c44", "#1f4b9c"


def main():
    style.use()
    z = np.load(style.ALTGEN / "sec4_ebcov.npz", allow_pickle=True)
    star = float(z["eb_star"]); hat = np.asarray(z["eb_hat"]); sig = np.asarray(z["eb_sig"])
    plo = np.asarray(z["prof_lo"]); phi = np.asarray(z["prof_hi"])
    gcov = np.asarray(z["gauss_cover"]).astype(bool); pcov = np.asarray(z["prof_cover"]).astype(bool)
    o = np.argsort(hat); n = len(hat)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.0, 5.6), gridspec_kw=dict(width_ratios=[2.1, 1]))
        xmin = min((hat - sig).min(), -0.5)
        ax.axvspan(xmin - 0.1, 0.0, color="0.9", zorder=0)                       # unphysical Eb<0
        ax.axvline(0, color="0.5", lw=1.0); ax.axvline(star, color="k", lw=1.2, ls="--")
        ax.text(star, n + 0.5, "truth", fontsize=8, ha="center", va="bottom")
        ax.text(xmin + 0.02, n - 1, "unphysical\n$E_b<0$", fontsize=8, color="0.45", va="top")
        for row, i in enumerate(o):
            ax.plot([hat[i] - sig[i], hat[i] + sig[i]], [row + 0.18] * 2,
                    color=C_GAUSS, lw=1.6, alpha=0.5 if gcov[i] else 1.0)
            ax.plot([plo[i], phi[i]], [row - 0.18] * 2, color=C_PROF, lw=1.6, alpha=0.5 if pcov[i] else 1.0)
            if not gcov[i]:
                ax.plot(hat[i], row + 0.18, "x", color=C_GAUSS, ms=4)
        ax.set_yticks([]); ax.set_ylim(-1, n); ax.set_xlim(xmin - 0.1, (hat + sig).max() + 0.1)
        ax.set_xlabel(r"$E_b$ shift  [MeV]", fontsize=10); ax.set_ylabel("toys (sorted by best-fit)", fontsize=9)
        ax.plot([], [], color=C_GAUSS, lw=1.6, label=r"Gaussian $[\hat\theta\pm\sigma]$")
        ax.plot([], [], color=C_PROF, lw=1.6, label=r"profile $[\Delta\chi^2{=}1]$")
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
        ax.set_title("(a)  per-toy 68% intervals at the wall", fontsize=10.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        cg, cp = gcov.mean(), pcov.mean()
        bx.bar([0, 1], [cg, cp], color=[C_GAUSS, C_PROF], width=0.6, alpha=0.85)
        bx.axhline(0.68, color="k", lw=1.2, ls=":"); bx.text(1.5, 0.685, "nominal 68%", fontsize=8, ha="right")
        for xi, cv in [(0, cg), (1, cp)]:
            bx.text(xi, cv + 0.02, f"{cv:.0%}", ha="center", fontsize=10, fontweight="bold")
        bx.set_xticks([0, 1]); bx.set_xticklabels(["Gaussian", "profile"], fontsize=9.5)
        bx.set_ylim(0, 1.05); bx.set_ylabel("empirical 68% coverage", fontsize=9.5)
        bx.set_title(f"(b)  coverage ({n} toys)", fontsize=10.5, loc="left")
        for sp in ("top", "right"):
            bx.spines[sp].set_visible(False)

        fig.suptitle(r"$E_b$ at the wall: Gaussian intervals go unphysical; the profile respects the boundary",
                     fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        style.save(fig, "sec4_fig43b_ebcov")
    print(f"  Gaussian coverage {gcov.mean():.0%} | profile coverage {pcov.mean():.0%}  ({n} toys)")


if __name__ == "__main__":
    main()
