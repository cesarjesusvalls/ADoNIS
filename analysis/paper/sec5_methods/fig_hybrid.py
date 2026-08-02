"""Sec 5.4 -- hybrid grid-bad x Taylor-good: exact vs pure-Taylor vs hybrid, at the E_b wall.

Reads sec5_hybrid.npz.  For each good dial X, the (E_b, X) 68/95% contour three ways:
  * exact       -- the profiled grid (ground truth; walled).
  * pure Taylor -- the 2nd-order-model chi2 profiled over the nuisances (from J + d2m).  Analytic ->
                   it CANNOT see the E_b>=0 wall, so it extends symmetrically into unphysical E_b<0.
  * hybrid      -- keep Taylor for the X (good) direction but swap the smooth Taylor E_b backbone for the
                   EXACT 1-D E_b profile:  chi2_hybrid = [chi2_Taylor - min_X chi2_Taylor] + eb_1d(E_b).
                   Exact in the one bad dial, Taylor in the good one -> matches exact, at grid-of-1 cost.

Usage:  python -m analysis.paper.sec5_methods.fig_hybrid
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.sec4_closure.corner_taylor import profile_pair

LEV = [2.30, 6.17]                                                # 2-D 68% / 95%
C_EXACT, C_TAYLOR, C_HYBRID = "k", style.FIT_EC, "#1f4b9c"


def main():
    style.use()
    z = np.load(style.ALTGEN / "sec5_hybrid.npz", allow_pickle=True)
    sub = [int(k) for k in z["subset"]]; pn = [str(x) for x in z["pnames"]]
    Jb = np.asarray(z["Jb"]); Bb = np.asarray(z["Bb"]); W = np.asarray(z["W"]); bfp = np.asarray(z["bfp"])
    eb_pos = int(z["eb_pos"]); eb_k = sub[eb_pos]
    eb_ax = np.asarray(z["eb_axis"]); eb_1d = np.asarray(z["eb_1d"]); xnames = [str(x) for x in z["xnames"]]

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(1, len(xnames), figsize=(6.0 * len(xnames), 5.2), squeeze=False)
        for ax, xn in zip(axes[0], xnames):
            xk = pn.index(xn); xpos = sub.index(xk)
            xa = np.asarray(z[f"xax_{xn}"]); exact = np.asarray(z[f"grid_{xn}"])
            taylor = profile_pair(Jb, Bb, W, eb_pos, xpos, eb_ax - bfp[eb_k], xa - bfp[xk])
            hybrid = (taylor - taylor.min(axis=1, keepdims=True)) + eb_1d[:, None]
            EB, X = np.meshgrid(eb_ax, xa, indexing="ij")
            ax.axvspan(eb_ax.min(), 0.0, color="0.92", zorder=0)
            ax.contour(EB, X, exact, levels=LEV, colors=C_EXACT, linewidths=1.5)
            ax.contour(EB, X, taylor, levels=LEV, colors=C_TAYLOR, linewidths=1.3, linestyles="--")
            ax.contour(EB, X, hybrid, levels=LEV, colors=C_HYBRID, linewidths=1.3, linestyles=":")
            ax.axvline(0, color="0.5", lw=0.8)
            ax.set_xlabel(r"$E_b$ shift [MeV]", fontsize=10); ax.set_ylabel(style.plab(xn), fontsize=11)
            ax.set_title(f"$E_b$ (bad) $\\times$ {style.plab(xn)} (good)", fontsize=10.5, loc="left")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0, 0].plot([], [], C_EXACT, lw=1.5, label="exact")
        axes[0, 0].plot([], [], C_TAYLOR, lw=1.3, ls="--", label="pure Taylor (misses wall)")
        axes[0, 0].plot([], [], C_HYBRID, lw=1.3, ls=":", label="hybrid (grid $E_b$ $\\times$ Taylor)")
        axes[0, 0].legend(fontsize=8.5, loc="upper right", framealpha=0.95)
        fig.suptitle("Sec 5.4  hybrid: grid the bad dial ($E_b$ wall), Taylor the good one",
                     fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        style.save(fig, "sec5_fig_hybrid")


if __name__ == "__main__":
    main()
