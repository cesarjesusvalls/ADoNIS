"""Section 4 profile figure: exact nonlinear likelihood vs the Gauss-Newton parabola, per dial.

Reads <label>_profile.npz (analysis.paper.physfit.multisample_profile).  For each fitted dial, the EXACT
profiled Delta(objective) -- chi2_data + prior, re-minimised over the other 15 dials at each pinned value --
is overlaid on the quadratic the fit assumes, Delta = (theta_k - BFP_k)^2 / sigma_post,k^2 = (grid)^2.

Agreement over +/-3 sigma == the marginal error the closure quotes IS the likelihood (locally Gaussian);
a dial whose exact curve sits ABOVE the parabola is over-covered (true error smaller), BELOW is under-
covered, and asymmetry flags a skewed / boundary-limited dial (Eb_shift is the usual suspect).

Usage:  python -m analysis.paper.sec4_closure.profile_fig [label]     (default sec4_closure_random16)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_EXACT, C_PAR = "#1f4b9c", style.FIT_EC


def main(label="sec4_closure_random16"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_profile.npz", allow_pickle=True)
    subset = [int(k) for k in z["subset"]]; pnames = [str(x) for x in z["pnames"]]
    grid = np.asarray(z["grid_sigma"]); prof = np.asarray(z["prof_dobj"])
    order = sorted(range(len(subset)), key=lambda c: (style.knob_group(pnames[subset[c]]), subset[c]))
    par = grid**2                                          # the GN parabola in sigma_post units
    xx = np.linspace(grid.min(), grid.max(), 200)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(4, 4, figsize=(12.5, 10.0), sharex=True, sharey=True)
        for ax, c in zip(axes.ravel(), order):
            k = subset[c]; gcol = style.KNOB_GROUP_COLOR[style.knob_group(pnames[k])]
            ax.plot(xx, xx**2, "-", color=C_PAR, lw=1.6, zorder=2, label="GN parabola")
            ax.plot(grid, prof[c], "o", color=C_EXACT, ms=4, zorder=3, label="exact profile")
            ax.axhline(1, color="0.75", lw=0.7, ls=":")     # 1 sigma  (Delta = 1)
            ax.axhline(4, color="0.85", lw=0.7, ls=":")     # 2 sigma  (Delta = 4)
            ax.set_ylim(-0.3, 10.5); ax.set_xlim(grid.min() - 0.2, grid.max() + 0.2)
            ax.text(0.5, 0.93, style.plab(pnames[k]), transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, color=gcol, fontweight="bold")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$(\theta-\hat\theta)/\sigma_{\rm post}$", fontsize=9)
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\Delta$ objective", fontsize=9)
        axes[0, 0].legend(fontsize=8, loc="upper center", framealpha=0.9)
        fig.suptitle(f"S4 quadratic-error validation — exact profile likelihood vs Gauss-Newton parabola "
                     f"({label})", fontsize=11.5)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        style.save(fig, f"{label}_profile")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
