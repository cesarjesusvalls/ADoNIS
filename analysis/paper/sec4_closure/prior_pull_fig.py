"""Section 4 prior-pull figure: the fit is unbiased; BFP != truth is purely the Gate-I prior.

Overlays two fits of the SAME closure on the same 16 dials:
  * MLE  (prior OFF, sec4_closure_r16_noprior) -- the best-fit lands EXACTLY on the injected truth
    (chi2_data = 0), so the differentiable multisample fit is bias-free.
  * MAP  (Gate-I prior ON, sec4_closure_random16) -- the best-fit is pulled toward nominal by the prior,
    with its (marginalised) error bar; the pull size tracks the dial's Gate-I shrinkage.

The gap between the orange MAP point and the star/diamond IS the prior pull; it is largest for the weakly-
constrained dials (shrinkage near 0.5) and ~0 for the well-measured ones (shrinkage << 0.5).

Usage:  python -m analysis.paper.sec4_closure.prior_pull_fig
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.physical_fit import theta_nominal
from adonis.reweight.reweight_model import nominal_knobs

C_MLE, C_MAP = "#1f4b9c", style.FIT_EC


def main(mle="sec4_closure_r16_noprior", mapl="sec4_closure_random16", gate="multisample_carbon"):
    style.use()
    zm = np.load(style.ALTGEN / f"{mle}.npz", allow_pickle=True)
    za = np.load(style.ALTGEN / f"{mapl}.npz", allow_pickle=True)
    zg = np.load(style.ALTGEN / f"{gate}.npz", allow_pickle=True)
    pnames = [str(x) for x in za["pnames"]]
    subset = [int(k) for k in za["subset"]]
    truth = np.asarray(za["truth"]); prior = np.asarray(za["prior"]); shrink = np.asarray(zg["shrink"])
    nom = np.asarray(theta_nominal(nominal_knobs()))
    th_mle = np.asarray(zm["fit_th"]); th_map = np.asarray(za["fit_th"])
    sig = np.sqrt(np.abs(np.diag(np.asarray(za["fit_V"]))))

    order = sorted(range(len(subset)), key=lambda c: (style.knob_group(pnames[subset[c]]), subset[c]))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(11.0, 7.4))
        yy = np.arange(len(order))
        gid = [style.knob_group(pnames[subset[c]]) for c in order]
        for i, c in enumerate(order):
            k = subset[c]; p = max(prior[k], 1e-12)
            xt = (truth[k] - nom[k]) / p; xa = (th_map[k] - nom[k]) / p; xe = sig[c] / p
            ax.plot([xt, xa], [yy[i], yy[i]], color="0.7", lw=1.0, zorder=1)          # the pull connector
            ax.plot(xt, yy[i], marker="*", ms=14, color="k", zorder=5, ls="none")     # injected truth
            ax.plot((th_mle[k] - nom[k]) / p, yy[i], marker="D", ms=6, color=C_MLE,
                    mec="white", mew=0.6, zorder=6, ls="none")                         # MLE (no prior) -> truth
            ax.errorbar(xa, yy[i], xerr=xe, fmt="o", color=C_MAP, ms=6, capsize=3, lw=1.4, zorder=4)  # MAP
            ax.text(1.02, yy[i], f"{shrink[k]:.2f}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color="0.4")
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.text(1.02, -1.0, "shrink", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                color="0.4", fontweight="bold")
        ax.set_yticks(yy); ax.set_yticklabels([style.plab(pnames[subset[c]]) for c in order], fontsize=9)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel(r"value $-$ nominal  (prior $\sigma$ units)", fontsize=10)
        style.knob_group_tabs(ax, gid, tabx=-0.20, tabw=0.02, labx=-0.25, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.plot([], [], "*", color="k", ms=12, label="injected truth")
        ax.plot([], [], "D", color=C_MLE, mec="white", mew=0.6, ms=7, label="MLE (no prior) — lands on truth")
        ax.plot([], [], "o", color=C_MAP, ms=7, label=r"MAP (Gate-I prior) $\pm\sigma$ — prior-pulled")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
        ax.set_title("S4: the fit is unbiased — BFP $\\ne$ truth is the prior pull "
                     "(gap $\\propto$ shrinkage)", fontsize=11.5, loc="left")
        fig.tight_layout()
        style.save(fig, "sec4_prior_pull")


if __name__ == "__main__":
    main()
