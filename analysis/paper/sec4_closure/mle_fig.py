"""Section 4 MLE closure figure: unregularised best-fit (no prior) with its data-only errors, vs truth.

MLE = Maximum Likelihood Estimate: the fit with the Gate-I prior switched OFF (pure chi2_data).  On a
noiseless closure it lands EXACTLY on the injected truth (chi2_data = 0), so this figure is the clean
statement that the differentiable multisample fit is unbiased.  The error bar is the DATA-ONLY covariance
sqrt(diag((J^T W J)^-1)) -- what the 20 samples alone determine about each dial, with no prior tightening.

Reads the no-prior run (S4_PRIOR_SCALE=0) npz.  The saved `prior` there is the scaled one, so the axis /
error normalisation uses the canonical physical_fit.PRIOR instead.

Usage:  python -m analysis.paper.sec4_closure.mle_fig [label]     (default sec4_closure_r16_noprior)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.physical_fit import PRIOR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs

C_MLE = "#1f4b9c"


def main(label="sec4_closure_r16_noprior"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    pnames = [str(x) for x in z["pnames"]]
    subset = [int(k) for k in z["subset"]]
    truth = np.asarray(z["truth"]); nom = np.asarray(theta_nominal(nominal_knobs()))
    prior = np.asarray(PRIOR)                                    # canonical prior sigma (NOT the scaled one)
    th = np.asarray(z["fit_th"]); sig = np.sqrt(np.abs(np.diag(np.asarray(z["fit_V"]))))

    order = sorted(range(len(subset)), key=lambda c: (style.knob_group(pnames[subset[c]]), subset[c]))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        yy = np.arange(len(order))
        gid = [style.knob_group(pnames[subset[c]]) for c in order]
        for i, c in enumerate(order):
            k = subset[c]; p = max(prior[k], 1e-12)
            ax.plot((truth[k] - nom[k]) / p, yy[i], marker="*", ms=14, color="k", zorder=3, ls="none")
            ax.errorbar((th[k] - nom[k]) / p, yy[i], xerr=sig[c] / p, fmt="D", color=C_MLE, ms=6,
                        mec="white", mew=0.6, capsize=3.5, lw=1.6, zorder=4)
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_yticks(yy); ax.set_yticklabels([style.plab(pnames[subset[c]]) for c in order], fontsize=9)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlabel(r"value $-$ nominal  (prior $\sigma$ units)", fontsize=10)
        style.knob_group_tabs(ax, gid, tabx=-0.185, tabw=0.02, labx=-0.235, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.plot([], [], "*", color="k", ms=12, label="injected truth")
        ax.plot([], [], "D", color=C_MLE, mec="white", mew=0.6, ms=7,
                label=r"MLE best-fit $\pm\sigma$ (data-only, no prior)")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
        cd = float(z["fit_chi2data"])
        ax.set_title(f"S4 MLE closure — unregularised fit recovers the injected truth "
                     f"($\\chi^2_{{\\rm data}}$ = {cd:.2g})", fontsize=11.5, loc="left")
        fig.tight_layout()
        style.save(fig, "sec4_mle_closure")


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(pos[:1] or []))
