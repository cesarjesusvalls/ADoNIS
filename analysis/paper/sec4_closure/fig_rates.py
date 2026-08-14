"""FIGURE D -- the binned event rate in every sample, before and after the closure fit.

Three things per panel, all from the reference closure npz:

  data        the Asimov data, m(theta_true), with the fit's own per-bin sigma (5% syst; bins whose MC
              error exceeded 5% of the central value carry sigma=inf and are dropped from the fit -- they
              are drawn hollow here so the masking is visible rather than implied)
  pre-fit     m(theta_nominal), the prediction before any fitting
  post-fit    m(theta_hat), the prediction the fit arrives at

The point of the figure: the injected truth is a long way from nominal (37 sigma across 17 dials at the
P1 point), so the pre-fit curve misses the data visibly in the samples that constrain those dials, and
the post-fit curve lands on it.  chi2 per sample is quoted before and after, over LIVE bins only.

This is an Asimov closure, so "post-fit lands on the data" is exact by construction -- the residual is
~1e-27.  That is the statement being made: the model can reproduce its own truth from a blind start.  It
is NOT a goodness-of-fit test, which needs the toy ensemble (figure A, panel b).

Usage:  python -m analysis.paper.sec4_closure.fig_rates [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_PRE, C_POST, C_DATA = "#c8842a", "#1f4b9c", "0.15"


def main(label="sec4_P1"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    data = np.asarray(z["data"]); sig = np.asarray(z["sigma"])
    pre = np.asarray(z["model_nom"]); post = np.asarray(z["fit_model"])
    row0 = np.asarray(z["row0"]); dsk = [str(x) for x in z["dskeys"]]
    live = np.isfinite(sig) & (sig > 0)
    print(f"{len(dsk)} samples, {int(live.sum())}/{len(sig)} live bins")

    # 3 x 7 fits the 21 samples exactly; the old 4-wide grid left three empty slots.
    nd = len(dsk); nc = 3; nr = int(np.ceil(nd / nc))
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans"}):
        fig, ax = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.05 * nr))
        for i, k in enumerate(dsk):
            A = ax.flat[i]; a, b = row0[i], row0[i + 1]
            m = live[a:b]
            # bin centres from the stored edges when present, else the bin index -- never silently
            # renumber, since several samples share an observable name across experiments
            ek = f"{k}_edges"
            if ek in z.files:
                e = np.asarray(z[ek], float); x = 0.5 * (e[1:] + e[:-1]); w = np.diff(e)
            else:
                x = np.arange(b - a) + 0.5; w = np.ones(b - a)
            d_, p_, q_, s_ = data[a:b], pre[a:b], post[a:b], sig[a:b]
            A.step(x, p_, where="mid", color=C_PRE, lw=1.5, label="pre-fit (nominal)")
            A.step(x, q_, where="mid", color=C_POST, lw=1.5, ls="--", label="post-fit")
            A.errorbar(x[m], d_[m], yerr=s_[m], fmt="o", ms=2.8, lw=0, elinewidth=0.9,
                       color=C_DATA, label="Asimov data", zorder=5)
            if (~m).any():          # masked bins drawn hollow: the cut is visible, not implied
                A.plot(x[~m], d_[~m], "o", ms=2.8, mfc="none", mec="0.6", mew=0.7, zorder=4)
            c_pre = float(np.sum(((p_[m] - d_[m]) / s_[m]) ** 2))
            c_post = float(np.sum(((q_[m] - d_[m]) / s_[m]) ** 2))
            A.set_title(f"{k}\n$\\chi^2$ {c_pre:.0f} $\\to$ {c_post:.2g}   ({int(m.sum())} bins)",
                        fontsize=7.5)
            A.tick_params(labelsize=6.5, top=False, right=False)
            A.set_ylim(bottom=0)
            if i == 0:
                A.legend(fontsize=6, frameon=False, loc="best")
            print(f"  {k:>26} chi2 {c_pre:10.1f} -> {c_post:9.2e}   live {int(m.sum()):3d}/{b-a}")
        for j in range(nd, nr * nc):
            ax.flat[j].axis("off")
        fig.supylabel("rate per bin  (sample units)", fontsize=9)
        # explicit padding: the per-panel titles carry two lines (name + chi2), and at the
        # default pad they collided with the axis above.
        fig.tight_layout(pad=0.9, h_pad=1.5, w_pad=1.1)
        style.save(fig, f"{label}_figD")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:1] or []))
