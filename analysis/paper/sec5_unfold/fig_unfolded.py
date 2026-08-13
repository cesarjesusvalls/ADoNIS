"""FIGURE F -- the unfolded result, for a closure and for an injected signal excess.

(a) the templates themselves: c_j with its full uncertainty, against the injected truth.  This is the
    measurement -- everything else in the section is a way of reading it.
(b,c) the same result as a physical rate, c_j N_j projected onto delta-p_T and delta-alpha_T, against the
    generator truth.  Projections are taken in TRUTH space, which is why the truth grid had to be one the
    detector can support.

The two studies are shown together because the point is that they behave identically: an injected 20%
signal excess is recovered with the same errors as the closure, not with degraded ones.

Usage:  python -m analysis.paper.sec5_unfold.fig_unfolded [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

C_A, C_S = "#1f4b9c", "#e08214"
STUDIES = [("asimov", "closure ($c=1$)", C_A, "o"), ("sig120", r"signal $\times1.20$", C_S, "s")]


def _project(v, ndpt, ndat, axis):
    M = np.asarray(v).reshape(ndpt, ndat)
    return M.sum(axis=1) if axis == "dpt" else M.sum(axis=0)


def main(label="sec5"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    tdpt, tdat = np.asarray(z["true_dpt"]), np.asarray(z["true_dat"])
    ndpt, ndat = len(tdpt) - 1, len(tdat) - 1
    N = np.asarray(z["n_true"])
    nt = ndpt * ndat

    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(nt)
    for st, lab, col, mk in STUDIES:
        c, e, ct = z[f"{st}_c"], z[f"{st}_c_err"], z[f"{st}_c_true"]
        off = -0.13 if st == "asimov" else 0.13
        ax.errorbar(x + off, c, yerr=e, fmt=mk, ms=4, color=col, lw=1.2, capsize=2, label=lab)
        ax.axhline(ct[0], color=col, ls=":", lw=1.0)
    ax.set_xlabel("truth cell"); ax.set_ylabel(r"template $c_j$")
    ax.set_xticks(x); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("(a) unfolded templates", fontsize=9, loc="left")

    for k, (axis, edges, xlab, ttl) in enumerate(
            [("dpt", tdpt, r"$\delta p_T$ [MeV/c]", r"(b) $\delta p_T$"),
             ("dat", tdat, r"$\delta\alpha_T$ [rad]", r"(c) $\delta\alpha_T$")]):
        ax = fig.add_subplot(gs[0, 1 + k])
        e = np.asarray(edges, float).copy()
        if not np.isfinite(e[-1]):                       # open bin: draw it at twice the last width
            e[-1] = e[-2] + (e[-2] - e[-3])
        ctr, w = 0.5 * (e[:-1] + e[1:]), np.diff(e)
        for st, lab, col, mk in STUDIES:
            c, err, ct = z[f"{st}_c"], z[f"{st}_c_err"], z[f"{st}_c_true"]
            # EACH STUDY AGAINST ITS OWN TRUTH.  A single nominal curve would leave the injected study
            # sitting 20% above it and reading as a discrepancy, when it is an exact recovery.
            ax.step(np.r_[e[0], e], np.r_[0, _project(ct * N, ndpt, ndat, axis) / w, 0][:len(e) + 1],
                    where="post", color=col, lw=1.0, alpha=0.55, zorder=1)
            val = _project(c * N, ndpt, ndat, axis) / w
            # errors added in quadrature within the projection -- the correlations between cells of the
            # same projected bin are shown in figure H, not smuggled in here
            ev = np.sqrt(_project((err * N) ** 2, ndpt, ndat, axis)) / w
            ax.errorbar(ctr, val, yerr=ev, fmt=mk, ms=4, color=col, lw=1.2, capsize=2, label=lab,
                        zorder=3)
        ax.set_xlabel(xlab); ax.set_ylabel("rate / bin width")
        ax.set_title(f"{ttl} projection  (lines = truth)", fontsize=8.5, loc="left")
        if k == 0:
            ax.legend(frameon=False, fontsize=8)

    style.save(fig, f"{label}_figF")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
