"""FIGURE F (lepton kinematics) -- unfolded T2K CC0pi result, in two cos(theta_mu) slices of the
published binning (arXiv:2002.09323), in 1e-38 cm^2/nucleon units.

Each panel plots d^2(sigma)/dp_mu dcos(theta_mu) for one slice: the p_mu bins are the published ones and
each point's error is its own diagonal covariance entry -- no projection, no covariance collapse.
Cell-to-cell correlations are fig_correlation's subject.

Usage:  ADONIS_UNFOLD_OBS=lep python -m analysis.paper.unfolding.fig_unfolded_lep [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis._cli import fig_cfg
from analysis.paper import style

C_A, C_S = style.C_QE, style.C_RES
STUDIES = [("asimov", "closure", C_A, "o"), ("sig120", r"signal $\times1.20$", C_S, "s")]

W0_TO_CM2, A_NUCLEON, XS_UNIT = 1e-33, 13.0, 1e39

SLICES = [(0.20, 0.60), (0.94, 0.98)]


def _legend():
    """Legend distinguishing line (truth, input) from marker (unfolded result), in addition to the
    per-study colour.
    """
    h = [Line2D([], [], color=c, lw=1.2) for _s, _l, c, _m in STUDIES]
    h += [Line2D([], [], color="0.25", lw=1.2),
          Line2D([], [], color="0.25", lw=0.0, marker="o", ms=3.4)]
    lab = [l for _s, l, _c, _m in STUDIES] + ["truth (input)", "unfolded (result)"]
    return h, lab


def main(label="t2k_cc0pi_lep"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    if str(z["true_kind"]) != "staircase":
        raise SystemExit(f"{label} is not a staircase (lepton) unfolding -- use fig_unfolded instead")
    N = np.asarray(z["n_true"])
    cl, ch = np.asarray(z["true_c_lo"]), np.asarray(z["true_c_hi"])
    pl, ph = np.asarray(z["true_p_lo"]), np.asarray(z["true_p_hi"])
    nt = len(N)
    xs = W0_TO_CM2 / A_NUCLEON * XS_UNIT / float(z["norm_scale"])
    slices = [tuple(v) for v in fig_cfg(z).get("slices", SLICES)]

    fig = plt.figure(figsize=(5.04, 2.72))
    gs = fig.add_gridspec(1, len(slices), width_ratios=[1] * len(slices), wspace=0.34)

    for k, (c0, c1) in enumerate(slices):
        ax = fig.add_subplot(gs[0, k])
        m = np.flatnonzero((cl == c0) & (ch == c1))
        edges = np.append(pl[m], ph[m][-1])
        dp, dcos = np.diff(edges), c1 - c0
        for st, lab, col, mk in STUDIES:
            c, ct = np.asarray(z[f"{st}_c"]), np.asarray(z[f"{st}_c_true"])
            C = np.asarray(z[f"{st}_cov"])[:nt, :nt]
            base = N[m] * xs / (dp * dcos)
            ax.step(np.r_[edges[0], edges], np.r_[0, ct[m] * base, 0][:len(edges) + 1], where="post",
                    color=col, lw=1.0, zorder=1)
            ax.errorbar(0.5 * (edges[:-1] + edges[1:]), c[m] * base,
                        yerr=np.sqrt(np.diag(C)[m]) * base, xerr=0.5 * dp,
                        fmt=mk, ms=3, lw=0.0, elinewidth=0.9, color=col, zorder=3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(edges[0], edges[-2])
        ax.set_xlabel(r"$p_\mu$ [GeV/$c$]")
        if k == 0:
            ax.set_ylabel(r"$d^2\sigma/dp_\mu\,d\cos\theta_\mu$" "\n"
                          r"[$10^{-39}$cm$^2$/nucleon/(GeV/$c$)]", fontsize=6.6)
        ax.set_title(rf"${c0:.2f}<\cos\theta_\mu<{c1:.2f}$", fontsize=7.5, pad=4)

    h, lab = _legend()
    fig.legend(h, lab, frameon=False, fontsize=6.4, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.045), handlelength=1.9, columnspacing=1.3, borderpad=0.2)
    fig.subplots_adjust(top=0.885, bottom=0.17)
    style.save(fig, "unfolding_example")


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or [])))
