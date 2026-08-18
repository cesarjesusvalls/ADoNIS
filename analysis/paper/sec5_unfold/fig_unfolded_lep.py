"""FIGURE F (lepton kinematics) -- the unfolded T2K CC0pi double-differential result, projected to 1-D.

(a) d(sigma)/dp_mu and (b) d(sigma)/dcos theta_mu, in the 1e-38 cm^2/nucleon units of the published
    measurement (arXiv:2002.09323).  The 58 templates themselves are the measurement; these are the two
    1-D readings of it.

TWO THINGS THE PROJECTION HAS TO GET RIGHT, both of which the rectangular-grid version got wrong:

  * FULL COVARIANCE, not quadrature.  Cells folded into one projected bin are correlated, and ignoring
    that is wrong in BOTH directions -- measured, quadrature understated the 3x3 STV projection by ~30%
    (correlations positive) and overstated the 5x6 one by up to 3.6x (adjacent cells anti-correlated
    because the detector could not resolve them).  With P the projection operator the value is P c and
    the covariance is exactly P C P^T, so this uses that.
  * THE BACKWARD CATCH-ALL IS EXCLUDED FROM (a).  The cos < 0.2 slice is a single bin spanning
    0-30 GeV/c and holds 27.6% of the signal.  Spreading it across a p_mu axis in proportion to width
    -- the flat-within-cell assumption every projection makes -- would invent a momentum spectrum for a
    quarter of the rate.  It is dropped from (a) and stated on the panel; (b) keeps it, because the cos
    slices are common to every cell and that projection needs no assumption at all.

Usage:  ADONIS_UNFOLD_OBS=lep python -m analysis.paper.sec5_unfold.fig_unfolded_lep [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from adonis.unfold.binning import t2k_truth_grid

# SECTION-1 PALETTE.  The paper already has a colour language -- IBM colourblind-safe, blue for the
# first series and orange for the second -- and a section that invents its own makes the reader relearn
# it.  The previous pair (#1f4b9c / #e08214) was neither the section-1 pair nor colourblind-safe.
C_A, C_S = style.C_QE, style.C_RES
STUDIES = [("asimov", "closure", C_A, "o"), ("sig120", r"signal $\times1.20$", C_S, "s")]

# Same chain as analysis/paper/info_content.load_cc0pi, so figure and data share one convention:
#   sigma [1e-38 cm^2/nucleon] = (sum w0) * 1e-33 / 12 * 1e38
W0_TO_CM2, A_CARBON, XS_UNIT = 1e-33, 12.0, 1e38

# Common p_mu axis for panel (b).  Chosen from the edges the slices actually share, so most cells map
# onto target bins exactly and the flat-within-cell assumption does as little work as possible.
PMU_TARGET = [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 2.0, 3.25, 5.625]


def _legend(ax):
    """Two things need distinguishing, and colour alone does not do it.

    The reader has to know that the STEP is the generator truth -- what the fit should return -- and the
    MARKER is what it did return.  A legend keyed only on the two studies leaves that to be inferred
    from the drawing, and the whole claim of the panel is that the two agree, so the distinction is the
    figure's subject rather than a detail of its styling.  Hence one legend column for the studies
    (colour) and one for truth-vs-result (line against marker).
    """
    h = [Line2D([], [], color=c, lw=1.2, alpha=0.7) for _s, _l, c, _m in STUDIES]
    h += [Line2D([], [], color="0.25", lw=1.2, alpha=0.7),
          Line2D([], [], color="0.25", lw=0.0, marker="o", ms=3.4)]
    lab = [l for _s, l, _c, _m in STUDIES] + ["truth (input)", "unfolded (result)"]
    ax.legend(h, lab, frameon=False, fontsize=6.8, ncol=1, loc="lower left", handlelength=1.9,
              labelspacing=0.28, borderpad=0.2)


def main(label="sec5lep"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    if str(z["true_kind"]) != "staircase":
        raise SystemExit(f"{label} is not a staircase (lepton) unfolding -- use fig_unfolded instead")
    G = t2k_truth_grid()
    N = np.asarray(z["n_true"])
    nt = G.n
    cl, ch = np.asarray(z["true_c_lo"]), np.asarray(z["true_c_hi"])
    xs = W0_TO_CM2 / A_CARBON * XS_UNIT / float(z["norm_scale"])

    # 30% SMALLER CANVAS.  Font sizes are absolute points, so shrinking the canvas without touching
    # them makes every label relatively larger -- which is what a figure reproduced at column width in a
    # paper actually needs.
    fig = plt.figure(figsize=(5.04, 2.52))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.34)

    # ---- (a) d sigma / d p_mu, backward slice excluded --------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    keep = cl >= 0.2
    te = np.array(PMU_TARGET); wid = np.diff(te)
    for st, lab, col, mk in STUDIES:
        c, ct = np.asarray(z[f"{st}_c"]), np.asarray(z[f"{st}_c_true"])
        C = np.asarray(z[f"{st}_cov"])[:nt, :nt]
        P = G.projector("pmu", te, N * keep)               # zeroing the excluded cells' weights
        ax.step(np.r_[te[0], te], np.r_[0, (P @ ct) * xs / wid, 0][:len(te) + 1], where="post",
                color=col, lw=1.0, alpha=0.55, zorder=1)
        val = (P @ c) * xs / wid
        err = np.sqrt(np.diag(P @ C @ P.T)) * xs / wid
        ax.errorbar(0.5 * (te[:-1] + te[1:]), val, yerr=err, xerr=0.5 * wid, fmt=mk, ms=3,
                    lw=0.0, elinewidth=0.9, color=col, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_\mu$ [GeV/$c$]")
    ax.set_ylabel(r"$d\sigma/dp_\mu$ [$10^{-38}$cm$^2$/nucleon/(GeV/$c$)]", fontsize=7.5)
    ax.set_xlim(te[0], te[-1])          # bin edges flush against the axes, no padding
    _legend(ax)

    # ---- (b) d sigma / d cos theta_mu, everything included ----------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    ce = G.cos_edges; cw = np.diff(ce)
    for st, lab, col, mk in STUDIES:
        c, ct = np.asarray(z[f"{st}_c"]), np.asarray(z[f"{st}_c_true"])
        C = np.asarray(z[f"{st}_cov"])[:nt, :nt]
        P = G.projector("cos", ce, N)
        ax.step(np.r_[ce[0], ce], np.r_[0, (P @ ct) * xs / cw, 0][:len(ce) + 1], where="post",
                color=col, lw=1.0, alpha=0.55, zorder=1)
        val = (P @ c) * xs / cw
        err = np.sqrt(np.diag(P @ C @ P.T)) * xs / cw
        ax.errorbar(0.5 * (ce[:-1] + ce[1:]), val, yerr=err, xerr=0.5 * cw, fmt=mk, ms=3,
                    lw=0.0, elinewidth=0.9, color=col, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\cos\theta_\mu$")
    ax.set_ylabel(r"$d\sigma/d\cos\theta_\mu$ [$10^{-38}$cm$^2$/nucleon]", fontsize=7.5)
    ax.set_xlim(ce[0], ce[-1])          # cos runs exactly -1 to 1, edge to edge

    style.save(fig, f"{label}_figF")


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or [])))
