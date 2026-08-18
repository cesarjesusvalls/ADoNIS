"""FIGURE F -- the unfolded result, for a closure and for an injected signal excess.

(a) the templates themselves: c_j with its full uncertainty, against the injected truth.  This is the
    measurement -- everything else in the section is a way of reading it.
(b,c) the same result projected onto delta-p_T and delta-alpha_T as a DIFFERENTIAL CROSS SECTION, in the
    1e-38 cm^2/nucleon units of the published T2K measurement.  Panel (a) stays a multiplicative factor
    because linearised truth cells have no axis to be differential in.  Projections are taken in TRUTH
    space, which is why the truth grid had to be one the detector can support.

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

# BANK WEIGHT -> CROSS SECTION, the same chain analysis/paper/info_content.load_cc0pi uses for the T2K
# CC0pi STV data, so the figure and the data it is meant to be read against share one convention:
#
#     sigma [1e-38 cm^2 / nucleon]  =  (sum w0) * W0_TO_CM2 / A_CARBON * 1e38
#
# The /12 is the per-NUCLEON convention: the T2K STV release and the ADoNIS bank are per nucleon while
# the generator cross section is on the 12C nucleus.  delta-p_T is reported per GeV/c (not per MeV/c)
# because that is the unit of the published measurement, so its widths are converted from MeV.
W0_TO_CM2 = 1e-33
A_CARBON = 12.0
XS_UNIT = 1e38               # express the result in 1e-38 cm^2
STUDIES = [("asimov", "closure ($c=1$)", C_A, "o"), ("sig120", r"signal $\times1.20$", C_S, "s")]


def _boxes(ax, lo, hi, x0, x1, color, alpha=0.55, lw=0.9, zorder=3, label=None):
    """Uncertainty as a BOX spanning the bin, not a capped bar.

    A bar marks a point with whiskers; a box says "this bin, this interval", which is what a binned
    measurement actually claims.  It also stops the eye reading the marker as more precise than the
    interval, which matters here because several intervals are wider than the spacing between bins.
    """
    for l, h, a, b in zip(np.atleast_1d(lo), np.atleast_1d(hi), np.atleast_1d(x0), np.atleast_1d(x1)):
        ax.add_patch(plt.Rectangle((a, l), b - a, max(h - l, 1e-12), facecolor=color, alpha=alpha,
                                   edgecolor=color, lw=lw, zorder=zorder, label=label))
        label = None


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
    for i, (st, lab, col, mk) in enumerate(STUDIES):
        c, e, ct = z[f"{st}_c"], z[f"{st}_c_err"], z[f"{st}_c_true"]
        off = -0.21 + 0.42 * i
        _boxes(ax, c - e, c + e, x + off - 0.19, x + off + 0.19, col, label=lab)
        ax.plot(x + off, c, "_", color="k", ms=7, mew=1.1, zorder=5)
        ax.axhline(ct[0], color=col, ls=":", lw=1.0, zorder=1)
    ax.set_xlabel("truth cell"); ax.set_ylabel(r"template $c_j$")
    ax.set_xticks(x); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-0.6, nt - 0.4)
    ax.set_title("(a)  unfolded templates", fontsize=9.5, loc="left")

    for k, (axis, edges, xlab, ttl) in enumerate(
            [("dpt", tdpt, r"$\delta p_T$ [MeV/c]", r"(b) $\delta p_T$"),
             ("dat", tdat, r"$\delta\alpha_T$ [rad]", r"(c) $\delta\alpha_T$")]):
        ax = fig.add_subplot(gs[0, 1 + k])
        e = np.asarray(edges, float).copy()
        if not np.isfinite(e[-1]):                       # open bin: draw it at twice the last width
            e[-1] = e[-2] + (e[-2] - e[-3])
        ctr, w = 0.5 * (e[:-1] + e[1:]), np.diff(e)
        # PROJECT AS A MULTIPLICATIVE FACTOR, not as a rate.  The quantity reported for a template is
        # the scale on the nominal prediction: 1 is nominal, and a band from 0.8 to 1.2 says directly
        # that the data allow this projection to move by +-20%.  Dividing c_j N_j by a bin width
        # instead gives events per MeV/c in one panel and events per radian in the other -- two
        # different units, neither comparable to panel (a), and an absolute scale that only means
        # anything relative to the arbitrary 50 000-event normalisation.
        #
        # The N-weighted mean is the right collapse: r_i = sum_j c_j N_j / sum_j N_j over the cells j
        # folded into projected bin i.  It reduces to c_j when a projected bin holds one cell, and it
        # is exactly the factor by which that projected bin's RATE has moved.
        #
        # It also removes the open-bin artefact.  The last delta-p_T bin runs to infinity, so its
        # "width" was a drawing substitution and its height was a rate over an arbitrary denominator,
        # not a density.  A ratio has no denominator to fake.
        Nproj = _project(N, ndpt, ndat, axis)
        # CROSS-SECTION UNITS for the 1-D projections.  Panel (a) stays a multiplicative factor: with
        # the truth cells linearised into one index there is no axis to be differential in, and the
        # scale factor is what is actually reported per template.  In 1-D there IS an axis, so the
        # projections are shown as d(sigma)/dx, which is the form the published measurement takes.
        #
        # `norm_scale` undoes the 50 000-event display normalisation to recover the raw summed w0.
        xs = W0_TO_CM2 / A_CARBON * XS_UNIT / float(z["norm_scale"])
        wid = np.diff(e) / (1000.0 if axis == "dpt" else 1.0)      # dpt: MeV -> GeV/c
        ylab = (r"$d\sigma/d\delta p_T$  [$10^{-38}$cm$^2$/nucleon/(GeV/$c$)]" if axis == "dpt"
                else r"$d\sigma/d\delta\alpha_T$  [$10^{-38}$cm$^2$/nucleon/rad]")
        for st, lab, col, mk in STUDIES:
            c, err, ct = z[f"{st}_c"], z[f"{st}_c_err"], z[f"{st}_c_true"]
            # EACH STUDY AGAINST ITS OWN TRUTH.  A single nominal curve would leave the injected study
            # sitting 20% above it and reading as a discrepancy, when it is an exact recovery.
            ax.step(np.r_[e[0], e],
                    np.r_[0, _project(ct * N, ndpt, ndat, axis) * xs / wid, 0][:len(e) + 1],
                    where="post", color=col, lw=1.0, alpha=0.55, zorder=1)
            val = _project(c * N, ndpt, ndat, axis) * xs / wid
            # FULL COVARIANCE, not quadrature.  The cells folded into one projected bin are correlated,
            # and ignoring that is wrong in BOTH directions, so it is not a conservative shortcut:
            # measured on the 3x3 grid the template correlations are positive (mean +0.27) and
            # quadrature UNDERSTATES the projected error by ~30%; on the 5x6 grid adjacent delta-p_T
            # cells are strongly anti-correlated -- the fit cannot split rate between cells the
            # detector does not resolve, so it moves them in opposite directions -- and quadrature
            # OVERSTATES by up to 3.6x.  With P the projection operator, sigma^2 = diag(P C P^T).
            P = np.zeros((len(Nproj), nt))
            for k in range(nt):
                ii, jj = divmod(k, ndat)
                P[(jj if axis == "dat" else ii), k] = N[k]
            ev = np.sqrt(np.diag(P @ z[f"{st}_cov"][:nt, :nt] @ P.T)) * xs / wid
            _boxes(ax, val - ev, val + ev, e[:-1], e[1:], col, alpha=0.4, label=lab)
            ax.plot(ctr, val, "_", color="k", ms=8, mew=1.1, zorder=5)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab, fontsize=8)
        if axis == "dpt":
            # THE LAST delta-p_T BIN IS OPEN (edge at infinity).  A differential cross section needs a
            # width, so the bar is drawn at the previous bin's width and hatched: its HEIGHT is not a
            # density and must not be compared with the others.
            ax.add_patch(plt.Rectangle((e[-2], ax.get_ylim()[0]), e[-1] - e[-2],
                                       ax.get_ylim()[1] - ax.get_ylim()[0], facecolor="none",
                                       edgecolor="0.45", hatch="///", lw=0.0, alpha=0.5, zorder=6))
            ax.annotate("open bin", (0.5 * (e[-2] + e[-1]), ax.get_ylim()[1]), ha="center",
                        va="top", fontsize=6.5, color="0.35")
        ax.set_title(f"{ttl} projection  (lines = truth)", fontsize=9.5, loc="left")
        if k == 0:
            ax.legend(frameon=False, fontsize=8)

    style.save(fig, f"{label}_figF")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
