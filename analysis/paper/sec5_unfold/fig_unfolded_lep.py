"""FIGURE F (lepton kinematics) -- the unfolded T2K CC0pi result, in two angular slices.

Each panel is d^2(sigma) / dp_mu dcos(theta_mu) for one cos theta_mu slice of the published binning
(arXiv:2002.09323), in the 1e-38 cm^2/nucleon units of the measurement.

WHY SLICES RATHER THAN PROJECTIONS.  Every 1-D projection of this measurement needs an assumption or an
omission, and slices need neither:

  * a p_mu projection has to collapse cells whose p_mu edges differ between cos slices, which means
    distributing each cell's rate across a common axis in proportion to width -- flat-within-cell --
    and it has to drop the cos < 0.2 catch-all, a single bin spanning 0-30 GeV/c holding 27.6% of the
    signal, because spreading THAT flat would invent a momentum spectrum for a quarter of the rate;
  * a cos theta_mu projection has no good axis.  The slices are near-uniform in angle but wildly
    unequal in cosine -- the backward slice alone is 1.2 of the 2.0 range, leaving six of the nine
    slices inside the forward 0.2 -- so a linear cos axis hides the region the measurement is about,
    and plotting against theta instead spends more than half the axis on one backward bin.

Inside a slice none of that arises: the p_mu bins ARE the published ones, each point is a single
template, and its error is the square root of its own diagonal entry -- no projection operator, no
covariance collapse, no assumption.  The cell-to-cell correlations are figure I's subject.

Slices chosen on measured content, not by eye: [0.20, 0.60] has the highest occupancy (4013 events,
median sigma(c) 0.22) and [0.94, 0.98] has the most bins and the widest momentum reach (10 bins out to
3.25 GeV/c).  Between them they bracket the angular range.

Usage:  ADONIS_UNFOLD_OBS=lep python -m analysis.paper.sec5_unfold.fig_unfolded_lep [label]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style

# SECTION-1 PALETTE.  The paper already has a colour language -- IBM colourblind-safe, blue for the
# first series and orange for the second -- and a section that invents its own makes the reader relearn
# it.  The previous pair (#1f4b9c / #e08214) was neither the section-1 pair nor colourblind-safe.
C_A, C_S = style.C_QE, style.C_RES
STUDIES = [("asimov", "closure", C_A, "o"), ("sig120", r"signal $\times1.20$", C_S, "s")]

# sigma [1e-38 cm^2/nucleon] = (sum w0) * W0_TO_CM2 / A_NUCLEON * 1e38
#
# A_NUCLEON = 13, NOT 12.  T2K:2020sbd reports per nucleon on CH, so the divisor is the 13 nucleons of
# a CH unit; info_content.load_cc0pi uses 12 because the STV release it loads is quoted on carbon.
# Free hydrogen has no neutrons and so contributes essentially nothing to nu_mu CC0pi, which is why
# this is a divisor change and not a missing event source -- the carbon bank is still the whole
# prediction.  It is an 8% effect and the only number here worth re-checking against the release.
# XS_UNIT = 1e39, i.e. the result is quoted in 1e-39 cm^2.  1e-38 left both panels needing a
# matplotlib scale-factor label (x10^-1 on the left), which is drawn at the top-left of the axes and
# collided with the slice title.  In 1e-39 the values run 1.1-7.3 and 1.2-14, so no offset is needed
# and the axis reads directly.
W0_TO_CM2, A_NUCLEON, XS_UNIT = 1e-33, 13.0, 1e39

SLICES = [(0.20, 0.60), (0.94, 0.98)]        # fallback; the config's figures.slices wins


def _fig_cfg(z):
    """The `figures:` block of the config that produced `z`.

    Read from the npz, NOT from the yaml: the figure scripts are pure consumers of a persisted fit --
    that is what stops a plot disagreeing with the numbers it claims to show -- and a yaml can change
    after the fit has run.  Older files carry no cfg_json, so the module defaults stand in.
    """
    import json
    if "cfg_json" in z.files:
        return json.loads(str(z["cfg_json"])).get("figures", {}) or {}
    return {}


def _legend():
    """Say what a LINE is and what a MARKER is.

    The reader has to know the step is the generator truth -- what the fit should return -- and the
    marker is what it did return.  A legend keyed only on the two studies leaves that to be inferred
    from the drawing, and the panel's whole claim is that the two coincide, so the distinction is the
    subject rather than a styling detail.
    """
    h = [Line2D([], [], color=c, lw=1.2) for _s, _l, c, _m in STUDIES]
    h += [Line2D([], [], color="0.25", lw=1.2),
          Line2D([], [], color="0.25", lw=0.0, marker="o", ms=3.4)]
    lab = [l for _s, l, _c, _m in STUDIES] + ["truth (input)", "unfolded (result)"]
    # ABOVE THE PANELS, not inside one.  Placed in the axes it landed on the curves it describes --
    # the spectrum peaks at upper-centre-left in both slices, so there is no free corner.  A
    # figure-level legend also stops it being attached to whichever panel happened to have room.
    return h, lab


def main(label="sec5lep"):
    style.use()
    z = np.load(style.ALTGEN / f"{label}_unfold.npz", allow_pickle=True)
    if str(z["true_kind"]) != "staircase":
        raise SystemExit(f"{label} is not a staircase (lepton) unfolding -- use fig_unfolded instead")
    N = np.asarray(z["n_true"])
    cl, ch = np.asarray(z["true_c_lo"]), np.asarray(z["true_c_hi"])
    pl, ph = np.asarray(z["true_p_lo"]), np.asarray(z["true_p_hi"])
    nt = len(N)
    xs = W0_TO_CM2 / A_NUCLEON * XS_UNIT / float(z["norm_scale"])
    slices = [tuple(v) for v in _fig_cfg(z).get("slices", SLICES)]

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
            # d^2 sigma / dp dcos.  Each point is ONE template, so its uncertainty is its own diagonal
            # entry -- there is no projection here and nothing to collapse.
            base = N[m] * xs / (dp * dcos)
            ax.step(np.r_[edges[0], edges], np.r_[0, ct[m] * base, 0][:len(edges) + 1], where="post",
                    color=col, lw=1.0, zorder=1)
            ax.errorbar(0.5 * (edges[:-1] + edges[1:]), c[m] * base,
                        yerr=np.sqrt(np.diag(C)[m]) * base, xerr=0.5 * dp,
                        fmt=mk, ms=3, lw=0.0, elinewidth=0.9, color=col, zorder=3)
        # LINEAR y.  Within a single angular slice the cross section spans about ONE decade (2e-2 to
        # 4e-1 on the left, 1.5e-2 to 1.2e-1 on the right), so a log axis compresses the differences
        # the panel exists to show and buys nothing -- and it dragged in a log formatter that labels
        # only decades, leaving two or three labels on the axis.  Log belongs on the p_mu spectrum
        # integrated over angle, which really does fall by orders of magnitude; it does not belong here.
        ax.set_ylim(bottom=0.0)
        # THE OVERFLOW BIN IS NOT DRAWN.  It runs to 30 GeV/c; shown to scale it takes ~90% of the
        # axis, and even compressed behind an axis break it spends real width on a bin whose per-GeV
        # density is ~100x below its neighbours.  Its CONTENT is in the fit and stored in the npz --
        # only its display is dropped.  NOTHING ON THE FIGURE SAYS SO, so the omitted range belongs in
        # the LaTeX caption: the axis stopping at 0.6 or 3 GeV/c is not self-explanatory.
        ax.set_xlim(edges[0], edges[-2])
        ax.set_xlabel(r"$p_\mu$ [GeV/$c$]")
        if k == 0:
            ax.set_ylabel(r"$d^2\sigma/dp_\mu\,d\cos\theta_\mu$" "\n"
                          r"[$10^{-39}$cm$^2$/nucleon/(GeV/$c$)]", fontsize=6.6)
        # SLICE DEFINITION AS A HEADER, not inside the axes.  In the panel it competes with the data
        # for space -- bottom-left sat on the first bin's step, top-right on the falling edge -- and
        # which corner is free depends on the slice, so any in-axes position is a per-slice choice
        # that breaks as soon as SLICES changes.
        ax.set_title(rf"${c0:.2f}<\cos\theta_\mu<{c1:.2f}$", fontsize=7.5, pad=4)

    h, lab = _legend()
    fig.legend(h, lab, frameon=False, fontsize=6.4, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.045), handlelength=1.9, columnspacing=1.3, borderpad=0.2)
    fig.subplots_adjust(top=0.885, bottom=0.17)
    # FIXED NAME, no run label.  The output used to be "{label}_figF" and was renamed by hand when
    # copied to the paper repo, so the same figure had two names and neither told you what it showed.
    # Emitting the paper's name directly means the two repositories cannot drift.  A re-run under a
    # different label overwrites this file, which is the intended behaviour: there is one current
    # figure, and the label lives in the npz next to the numbers.
    style.save(fig, "unfolding_example")


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or [])))
