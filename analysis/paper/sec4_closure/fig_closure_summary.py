"""Figure A -- the whole closure argument in one figure.

Four panels, one claim each, all from the SAME reference fit (one truth, MLE, no prior):

  (a) RECOVERY + quoted uncertainty.  Asimov fit (no statistical fluctuations): every dial comes back on
      its injected truth, with the profile interval beside the Gaussian sigma, and beside the interval of
      the refit ensemble.  Both are WATER-FILLED (highest-density for the profile, shortest for the toys)
      at the SAME mass, so they are the same construction and can be read against each other directly.
  (b) GOODNESS OF FIT.  chi2_data at the best fit across the refits, against chi2(ndf) with ndf counting
      only bins that CONSTRAIN (empty bins carry sigma=inf and are not degrees of freedom).
  (c) THE BOUNDARY CASE.  E_b sits a couple of sigma from its physical wall (-1.72 at the P1 truth), where
      the model clamps and the likelihood goes flat.  The Gaussian leaks below the wall; the
      likelihood-based interval does not, and the estimator's sampling distribution is an ATOM on the
      boundary plus a truncated continuum.
  (d) THE NON-GAUSSIAN CASE.  C5A (res_axial_strength) away from any boundary: its profile is visibly
      non-parabolic, so the quadratic sigma is the wrong summary even where nothing is clamped.
(c) and (d) share one recipe -- toy histogram, profile curve, quadratic Gaussian, water-filled bars, x in
sigma about the best fit -- so the only difference a reader has to absorb is the physics.

(a) says "here is the uncertainty"; (b) says the fit describes the data; (c) and (d) are the two ways the
quadratic error fails -- at a boundary and at curvature -- with the profile tracking the toys in both.
Superseding the old split of 4.1 / 4.3 / 4.1b into three disconnected figures built on different
ensembles.

The interval MASS is one argument and moves every panel together (see MASS below).  At 68% the shortest
-interval estimator is mode-seeking and noisy: on the 13 well-behaved dials it wanders by 1.5-3 bootstrap
sigma, which is estimator noise and not a real profile-vs-toys disagreement.  At 95% the endpoints sit in
the smooth tails and the median endpoint discrepancy falls from 1.62 to 1.12 bootstrap sigma -- while
M_A_res and delta_strength stay discrepant (4-6 sigma), so the wider level does not hide the real problem.

Usage:  python -m analysis.paper.sec4_closure.fig_closure_summary [label] [ens_label] [mass]
        e.g.  ... sec4_P1 sec4_P1_ens          -> 68% (1 sigma), the default
              ... sec4_P1 sec4_P1_ens 0.9545   -> 95% (2 sigma), written to a separate file
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.physical_fit import PRIOR, theta_nominal
from adonis.reweight.reweight_model import nominal_knobs
from adonis.analysis import knobs as K

C_FIT, C_GAUSS, C_ENS = "#1f4b9c", "0.55", "#c8842a"
# The best-fit marker must NOT reuse the colour of any interval: it used to be C_FIT, so a reader could
# not tell whether the diamond belonged to the profile bar or was its own statement.
C_BFP = "#1a9e57"

# Panel (d) shows ONE dial in detail as the non-Gaussian counterpart to the boundary case in (c).
# C5A is the right choice: it is far from every bound, so its departure from the quadratic cannot be
# blamed on clamping, and its profile and toys agree -- which is what makes it a demonstration that the
# profile is right rather than an open question (M_A_res and delta_strength, whose profiles carry the
# mirror basin of the quadratic RES weight, do not yet agree and belong in their own figure).
NONGAUSS_DIAL = "res_axial_strength"

# MASS carried by every interval bar in the figure.  0.6827 = "1 sigma", 0.9545 = "2 sigma".  Set once
# here (and overridable from the command line) because (a), (c) and (d) must all quote the SAME level --
# a 1-sigma bar in (a) beside a 2-sigma bar in (d) would be unreadable.
MASS = 0.6827
MASS_NAME = {0.6827: r"68% ($1\sigma$)", 0.9545: r"95% ($2\sigma$)", 0.90: "90%"}


def _credible(grid, prof, mass=0.6827):
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(grid, prof)
    xf = np.linspace(grid.min(), grid.max(), 2001)
    dens = np.exp(-0.5 * np.maximum(spl(xf), 0.0)); dens /= np.trapezoid(dens, xf)
    o = np.argsort(dens)[::-1]
    cum = np.cumsum(dens[o]) * (xf[1] - xf[0])
    thr = dens[o][min(np.searchsorted(cum, mass), len(o) - 1)]
    sel = dens >= thr
    return float(xf[np.argmax(dens)]), float(xf[sel].min()), float(xf[sel].max())




def _dchi2_interval(grid, prof, level=1.0):
    """{theta : Dchi2_profile < level} -- the standard profile-likelihood (MINOS/Wilks) interval.

    Distinct from _credible above, which normalises exp(-Dchi2/2) as if it were a density and takes its
    68% highest-density region.  That is neither a credible interval (nothing is marginalised) nor the
    standard profile interval; the two coincide only for a parabolic profile.  This one is the object
    whose coverage was measured against the FC belts (66.0% / 66.7% vs 68.27% for M_A_res / S_Delta).
    """
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(grid, prof)
    xf = np.linspace(grid.min(), grid.max(), 4001)
    y = spl(xf) - level
    imin = int(np.argmin(spl(xf)))
    lo, hi = xf[0], xf[-1]
    left = np.where(y[:imin] > 0)[0]
    if len(left):
        i = left[-1]; lo = float(np.interp(0.0, [y[i], y[i + 1]], [xf[i], xf[i + 1]]))
    right = np.where(y[imin:] > 0)[0]
    if len(right):
        i = imin + right[0]; hi = float(np.interp(0.0, [y[i], y[i - 1]], [xf[i], xf[i - 1]]))
    return float(xf[imin]), lo, hi


def _half68_density(grid, prof, anchor=0.0, half=0.3413447):
    """Interval holding `half` of exp(-dchi2/2) on EACH side of `anchor` (the best fit).

    Anchoring matters: percentiles are anchored at the median and HPD at the mode, so neither is
    comparable to a toy interval measured from the fit.  Both objects are anchored at the SAME point here.
    """
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(grid, prof)
    xf = np.linspace(grid.min(), grid.max(), 4001)
    d = np.exp(-0.5 * np.maximum(spl(xf), 0.0))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(xf))])
    cdf /= cdf[-1]
    c0 = float(np.interp(anchor, xf, cdf))
    lo = float(np.interp(max(c0 - half, 0.0), cdf, xf))
    hi = float(np.interp(min(c0 + half, 1.0), cdf, xf))
    return lo, hi


def _half68_sample(x, anchor, half=0.3413447):
    """Same definition for a toy sample: `half` of the toys on each side of `anchor`."""
    x = np.sort(np.asarray(x))
    c0 = float(np.searchsorted(x, anchor) / len(x))
    q = lambda f: float(np.quantile(x, min(max(f, 0.0), 1.0)))
    return q(c0 - half), q(c0 + half)


def _hpd_density(grid, prof, mass=0.6827):
    """WATER-FILL: lower a horizontal level from the peak of exp(-Dchi2/2) until the enclosed mass
    reaches `mass`; return the extent of {density >= level} plus a flag for a disconnected region.

    Preferred over _half68_density because it always EXISTS.  "x% each side of the best fit" needs x%
    of the mass available on BOTH sides -- at 90% that is 45% a side, and the RES dials do not have it
    (M_A_res carries 65% of its profile mass below the best fit).  A naive implementation then clips to
    the end of the scan and silently reports the scan range as the interval.
    """
    from scipy.interpolate import CubicSpline
    spl = CubicSpline(grid, prof)
    xf = np.linspace(grid.min(), grid.max(), 4001)
    d = np.exp(-0.5 * np.maximum(spl(xf), 0.0)); d /= np.trapezoid(d, xf)
    o = np.argsort(d)[::-1]
    cum = np.cumsum(d[o]) * (xf[1] - xf[0]); cum /= cum[-1]
    thr = d[o][min(int(np.searchsorted(cum, mass)), len(o) - 1)]
    idx = np.where(d >= thr)[0]
    # CLIPPED: the region reached an end of the SCAN, so the reported endpoint is where the scan stopped,
    # not where the likelihood fell away.  Harmless at 68% (the adaptive reach is >=3 sigma) but the
    # binding constraint at 95%, which is why it is flagged rather than silently returned.
    clipped = bool(idx[0] == 0 or idx[-1] == len(xf) - 1)
    return float(xf[idx[0]]), float(xf[idx[-1]]), bool(np.any(np.diff(idx) > 1)), clipped


def _hpd_sample(x, mass=0.6827):
    """Sample analogue of the water-fill: the SHORTEST interval holding `mass` of the toys."""
    t = np.sort(np.asarray(x)); n = t.size
    w = max(int(np.ceil(mass * n)), 2)
    if w >= n:
        return float(t[0]), float(t[-1])
    k = int(np.argmin(t[w:] - t[:n - w]))
    return float(t[k]), float(t[k + w])


def _hpd_boot(x, mass=0.6827, nboot=400, seed=0):
    """Bootstrap error on the two endpoints: resample the toys with replacement and re-measure.

    The endpoints are order statistics, so with a finite ensemble they carry their own uncertainty --
    without it there is no way to tell a real profile-vs-toys discrepancy from ensemble noise.
    """
    x = np.asarray(x); n = len(x)
    rng = np.random.default_rng(seed)
    lo = np.empty(nboot); hi = np.empty(nboot)
    for b in range(nboot):
        lo[b], hi[b] = _hpd_sample(x[rng.integers(0, n, n)], mass)
    return float(lo.std(ddof=1)), float(hi.std(ddof=1))


def main(label="sec4_ref", ens="sec4_ens", mass=None):
    global MASS
    if mass is not None:
        MASS = float(mass)
    style.use()
    zp = np.load(style.ALTGEN / f"{label}_profile.npz", allow_pickle=True)
    sub = [int(k) for k in zp["subset"]]; pn = [str(x) for x in zp["pnames"]]
    grid = np.asarray(zp["grid_sigma"]); prof = np.asarray(zp["prof_dobj"])
    # PER-DIAL axes: a bounded dial is scanned from its boundary upward, so it does NOT share the nominal
    # +-3 sigma axis.  Old npz lack this -- warn rather than silently integrate a fictitious region.
    grids = np.asarray(zp["grids_sigma"]) if "grids_sigma" in zp.files else None
    if grids is None:
        print("[warn] profile npz predates the bounds-aware scan; credible intervals on BOUNDED dials "
              "integrate below the wall and are inflated (E_b was +34%)")
    bfp = np.asarray(zp["bfp"]); spost = np.asarray(zp["sigma_post"]); truth = np.asarray(zp["truth"])
    nom = np.asarray(theta_nominal(nominal_knobs())); prior = np.asarray(PRIOR)
    order = sorted(range(len(sub)), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    # the Gaussian bar is the +-z sigma that carries the SAME mass as the water-filled bars
    ZQ = float(stats.norm.ppf(0.5 + MASS / 2.0))
    MNAME = MASS_NAME.get(round(MASS, 4), f"{100*MASS:.1f}%")
    print(f"interval level: {MNAME}  (Gaussian bar = +-{ZQ:.3f} sigma)")

    F = sorted(glob.glob(str(style.ALTGEN / f"{ens}_*.npz")))
    Z = [np.load(f, allow_pickle=True) for f in F]
    e_fit = np.concatenate([z["th_fit"] for z in Z]) if Z else None
    e_chi2 = np.concatenate([z["chi2_data"] for z in Z]) if Z else None
    nlive = int(Z[0]["nbins_live"]) if Z and "nbins_live" in Z[0].files else None
    ndf = (nlive if nlive else (int(Z[0]["nbins"]) if Z else 0)) - len(sub)

    # (no separate `<label>_ebwall.npz` is read any more -- panel (d) is built from THIS fit's own E_b
    # profile below, which is what stops it going stale against a different truth.)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(11.5, 6.6))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], height_ratios=[1.0, 1.0, 1.0],
                              hspace=0.55, wspace=0.28, left=0.09, right=0.98, top=0.95, bottom=0.08)
        axA = fig.add_subplot(gs[:, 0])          # recovery spans all three rows
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 1])
        axD = fig.add_subplot(gs[2, 1])

        # ---- (a) recovery ------------------------------------------------------------------------ #
        yy = np.arange(len(order)); gid = [style.knob_group(pn[sub[c]]) for c in order]
        for i, c in enumerate(order):
            k = sub[c]; p = max(prior[k], 1e-12)
            _g = grids[c] if grids is not None else grid
            # ANCHOR EVERYTHING ON THE ASIMOV BEST FIT.  This used to sit at bfp + mode*sigma, the HPD
            # MODE of exp(-Dchi2/2).  On a near-degenerate direction the inner re-minimisation of the
            # profile scan stops short, which displaces that mode (measured: -0.32 sigma for C5A), and the
            # Gaussian bar, the profile bar and the marker all slid with it while the truth star did not --
            # making the closure look broken when the fit is exact (bfp == truth to 1e-15).
            _ok = np.isfinite(_g) & np.isfinite(prof[c])
            # PROFILE BAR = 68% WATER-FILL of exp(-Dchi2/2) -- the same construction as the toy band
            # below (shortest interval holding 68%), so the two are directly comparable.  Neither is
            # anchored on the best fit: an interval that must carry 34.1% on each side of the BFP does
            # not exist for the strongly one-sided RES profiles, so that construction was clipping.
            hlo, hhi, hdj, hcl = _hpd_density(_g[_ok], prof[c][_ok], MASS)
            xhat = (bfp[k] - nom[k]) / p
            axA.errorbar([xhat], [yy[i]], xerr=[[ZQ * spost[c] / p], [ZQ * spost[c] / p]],
                         fmt="none", ecolor=C_GAUSS, capsize=4.5, capthick=0.9, elinewidth=0.9,
                         zorder=2)
            # drawn as an explicit segment, not xerr about xhat: a water-filled interval is asymmetric
            # about the best fit and, on a bimodal profile, need not be centred on it at all.
            axA.plot([xhat + hlo * spost[c] / p, xhat + hhi * spost[c] / p], [yy[i]] * 2,
                     color=C_FIT, lw=2.2, solid_capstyle="butt", zorder=4)
            for xe in (xhat + hlo * spost[c] / p, xhat + hhi * spost[c] / p):
                axA.plot([xe] * 2, [yy[i] - 0.13, yy[i] + 0.13], color=C_FIT, lw=1.5, zorder=4)
            if hdj or hcl:
                # v = the water-filled region is DISCONNECTED; x = it ran into the end of the scan, so
                # that endpoint is where the scan stopped and is a lower bound on the true one.
                axA.plot(xhat, yy[i] - 0.30, marker=("v" if hdj else "x"), ms=2.6, color=C_FIT, zorder=4)
            if e_fit is not None:
                # LIKE FOR LIKE: the toys' SHORTEST 68% interval, at its own position.  This used to be
                # +-1 RMS about xhat, which forced a skewed toy distribution to share a centre with the
                # object it is compared against and hid its asymmetry; the intermediate "34.1% each side
                # of the BFP" fixed the anchor but does not exist on one-sided profiles.
                t_lo, t_hi = _hpd_sample(e_fit[:, c], MASS)
                axA.plot([(t_lo - nom[k]) / p, (t_hi - nom[k]) / p], [yy[i]] * 2,
                         color=C_ENS, lw=5.5, alpha=0.35, solid_capstyle="butt", zorder=1)
                # bootstrap error on the endpoints is NOT drawn (it was ~0.05-0.13 sigma at 2000 toys,
                # small enough that the ticks only added ink); still printed so it can be quoted.
                b_lo, b_hi = _hpd_boot(e_fit[:, c], MASS)
                print(f"  {pn[k]:>20}  profile [{hlo:+6.3f},{hhi:+6.3f}]  toys "
                      f"[{(t_lo-bfp[k])/spost[c]:+6.3f},{(t_hi-bfp[k])/spost[c]:+6.3f}]"
                      f"  (sigma_post units, +-{b_lo/spost[c]:.3f}/{b_hi/spost[c]:.3f} boot)"
                      + ("   DISJOINT" if hdj else "") + ("   CLIPPED-BY-SCAN" if hcl else ""))
            axA.plot(xhat, yy[i], "D", ms=4.5, color=C_BFP, mec="white", mew=0.5, zorder=6)
            axA.plot((truth[k] - nom[k]) / p, yy[i], marker="*", ms=11, color="k", zorder=5, ls="none")
        axA.axvline(0, color="0.8", lw=0.8, zorder=0)
        axA.set_yticks(yy); axA.set_yticklabels([style.plab(pn[sub[c]]) for c in order], fontsize=7)
        axA.set_ylim(len(order) - 0.5, -0.5)
        axA.set_xlabel(r"$(\theta-\theta_{\rm nom})\,/\,\sigma_{\rm ref}$", fontsize=9)
        bounds = [i for i in range(1, len(gid)) if gid[i] != gid[i - 1]]
        seg = [-.5] + [b - .5 for b in bounds] + [len(gid) - .5]
        yt = axA.get_yaxis_transform()
        for a_, b_ in zip(seg[:-1], seg[1:]):
            g = gid[int((a_ + b_) / 2 + .5)]
            axA.axhspan(a_, b_, facecolor=style.KNOB_GROUP_COLOR[g], alpha=0.10, zorder=0, lw=0)
            axA.text(0.982, a_ + 0.12, style.KNOB_GROUP_NAME[g].replace("\n", " "), transform=yt,
                     ha="right", va="top", fontsize=6, color="0.25", fontweight="bold", zorder=7)
        for b_ in bounds:
            axA.axhline(b_ - .5, color="0.35", lw=0.7, zorder=1)
        axA.tick_params(which="both", top=False, right=False, labelsize=7)
        h = [plt.Line2D([], [], color="k", marker="*", ls="", ms=10),
             plt.Line2D([], [], color=C_BFP, marker="D", ls="", ms=4.5, mec="white", mew=0.5),
             plt.Line2D([], [], color=C_FIT, lw=2.2),
             plt.Line2D([], [], color=C_GAUSS, lw=0.9),
             plt.Line2D([], [], color=C_ENS, lw=5, alpha=0.35)]
        axA.legend(h, ["injected truth", "BFP", "Profile", "Gaussian", "Toys"],
                   fontsize=6, loc="lower left", framealpha=0.92, borderpad=0.3, labelspacing=0.3)
        axA.set_title(f"(a)  recovery + quoted uncertainty  ({MNAME} intervals)",
                      fontsize=9.5, loc="left")

        # ---- (b) goodness of fit ------------------------------------------------------------------- #
        # (the old "ensemble RMS / quoted sigma" panel is gone: a single RMS ratio compresses the whole
        # interval comparison to one number and cannot show ASYMMETRY, which is the entire point on the
        # dials that fail.  Panel (a) already carries that comparison interval-to-interval.)
        if e_chi2 is not None:
            axB.hist(e_chi2, bins=30, density=True, color=C_FIT, alpha=0.75, edgecolor="white", lw=0.4)
            xx = np.linspace(max(0, e_chi2.min() * 0.85), e_chi2.max() * 1.1, 300)
            axB.plot(xx, stats.chi2.pdf(xx, ndf), "k-", lw=1.3, label=f"$\\chi^2$(ndf={ndf})")
            axB.legend(fontsize=7); axB.set_xlabel(r"$\chi^2_{\rm data}$ at the best fit", fontsize=8)
            axB.set_ylabel("density", fontsize=8)
            axB.set_title(f"(b)  goodness of fit  ({len(e_chi2)} refits)", fontsize=9.5, loc="left")
        axB.tick_params(which="both", top=False, right=False, labelsize=7)

        # ---- (c) the boundary case: the toys are a MIXTURE, and both parts are predictable ---------- #
        # Built from the SAME profile scan as panel (a) -- zp's per-dial axis for E_b -- so (a) and (c)
        # cannot disagree.  This used to read a separate `ebwall` scan with its own range, resolution and
        # script, whose only justification was showing the region below the wall; but the likelihood there
        # is meaningless (the model clamps, chi2 is flat) and the Gaussian is analytic, so the second scan
        # bought nothing and silently went stale against a different truth.
        #
        # A bounded MLE has no density: its sampling distribution is an ATOM at the wall plus a truncated
        # continuum (Chernoff 1954; Self & Liang 1987), with atom weight
        #        P(on wall) = Phi( -(theta* - b) / sigma )
        # -- the probability the UNCONSTRAINED estimator would have fallen below it.  Both pieces are
        # tested against the toys here.
        #
        # SAME RECIPE AS (d) -- toy histogram, profile curve, quadratic Gaussian, 68% water-fill bars,
        # x in units of sigma about the best fit -- so the two failure modes are read off identically and
        # the only difference between the panels is the physics.  The boundary adds two things: the
        # unphysical strip is shaded and the wall is marked, and the toys' point mass ON the wall is
        # excluded from the histogram (a MASS drawn as a density would tower over every curve) and
        # reported as text instead.  Nothing is rescaled by hand: the histogram is a density over ALL
        # toys, so its area is (1 - f_wall) by construction, and the Gaussian is normalised over the whole
        # line, so its area above the wall is (1 - p_atom).  The two are then directly comparable.
        cE = [c for c, k in enumerate(sub) if pn[k] == "Eb_shift"]
        if cE and e_fit is not None:
            from scipy.interpolate import CubicSpline
            c0 = cE[0]; k0 = sub[c0]
            lo = K.phys_lo("Eb_shift") or 0.0
            sig = float(spost[c0]); ebt = float(truth[k0])
            wall = (lo - bfp[k0]) / sig                                # the wall, in sigma about the BFP
            eb = e_fit[:, c0]; on = np.abs(eb - lo) < 1e-6
            f_on = float(on.mean()); n_on = int(on.sum())
            v = (eb - bfp[k0]) / sig
            g0 = (grids[c0] if grids is not None else grid); dch = prof[c0]
            ok = np.isfinite(g0) & np.isfinite(dch)
            xf = np.linspace(max(g0[ok].min(), wall), g0[ok].max(), 800)
            d = np.exp(-0.5 * np.maximum(CubicSpline(g0[ok], dch[ok])(xf), 0.0))
            d /= np.trapezoid(d, xf)                                   # physical region only
            p_atom = float(stats.norm.cdf((lo - ebt) / sig))

            axC.axvspan(wall - 1.6, wall, facecolor="0.5", alpha=0.25, lw=0, zorder=0)
            axC.hist(v[~on], bins=np.linspace(wall, max(v.max(), 3.2), 41), density=False,
                     weights=np.full(int((~on).sum()),
                                     1.0 / (len(v) * (max(v.max(), 3.2) - wall) / 40.0)),
                     histtype="stepfilled", color=C_ENS, alpha=0.40, label=f"toys (n={len(v)})")
            axC.hist(v[~on], bins=np.linspace(wall, max(v.max(), 3.2), 41), density=False,
                     weights=np.full(int((~on).sum()),
                                     1.0 / (len(v) * (max(v.max(), 3.2) - wall) / 40.0)),
                     histtype="step", lw=1.1, color="#8a5a12")
            axC.plot(xf, d * (1.0 - f_on), color=C_FIT, lw=1.7,
                     label=r"profile $\propto e^{-\Delta\chi^2/2}$")
            gx = np.linspace(wall - 1.6, max(v.max(), 3.2), 400)       # crosses the wall on purpose
            axC.plot(gx, np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi), color=C_GAUSS, lw=1.3, ls="--",
                     label=r"quadratic $N(0,\sigma)$")
            tl, tu = _hpd_sample(v, MASS)
            pl, pu, _, _ = _hpd_density(g0[ok][g0[ok] >= wall], dch[ok][g0[ok] >= wall], MASS)
            yt_ = max(axC.get_ylim()[1], d.max())
            axC.plot([tl, tu], [0.90 * yt_] * 2, color="#8a5a12", lw=2.4, solid_capstyle="butt", zorder=6)
            axC.plot([pl, pu], [0.80 * yt_] * 2, color=C_FIT, lw=2.4, solid_capstyle="butt", zorder=6)
            for xe, yv, cc in ((tl, 0.90, "#8a5a12"), (tu, 0.90, "#8a5a12"),
                               (pl, 0.80, C_FIT), (pu, 0.80, C_FIT)):
                axC.plot([xe] * 2, [(yv - 0.03) * yt_, (yv + 0.03) * yt_], color=cc, lw=1.4, zorder=6)
            axC.axvline(wall, color="k", lw=1.0, ls="--", zorder=5)
            axC.axvline(0.0, color="k", lw=0.8, ls=":")

            # THE ATOM, as text INSIDE the shaded strip -- a probability MASS and a probability DENSITY do
            # not share a y-scale, so it cannot be drawn as a bar among the curves.  Kept to three short
            # lines so it fits the 1.6-sigma strip without a box of its own.
            err = np.sqrt(max(f_on, 1e-9) * (1 - f_on) / len(eb))
            axC.text(wall - 0.8, 0.62 * yt_,
                     f"on boundary\n{100*f_on:.1f}$\\pm${100*err:.1f}%\npred {100*p_atom:.1f}%",
                     ha="center", va="center", fontsize=5.8, color="0.15", linespacing=1.25, zorder=7)
            axC.set_xlabel(r"$(E_b-\hat\theta)\,/\,\sigma$", fontsize=8)
            axC.set_ylabel("density", fontsize=8)
            axC.set_xlim(wall - 1.6, max(v.max(), 3.2)); axC.set_ylim(bottom=0)
            axC.legend(fontsize=5.8, loc="upper right", framealpha=0.92, borderpad=0.3,
                       handlelength=1.6, labelspacing=0.25)
            axC.set_title(r"(c)  at a boundary: $E_b$", fontsize=9.5, loc="left")
            print(f"  panel (c) Eb_shift: on the boundary {100*f_on:.1f}+-{100*err:.1f}%  "
                  f"Chernoff {100*p_atom:.1f}%  (n={len(eb)}, n_on={n_on})  boundary at {wall:+.2f} sigma\n"
                  f"                      toys 68% [{tl:+.3f},{tu:+.3f}]  profile 68% [{pl:+.3f},{pu:+.3f}]")
        axC.tick_params(which="both", top=False, right=False, labelsize=7)

        # ---- (d) the OTHER way the quadratic fails: curvature, with no boundary in sight ------------ #
        # C5A is the control for panel (c).  Nothing is clamped here -- it sits far from every bound --
        # yet exp(-Dchi2/2) is visibly narrower than the quadratic N(0,1) that the same fit's covariance
        # reports, because the RES axial block enters the weight quadratically.  So "the error bar is
        # wrong" is not a boundary artefact: the Hessian at the minimum only knows the curvature THERE,
        # and the profile is what carries the rest.  The toys are the arbiter and they follow the profile.
        cP = [c for c, k in enumerate(sub) if pn[k] == NONGAUSS_DIAL]
        if cP and e_fit is not None:
            from scipy.interpolate import CubicSpline
            c0 = cP[0]; k0 = sub[c0]; sig = float(spost[c0])
            v = (e_fit[:, c0] - bfp[k0]) / sig                 # toy MLE about the Asimov fit, in sigma
            g0 = (grids[c0] if grids is not None else grid); dch = prof[c0]
            ok = np.isfinite(g0) & np.isfinite(dch)
            xf = np.linspace(g0[ok].min(), g0[ok].max(), 800)
            d = np.exp(-0.5 * np.maximum(CubicSpline(g0[ok], dch[ok])(xf), 0.0))
            d /= np.trapezoid(d, xf)
            axD.hist(v, bins=40, density=True, histtype="stepfilled", color=C_ENS, alpha=0.40,
                     label=f"toys (n={len(v)})")
            axD.hist(v, bins=40, density=True, histtype="step", lw=1.1, color="#8a5a12")
            axD.plot(xf, d, color=C_FIT, lw=1.7, label=r"profile $\propto e^{-\Delta\chi^2/2}$")
            gx = np.linspace(-4, 4, 400)
            axD.plot(gx, np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi), color=C_GAUSS, lw=1.3, ls="--",
                     label=r"quadratic $N(0,\sigma)$")
            tl, tu = _hpd_sample(v, MASS); pl, pu, _, _ = _hpd_density(g0[ok], dch[ok], MASS)
            yt_ = max(axD.get_ylim()[1], d.max())
            axD.plot([tl, tu], [0.90 * yt_] * 2, color="#8a5a12", lw=2.4, solid_capstyle="butt", zorder=6)
            axD.plot([pl, pu], [0.80 * yt_] * 2, color=C_FIT, lw=2.4, solid_capstyle="butt", zorder=6)
            for xe, yv, cc in ((tl, 0.90, "#8a5a12"), (tu, 0.90, "#8a5a12"),
                               (pl, 0.80, C_FIT), (pu, 0.80, C_FIT)):
                axD.plot([xe] * 2, [(yv - 0.03) * yt_, (yv + 0.03) * yt_], color=cc, lw=1.4, zorder=6)
            axD.axvline(0.0, color="k", lw=0.8, ls=":")
            lo_, hi_ = np.percentile(v, [0.2, 99.8]); m_ = d > 1e-3 * d.max()
            if m_.any():
                lo_ = min(lo_, xf[m_].min()); hi_ = max(hi_, xf[m_].max())
            pad = 0.06 * (hi_ - lo_); axD.set_xlim(lo_ - pad, hi_ + pad); axD.set_ylim(bottom=0)
            axD.set_xlabel(r"$(\theta-\hat\theta)\,/\,\sigma$", fontsize=8)
            axD.set_ylabel("density", fontsize=8)
            axD.legend(fontsize=5.8, loc="upper right", framealpha=0.92, borderpad=0.3,
                       handlelength=1.6, labelspacing=0.25)
            axD.set_title(f"(d)  away from any boundary: {style.plab(NONGAUSS_DIAL)}",
                          fontsize=9.5, loc="left")
            print(f"  panel (d) {NONGAUSS_DIAL}: toys 68% [{tl:+.3f},{tu:+.3f}]  "
                  f"profile 68% [{pl:+.3f},{pu:+.3f}]  quadratic [-1.000,+1.000]")
        axD.tick_params(which="both", top=False, right=False, labelsize=7)
        for a_ in (axB, axC, axD):
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)

        # LABEL-tagged: the study points (P1 = all dials off nominal, P2 = nominal but E_b) are separate
        # figures and must not overwrite each other.  `sec4_ref` keeps the historical filename.
        base = "sec4_closure_summary" if label == "sec4_ref" else f"{label}_closure_summary"
        # a non-default interval level gets its own file, so the 1-sigma and 2-sigma versions can be
        # compared side by side instead of one silently replacing the other
        style.save(fig, base if abs(MASS - 0.6827) < 1e-6 else f"{base}_{round(100*MASS)}")


if __name__ == "__main__":
    # third arg = interval mass, e.g. 0.9545 for 2 sigma (default 0.6827)
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:3] or []))
