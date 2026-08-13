"""FIGURE A -- the whole closure argument in one figure.

Four panels, all from the SAME reference fit (one truth, MLE, no prior).  (a), (c) and (d) state what
uncertainty ONE dataset implies; (b) is the only panel that uses the toy ensemble, and it uses it for the
one question toys actually answer.

  (a) RECOVERY + quoted uncertainty.  Asimov fit: every dial returns on its injected truth, with three
      intervals on one line as nested ribbons -- the quadratic sigma, the Laplace marginal
      (profile x sqrt(det V_nuis)) and the exact marginal from NUTS.
  (b) DO THE INTERVALS COVER?  Dchi2 = chi2(theta_true) - chi2(theta_hat) over the toys, against chi2(k).
      This is the statistic whose distribution DEFINES coverage.  chi2(theta_true) costs nothing: the toy
      data is m(theta_true)+n, so it is the sum of squared pulls, reconstructed from the seed.
  (c) AT A BOUNDARY: E_b, a couple of sigma from its wall.  The Gaussian leaks below it; the
      likelihood-based intervals do not.  The text box carries the boundary atom (Chernoff 1954).
  (d) AWAY FROM ANY BOUNDARY: C5A.  Non-parabolic with nothing clamped, and the clearest case where the
      profile alone is NOT the marginal -- the Laplace correction lands on the NUTS histogram.

WHY NO TOY HISTOGRAM IN (a), (c), (d).  Earlier versions overlaid the spread of best-fit values on the
quoted interval.  Those are different objects: an interval is a coverage statement, a histogram of
theta_hat is a sampling spread, and they coincide only in the Gaussian limit.  Comparing them made a 0.2
sigma offset on the degenerate RES directions look like a defect when the measured coverage is correct
(67.0 +- 1.1% at a nominal 68.27%).  The toys belong in (b).

Goodness of fit is no longer plotted -- it is a different claim (does the MODEL fit) and is printed for
the caption instead: median chi2/ndf ~ 0.99 at the P1 point.

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
# the two objects that were missing from this figure: the Laplace (Occam) correction of the profile,
# and the EXACT marginal from NUTS.  They should land on top of each other -- that is the check.
C_OCC, C_NUTS = "#6a3d9a", "#e08214"

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


def _chi2_at_truth(label, F):
    """chi2 evaluated at the TRUE theta for every toy -- reconstructed from the seed, no model call.

    The toy data is  m(theta_true) + n , so chi2(theta_true) = sum (n/sigma)^2 exactly, and n is
    reproducible: rng(1e6+seed) drawn per sample block in engine order, which is what the ensemble did.
    Returned in the SAME order as np.concatenate([z["chi2_data"] for z in Z]) over sorted F.
    """
    z = np.load(style.ALTGEN / f"{label}.npz", allow_pickle=True)
    sig = np.asarray(z["sigma"]); row0 = np.asarray(z["row0"])
    ok = np.isfinite(sig) & (sig > 0)
    out = []
    for f in F:
        base = int(Path(f).stem.rsplit("_", 1)[1])
        for t in range(len(np.load(f, allow_pickle=True)["chi2_data"])):
            rng = np.random.default_rng(1_000_000 + base + t)
            p = np.zeros(len(sig))
            for i in range(len(row0) - 1):
                a, b = row0[i], row0[i + 1]
                sd = np.where(np.isfinite(sig[a:b]), sig[a:b], 0.0)
                nn = rng.normal(0.0, sd)
                p[a:b] = np.where(sd > 0, nn / np.where(sd > 0, sd, 1.0), 0.0)
            out.append(float(np.sum(p[ok] ** 2)))
    return np.asarray(out)


def _load_nuts(label):
    """NUTS samples in sigma_post units about the BFP, keyed by dial name.  Empty dict if absent."""
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_nutsown_*.npz")))
    if not fs:
        return {}, 0
    Z = [np.load(f, allow_pickle=True) for f in fs]
    n = min(len(np.asarray(z["u"])) for z in Z)
    U = np.concatenate([np.asarray(z["u"])[-n:] for z in Z])
    pnn = [str(x) for x in Z[0]["pnames"]]; sbn = [int(k) for k in Z[0]["subset"]]
    return {pnn[k]: U[:, c] for c, k in enumerate(sbn)}, len(U)


def _occam(grid, prof, logdet, mass):
    """Profile x sqrt(det V_nuis): the Laplace approximation to the MARGINAL, which is what NUTS
    computes exactly.  Profiling maximises over the nuisances, marginalising integrates over them, and
    to leading order the two differ by exactly this volume factor."""
    m = np.isfinite(grid) & np.isfinite(prof) & np.isfinite(logdet)
    if m.sum() < 5:
        return None
    from scipy.interpolate import CubicSpline
    xf = np.linspace(grid[m].min(), grid[m].max(), 4001)
    d = np.exp(-0.5 * np.maximum(CubicSpline(grid[m], prof[m])(xf), 0.0))
    lv = CubicSpline(grid[m], logdet[m])(xf)
    d = d * np.exp(0.5 * (lv - lv.max())); d /= np.trapezoid(d, xf)
    o = np.argsort(d)[::-1]; cum = np.cumsum(d[o]) * (xf[1] - xf[0]); cum /= cum[-1]
    thr = d[o][min(int(np.searchsorted(cum, mass)), len(o) - 1)]
    idx = np.where(d >= thr)[0]
    return float(xf[idx[0]]), float(xf[idx[-1]]), xf, d


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
    logdet = np.asarray(zp["logdet_Vnuis"]) if "logdet_Vnuis" in zp.files else None
    NU, n_nuts = _load_nuts(label)
    print(f"NUTS: {n_nuts} samples, {sum(1 for k in sub if pn[k] in NU)}/{len(sub)} dials")
    nom = np.asarray(theta_nominal(nominal_knobs())); prior = np.asarray(PRIOR)
    order = sorted(range(len(sub)), key=lambda c: (style.knob_group(pn[sub[c]]), sub[c]))
    # the Gaussian bar is the +-z sigma that carries the SAME mass as the water-filled bars
    ZQ = float(stats.norm.ppf(0.5 + MASS / 2.0))
    MNAME = MASS_NAME.get(round(MASS, 4), f"{100*MASS:.1f}%")
    print(f"interval level: {MNAME}  (Gaussian bar = +-{ZQ:.3f} sigma)")

    # SHARD FILES ONLY: `<ens>_<base>.npz` with a NUMERIC suffix.  A bare `{ens}_*.npz` also matches
    # sidecars that share the prefix -- `sec4_P1_ens_conv.npz`, the per-fit convergence record, has no
    # `th_fit` key and made this crash.  Match the contract, not the prefix.
    F = sorted(f for f in glob.glob(str(style.ALTGEN / f"{ens}_*.npz"))
               if Path(f).stem[len(Path(ens).name) + 1:].isdigit())
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
            sc = spost[c] / p
            # FOUR PREDICTIONS, all from the SAME single dataset, stacked so they can be read against
            # each other: the quadratic sigma, the profile, the profile corrected by the nuisance
            # volume (Laplace marginal), and the exact marginal from NUTS.  No toys -- a histogram of
            # best fits is a spread, not an interval, and belongs with the coverage test in (b).
            # PROFILE-ONLY is deliberately not drawn here: that it needs the volume factor is
            # the point of (c) and (d).  The three intervals share ONE line per dial, drawn as nested
            # ribbons -- widest/faintest behind, narrowest/solid in front -- so a dial occupies a single
            # row and the relative widths are read directly instead of across three stacked rows.
            rows = [(C_GAUSS, -ZQ, ZQ, 7.0, 0.45, 2)]
            oc = _occam(_g, prof[c], logdet[c], MASS) if logdet is not None else None
            if oc is not None:
                rows.append((C_OCC, oc[0], oc[1], 3.6, 0.95, 3))
            nu = NU.get(pn[k])
            a_ = b_ = None
            if nu is not None:
                a_, b_ = _hpd_sample(nu, MASS)
                rows.append((C_NUTS, a_, b_, 1.5, 1.0, 4))
            for col, lo_, hi_, lw, al, zo in rows:
                axA.plot([xhat + lo_ * sc, xhat + hi_ * sc], [yy[i]] * 2, color=col, lw=lw,
                         alpha=al, solid_capstyle="butt", zorder=zo)
            for xe in (xhat - ZQ * sc, xhat + ZQ * sc):      # Gaussian extent, readable when overlaid
                axA.plot([xe] * 2, [yy[i] - 0.22, yy[i] + 0.22], color=C_GAUSS, lw=0.9, zorder=2)
            if hdj or hcl:
                axA.plot(xhat, yy[i] - 0.34, marker=("v" if hdj else "x"), ms=2.4, color=C_FIT, zorder=4)
            print(f"  {pn[k]:>20}  prof [{hlo:+6.3f},{hhi:+6.3f}]"
                  + (f"  occam [{oc[0]:+6.3f},{oc[1]:+6.3f}]" if oc else "")
                  + (f"  nuts [{a_:+6.3f},{b_:+6.3f}]" if nu is not None else "")
                  + ("   DISJOINT" if hdj else "") + ("   CLIPPED" if hcl else ""))
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
             plt.Line2D([], [], color=C_GAUSS, lw=0.9),
             plt.Line2D([], [], color=C_OCC, lw=2.2),
             plt.Line2D([], [], color=C_NUTS, lw=2.2)]
        axA.legend(h, ["Injected truth", "BFP", "Gaussian",
                       "Marginal (Laplace)", "Marginal (NUTS)"],
                   fontsize=5.6, loc="lower left", framealpha=0.92, borderpad=0.3, labelspacing=0.25)
        axA.set_title(f"(a)  recovery + quoted uncertainty  ({MNAME} intervals)",
                      fontsize=9.5, loc="left")

        # ---- (b) do the intervals COVER? ---------------------------------------------------------
        # The likelihood ratio at the TRUE point, Dchi2 = chi2(theta_true) - chi2(theta_hat), is the
        # statistic whose distribution defines coverage: Wilks says chi2(k) for k fitted dials, so the
        # region {Dchi2 < q_cl} is a cl-confidence region.  This is NOT the goodness-of-fit chi2 that
        # used to live here -- that one (chi2 at the best fit vs chi2(ndf-k)) asks whether the MODEL
        # fits, which is a different claim and is quoted in the caption instead.
        if e_chi2 is not None:
            c2t = _chi2_at_truth(label, F)
            dch = c2t - e_chi2
            kk_ = len(sub)
            axB.hist(dch, bins=40, density=True, color=C_FIT, alpha=0.75, edgecolor="white", lw=0.4)
            xx = np.linspace(0, max(dch.max(), stats.chi2.ppf(0.999, kk_)) * 1.02, 400)
            axB.plot(xx, stats.chi2.pdf(xx, kk_), "k-", lw=1.3, label=f"$\\chi^2$({kk_})")
            txt = []
            for cl in (0.6827, 0.90, 0.95):
                q = stats.chi2.ppf(cl, kk_); cov = float(np.mean(dch < q))
                se = np.sqrt(cov * (1 - cov) / len(dch))
                axB.axvline(q, color="0.5", lw=0.7, ls=":")
                txt.append(f"{100*cl:.1f}%: {100*cov:.1f}$\\pm${100*se:.1f}%")
                print(f"  coverage nominal {100*cl:5.2f}% -> observed {100*cov:5.2f}+-{100*se:.2f}%")
            axB.text(0.97, 0.62, "coverage\n" + "\n".join(txt), transform=axB.transAxes,
                     ha="right", va="top", fontsize=6.2, color="0.15",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5, alpha=0.95))
            axB.legend(fontsize=7, loc="upper left")
            axB.set_xlabel(r"$\Delta\chi^2 = \chi^2(\theta_{\rm true})-\chi^2(\hat\theta)$", fontsize=8)
            axB.set_ylabel("density", fontsize=8)
            axB.set_title(f"(b)  do the intervals cover?  ({len(dch)} toys)", fontsize=9.5, loc="left")
            print(f"  GOF (for the caption): median chi2/ndf = {np.median(e_chi2):.1f}/{ndf} "
                  f"= {np.median(e_chi2)/ndf:.3f}")
        axB.tick_params(which="both", top=False, right=False, labelsize=7)

        # ---- (c) and (d): the four PREDICTIONS as curves, no toys -------------------------------
        # Same recipe for both, so the only difference a reader absorbs is the physics: (c) is E_b,
        # a couple of sigma from a hard wall; (d) is C5A, far from every bound.  Toys are deliberately
        # absent -- the histogram of best fits is a spread, not an interval, and the question "is the
        # interval right" is answered by the coverage panel (b), not by overlaying the two.
        def _panel(A, nm, tag, title, note=None):
            from scipy.interpolate import CubicSpline
            cc = [c for c, k in enumerate(sub) if pn[k] == nm]
            if not cc:
                A.axis("off"); return
            c0 = cc[0]; k0 = sub[c0]; sig = float(spost[c0])
            g0 = (grids[c0] if grids is not None else grid); dch = prof[c0]
            ok = np.isfinite(g0) & np.isfinite(dch)
            wall = None
            _lo = K.phys_lo(nm)
            if _lo is not None:
                xb = (_lo - bfp[k0]) / sig
                if g0[ok].min() - 1e-9 <= xb <= g0[ok].max() + 1e-9:
                    wall = xb
            xf = np.linspace(g0[ok].min(), g0[ok].max(), 800)
            d = np.exp(-0.5 * np.maximum(CubicSpline(g0[ok], dch[ok])(xf), 0.0))
            d /= np.trapezoid(d, xf)
            gx = np.linspace(min(xf[0], (wall - 1.6) if wall is not None else xf[0]), xf[-1], 500)
            A.plot(gx, np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi), color=C_GAUSS, lw=1.3, ls="--",
                   label="Gaussian")
            A.plot(xf, d, color=C_FIT, lw=1.7, label="Profile")
            if logdet is not None:
                oc = _occam(g0, prof[c0], logdet[c0], MASS)
                if oc is not None:
                    A.plot(oc[2], oc[3], color=C_OCC, lw=1.5, ls=(0, (5, 1.6)),
                           label="Marginal (Laplace)")
            nu = NU.get(nm)
            if nu is not None:
                h_, e_ = np.histogram(nu, bins=40, range=(xf[0], xf[-1]), density=True)
                A.step(0.5 * (e_[1:] + e_[:-1]), h_, where="mid", color=C_NUTS, lw=1.3,
                       label="Marginal (NUTS)")
            # FIXED -3..3 WINDOW for both panels.  The scanned span is adaptive (E_b runs to +5.4
            # sigma, C5A to +5.4) and letting it set the axis squeezed all the structure into the left
            # third.  Everything that matters at 68/90% lives well inside +-3.
            A.set_xlim(-3.0, 3.0)
            if wall is not None:
                # the ENTIRE region below the wall is unphysical, not a token strip
                A.axvspan(-3.0, wall, facecolor="0.5", alpha=0.30, lw=0, zorder=3)
                A.axvline(wall, color="k", lw=0.9, ls="--", zorder=4)
            A.set_ylim(bottom=0)
            A.set_xlabel(r"$(\theta-\hat\theta)\,/\,\sigma$", fontsize=8)
            A.set_ylabel("density", fontsize=8)
            A.legend(fontsize=5.8, loc="upper right", framealpha=0.92, borderpad=0.3,
                     handlelength=1.6, labelspacing=0.25)
            if note and wall is not None:
                A.text(0.5 * (-3.0 + wall), 0.62 * A.get_ylim()[1], note, ha="center",
                       va="center", fontsize=5.8, color="0.15", linespacing=1.25, zorder=7)
            A.set_title(f"({tag})  {title}", fontsize=9.5, loc="left")
            A.tick_params(which="both", top=False, right=False, labelsize=7)

        _note = None
        cE = [c for c, k in enumerate(sub) if pn[k] == "Eb_shift"]
        if cE and e_fit is not None:
            _c0 = cE[0]; _k0 = sub[_c0]; _w = K.phys_lo("Eb_shift") or 0.0
            _on = np.abs(e_fit[:, _c0] - _w) < 1e-6
            _f = float(_on.mean()); _e = np.sqrt(max(_f, 1e-9) * (1 - _f) / len(e_fit))
            _pa = float(stats.norm.cdf((_w - float(truth[_k0])) / float(spost[_c0])))
            _note = f"on boundary\n{100*_f:.1f}$\\pm${100*_e:.1f}%\npred {100*_pa:.1f}%"
            print(f"  panel (c) E_b atom: {100*_f:.1f}+-{100*_e:.1f}%  Chernoff {100*_pa:.1f}%")
        _panel(axC, "Eb_shift", "c", r"at a boundary: $E_b$", note=_note)
        _panel(axD, NONGAUSS_DIAL, "d", f"away from any boundary: {style.plab(NONGAUSS_DIAL)}")

        for a_ in (axB, axC, axD):
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)

        # LABEL-tagged: the study points (P1 = all dials off nominal, P2 = nominal but E_b) are separate
        # figures and must not overwrite each other.  `sec4_ref` keeps the historical filename.
        base = f"{label}_figA"
        # a non-default interval level gets its own file, so the 1-sigma and 2-sigma versions can be
        # compared side by side instead of one silently replacing the other
        style.save(fig, base if abs(MASS - 0.6827) < 1e-6 else f"{base}_{round(100*MASS)}")


if __name__ == "__main__":
    # third arg = interval mass, e.g. 0.9545 for 2 sigma (default 0.6827)
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:3] or []))
