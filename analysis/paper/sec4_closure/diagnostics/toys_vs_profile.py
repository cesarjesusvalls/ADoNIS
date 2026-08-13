"""Toy distribution vs the profile likelihood, per dial (sec4_N study point)."""
import os, sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from scipy.interpolate import CubicSpline
style.use()

HALF = 0.3413447          # 34.1% each side -> 68.27% (one Gaussian sigma)
HALF90 = 0.45             # 45.0% each side -> 90% (1.645 sigma)


def hpd_density(xf, d, mass):
    """Water-fill: lower a horizontal threshold from the peak until the enclosed area reaches `mass`.
    Returns (lo, hi, disjoint) -- the extent of {d >= thr}, and whether that region is disconnected.

    Unlike "x% each side of the best fit", this always EXISTS.  On a skewed profile the best fit can sit
    far from the median (measured: 65.2% of M_A_res's density lies below it), so a symmetric-mass 90%
    interval about it is impossible and a naive implementation silently clips to the edge of the scan.
    """
    o = np.argsort(d)[::-1]
    cum = np.cumsum(d[o]) * (xf[1] - xf[0]); cum /= cum[-1]
    thr = d[o][min(int(np.searchsorted(cum, mass)), len(o) - 1)]
    sel = d >= thr
    idx = np.where(sel)[0]
    disjoint = bool(np.any(np.diff(idx) > 1))
    return float(xf[idx[0]]), float(xf[idx[-1]]), disjoint


def hpd_sample(v, mass):
    """Sample analogue: the SHORTEST interval containing `mass` of the toys."""
    t = np.sort(np.asarray(v)); n = t.size; w = max(int(np.ceil(mass * n)), 2)
    if w >= n: return float(t[0]), float(t[-1])
    k = int(np.argmin(t[w:] - t[:n - w]))
    return float(t[k]), float(t[k + w])


def half68_sample(v, anchor=0.0, half=HALF):
    """`half` of the TOYS on each side of `anchor` (the best fit).  Third value flags a side that does
    not HOLD that much mass -- clipping there would silently return the sample's extreme."""
    t = np.sort(np.asarray(v)); c0 = np.searchsorted(t, anchor) / len(t)
    short = (c0 < half) or (1 - c0 < half)
    return (float(np.quantile(t, max(c0 - half, 0.0))),
            float(np.quantile(t, min(c0 + half, 1.0))), short)


def half68_density(xf, d, anchor=0.0, half=HALF):
    """`half` of exp(-dchi2/2) on each side of `anchor` -- the SAME construction as half68_sample, so the
    profile and the toys are anchored at the same point and are directly comparable (an HPD is anchored at
    the mode and a percentile at the median, so neither would be)."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(xf))]); cdf /= cdf[-1]
    c0 = float(np.interp(anchor, xf, cdf))
    short = (c0 < half) or (1 - c0 < half)
    return (float(np.interp(max(c0 - half, 0.0), cdf, xf)),
            float(np.interp(min(c0 + half, 1.0), cdf, xf)), short)

LAB = sys.argv[1] if len(sys.argv) > 1 else "sec4_N"
ENS = sys.argv[2] if len(sys.argv) > 2 else "sec4_N_ens"

z = np.load(style.ALTGEN / f"{LAB}.npz", allow_pickle=True)
pn = [str(x) for x in z["pnames"]]; sub = [int(k) for k in z["subset"]]
sp = np.sqrt(np.abs(np.diag(np.asarray(z["fit_V"])))); names = [pn[k] for k in sub]

zp = np.load(style.ALTGEN / f"{LAB}_profile.npz", allow_pickle=True)
prof = np.asarray(zp["prof_dobj"]); gs = np.asarray(zp["grids_sigma"]); spp = np.asarray(zp["sigma_post"])
# NUISANCE VOLUME.  Profiling MAXIMISES over the other 16 dials, marginalising INTEGRATES over them; to
# Laplace order the two differ by sqrt(det V_nuis(theta_k)), which multisample_profile already records at
# every scan node.  Drawing profile x Occam next to the raw profile turns "profile and marginal disagree"
# into a statement that can be checked rather than asserted.
LD = np.asarray(zp["logdet_Vnuis"]) if "logdet_Vnuis" in zp.files else None

# NUTS 1-D marginals on the SAME axis.  Third object in the comparison: the toys are a SAMPLING
# distribution, the profile a likelihood ratio, and this a POSTERIOR marginal -- they need not agree,
# and where they do not is the point.  Samples are already in sigma_post units about the BFP.
NUTS = sorted(glob.glob(str(style.ALTGEN / f"{LAB}_nutsown_*.npz")))
UU = None
if NUTS:
    Zn = [np.load(f, allow_pickle=True) for f in NUTS]
    nmin = min(len(np.asarray(z["u"])) for z in Zn)
    UU = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])
    _np_ = [str(x) for x in Zn[0]["pnames"]]; _ns_ = [int(k) for k in Zn[0]["subset"]]
    NCOL = {_np_[k]: c for c, k in enumerate(_ns_)}
    print(f"{len(NUTS)} NUTS chains x {nmin} = {len(UU)} samples")

_ef = [f for f in sorted(glob.glob(str(style.ALTGEN / f"{ENS}_*.npz"))) if "_conv" not in f]
E = [np.load(f, allow_pickle=True) for f in _ef]
th = np.concatenate([e["th_fit"] for e in E]); st = np.concatenate([e["th_star"] for e in E])
sg = np.concatenate([e["sig_fit"] for e in E])
# SEEDS, reconstructed the same way the ensemble assigned them: seed = <file base> + row.  The npz do
# not store them, so this is the only join back to the per-fit convergence record scraped from the logs.
seeds = np.concatenate([int(f.rsplit("_", 1)[1].split(".")[0]) + np.arange(len(e["th_fit"]))
                        for f, e in zip(_ef, E)])
GAPCUT = float(os.environ.get("TOY_GAPCUT", "0") or 0)
keep = np.ones(len(th), bool)
if GAPCUT > 0:
    _cf = style.ALTGEN / f"{ENS}_conv.npz"
    zc = np.load(_cf, allow_pickle=True)
    m = {int(a): (int(b), float(c)) for a, b, c in zip(zc["seed"], zc["nfev"], zc["gap"])}
    # DROP a fit if it exhausted the iteration budget OR still predicts more than GAPCUT of chi2 left.
    # gap (the Newton decrement) is the criterion that matters: 4% of fits hit nfev=200 yet leave <0.03,
    # while some early exits leave 6.6 -- so the iteration count alone is the wrong filter.
    keep = np.array([(s_ in m) and (m[s_][0] < 200) and (m[s_][1] <= GAPCUT) for s_ in seeds])
    print(f"convergence cut gap<={GAPCUT:g} and nfev<200: keeping {keep.sum()}/{len(keep)} toys "
          f"({100*keep.mean():.1f}%)")
# DROP TOYS PARKED ON THE E_b WALL.  A clamped E_b cannot absorb its share of the noise, so the other
# 16 dials re-optimise around a frozen value -- the censoring is not confined to E_b's own marginal.
# Dropping the whole toy (not just its E_b entry) is the only way to ask what the ensemble looks like
# when every dial was free to move; it also removes the point mass that no continuous density can match.
if os.environ.get("TOY_DROP_EBWALL", "") == "1":
    from adonis.analysis import knobs as _K
    _ce = names.index("Eb_shift"); _w = _K.phys_lo("Eb_shift") or 0.0
    _on = np.abs(th[:, _ce] - _w) < 1e-6
    print(f"E_b-wall cut: dropping {_on.sum()} toys on the boundary ({100*_on.mean():.1f}%)")
    keep = keep & (~_on)
th_all = th.copy()
th, st, sg = th[keep], st[keep], sg[keep]
print(f"{len(th)} toys, {len(names)} dials")

ROWS = []
nd = len(names); nc = 4; nr = int(np.ceil(nd / nc))
fig, ax = plt.subplots(nr, nc, figsize=(3.9 * nc, 2.5 * nr))
for c, nm in enumerate(names):
    A = ax.flat[c]
    v = (th[:, c] - st[:, c]) / spp[c]              # toy MLE minus ITS OWN truth, in sigma_post
    A.hist(v, bins=40, density=True, histtype="stepfilled", color="#2b6ea8", alpha=.45,
           label=f"toys ({len(v)})")
    A.hist(v, bins=40, density=True, histtype="step", lw=1.2, color="#14406b")
    g = gs[c]; ok = np.isfinite(g) & np.isfinite(prof[c])
    xf = np.linspace(g[ok].min(), g[ok].max(), 800)
    d = np.exp(-0.5 * np.maximum(CubicSpline(g[ok], prof[c][ok])(xf), 0)); d /= np.trapezoid(d, xf)
    A.plot(xf, d, color="#c33", lw=1.7, label=r"profile $e^{-\Delta\chi^2/2}$")
    gx = np.linspace(-4, 4, 400)
    A.plot(gx, np.exp(-0.5 * gx**2) / np.sqrt(2 * np.pi), color="k", lw=1.0, ls="--",
           label="quadratic (Gaussian)")
    nl = nu = None
    if UU is not None and nm in NCOL:
        un = UU[:, NCOL[nm]]
        A.hist(un, bins=60, density=True, histtype="step", lw=1.4, color="#e08214",
               label="NUTS marginal")
        nl, nu = hpd_sample(un, 0.6827); n9l, n9u = hpd_sample(un, 0.90)
    ol = ou = o9l = o9u = None
    if LD is not None and np.isfinite(LD[c][ok]).sum() > 4:
        _m = ok & np.isfinite(LD[c])
        lv = CubicSpline(g[_m], LD[c][_m])(xf)
        do = d * np.exp(0.5 * (lv - np.nanmax(lv)))
        if np.trapezoid(do, xf) > 0:
            do = do / np.trapezoid(do, xf)
            A.plot(xf, do, color="#6a3d9a", lw=1.5, ls=(0, (5, 1.6)),
                   label="profile x nuisance volume")
            ol, ou, _ = hpd_density(xf, do, 0.6827); o9l, o9u, _ = hpd_density(xf, do, 0.90)
    # 34.1% EACH SIDE OF THE BEST FIT, for both objects, drawn on a common anchor (x=0 is the BFP).
    # WATER-FILL (HPD) for BOTH: lower a level until the enclosed mass reaches the target.  Unlike
    # "x% each side of the fit" this always exists -- that construction needs 45% on each side, which
    # M_A_res and delta_strength do not have (their profiles are strongly one-sided).
    tl, tu = hpd_sample(v, 0.6827);  pl, pu, d1 = hpd_density(xf, d, 0.6827)
    t9l, t9u = hpd_sample(v, 0.90);  p9l, p9u, d2 = hpd_density(xf, d, 0.90)
    s90t = s90p = False; dj68 = d1; dj90 = d2
    ytop = A.get_ylim()[1] if A.get_ylim()[1] > 0 else max(d.max(), 1e-9)
    # 90% drawn thin UNDER the thick 68% bar, both anchored at the best fit (x=0)
    if not s90t:
        A.errorbar([0.0], [0.86 * ytop], xerr=[[-t9l], [t9u]], fmt="none", ecolor="#14406b",
                   capsize=2, elinewidth=0.9, alpha=0.85, zorder=5)
    A.errorbar([0.0], [0.86 * ytop], xerr=[[-tl], [tu]], fmt="none", ecolor="#14406b",
               capsize=3, elinewidth=2.4, zorder=6)
    if not s90p:
        A.errorbar([0.0], [0.74 * ytop], xerr=[[-p9l], [p9u]], fmt="none", ecolor="#c33",
                   capsize=2, elinewidth=0.9, alpha=0.85, zorder=5)
    if dj68 or dj90:
        A.text(0.98, 0.62, "disjoint", transform=A.transAxes, fontsize=5.5,
               ha="right", va="top", color="0.35")
    A.errorbar([0.0], [0.74 * ytop], xerr=[[-pl], [pu]], fmt="none", ecolor="#c33",
               capsize=3, elinewidth=2.4, zorder=6)
    if ol is not None:
        A.errorbar([0.0], [0.62 * ytop], xerr=[[-o9l], [o9u]], fmt="none", ecolor="#6a3d9a",
                   capsize=2, elinewidth=0.9, alpha=0.85, zorder=5)
        A.errorbar([0.0], [0.62 * ytop], xerr=[[-ol], [ou]], fmt="none", ecolor="#6a3d9a",
                   capsize=3, elinewidth=2.4, zorder=6)
    if nl is not None:
        A.errorbar([0.0], [0.50 * ytop], xerr=[[-n9l], [n9u]], fmt="none", ecolor="#e08214",
                   capsize=2, elinewidth=0.9, alpha=0.85, zorder=5)
        A.errorbar([0.0], [0.50 * ytop], xerr=[[-nl], [nu]], fmt="none", ecolor="#e08214",
                   capsize=3, elinewidth=2.4, zorder=6)
    A.plot([0, 0], [0.48 * ytop, 0.88 * ytop], color="k", lw=0.8, zorder=7)
    ROWS.append((nm, tl, tu, pl, pu, nl, nu, ol, ou))
    A.axvline(0, color="k", lw=.8, ls=":")
    lo_, hi_ = np.percentile(v, [0.2, 99.8]); m_ = d > 1e-3 * d.max()
    if m_.any(): lo_ = min(lo_, xf[m_].min()); hi_ = max(hi_, xf[m_].max())
    pad = .06 * (hi_ - lo_); A.set_xlim(lo_ - pad, hi_ + pad)
    A.set_title(f"{style.plab(nm)}   pull sd={np.nanstd((th[:,c]-st[:,c])/np.where(sg[:,c]>0,sg[:,c],np.nan)):.3f}",
                fontsize=8.5)
    A.tick_params(labelsize=7)
    if c == 0: A.legend(fontsize=6.5, frameon=False)
for j in range(nd, nr * nc): ax.flat[j].axis("off")
fig.suptitle(f"Toys vs profile vs quadratic  ({LAB}: {len(names)} dials, nominal point, 20% priors, $E_b$ free)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97)); style.save(fig, f"{LAB}_toys_vs_profile_nuts" + ("_conv" if GAPCUT > 0 else "")
           + ("_noebwall" if os.environ.get("TOY_DROP_EBWALL", "") == "1" else ""))
print(f"\nHPD / water-fill to 68% and 90%, sigma_post units (toys = shortest interval holding that mass)")
print("68% water-fill, sigma_post units")
print(f"{'dial':>20} {'toys':>18} {'profile':>18} {'prof x Occam':>18} {'NUTS':>18}")
for nm, tl, tu, pl, pu, nl, nu, ol, ou in ROWS:
    f_ = lambda a, b: (f"[{a:+7.3f},{b:+7.3f}]" if a is not None else " " * 18)
    print(f"{nm:>20} {f_(tl,tu)} {f_(pl,pu)} {f_(ol,ou)} {f_(nl,nu)}")
