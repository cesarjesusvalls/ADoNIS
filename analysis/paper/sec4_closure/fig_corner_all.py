"""FIGURE B -- the 2-D companion to figure A: Gaussian vs Laplace marginal vs NUTS marginal.

The section's claim is that gradients make it cheap to CHECK the quoted error rather than assume it.  In
1-D that check is fig A; this is the 2-D version, and it puts the three objects on the same panel so the
places they disagree are visible instead of inferred from three separate figures.

They are NOT the same object, and the legend says so:

  * GAUSSIAN   -- the Gauss-Newton curvature at the best fit, (J^T W J + prior)^-1, drawn at the same
                  Delta chi2 levels.  This is what a Minuit/HESSE analysis quotes.
  * PROFILE    -- exact chi2 minimised over the other 14 dials at every grid node: a 2-D CONFIDENCE
                  region, Delta chi2 = 2.30 (68%) / 4.61 (90%).  Frequentist.
  * NUTS       -- the 2-D marginal of the 16-D posterior, integrated (not minimised) over the other 14,
                  drawn at the 68%/90% HPD levels: a 2-D CREDIBLE region.  Bayesian.

Profile and marginal answer different questions and need not coincide -- they differ to Laplace order by
the nuisance-volume (Occam) factor -- so agreement is evidence the posterior is close to Gaussian in the
integrated directions, and disagreement localises where it is not.  Where all three coincide the cheap
Gaussian error is doing its job.

Usage:  python -m analysis.paper.sec4_closure.fig_corner_all [prof_label] [nuts_label]
"""
import os
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.sec4_closure.fig_corner_prof import load_views, snap_axis, view_for
from adonis.analysis import knobs as K

L68, L90 = 2.30, 4.61                  # 2-D Delta-chi2 levels (68% / 90% of a 2-D Gaussian)
C_ARROW = "#186"                       # gradient field (was figure C's colour)
VIEW_SIG = 3.2                         # display window, in sigma_post, for the contour half
# The corner quotes every dial in PHYSICAL units, so the E_b axis is the binding energy itself and the
# "Delta" that style.plab carries (correct wherever the dial is shown as an offset) would misname it
# here.  Local to this figure: style.plab stays as it is for the sections that do show the shift.
_LBL = {"Eb_shift": r"$E_b$"}
def _lab(nm):
    return _LBL.get(nm, style.plab(nm))
C_LAP, C_NUTS, C_GAUS, C_BFP = "#1f4b9c", "#e08214", "#0b8f8f", "#d24"
# ONE style everywhere: the Laplace marginal is the filled blue SURFACE (68% dark, 90% light),
# NUTS is orange LINES (solid 68 / dashed 90) laid over it, and the Gaussian is teal lines in
# the same two styles.  The raw profile is not drawn in the corner -- (c)/(d) of figure A make
# the point that it needs the volume factor, and repeating it here only adds a fourth object.


NB_MAX, NB_MIN, NB_TARGET = 36, 12, 250     # adaptive 2-D binning: see _nbins
# Gaussian smoothing of the NUTS histogram before the HPD level is taken, in grid cells (0 disables).
# UNLIKE the colour background of the gradient figure, this touches a QUANTITATIVE object: the contour
# is a credible region, and smoothing slightly inflates narrow features.  Measured cost is printed at
# render time as the change in enclosed area, and at 1 cell it is well under the sampling error.
NUTS_SMOOTH = float(os.environ.get("NUTS_SMOOTH", "1.0"))     # largest kernel allowed
NUTS_SMOOTH_MAXBIAS = float(os.environ.get("NUTS_SMOOTH_MAXBIAS", "0.10"))   # cap on the area change


def _smooth_hist(H):
    """Gaussian-smooth a 2-D count histogram, but only as far as it stays honest.

    A fixed kernel cannot be right for every panel: 1 cell costs +5-9% of enclosed area on the fat
    uncorrelated pairs (where the contour is genuinely noisy and the smoothing is nearly free) but +30
    to +82% on the near-degenerate ones, whose blade is barely wider than the kernel itself.  So walk
    the kernel down until the 68% area moves by less than NUTS_SMOOTH_MAXBIAS, and report what was used.
    Returns (H_smoothed, sigma_used, area_change).
    """
    a0 = float((H >= hpd_levels(H)[0]).sum())
    if NUTS_SMOOTH <= 0 or a0 <= 0:
        return H, 0.0, 0.0
    from scipy.ndimage import gaussian_filter
    sg = NUTS_SMOOTH
    while sg > 0.05:
        Hs = gaussian_filter(H, sg, mode="constant")
        a1 = float((Hs >= hpd_levels(Hs)[0]).sum())
        if abs(a1 / a0 - 1.0) <= NUTS_SMOOTH_MAXBIAS:
            return Hs, sg, a1 / a0 - 1.0
        sg -= 0.1
    return H, 0.0, 0.0


def _nbins(sx, sy, rng):
    """Bins per axis, chosen so the PEAK bin reaches NB_TARGET counts.

    A fixed 36x36 is fine for the near-degenerate pairs, whose samples pile into a thin blade (peak ~385
    counts, density falling ~160 counts per bin across the narrow direction, so the contour barely moves).
    The uncorrelated pairs spread the same samples over a shallow disc -- peak 77, gradient 9 counts/bin --
    and the HPD contour then wanders by ~0.5 bins.  Contour jitter goes as sqrt(N_bin)/|dN/dx| ~ 1/w, so
    coarsening is the direct fix: measured on E_b x S_Delta it takes the jitter from 0.101 to 0.052 sigma
    while leaving the degenerate panels at 36.
    """
    nb = NB_MAX
    while nb > NB_MIN:
        H, _, _ = np.histogram2d(sx, sy, bins=nb, range=rng)
        if H.max() >= NB_TARGET:
            break
        nb -= 2
    return nb


def hpd_levels(H, fracs=(0.6827, 0.90)):
    """Density levels enclosing `fracs` of the mass -- the 2-D HPD contours of a grid.

    Normalise by the ACTUAL total, not max(total, 1).  That floor was written for integer count
    histograms, where the sum is always >= 1, and it silently broke on the Laplace density, whose
    normalisation is arbitrary: M_A_res x S_Delta sums to 0.25, so the cumulative never reached 0.68,
    searchsorted ran off the end and the level came back 0 -- contour(0) then traced the edge of the
    zero region instead of a credible region.
    """
    tot = float(np.sum(H))
    if not np.isfinite(tot) or tot <= 0:
        return [0.0 for _ in fracs]
    h = np.sort(H.ravel())[::-1]
    cs = np.cumsum(h) / tot
    return [h[min(int(np.searchsorted(cs, f)), len(h) - 1)] for f in fracs]


def _diag(A, nm, dax, dcol, sub, pn, bfp, spost, prof1, U, ncol):
    """One diagonal cell: the SAME three statements as the 2-D panels, in 1-D.

    profile  exp(-Dchi2/2) from the 1-D scan   (minimised over the other 16)
    NUTS     the 1-D marginal histogram        (integrated over the other 16)
    Gaussian N(0,1) in sigma_post units        (the quoted Gauss-Newton curvature)

    All three are normalised to unit AREA over the panel window -- ordinary densities.  Unit PEAK, which
    this used to do, pins every curve to the same height and throws away the very thing the panel is for:
    a narrower distribution should look TALLER, and with peak normalisation the Gaussian's excess width
    was only readable from the tails.  The windows here comfortably contain all three curves, so area
    normalisation is not distorted by truncation.
    """
    from scipy.interpolate import CubicSpline
    cc = dcol[nm]; kk = sub[cc]; aa = dax[nm]
    wall = None
    _lo = K.phys_lo(pn[kk])
    if _lo is not None:
        xb = (_lo - bfp[kk]) / spost[cc]
        if aa[0] - 1e-9 <= xb <= aa[-1] + 1e-9:
            wall = xb
    top = 0.0
    # LAPLACE MARGINAL as the filled surface, same object and same two levels as the 2-D panels
    if nm in prof1 and prof1[nm][2] is not None:
        g_, d_, l_ = prof1[nm]
        xf = np.linspace(max(g_.min(), aa[0]), min(g_.max(), aa[-1]), 800)
        y = np.exp(-0.5 * np.maximum(CubicSpline(g_, d_)(xf), 0.0))
        lv = CubicSpline(g_, l_)(xf)
        y = y * np.exp(0.5 * (lv - lv.max()))
        if y.max() > 0 and np.trapezoid(y, xf) > 0:
            y = y / np.trapezoid(y, xf); top = max(top, y.max())
            o = np.argsort(y)[::-1]; cum = np.cumsum(y[o]) * (xf[1] - xf[0]); cum /= cum[-1]
            for frac, al in ((0.90, 0.22), (0.6827, 0.45)):
                thr = y[o][min(int(np.searchsorted(cum, frac)), len(o) - 1)]
                A.fill_between(xf, 0, y, where=(y >= thr), color=C_LAP, alpha=al, lw=0)
            A.plot(xf, y, color=C_LAP, lw=1.2)
    if nm in ncol:                                    # NUTS marginal, orange
        s_ = U[:, ncol[nm]]
        # SMOOTHED, like the 2-D panels.  A 40-bin step histogram of 24k samples is mostly sampling
        # noise at this panel size, and it read as structure next to the smooth Laplace curve it is
        # meant to be compared with.  Finer bins + a narrow Gaussian: the same information, without the
        # staircase.  The kernel is small (1.5 cells of a 120-bin grid = 1/80 of the range), so the
        # 68% mass moves by well under a percent -- this smooths the noise, not the distribution.
        h, e = np.histogram(s_, bins=120, range=(aa[0], aa[-1]), density=True)
        if h.max() > 0:
            from scipy.ndimage import gaussian_filter1d
            c_ = 0.5 * (e[1:] + e[:-1])
            hs = gaussian_filter1d(h, 1.5, mode="nearest")
            A.plot(c_, hs, color=C_NUTS, lw=1.5); top = max(top, hs.max())
    gx = np.linspace(aa[0], aa[-1], 300)              # Gaussian, teal
    gy = np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi)
    A.plot(gx, gy, color=C_GAUS, lw=1.3, ls="--"); top = max(top, gy.max())
    if wall is not None:
        span = aa[-1] - aa[0]
        A.axvspan(wall - 0.04 * span, wall, facecolor="0.5", alpha=0.30, lw=0, zorder=3)
        A.axvline(wall, color="k", lw=0.9, ls="--", zorder=4)
    A.set_ylim(0, 1.18 * max(top, 1e-9))


def _load_grad(label):
    """The gradient2d shards, keyed by dial-index pair.  Returns {} when the run has none, so the
    figure degrades to the plain lower-triangle corner rather than failing."""
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_corner2d_grad_*.npz")))
    if not fs:
        return {}, None
    Z = [np.load(f, allow_pickle=True) for f in fs]
    G = {}
    for z in Z:
        for i, (a, b) in enumerate(np.asarray(z["pair_idx"])):
            G[(int(a), int(b))] = (np.asarray(z["axes_phys"])[i], np.asarray(z["dchi2"])[i],
                                   np.asarray(z["gn_step"])[i])
    return G, Z[0]


def _grad_panel(A, G, i, j, sub, pos, pn, bfp, crop=None, xbot=True, yleft=True):
    """Gauss-Newton field for pair (i, j): x = dial i, y = dial j, PHYSICAL units, full allowed range.

    A copy of fig_corner_grad's methodology -- same key, same smoothing, same normalisation, same
    subsampling -- because this half of the figure is that figure.

    `crop` = (x0, x1, y0, y1) in physical units: the window the contour panel opposite it shows.  Drawn
    as a rectangle, it is the point of putting the two halves together -- it says how small the region
    the data constrains is inside the range the dial is allowed to take.
    """
    (axa, axb), d, F = G[(i, j)]
    _c = np.log10(np.maximum(d, 1e-3))
    if np.isfinite(_c).all():
        from scipy.ndimage import gaussian_filter
        _c = gaussian_filter(_c, 1.0, mode="nearest")      # colour only; the arrows are untouched
    A.imshow(np.ma.masked_invalid(_c).T, cmap="Greys", origin="lower", aspect="auto",
             extent=(float(axa[0]), float(axa[-1]), float(axb[0]), float(axb[-1])),
             interpolation="bicubic", rasterized=True, alpha=0.75, zorder=0)
    X, Y = np.meshgrid(axa, axb, indexing="ij")
    U, V = F[..., 0], F[..., 1]
    n = np.hypot(U, V); n = np.where(n > 0, n, 1.0)
    s_ = max(1, len(axa) // 9)
    A.quiver(X[::s_, ::s_], Y[::s_, ::s_], (U / n)[::s_, ::s_], (V / n)[::s_, ::s_],
             angles="xy", scale=17, width=0.008, color=C_ARROW, alpha=0.9, zorder=2)
    if crop is not None:
        x0, x1, y0, y1 = crop
        A.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=C_LAP, lw=1.2,
                                  zorder=6))
    A.plot(bfp[sub[pos[i]]], bfp[sub[pos[j]]], "*", color=C_BFP, ms=10, mec="white", mew=0.6, zorder=8)
    A.set_xlim(axa[0], axa[-1]); A.set_ylim(axb[0], axb[-1])
    A.tick_params(labelsize=8, top=False, right=False, labelbottom=xbot, labelleft=yleft, length=2)
    from matplotlib.ticker import MaxNLocator
    A.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
    A.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
    if xbot:
        A.set_xlabel(_lab(pn[sub[pos[i]]]), fontsize=11)
    if yleft:
        A.set_ylabel(_lab(pn[sub[pos[j]]]), fontsize=11)


def main(label="sec4_A", nuts_label="sec4_B", allow_partial=False):
    style.use()
    views, meta = load_views(label, allow_partial)
    pn, sub, bfp, spost, V0, dials = (meta["pn"], meta["sub"], meta["bfp"], meta["spost"],
                                      meta["V0"], meta["dials"])

    CROP = {}
    GRAD, gz = _load_grad(label)
    gpos = None
    if GRAD:
        gd = [str(x) for x in gz["dials"]]
        gsel = [int(q) for q in gz["sel_pos"]]
        # the grad run's dial ORDER need not match the corner's; index by NAME, never by position
        try:
            gidx = [gd.index(d_) for d_ in dials]
            GRAD = {(gidx.index(i), gidx.index(j)) if False else (ii, jj): GRAD[(gidx[ii], gidx[jj])]
                    for ii in range(len(dials)) for jj in range(len(dials))
                    if (gidx[ii], gidx[jj]) in GRAD for (ii, jj) in [(ii, jj)]}
            gpos = [gsel[k] for k in gidx]
        except ValueError:
            print("[warn] gradient run covers different dials; upper triangle left empty")
            GRAD = {}
    if GRAD:
        print(f"[grad] upper triangle: {len(GRAD)} panel(s)")

    fs = sorted(glob.glob(str(style.ALTGEN / f"{nuts_label}_nutsown_*.npz")))
    if not fs:
        raise SystemExit(f"no NUTS chains for {nuts_label}")
    Zn = [np.load(f, allow_pickle=True) for f in fs]
    nmin = min(len(np.asarray(z["u"])) for z in Zn)
    U = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])      # (nchain*n, ndial), sigma units
    # The chains are indexed by SUBSET POSITION, same as sigma_post -- map dial name -> that position via
    # the chain's own subset, never by assuming it matches the corner file's ordering.
    npn = [str(x) for x in Zn[0]["pnames"]]; nsub = [int(k) for k in Zn[0]["subset"]]
    ncol = {npn[k]: c for c, k in enumerate(nsub)}
    print(f"[nuts] {len(fs)} chains x {nmin} = {len(U)} samples; "
          f"{sum(nm in ncol for nm in dials)}/{len(dials)} dials matched")

    nd = len(dials)

    # ---- per-dial axis + column position, ONCE ---------------------------------------------------- #
    # A corner only reads as one if column k has a single x-axis in every row.  Deriving the limits
    # per panel (as the pairs-only version did) is safe when each dial appears with one span, but it
    # cannot align a column against the 1-D panel on the diagonal, which has no partner to inherit from.
    dax, dcol = {}, {}
    for (ni, nj), w in views.items():
        for nm, ax_, cc in ((ni, w["axi"], w["ci"]), (nj, w["axj"], w["cj"])):
            if nm not in dax or len(ax_) > len(dax[nm]):
                dax[nm] = np.asarray(ax_, float); dcol[nm] = cc
    for nm in dax:
        dax[nm] = snap_axis(dax[nm], sub[dcol[nm]], dcol[nm], pn, bfp, spost)

    def _lim(nm):
        """x/y limits for a dial: its scanned span, widened to show the wall band when there is one."""
        aa = dax[nm]; cc = dcol[nm]; kk = sub[cc]; span = aa[-1] - aa[0]
        # CLAMP to +-VIEW_SIG.  The 1-D profile scan is adaptive and reaches until the density decays,
        # which for M_A_res is -6.5 sigma; letting that set the axis pushes every contour into a corner
        # and drags the physical tick labels far below anything the fit supports.  The data is unchanged
        # -- this is the window we look through.
        lo_, hi_ = max(aa[0], -VIEW_SIG), min(aa[-1], VIEW_SIG)
        for _b, side in ((K.phys_lo(pn[kk]), -1), (K.phys_hi(pn[kk]), +1)):
            if _b is None: continue
            xb = (_b - bfp[kk]) / spost[cc]
            if not (aa[0] - 1e-9 <= xb <= aa[-1] + 1e-9): continue
            lo_ = min(lo_, xb - 0.04 * span) if side < 0 else lo_
            hi_ = max(hi_, xb + 0.04 * span) if side > 0 else hi_
        return lo_, hi_

    # 1-D PROFILE for the diagonal: the SAME scan Fig A uses, so the diagonal of the corner and the
    # bars in Fig A are the same object rather than two independent calculations of "the profile".
    prof1 = {}
    _pf = style.ALTGEN / f"{label}_profile.npz"
    if _pf.exists():
        _z = np.load(_pf, allow_pickle=True)
        _gs = np.asarray(_z["grids_sigma"]); _pd = np.asarray(_z["prof_dobj"])
        _sb = [int(k) for k in _z["subset"]]; _pnp = [str(x) for x in _z["pnames"]]
        _ldp = np.asarray(_z["logdet_Vnuis"]) if "logdet_Vnuis" in _z.files else None
        for c_, k_ in enumerate(_sb):
            ok_ = np.isfinite(_gs[c_]) & np.isfinite(_pd[c_])
            if _ldp is not None:
                ok_ = ok_ & np.isfinite(_ldp[c_])
            if ok_.sum() > 3:
                prof1[_pnp[k_]] = (_gs[c_][ok_], _pd[c_][ok_],
                                   None if _ldp is None else _ldp[c_][ok_])
    else:
        print(f"[warn] {_pf.name} missing -- diagonal will show NUTS + Gaussian only")

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        # nd x nd, NOT sharey: the diagonal's y is a density, the off-diagonal's y is a dial, so a shared
        # row axis would force one onto the other.  Columns are aligned by explicit set_xlim instead.
        fig, axes = plt.subplots(nd, nd, figsize=(max(4.0, 1.74 * nd), max(3.9, 1.74 * nd)),
                                 squeeze=False)
        for a in range(nd):
            for b in range(nd):
                A = axes[a, b]
                if b < a:
                    # LOWER TRIANGLE: the gradient field over the FULL physical range, in figure C's
                    # own orientation (x = dial b, y = dial a) so the drawing code is a copy rather
                    # than a transposition.  Mirroring a vector field means swapping its components as
                    # well as its axes, and getting that half-right silently rotates every arrow.
                    # CROP is keyed by the CONTOUR panel's orientation, which is this panel's
                    # transpose: the contour at (row b, col a) stores (x=dial a, y=dial b) while this
                    # panel is (x=dial b, y=dial a).  Swap the pairs, or the rectangle comes out
                    # rotated -- and being a rectangle it would look perfectly plausible.
                    _cr = CROP.get((a, b))
                    _cr = (_cr[2], _cr[3], _cr[0], _cr[1]) if _cr else None
                    if (b, a) in GRAD:
                        _grad_panel(A, GRAD, b, a, sub, gpos, pn, bfp, crop=_cr,
                                    xbot=(a == nd - 1), yleft=(b == 0))
                    else:
                        A.axis("off")
                    continue
                if b == a:
                    _diag(A, dials[a], dax, dcol, sub, pn, bfp, spost, prof1, U, ncol)
                    A.set_xlim(*_lim(dials[a]))
                    # the dial NAME lives on the diagonal, so neither triangle has to carry it twice
                    _side = "left" if a < nd // 2 else "right"
                    A.text(0.04 if _side == "left" else 0.96, 0.94, _lab(dials[a]),
                           transform=A.transAxes, ha=_side, va="top", fontsize=13)
                    A.tick_params(labelsize=8, top=False, right=False, left=False, labelleft=False)
                    # no x ticks or label on ANY diagonal: the name is in the corner, and the bottom
                    # row's scale belongs to the gradient panels beside it, which are a different range
                    A.tick_params(labelbottom=False)
                    continue
                def _phys_ticks(axis, k_, c_, lo_s, hi_s, n=3):
                    """Ticks at ROUND PHYSICAL values, placed at their sigma positions.

                    Locating on the sigma axis and relabelling gave round sigma values and therefore
                    arbitrary physical ones (0.53, 1.27, 1.92).  Choosing the numbers in physical space
                    first and mapping them back puts them where a reader expects.
                    """
                    from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
                    p0, p1 = bfp[k_] + lo_s * spost[c_], bfp[k_] + hi_s * spost[c_]
                    vals = [v for v in MaxNLocator(n).tick_values(p0, p1) if p0 <= v <= p1]
                    axis.set_major_locator(FixedLocator([(v - bfp[k_]) / spost[c_] for v in vals]))
                    axis.set_major_formatter(FixedFormatter([f"{v:g}" for v in vals]))

                i, j = b, a
                w = view_for(views, dials[i], dials[j])
                if w is None:
                    A.axis("off"); continue
                ci, cj = w["ci"], w["cj"]
                ki, kj = sub[ci], sub[cj]
                axi = snap_axis(w["axi"], ki, ci, pn, bfp, spost)
                axj = snap_axis(w["axj"], kj, cj, pn, bfp, spost)
                # THE DRAWING STAYS IN SIGMA.  Everything that lands on these axes -- the NUTS 2-D
                # histogram, the Laplace surface, the Gaussian ellipse -- is built in sigma about the
                # BFP, so converting the axis arrays alone left the data at sigma coordinates on
                # physical axes and threw most of it off the panel.  Physical units are applied as a
                # tick FORMATTER instead: the numbers read physical, the recipe is untouched.
                # from the DISPLAYED limits, not the scanned span: the axes are clamped to +-VIEW_SIG,
                # so a rectangle built from aa[0]/aa[-1] would outline a window the panel never shows.
                _cx, _cy = _lim(dials[i]), _lim(dials[j])
                CROP[(i, j)] = (float(bfp[ki] + _cx[0] * spost[ci]),
                                float(bfp[ki] + _cx[1] * spost[ci]),
                                float(bfp[kj] + _cy[0] * spost[cj]),
                                float(bfp[kj] + _cy[1] * spost[cj]))
                X, Y = np.meshgrid(axi, axj, indexing="ij")
                d = np.array(w["d"], float)
                for which, (kk, cc, aa) in (("x", (ki, ci, axi)), ("y", (kj, cj, axj))):
                    _lo = K.phys_lo(pn[kk])
                    if _lo is None: continue
                    _xb = (_lo - bfp[kk]) / spost[cc] - 1e-2 * abs(aa[1] - aa[0])
                    if which == "x": d[X < _xb] = np.nan
                    else: d[Y < _xb] = np.nan
                if not np.isfinite(d).any():
                    A.axis("off"); continue
                d = d - np.nanmin(d)

                # ---- LAPLACE MARGINAL: filled surface at its 68/90% HPD levels ---------------------
                _ld = w.get("ld")
                if _ld is not None and np.isfinite(_ld).mean() > 0.99:
                    dm = np.exp(-0.5 * np.maximum(d, 0.0)) * np.exp(0.5 * (_ld - np.nanmax(_ld)))
                    dm = np.where(np.isfinite(dm), dm, 0.0)
                    l68, l90 = hpd_levels(dm)
                    if dm.max() > 0 and l90 < l68:
                        A.contourf(X, Y, dm, levels=[l90, l68], colors=[C_LAP], alpha=0.22)
                        A.contourf(X, Y, dm, levels=[l68, dm.max()], colors=[C_LAP], alpha=0.45)

                # ---- NUTS: orange lines over it ----------------------------------------------------
                if dials[i] in ncol and dials[j] in ncol:
                    sx, sy = U[:, ncol[dials[i]]], U[:, ncol[dials[j]]]
                    # bin over the SAME window as the profile so the two are read on one footing
                    rng = [[axi[0], axi[-1]], [axj[0], axj[-1]]]
                    H, xe, ye = np.histogram2d(sx, sy, bins=_nbins(sx, sy, rng), range=rng)
                    if H.sum() > 0:
                        lv = hpd_levels(H)
                        Xc = 0.5 * (xe[1:] + xe[:-1]); Yc = 0.5 * (ye[1:] + ye[:-1])
                        # levels must increase for contour(); HPD levels come out descending
                        H, _sg, _db = _smooth_hist(H)
                        lv = hpd_levels(H)
                        print(f"[smooth] {dials[i]} x {dials[j]}: sigma={_sg:.1f} cells, "
                              f"68% area {100*_db:+.1f}%")
                        _u = sorted(set(lv))
                        A.contour(Xc, Yc, H.T, levels=_u, colors=C_NUTS,
                                  linewidths=[1.1, 1.6][:len(_u)],
                                  linestyles=["--", "-"][:len(_u)])
                        frac = np.mean((sx >= axi[0]) & (sx <= axi[-1])
                                       & (sy >= axj[0]) & (sy <= axj[-1]))
                        if frac < 0.99:      # never let a clipped marginal masquerade as the whole thing
                            A.text(0.03, 0.95, f"{100*(1-frac):.0f}% off-panel", transform=A.transAxes,
                                   fontsize=5.5, color=C_NUTS, va="top")

                # ---- GAUSSIAN: the quoted Gauss-Newton curvature ------------------------------------
                Vp = V0[np.ix_([ci, cj], [ci, cj])]
                Dp = np.diag(1.0 / np.array([spost[ci], spost[cj]]))
                Rp = Dp @ Vp @ Dp
                ev, evec = np.linalg.eigh(np.linalg.inv(Rp))
                tt = np.linspace(0, 2 * np.pi, 361)
                for lev, ls, lw in ((L68, "-", 1.4), (L90, "--", 1.0)):
                    r = np.stack([np.cos(tt) / np.sqrt(ev[0]), np.sin(tt) / np.sqrt(ev[1])]) * np.sqrt(lev)
                    xy = evec @ r
                    A.plot(xy[0], xy[1], ls, color=C_GAUS, lw=lw, zorder=6)

                # ---- walls, best fit, limits --------------------------------------------------------
                # the wall BAND and line; the panel limits themselves now come from _lim() so that a
                # column shares one x-axis with the 1-D panel on the diagonal
                for which, (kk, cc, aa) in (("x", (ki, ci, axi)), ("y", (kj, cj, axj))):
                    span = aa[-1] - aa[0]
                    for _b, side in ((K.phys_lo(pn[kk]), -1), (K.phys_hi(pn[kk]), +1)):
                        if _b is None: continue
                        xb = (_b - bfp[kk]) / spost[cc]
                        if not (aa[0] - 1e-9 <= xb <= aa[-1] + 1e-9): continue
                        e = xb - 0.04 * span if side < 0 else xb + 0.04 * span
                        (A.axvspan if which == "x" else A.axhspan)(
                            e if side < 0 else xb, xb if side < 0 else e,
                            facecolor="0.5", alpha=0.30, zorder=3, lw=0)
                        (A.axvline if which == "x" else A.axhline)(xb, color="k", lw=0.9, ls="--",
                                                                   zorder=4)
                A.plot(0, 0, "*", color=C_BFP, ms=10, mec="white", mew=0.6, zorder=8)
                # limits come from the shared per-dial spans, so every panel in a column has one x-axis
                # and every panel in a row has one y-axis -- the property that makes a corner readable
                A.set_xlim(*_lim(dials[i])); A.set_ylim(*_lim(dials[j]))
                # PHYSICAL numbers on sigma axes: the formatter converts at draw time, so the data and
                # every contour recipe stay in the units they were built in.
                _xl, _yl = _lim(dials[i]), _lim(dials[j])
                _phys_ticks(A.xaxis, ki, ci, *_xl)
                _phys_ticks(A.yaxis, kj, cj, *_yl)
                # this half sits ABOVE the diagonal, so its outer edge is the top row / right column
                A.tick_params(labelsize=8, top=True, right=True, bottom=False, left=False,
                              labelbottom=False, labelleft=False,
                              labeltop=(a == 0), labelright=(b == nd - 1), length=2)
                if a == 0:
                    A.set_xlabel(_lab(dials[i]), fontsize=11); A.xaxis.set_label_position("top")
                if b == nd - 1:
                    A.set_ylabel(_lab(dials[j]), fontsize=11); A.yaxis.set_label_position("right")

        # SHORT labels, placed inside the empty upper-right block.  What each object IS belongs in
        # the caption; the legend only has to let a reader tell the three apart on the panel.
        h = [plt.Rectangle((0, 0), 1, 1, fc=C_LAP, alpha=0.45),
             plt.Line2D([], [], color=C_NUTS, lw=1.6),
             plt.Line2D([], [], color=C_GAUS, lw=1.4),
             plt.Line2D([], [], color=C_BFP, marker="*", ls="", ms=11)]
        # ABOVE the grid, horizontal.  The upper-right corner used to be empty in a corner plot and
        # held this legend; the gradient panels now live there, so it has to come out.
        fig.legend(h, ["Laplace", "NUTS", "Gaussian", "BFP"],
                   loc="upper center", ncol=4, fontsize=8.5, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handletextpad=0.4, columnspacing=1.6)
        # what each half of the figure is
        fig.tight_layout(rect=(0, 0, 1, 0.968))
        fig.subplots_adjust(hspace=0.10, wspace=0.10)
        style.save(fig, f"{label}_figB")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    # --allow-partial: draw from a shard set the merge has judged incomplete.  Marked in the log, and
    # never the default: a downgraded panel is indistinguishable from a healthy one by eye.
    main(*(p[:2] or ["sec4_A"]), allow_partial="--allow-partial" in sys.argv)
