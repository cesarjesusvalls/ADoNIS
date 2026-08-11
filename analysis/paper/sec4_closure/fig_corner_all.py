"""Figure D2 -- the three 2-D uncertainty statements on one set of axes: NUTS, profile, Gaussian.

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
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from analysis.paper.sec4_closure.fig_corner_prof import load_views, snap_axis
from adonis.analysis import knobs as K

L68, L90 = 2.30, 4.61                  # 2-D Delta-chi2 levels (68% / 90% of a 2-D Gaussian)
C_PROF, C_NUTS, C_GAUS, C_BFP = "#1f4b9c", "#e08214", "#d81b8c", "#d24"


def hpd_levels(H, fracs=(0.6827, 0.90)):
    """Density levels enclosing `fracs` of the samples -- the 2-D HPD contours of a histogram."""
    h = np.sort(H.ravel())[::-1]
    cs = np.cumsum(h) / max(H.sum(), 1)
    return [h[min(np.searchsorted(cs, f), len(h) - 1)] for f in fracs]


def main(label="sec4_A", nuts_label="sec4_B"):
    style.use()
    views, meta = load_views(label)
    pn, sub, bfp, spost, V0, dials = (meta["pn"], meta["sub"], meta["bfp"], meta["spost"],
                                      meta["V0"], meta["dials"])

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
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(nd - 1, nd - 1, figsize=(max(4.6, 2.25 * (nd - 1)),
                                                          max(4.4, 2.25 * (nd - 1))),
                                 sharex="col", sharey="row", squeeze=False)
        for a in range(nd - 1):
            for b in range(nd - 1):
                A = axes[a, b]
                i, j = b, a + 1
                key = (dials[i], dials[j])
                if j <= i or key not in views:
                    A.axis("off"); continue
                w = views[key]; ci, cj = w["ci"], w["cj"]
                ki, kj = sub[ci], sub[cj]
                axi = snap_axis(w["axi"], ki, ci, pn, bfp, spost)
                axj = snap_axis(w["axj"], kj, cj, pn, bfp, spost)
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

                # ---- PROFILE: 2-D confidence region -------------------------------------------------
                A.contourf(X, Y, d, levels=[0, L68], colors=[C_PROF], alpha=0.22)
                A.contour(X, Y, d, levels=[L68, L90], colors=[C_PROF, C_PROF],
                          linewidths=[1.5, 1.0], linestyles=["-", "--"])

                # ---- NUTS: 2-D marginal credible region ---------------------------------------------
                if dials[i] in ncol and dials[j] in ncol:
                    sx, sy = U[:, ncol[dials[i]]], U[:, ncol[dials[j]]]
                    # bin over the SAME window as the profile so the two are read on one footing
                    rng = [[axi[0], axi[-1]], [axj[0], axj[-1]]]
                    H, xe, ye = np.histogram2d(sx, sy, bins=36, range=rng)
                    if H.sum() > 0:
                        lv = hpd_levels(H)
                        Xc = 0.5 * (xe[1:] + xe[:-1]); Yc = 0.5 * (ye[1:] + ye[:-1])
                        # levels must increase for contour(); HPD levels come out descending
                        A.contour(Xc, Yc, H.T, levels=sorted(set(lv)), colors=C_NUTS,
                                  linewidths=[1.0, 1.5][:len(set(lv))],
                                  linestyles=["--", "-"][:len(set(lv))])
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
                for lev, ls, lw in ((L68, "-", 1.4), (L90, "--", 0.9)):
                    r = np.stack([np.cos(tt) / np.sqrt(ev[0]), np.sin(tt) / np.sqrt(ev[1])]) * np.sqrt(lev)
                    xy = evec @ r
                    A.plot(xy[0], xy[1], ls, color=C_GAUS, lw=lw, zorder=6)

                # ---- walls, best fit, limits --------------------------------------------------------
                lim = {"x": [axi[0], axi[-1]], "y": [axj[0], axj[-1]]}
                for which, (kk, cc, aa) in (("x", (ki, ci, axi)), ("y", (kj, cj, axj))):
                    span = aa[-1] - aa[0]
                    for _b, side in ((K.phys_lo(pn[kk]), -1), (K.phys_hi(pn[kk]), +1)):
                        if _b is None: continue
                        xb = (_b - bfp[kk]) / spost[cc]
                        if not (aa[0] - 1e-9 <= xb <= aa[-1] + 1e-9): continue
                        e = xb - 0.04 * span if side < 0 else xb + 0.04 * span
                        lim[which][0 if side < 0 else 1] = (min(lim[which][0], e) if side < 0
                                                            else max(lim[which][1], e))
                        (A.axvspan if which == "x" else A.axhspan)(
                            e if side < 0 else xb, xb if side < 0 else e,
                            facecolor="0.5", alpha=0.30, zorder=3, lw=0)
                        (A.axvline if which == "x" else A.axhline)(xb, color="k", lw=0.9, ls="--",
                                                                   zorder=4)
                A.plot(0, 0, "*", color=C_BFP, ms=10, mec="white", mew=0.6, zorder=8)
                A.set_xlim(*lim["x"]); A.set_ylim(*lim["y"])
                A.tick_params(labelsize=6, top=False, right=False)
                if a == nd - 2: A.set_xlabel(style.plab(dials[i]), fontsize=8)
                if b == 0: A.set_ylabel(style.plab(dials[j]), fontsize=8)

        h = [plt.Line2D([], [], color=C_PROF, lw=1.5),
             plt.Line2D([], [], color=C_NUTS, lw=1.5),
             plt.Line2D([], [], color=C_GAUS, lw=1.4),
             plt.Line2D([], [], color="k", lw=1.0, ls="--"),
             plt.Line2D([], [], color=C_BFP, marker="*", ls="", ms=10),
             plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.30)]
        fig.legend(h, ["profile: 2-D confidence, min over other 14 " r"($\Delta\chi^2$=2.30/4.61)",
                       "NUTS: 2-D marginal credible, integrated over other 14 (68%/90% HPD)",
                       "Gaussian: Gauss-Newton curvature at the best fit",
                       "dashed = 90%, solid = 68%", "best fit", "unphysical (model clamps)"],
                   loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(0.995, 0.955))
        fig.supxlabel(r"$(\theta-\hat\theta)/\sigma_{\rm post}$", fontsize=9)
        fig.suptitle("2-D uncertainty three ways", fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        style.save(fig, "sec4_corner_all")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:2] or ["sec4_A"]))
