"""2-D PROFILED confidence contours -- NO LONGER A PAPER FIGURE.

Superseded by fig_corner_all (figure B), which shows the Gaussian, the Laplace marginal and
the NUTS marginal on one set of axes; the raw profile is not drawn there because figure A
(c)/(d) already make the point that it needs the nuisance-volume factor.  This module is kept
because load_views / snap_axis / view_for live here and figure B imports them -- the shard
merge, axis and wall conventions must exist in exactly one place.  Still runnable as a
diagnostic.

Original description:

For every dial pair: chi2 minimised over the other 14 dials at each grid node, so the levels really are
2-D confidence regions:  Dchi2 = 2.30 (68%) and 4.61 (90%).

This is the ONLY corner in the section whose contours mean that.  fig_corner_minimizer shows a CONDITIONAL
surface (others frozen), whose levels are far tighter than the marginal errors -- measured factor ~12 for
M_A_res -- and fig_corner_grad shows gradient fields with no contours at all.  Keeping the distinction is
the point: a conditional slice read as a confidence region overstates the precision badly.

Physical bounds are SHADED, not masked: the model clamps beyond them so the flat chi2 there is real.

Usage:  python -m analysis.paper.sec4_closure.fig_corner_prof [label]
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from adonis.analysis import knobs as K
from adonis.fit import merge as MG

L68, L90 = 2.30, 4.61          # 2-D Delta-chi2 levels
C68, C90, C_BFP = "#1f4b9c", "#7aa7dd", "#d24"


def load_views(label, allow_partial=False):
    """Assemble the per-pair profiled chi2 surfaces for `label`.

    Shared with fig_corner_all so the axis / merge / wall-snap conventions live in ONE place -- each of
    them has already produced a wrong figure once (global +-3 sigma axes on an adaptive scan, node-count
    preference reverting to a narrower run, and a rounding-level wall test deleting a whole edge row).

    Returns (views, meta) where views maps (name_i, name_j) -> dict(axi, axj, d, ci, cj) with axi/axj in
    sigma_post about the BFP, and meta carries pn/sub/bfp/spost/V0/dials.
    """
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_corner2d_prof_*.npz")))
    if not fs:
        raise SystemExit(f"no prof shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    # ONE RUN DEFINITION for all shards, and none of them preempted mid-grid.  The axes agreeing is not
    # evidence of this: two campaigns with different injected truth, sigma or estimator produce identical
    # axes and merge without a murmur.
    rep = MG.check(fs, Z, what=f"{label} corner2d")
    rep.raise_if_bad(allow_partial)
    # Views are keyed by DIAL NAMES, not by pair index, so several corner RUNS can contribute to one
    # figure: the 6-dial N=21 sweep plus the dedicated N=41 scan of M_A_res x S_Delta, whose degeneracy is
    # so tight (corr -0.995) that at N=21 the 68% region breaks into disconnected diamonds.  Where a pair
    # appears in more than one run the FINER grid wins, and each panel carries its own axis.
    pnames_all = [str(x) for x in Z[0]["pnames"]]
    sub_all = [int(k) for k in Z[0]["subset"]]
    V0 = np.asarray(Z[0]["V"]); spost = np.asarray(Z[0]["sigma_post"]); bfp = np.asarray(Z[0]["bfp"])
    pn = pnames_all; sub = sub_all

    views = {}            # (name_i, name_j) -> dict(axi, axj, d, ci, cj)
    cands = {}            # (name_i, name_j) -> {grid signature -> candidate view}
    for z in Z:
        dl = [str(x) for x in z["dials"]]
        pos = [int(q) for q in z["sel_pos"]]
        _bf = np.asarray(z["bfp"]); _sp = np.asarray(z["sigma_post"]); _sb = [int(k) for k in z["subset"]]
        # PER-DIAL axis.  `axis_sigma` is the single global +-S4_CORNER_RANGE grid and is only correct when
        # every dial shares it; with S4_CORNER_FROM_PROFILE=1 each dial gets the 1-D profile's own adaptive
        # span (M_A_res [-6.53,+5.40], S_Delta [-3.00,+9.72], ...) and the two axes of a panel DIFFER.
        # Plotting both against `axis_sigma` mislabelled every panel by a different factor -- which is why
        # the contours changed size relative to the (correctly drawn, true-sigma) Gaussian ellipse.
        # `axes_phys[t]` carries the PHYSICAL grid of BOTH dials of pair t; convert each to sigma_post.
        _ap = np.asarray(z["axes_phys"]) if "axes_phys" in z.files else None
        for t, (ii, jj) in enumerate(np.asarray(z["pair_idx"])):
            key = (dl[int(ii)], dl[int(jj)])
            # RAW chi2 when the shard carries it: row shards each subtract their OWN nanmin, so merging
            # `dchi2` across them stitches surfaces that sit at different offsets.  `chi2_abs` has no
            # offset, and the global minimum is taken once, after the merge, below.
            _raw = "chi2_abs" in z.files
            v = np.asarray(z["chi2_abs" if _raw else "dchi2"])[t]
            # LOG-DET of the nuisance covariance at each node -- the ingredient the MARGINAL needs.
            # Absent from shards written before it was stored; those panels simply get no Laplace
            # contour rather than a wrong one.
            _ld = np.asarray(z["logdet_Vnuis"])[t] if "logdet_Vnuis" in z.files else None
            if _ap is not None and np.isfinite(_ap[t]).all():
                axi, axj = ((_ap[t, e] - _bf[_sb[pos[int(a)]]]) / _sp[pos[int(a)]]
                            for e, a in ((0, ii), (1, jj)))
            else:
                axi = axj = np.asarray(z["axis_sigma"])
            # Group by GRID SIGNATURE first: row shards of one run merge, distinct runs stay separate
            # candidates.  Choosing between runs BEFORE knowing which are finished let an in-progress
            # finer scan shadow a complete coarser one, and the panel then vanished as "incomplete".
            sig = (len(axi), round(float(axi[0]), 6), round(float(axi[-1]), 6),
                   round(float(axj[0]), 6), round(float(axj[-1]), 6))
            # Which ROWS this shard was assigned, from its own stamp.  Completeness is then a statement
            # about the work rather than about NaNs, and a gap can be NAMED.
            _p = MG.provenance.read(z)
            _blk = (int(_p["row_base"]), int(_p["n_row"])) if _p and "n_row" in _p else None
            c = cands.setdefault(key, {}).get(sig)
            if c is None:
                cands[key][sig] = dict(axi=axi, axj=axj, d=v.copy(), raw=_raw,
                                       ld=(None if _ld is None else _ld.copy()),
                                       ci=pos[int(ii)], cj=pos[int(jj)],
                                       blocks=([_blk] if _blk else []), nshard=1)
            else:
                m = np.isfinite(v); c["d"][m] = v[m]; c["raw"] = c["raw"] and _raw
                c["nshard"] += 1
                if _blk:
                    c["blocks"].append(_blk)
                if c["ld"] is not None and _ld is not None:
                    m2 = np.isfinite(_ld); c["ld"][m2] = _ld[m2]
                else:
                    c["ld"] = None

    # Re-zero AFTER merging every shard of a candidate.  Only legal on raw-chi2 candidates; a mixed set
    # (some shards pre-dating chi2_abs) is left alone rather than silently mis-stitched.
    for by_sig in cands.values():
        for w in by_sig.values():
            if w["raw"] and np.isfinite(w["d"]).any():
                w["d"] = w["d"] - np.nanmin(w["d"])

    # PREFERENCE, independent of completeness: the WIDER span (the adaptive scan), then the finer grid.
    # ROUND the span before comparing.  Two runs over the SAME window reconstruct it from linspace grids
    # of different length, so the spans differ in the last ulp (151.69462533697404 vs 151.694625336974) --
    # enough for max() to settle it on the span and never reach the node-count tie-break, silently
    # preferring N=21 over the complete N=81 scan.
    def _rank(w):
        return (round((w["axi"][-1] - w["axi"][0]) * (w["axj"][-1] - w["axj"][0]), 6), len(w["axi"]))

    def _complete(w, key):
        """Did this candidate compute everything it was asked to?

        With stamps, that is an exact statement: the assigned row blocks must tile range(N).  Without
        them (every shard of the sec4_P1 campaign), fall back to the old NaN-fraction proxy -- which is
        why the proxy survives here at all.
        """
        if w["blocks"]:
            sub = MG.Report(what="")
            sub.rows(w["blocks"], grid=len(w["axi"]), what=f"{key[0]} x {key[1]} (N={len(w['axi'])})")
            return (not sub.errors), (sub.errors + sub.warnings)
        frac = float(np.isfinite(w["d"]).mean())
        return frac >= 0.999, ([] if frac >= 0.999 else
                               [f"{key[0]} x {key[1]} (N={len(w['axi'])}): {100*frac:.1f}% of nodes "
                                f"finite, and the shards carry no row assignment to check against"])

    drop = []
    for key, by_sig in cands.items():
        ranked = sorted(by_sig.values(), key=_rank, reverse=True)
        ok, why = [], []
        for w in ranked:
            good, msg = _complete(w, key)
            if good:
                ok.append(w)
            why += msg
        if not ok:
            drop.append((key, [f"{len(w['axi'])}^2 {100*np.isfinite(w['d']).mean():.0f}%"
                               for w in by_sig.values()]))
            rep.error(f"{key[0]} x {key[1]}: no complete grid; " + " | ".join(why))
            continue
        views[key] = ok[0]
        # THE SILENT DOWNGRADE.  If a better candidate exists but was rejected as incomplete, the figure
        # renders happily at the lower resolution and nothing about it looks wrong.  That is the failure
        # this check exists for, so it is an error and not the `[note]` it used to be.
        if _rank(ok[0]) < _rank(ranked[0]):
            rep.error(f"{key[0]} x {key[1]}: falling back to N={len(ok[0]['axi'])} because the "
                      f"preferred N={len(ranked[0]['axi'])} scan is incomplete -- " + " | ".join(why))
    if drop:
        print(f"[warn] {len(drop)} view(s) DROPPED, no complete grid: {drop}")
    rep.raise_if_bad(allow_partial)
    for key, w in views.items():
        n_part = len(cands[key]) - 1
        if n_part:
            other = ", ".join(f"N={len(v['axi'])} {100*np.isfinite(v['d']).mean():.0f}%"
                              for sg, v in cands[key].items() if v is not w)
            print(f"[note] {key[0]} x {key[1]}: using N={len(w['axi'])} "
                  f"span [{w['axi'][0]:+.2f},{w['axi'][-1]:+.2f}]x[{w['axj'][0]:+.2f},{w['axj'][-1]:+.2f}]"
                  f"; also present: {other}")
    dials = []
    for k in views:
        for nm in k:
            if nm not in dials:
                dials.append(nm)
    # RES trio adjacent, bounded dial last: the three that share the axial block read together,
    # and E_b -- the only one with a wall -- does not sit between them.
    order = ["M_A_res", "delta_strength", "res_axial_strength", "sabs", "f_NN_cex", "kF_sf",
             "Eb_shift"]
    dials = [d for d in order if d in dials] + [d for d in dials if d not in order]
    print(f"[ok]   {len(views)} view(s) complete; grids "
          f"{sorted({len(w['axi']) for w in views.values()})}; "
          + "spans " + ", ".join(f"{k[0]}x{k[1]}:[{w['axi'][0]:+.1f},{w['axi'][-1]:+.1f}]x"
                                 f"[{w['axj'][0]:+.1f},{w['axj'][-1]:+.1f}]"
                                 for k, w in list(views.items())[:3]))
    return views, dict(pn=pn, sub=sub, bfp=bfp, spost=spost, V0=V0, dials=dials)


def view_for(views, ni, nj):
    """The view for the ORDERED pair (x=ni, y=nj), transposing the stored one if it was keyed the
    other way round.

    Views are keyed in the order the CORNER RUN listed its dials (S4_CORNER_DIALS); the figures index
    them in DISPLAY order, and the two need not agree -- with S4_CORNER_DIALS=...,res_axial_strength,
    Eb_shift but res_axial_strength sorting last for display, the C5A x E_b panel was looked up as
    ("Eb_shift","res_axial_strength"), missed, and silently rendered blank.  Returns None if neither
    orientation is present, which is a genuinely absent pair.
    """
    if (ni, nj) in views:
        return views[(ni, nj)]
    w = views.get((nj, ni))
    if w is None:
        return None
    # `ld` (log-det of the nuisance covariance, the Laplace ingredient) must transpose WITH `d`.
    # It was dropped here, so a transposed lookup silently returned a view with no Laplace surface --
    # invisible while the contours were only ever looked up in the stored order, and total once the
    # merged corner moved them to the upper triangle, where every lookup is transposed.
    _ld = w.get("ld")
    return dict(axi=w["axj"], axj=w["axi"], d=np.asarray(w["d"]).T, ci=w["cj"], cj=w["ci"],
                ld=(None if _ld is None else np.asarray(_ld).T))


def snap_axis(aa, kk, cc, pn, bfp, spost):
    """Pull an endpoint that sits a rounding error outside a wall back ONTO the wall (see fig_corner_all).
    The adaptive scan ends AT the bound but its edge node returns as e.g. Eb=0.00999 vs phys_lo=0.01; a
    bare `< wall` mask then NaNs the whole edge row and the region stops a full grid cell short."""
    aa = np.array(aa, float)
    tol = 1e-2 * abs(aa[1] - aa[0])
    for _b in (K.phys_lo(pn[kk]), K.phys_hi(pn[kk])):
        if _b is None: continue
        xb = (_b - bfp[kk]) / spost[cc]
        for e in (0, -1):
            if abs(aa[e] - xb) < tol: aa[e] = xb
    return aa


def main(label="sec4_ref", allow_partial=False):
    style.use()
    views, meta = load_views(label, allow_partial)
    pn, sub, bfp, spost, V0, dials = (meta["pn"], meta["sub"], meta["bfp"], meta["spost"],
                                      meta["V0"], meta["dials"])

    nd = len(dials)
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6}):
        # squeeze=False: a 2-dial corner is subplots(1,1), which otherwise returns a bare Axes and
        # breaks the axes[a, b] indexing below.
        fig, axes = plt.subplots(nd - 1, nd - 1, figsize=(max(4.6, 2.1 * (nd - 1)),
                                                          max(4.4, 2.1 * (nd - 1))),
                                 sharex="col", sharey="row", squeeze=False)
        for a in range(nd - 1):
            for b in range(nd - 1):
                A = axes[a, b]
                i, j = b, a + 1
                w = None if j <= i else view_for(views, dials[i], dials[j])
                if w is None:
                    A.axis("off"); continue
                axi, axj = w["axi"], w["axj"]; ci, cj = w["ci"], w["cj"]
                ki, kj = sub[ci], sub[cj]

                def _snap(aa, kk, cc):
                    """Pull an endpoint that sits a rounding error outside a wall back ONTO the wall.
                    The adaptive scan ends AT the bound, but its last node comes back as e.g. Eb=0.00999
                    against phys_lo=0.01 (linspace/clip_phys rounding).  A bare `< wall` mask then NaNs the
                    whole first row, contourf starts one node in, and the region appears to stop a full
                    grid cell short of the wall -- 0.55 sigma for E_b, 0.60 for M_A_res.  Those nodes are
                    NOT unphysical: the driver clipped theta before evaluating, so the value stored there
                    is the value AT the wall.  Snap the coordinate instead of throwing the node away."""
                    aa = np.array(aa, float)
                    tol = 1e-2 * abs(aa[1] - aa[0])
                    for _b in (K.phys_lo(pn[kk]), K.phys_hi(pn[kk])):
                        if _b is None: continue
                        xb = (_b - bfp[kk]) / spost[cc]
                        for e in (0, -1):
                            if abs(aa[e] - xb) < tol: aa[e] = xb
                    return aa
                axi = _snap(axi, ki, ci); axj = _snap(axj, kj, cj)
                X, Y = np.meshgrid(axi, axj, indexing="ij")     # driver fills chi[pair, i_x, j_y]
                d = np.array(w["d"], float)
                # mask only nodes GENUINELY beyond a wall (the driver clips theta but stores the
                # unclipped coordinate); the tolerance keeps the snapped edge node alive.
                for which, (kk, cc, aa) in (("x", (ki, ci, axi)), ("y", (kj, cj, axj))):
                    _lo = K.phys_lo(pn[kk])
                    if _lo is None: continue
                    _xb = (_lo - bfp[kk]) / spost[cc] - 1e-2 * abs(aa[1] - aa[0])
                    if which == "x": d[X < _xb] = np.nan
                    else: d[Y < _xb] = np.nan
                if not np.isfinite(d).any():
                    A.axis("off"); continue
                d = d - np.nanmin(d)          # zero at the best PHYSICAL point, after masking
                A.contourf(X, Y, d, levels=[0, L68, L90], colors=[C68, C90], alpha=0.75)
                A.contour(X, Y, d, levels=[L68, L90], colors=["k", "0.35"], linewidths=[1.0, 0.7])
                Vp = V0[np.ix_([ci, cj], [ci, cj])]
                Dp = np.diag(1.0 / np.array([spost[ci], spost[cj]]))
                Rp = Dp @ Vp @ Dp
                ev, evec = np.linalg.eigh(np.linalg.inv(Rp))
                tt = np.linspace(0, 2 * np.pi, 361)
                for lev, ls in ((L68, "-"), (L90, "--")):
                    r = np.stack([np.cos(tt) / np.sqrt(ev[0]), np.sin(tt) / np.sqrt(ev[1])]) * np.sqrt(lev)
                    xy = evec @ r
                    A.plot(xy[0], xy[1], ls, color="#e8b", lw=1.4, zorder=6)
                # WALLS.  The adaptive scan stops EXACTLY AT a physical bound, so the wall coincides with
                # the axis edge and a strict `aa[0] < xb < aa[-1]` test silently drops it -- every bounded
                # dial (M_A_res at 0.01, Eb_shift at its floor) lost its marker.  Accept the closed
                # interval and pad the view a little past the wall so the forbidden strip stays visible.
                lim = {"x": [axi[0], axi[-1]], "y": [axj[0], axj[-1]]}
                for which, (kk, cc, aa) in (("x", (ki, ci, axi)), ("y", (kj, cj, axj))):
                    _sp = aa[-1] - aa[0]
                    for _b, _side in ((K.phys_lo(pn[kk]), -1), (K.phys_hi(pn[kk]), +1)):
                        if _b is None: continue
                        _xb = (_b - bfp[kk]) / spost[cc]
                        if not (aa[0] - 1e-9 <= _xb <= aa[-1] + 1e-9): continue
                        _e = _xb - 0.04 * _sp if _side < 0 else _xb + 0.04 * _sp
                        lim[which][0 if _side < 0 else 1] = (min(lim[which][0], _e) if _side < 0
                                                             else max(lim[which][1], _e))
                        (A.axvspan if which == "x" else A.axhspan)(
                            _e if _side < 0 else _xb, _xb if _side < 0 else _e,
                            facecolor="0.5", alpha=0.30, zorder=3, lw=0)
                        (A.axvline if which == "x" else A.axhline)(_xb, color="k", lw=0.9, ls="--", zorder=4)
                A.plot(0, 0, "*", color=C_BFP, ms=11, mec="white", mew=0.6, zorder=8)
                A.set_xlim(*lim["x"]); A.set_ylim(*lim["y"])
                A.tick_params(labelsize=6, top=False, right=False)
                if a == nd - 2: A.set_xlabel(style.plab(dials[i]), fontsize=8)
                if b == 0: A.set_ylabel(style.plab(dials[j]), fontsize=8)
        h = [plt.Line2D([], [], color="#e8b", lw=1.4),
             plt.Rectangle((0, 0), 1, 1, fc=C68, alpha=0.75),
             plt.Rectangle((0, 0), 1, 1, fc=C90, alpha=0.75),
             plt.Line2D([], [], color=C_BFP, marker="*", ls="", ms=11),
             plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.30)]
        fig.legend(h, ["Gaussian ellipse (2x2 covariance)", r"68%  ($\Delta\chi^2=2.30$)", r"90%  ($\Delta\chi^2=4.61$)", "best fit",
                       "unphysical (model clamps)"],
                   loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(0.99, 0.99))
        fig.supxlabel(r"$(\theta-\hat\theta)/\sigma_{\rm post}$", fontsize=9)
        fig.suptitle("2-D profiled confidence regions (minimised over the other 14 dials)",
                     fontsize=11, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        style.save(fig, "sec4_corner_prof")


if __name__ == "__main__":
    p = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(p[:1] or []), allow_partial="--allow-partial" in sys.argv)
