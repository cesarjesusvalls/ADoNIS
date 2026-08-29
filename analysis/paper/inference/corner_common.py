"""Shared loaders for the corner figures: load_views, snap_axis, view_for.

Also draws 2-D profiled confidence contours (chi2 minimised over the other dials at each grid node).
Physical bounds are shaded rather than masked: the model clamps beyond them, so the flat chi2 there
is real.  Imported by fig_corner_all; not a figure driver itself.
"""
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.paper import style
from adonis.reweight import knobs as K
from adonis.fit import merge as MG

L68, L90 = 2.30, 4.61
C68, C90, C_BFP = "#1f4b9c", "#7aa7dd", "#d24"


def load_views(label, allow_partial=False):
    """Assemble the per-pair profiled chi2 surfaces for `label`.  Shared with fig_corner_all so the
    axis/merge/wall-snap conventions live in one place.

    Returns (views, meta): views maps (name_i, name_j) -> dict(axi, axj, d, ci, cj) with axi/axj in
    sigma_post about the BFP; meta carries pn/sub/bfp/spost/V0/dials.
    """
    fs = sorted(glob.glob(str(style.ALTGEN / f"{label}_corner2d_prof_*.npz")))
    if not fs:
        raise SystemExit(f"no prof shards for {label}")
    Z = [np.load(f, allow_pickle=True) for f in fs]
    rep = MG.check(fs, Z, what=f"{label} corner2d")
    rep.raise_if_bad(allow_partial)
    pnames_all = [str(x) for x in Z[0]["pnames"]]
    sub_all = [int(k) for k in Z[0]["subset"]]
    V0 = np.asarray(Z[0]["V"]); spost = np.asarray(Z[0]["sigma_post"]); bfp = np.asarray(Z[0]["bfp"])
    pn = pnames_all; sub = sub_all

    views = {}
    cands = {}
    for z in Z:
        dl = [str(x) for x in z["dials"]]
        pos = [int(q) for q in z["sel_pos"]]
        _bf = np.asarray(z["bfp"]); _sp = np.asarray(z["sigma_post"]); _sb = [int(k) for k in z["subset"]]
        _ap = np.asarray(z["axes_phys"]) if "axes_phys" in z.files else None
        for t, (ii, jj) in enumerate(np.asarray(z["pair_idx"])):
            key = (dl[int(ii)], dl[int(jj)])
            _raw = "chi2_abs" in z.files
            v = np.asarray(z["chi2_abs" if _raw else "dchi2"])[t]
            _ld = np.asarray(z["logdet_Vnuis"])[t] if "logdet_Vnuis" in z.files else None
            if _ap is not None and np.isfinite(_ap[t]).all():
                axi, axj = ((_ap[t, e] - _bf[_sb[pos[int(a)]]]) / _sp[pos[int(a)]]
                            for e, a in ((0, ii), (1, jj)))
            else:
                axi = axj = np.asarray(z["axis_sigma"])
            sig = (len(axi), round(float(axi[0]), 6), round(float(axi[-1]), 6),
                   round(float(axj[0]), 6), round(float(axj[-1]), 6))
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

    for by_sig in cands.values():
        for w in by_sig.values():
            if w["raw"] and np.isfinite(w["d"]).any():
                w["d"] = w["d"] - np.nanmin(w["d"])

    def _rank(w):
        return (round((w["axi"][-1] - w["axi"][0]) * (w["axj"][-1] - w["axj"][0]), 6), len(w["axi"]))

    def _complete(w, key):
        """Did this candidate compute everything it was asked to?  With stamps, the assigned row blocks
        must tile range(N); without them, fall back to the NaN-fraction proxy.
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
    """The view for the ordered pair (x=ni, y=nj), transposing the stored one if it was keyed the other
    way round (views are keyed in the corner run's dial order, not necessarily display order).  Returns
    None if neither orientation is present.
    """
    if (ni, nj) in views:
        return views[(ni, nj)]
    w = views.get((nj, ni))
    if w is None:
        return None
    _ld = w.get("ld")
    return dict(axi=w["axj"], axj=w["axi"], d=np.asarray(w["d"]).T, ci=w["cj"], cj=w["ci"],
                ld=(None if _ld is None else np.asarray(_ld).T))


def snap_axis(aa, kk, cc, pn, bfp, spost):
    """Pull an endpoint that sits a rounding error outside a physical wall back onto the wall (the
    adaptive scan can return an edge node a hair beyond phys_lo/phys_hi)."""
    aa = np.array(aa, float)
    tol = 1e-2 * abs(aa[1] - aa[0])
    for _b in (K.phys_lo(pn[kk]), K.phys_hi(pn[kk])):
        if _b is None: continue
        xb = (_b - bfp[kk]) / spost[cc]
        for e in (0, -1):
            if abs(aa[e] - xb) < tol: aa[e] = xb
    return aa
