"""Pick the bin count per panel so every panel's contour sits on comparable statistics.

Contour jitter goes as sqrt(N_bin)/|dN/dx| ~ 1/w, so coarser bins pin the contour -- at the cost of
resolution.  Rather than a global 36, choose nb per panel so the PEAK bin reaches a target count: that
automatically coarsens the fat uncorrelated panels and leaves the thin degenerate ones alone.
"""
import sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from analysis.paper.sec4_closure.fig_corner_prof import load_views, snap_axis, view_for
from analysis.paper.sec4_closure.fig_corner_all import hpd_levels
views, meta = load_views("sec4_P1")
pn, sub, bfp, spost, dials = meta["pn"], meta["sub"], meta["bfp"], meta["spost"], meta["dials"]
Zn = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_nutsown_*.npz")))]
nmin = min(len(np.asarray(z["u"])) for z in Zn)
U = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])
npn = [str(x) for x in Zn[0]["pnames"]]; nsb = [int(k) for k in Zn[0]["subset"]]
NC = {npn[k]: c for c, k in enumerate(nsb)}
prs = [(dials[b], dials[a]) for a in range(len(dials)) for b in range(a) if view_for(views, dials[b], dials[a])]
TARGET = 250
print(f"{'panel':36s} {'nb':>4} {'max':>5} {'lvl68':>6} {'grad/bin':>9} {'jitter(bins)':>12} {'jitter(sig)':>11}")
for ni, nj in prs:
    w = view_for(views, ni, nj)
    axi = snap_axis(w["axi"], sub[w["ci"]], w["ci"], pn, bfp, spost)
    axj = snap_axis(w["axj"], sub[w["cj"]], w["cj"], pn, bfp, spost)
    sx, sy = U[:, NC[ni]], U[:, NC[nj]]
    rng = [[axi[0], axi[-1]], [axj[0], axj[-1]]]
    for nb in (36, None):
        if nb is None:                       # adaptive: coarsen until the peak reaches TARGET
            nb = 36
            while nb > 10:
                H, _, _ = np.histogram2d(sx, sy, bins=nb, range=rng)
                if H.max() >= TARGET: break
                nb -= 2
        H, xe, ye = np.histogram2d(sx, sy, bins=nb, range=rng)
        lv = hpd_levels(H)[0]
        gy, gx = np.gradient(H)
        near = np.abs(H - lv) < 0.25 * lv
        gmag = np.median(np.hypot(gx, gy)[near]) if near.sum() else np.nan
        jit = np.sqrt(lv) / gmag if gmag > 0 else np.nan
        wpx = (xe[1] - xe[0])
        print(f"{(ni[:16]+' x '+nj[:16]):36s} {nb:4d} {int(H.max()):5d} {lv:6.0f} {gmag:9.1f} "
              f"{jit:12.2f} {jit*wpx:11.3f}")
    print()
