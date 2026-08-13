import sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from analysis.paper.sec4_closure.fig_corner_prof import load_views, snap_axis, view_for
from analysis.paper.sec4_closure.fig_corner_all import hpd_levels
from adonis.analysis import knobs as K

views, meta = load_views("sec4_P1")
pn, sub, bfp, spost, dials = meta["pn"], meta["sub"], meta["bfp"], meta["spost"], meta["dials"]
Zn = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_nutsown_*.npz")))]
nmin = min(len(np.asarray(z["u"])) for z in Zn)
U = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])
npn = [str(x) for x in Zn[0]["pnames"]]; nsub = [int(k) for k in Zn[0]["subset"]]
ncol = {npn[k]: c for c, k in enumerate(nsub)}

def hpd_area(sx, sy, rng, nb=36):
    H, xe, ye = np.histogram2d(sx, sy, bins=nb, range=rng)
    if H.sum() == 0: return np.nan, 0
    lv = hpd_levels(H, (0.6827,))[0]
    cell = (xe[1]-xe[0])*(ye[1]-ye[0])
    return float((H >= lv).sum()*cell), int((H > 0).sum())

print(f"{'panel':34s} {'panelbox':>13s} {'samplebox':>13s} {'occ%':>5s} {'filled':>7s} "
      f"{'A68prof':>8s} {'A68nutsPANEL':>12s} {'A68nutsTIGHT':>12s} {'ratioP':>7s} {'ratioT':>7s}")
for a in range(len(dials)):
    for b in range(a):
        ni, nj = dials[b], dials[a]
        w = view_for(views, ni, nj)
        if w is None: continue
        ci, cj = w["ci"], w["cj"]; ki, kj = sub[ci], sub[cj]
        axi = snap_axis(w["axi"], ki, ci, pn, bfp, spost)
        axj = snap_axis(w["axj"], kj, cj, pn, bfp, spost)
        X, Y = np.meshgrid(axi, axj, indexing="ij")
        d = np.array(w["d"], float)
        for which,(kk,cc,aa) in (("x",(ki,ci,axi)),("y",(kj,cj,axj))):
            lo_ = K.phys_lo(pn[kk])
            if lo_ is None: continue
            xb = (lo_-bfp[kk])/spost[cc] - 1e-2*abs(aa[1]-aa[0])
            if which=="x": d[X<xb]=np.nan
            else: d[Y<xb]=np.nan
        d = d - np.nanmin(d)
        cell = (axi[1]-axi[0])*(axj[1]-axj[0])
        Aprof = float(np.nansum(d <= 2.30)*cell)
        sx, sy = U[:, ncol[ni]], U[:, ncol[nj]]
        pbox = [[axi[0], axi[-1]], [axj[0], axj[-1]]]
        qx = np.percentile(sx,[0.5,99.5]); qy = np.percentile(sy,[0.5,99.5])
        tbox = [[qx[0],qx[1]],[qy[0],qy[1]]]
        Ap, fp = hpd_area(sx, sy, pbox)
        At, ft = hpd_area(sx, sy, tbox)
        occ = 100*((qx[1]-qx[0])*(qy[1]-qy[0]))/((axi[-1]-axi[0])*(axj[-1]-axj[0]))
        print(f"{ni[:15]+' x '+nj[:15]:34s} "
              f"[{axi[0]:+.1f},{axi[-1]:+.1f}]".rjust(13)+" "
              f"[{qx[0]:+.1f},{qx[1]:+.1f}]".rjust(13)+
              f" {occ:5.1f} {fp:4d}/1296 {Aprof:8.2f} {Ap:12.2f} {At:12.2f} "
              f"{Ap/Aprof:7.2f} {At/Aprof:7.2f}")
