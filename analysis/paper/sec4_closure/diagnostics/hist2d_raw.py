"""The raw 36x36 NUTS histogram behind each corner panel, with the HPD levels drawn on it."""
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from analysis.paper.sec4_closure.fig_corner_prof import load_views, snap_axis, view_for
from analysis.paper.sec4_closure.fig_corner_all import hpd_levels
style.use()

views, meta = load_views("sec4_P1")
pn, sub, bfp, spost, dials = meta["pn"], meta["sub"], meta["bfp"], meta["spost"], meta["dials"]
Zn = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_nutsown_*.npz")))]
nmin = min(len(np.asarray(z["u"])) for z in Zn)
U = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])
npn = [str(x) for x in Zn[0]["pnames"]]; nsb = [int(k) for k in Zn[0]["subset"]]
NC = {npn[k]: c for c, k in enumerate(nsb)}

prs = [(dials[b], dials[a]) for a in range(len(dials)) for b in range(a) if view_for(views, dials[b], dials[a])]
fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
print(f"{len(U)} samples, 36x36 bins over the PANEL span\n")
print(f"{'panel':38s} {'in range':>9} {'filled':>7} {'max':>5} {'lvl68':>6} {'lvl90':>6} "
      f"{'nbin>68':>8} {'nbin>90':>8}")
for i, (ni, nj) in enumerate(prs):
    A = ax.flat[i]; w = view_for(views, ni, nj)
    axi = snap_axis(w["axi"], sub[w["ci"]], w["ci"], pn, bfp, spost)
    axj = snap_axis(w["axj"], sub[w["cj"]], w["cj"], pn, bfp, spost)
    sx, sy = U[:, NC[ni]], U[:, NC[nj]]
    rng = [[axi[0], axi[-1]], [axj[0], axj[-1]]]
    H, xe, ye = np.histogram2d(sx, sy, bins=36, range=rng)
    lv = hpd_levels(H)
    m = A.pcolormesh(xe, ye, H.T, cmap="Blues", shading="flat")
    Xc, Yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    A.contour(Xc, Yc, H.T, levels=sorted(set(lv)), colors="#c33",
              linewidths=[1.0, 1.6][:len(set(lv))], linestyles=["--", "-"][:len(set(lv))])
    fig.colorbar(m, ax=A, fraction=0.046, pad=0.02).set_label("counts/bin", fontsize=7)
    inr = int(((sx >= axi[0]) & (sx <= axi[-1]) & (sy >= axj[0]) & (sy <= axj[-1])).sum())
    A.set_title(f"{style.plab(ni)} x {style.plab(nj)}\nmax={int(H.max())}  "
                f"lvl68={lv[0]:.0f} lvl90={lv[1]:.0f}  filled={int((H>0).sum())}/1296", fontsize=8)
    A.tick_params(labelsize=7)
    print(f"{ni[:17]+' x '+nj[:17]:38s} {inr:9d} {int((H>0).sum()):7d} {int(H.max()):5d} "
          f"{lv[0]:6.0f} {lv[1]:6.0f} {int((H>=lv[0]).sum()):8d} {int((H>=lv[1]).sum()):8d}")
fig.suptitle(f"Raw 36x36 NUTS histograms behind the corner panels ({len(U)} samples); "
             f"red = the 68%/90% HPD levels", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
style.save(fig, "sec4_P1_nuts_hist2d")
