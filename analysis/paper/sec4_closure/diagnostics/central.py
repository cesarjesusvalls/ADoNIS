"""Central (equal-tailed) 68% instead of the shortest interval, for every object.

The shortest-interval / water-fill statistic is mode-seeking: on a flat direction it chases wherever the
histogram happens to peak, and we measured it swinging +-0.25 sigma on S_Delta depending on which 5-12%
of toys were dropped.  The 16th-84th percentile is anchored at the median and has none of that
sensitivity, so it can say whether S_Delta is really discrepant or merely unmeasurable.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from scipy.interpolate import CubicSpline
from adonis.analysis import knobs as K

zp = np.load(style.ALTGEN / "sec4_P1_profile.npz", allow_pickle=True)
prof = np.asarray(zp["prof_dobj"]); gs = np.asarray(zp["grids_sigma"]); sp = np.asarray(zp["sigma_post"])
LD = np.asarray(zp["logdet_Vnuis"]); sub = [int(k) for k in zp["subset"]]; pn = [str(x) for x in zp["pnames"]]
names = [pn[k] for k in sub]

ef = [f for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_ens_*.npz"))) if "_conv" not in f]
E = [np.load(f, allow_pickle=True) for f in ef]
th = np.concatenate([e["th_fit"] for e in E]); st = np.concatenate([e["th_star"] for e in E])
seeds = np.concatenate([int(f.rsplit("_",1)[1].split(".")[0]) + np.arange(len(e["th_fit"]))
                        for f, e in zip(ef, E)])
zc = np.load(style.ALTGEN / "sec4_P1_ens_conv.npz", allow_pickle=True)
cm = {int(a): (int(b), float(c)) for a, b, c in zip(zc["seed"], zc["nfev"], zc["gap"])}
conv = np.array([(s in cm) and (cm[s][0] < 200) and (cm[s][1] <= 0.01) for s in seeds])
ce = names.index("Eb_shift"); wall = K.phys_lo("Eb_shift") or 0.0
noeb = np.abs(th[:, ce] - wall) >= 1e-6
CUTS = [("all", np.ones(len(th), bool)), ("conv", conv), ("noEb", noeb), ("both", conv & noeb)]

Zn = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_nutsown_*.npz")))]
nmin = min(len(np.asarray(z["u"])) for z in Zn)
U = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])
npn = [str(x) for x in Zn[0]["pnames"]]; nsb = [int(k) for k in Zn[0]["subset"]]
NC = {npn[k]: c for c, k in enumerate(nsb)}

def dens_ct(g, d, ld=None):
    xf = np.linspace(g.min(), g.max(), 3000)
    y = np.exp(-0.5 * np.maximum(CubicSpline(g, d)(xf), 0.0))
    if ld is not None:
        y = y * np.exp(0.5 * (CubicSpline(g, ld)(xf) - CubicSpline(g, ld)(xf).max()))
    c = np.concatenate([[0], np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(xf))]); c /= c[-1]
    return float(np.interp(0.1587, c, xf)), float(np.interp(0.8413, c, xf))

print(f"CENTRAL 68% (16th-84th percentile), sigma_post units.  {len(th)} toys, {len(U)} NUTS samples\n")
print(f"{'dial':>20} " + " ".join(f"{n:>17}" for n, _ in CUTS) + f" {'profile':>17} {'p x Occam':>17} {'NUTS':>17}")
for c, k in enumerate(sub):
    ok = np.isfinite(gs[c]) & np.isfinite(prof[c])
    row = []
    for _, m in CUTS:
        v = (th[m, c] - st[m, c]) / sp[c]
        row.append(f"[{np.percentile(v,15.87):+6.3f},{np.percentile(v,84.13):+6.3f}]")
    a, b = dens_ct(gs[c][ok], prof[c][ok])
    ok2 = ok & np.isfinite(LD[c])
    a2, b2 = dens_ct(gs[c][ok2], prof[c][ok2], LD[c][ok2])
    u = U[:, NC[names[c]]]
    print(f"{names[c]:>20} " + " ".join(row) +
          f" [{a:+6.3f},{b:+6.3f}] [{a2:+6.3f},{b2:+6.3f}] "
          f"[{np.percentile(u,15.87):+6.3f},{np.percentile(u,84.13):+6.3f}]")
