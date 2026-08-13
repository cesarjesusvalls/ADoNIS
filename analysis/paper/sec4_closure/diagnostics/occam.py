"""Decisive test: is (marginal - profile) exactly the nuisance-volume (Occam) factor?

To Laplace order   p_marg(th_k) ~ exp(-Dchi2_prof(th_k)/2) * sqrt(det V_nuis(th_k)) .
multisample_profile already stores logdet_Vnuis at every scan node, so this needs no new computation.
"""
import sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.paper import style
from scipy.interpolate import CubicSpline

zp = np.load(style.ALTGEN / "sec4_P1_profile.npz", allow_pickle=True)
gs = np.asarray(zp["grids_sigma"]); pr = np.asarray(zp["prof_dobj"])
ld = np.asarray(zp["logdet_Vnuis"]); sub = [int(k) for k in zp["subset"]]
pn = [str(x) for x in zp["pnames"]]
Zn = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(style.ALTGEN / "sec4_P1_nutsown_*.npz")))]
nmin = min(len(np.asarray(z["u"])) for z in Zn)
U = np.concatenate([np.asarray(z["u"])[-nmin:] for z in Zn])
npn = [str(x) for x in Zn[0]["pnames"]]; nsub = [int(k) for k in Zn[0]["subset"]]
ncol = {npn[k]: c for c, k in enumerate(nsub)}

def hpd(x, d, m=0.6827):
    o = np.argsort(d)[::-1]; c = np.cumsum(d[o]) * (x[1]-x[0]); c /= c[-1]
    t = d[o][min(int(np.searchsorted(c, m)), len(o)-1)]; i = np.where(d >= t)[0]
    return float(x[i[0]]), float(x[i[-1]])
def hpd_s(v, m=0.6827):
    t = np.sort(v); n = t.size; w = max(int(np.ceil(m*n)), 2)
    if w >= n: return float(t[0]), float(t[-1])
    k = int(np.argmin(t[w:] - t[:n-w])); return float(t[k]), float(t[k+w])

print(f"{'dial':>20} {'profile 68%':>18} {'prof x Occam':>18} {'NUTS 68%':>18}   verdict")
for c, k in enumerate(sub):
    ok = np.isfinite(gs[c]) & np.isfinite(pr[c]) & np.isfinite(ld[c])
    if ok.sum() < 5 or pn[k] not in ncol: continue
    g, d, L = gs[c][ok], pr[c][ok], ld[c][ok]
    xf = np.linspace(g.min(), g.max(), 2000)
    dc = CubicSpline(g, d)(xf); lv = CubicSpline(g, L)(xf)
    p_prof = np.exp(-0.5*np.maximum(dc, 0.0))
    p_marg = p_prof * np.exp(0.5*(lv - lv.max()))          # sqrt(det V_nuis)
    a, b = hpd(xf, p_prof/np.trapezoid(p_prof, xf))
    a2, b2 = hpd(xf, p_marg/np.trapezoid(p_marg, xf))
    n1, n2 = hpd_s(U[:, ncol[pn[k]]])
    # does the Occam correction move the profile TOWARD the NUTS answer?
    before = abs(a-n1)+abs(b-n2); after = abs(a2-n1)+abs(b2-n2)
    v = "OCCAM EXPLAINS" if after < 0.5*before else ("closer" if after < before else "no/worse")
    print(f"{pn[k]:>20} [{a:+7.3f},{b:+7.3f}] [{a2:+7.3f},{b2:+7.3f}] [{n1:+7.3f},{n2:+7.3f}]   {v}")
