"""Decisive test: does the flux-correct mean-free-path rate (lam = rho*sigma*v_rel/v_pion,
which cancels the spurious 1/v_rel in the Oset cross section) bring the continuum cascade's
sigma_abs(p) onto the ACHILLES oracle -- especially in the low-p wings where formula A
(lam = rho*sigma) over-predicts?  Fully vectorised (precomputed scatter sigma on the W grid)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import adonis.fsi.oset_xsec as ox
from adonis.fsi.mb.anl_xsec import load_anl, _channel_sigma
from adonis.fsi.mb.cascade_mb import _CHANNELS

HBARC = ox.HBARC; M_N = ox.M_N; M_PIP = ox.M_PIP; MB = 0.1
rng = np.random.default_rng(0)
d = np.loadtxt("data/nuclear/c12_density.txt", comments="#"); rg, rp = d[:, 0], d[:, 1]
R = float(rg[rp > 1e-4 * rp[0]].max())
rho_p = lambda r: np.interp(r, rg, rp, left=rp[0], right=0.0)
kf = lambda r: np.cbrt(np.clip(rho_p(r), 0, None) * 3 * np.pi ** 2) * HBARC

# precompute pi+ scatter sigma (p/n averaged) on the W grid -> fast interp
Wt, amps = load_anl(0, 0)
_ss = np.zeros_like(Wt)
for nuc in ("p", "n"):
    for po, no, cg in _CHANNELS[(0, nuc)]:
        _ss += 0.5 * np.clip(_channel_sigma(amps, Wt, cg), 0, None)
scat = lambda W: np.interp(W, Wt, _ss, left=0, right=0)

orc = np.loadtxt("data/oracle/cascade_pip_c12_virt_abs.csv")


def run(plab, n=8000, step=0.05, rate="A"):
    b = R * np.sqrt(rng.uniform(size=n)); ang = 2 * np.pi * rng.uniform(size=n)
    pos = np.stack([b * np.cos(ang), b * np.sin(ang),
                    -np.sqrt(np.clip(R ** 2 - b ** 2, 0, None)) * 0.999], 1)
    Epi = np.sqrt(M_PIP ** 2 + plab ** 2); vpion = plab / Epi
    dhat = np.tile([0, 0, 1.], (n, 1)); p_pi = np.tile([Epi, 0, 0, plab], (n, 1)).astype(float)
    alive = np.ones(n, bool); absb = np.zeros(n, bool)
    for _ in range(int(2.5 * (R + 1) / step)):
        r = np.linalg.norm(pos, axis=1); esc = alive & (r > R) & (np.sum(pos * dhat, 1) > 0); alive &= ~esc
        if not alive.any(): break
        rt = 2 * rho_p(r); k = kf(r)
        dirn = rng.normal(size=(n, 3)); dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
        pm = k * rng.uniform(size=n) ** (1 / 3)
        pN = np.concatenate([np.sqrt(M_N ** 2 + pm ** 2)[:, None], dirn * pm[:, None]], 1)
        vrel = np.clip(np.linalg.norm(p_pi[:, 1:] / p_pi[:, 0:1] - pN[:, 1:] / pN[:, 0:1], axis=1), 1e-3, None)
        P = p_pi + pN; W = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, 1), 1., None))
        sa = np.clip(np.array(ox.abs_cross_section(p_pi[:, 0], M_PIP, plab + 0 * r, vrel,
                     np.clip(k, 1e-6, None), np.clip(rt, 1e-9, None))), 0, None)
        ss = scat(W)
        fac = (vrel / vpion) if rate == "B" else 1.0
        lam = rt * (sa + ss) * MB * fac
        pint = -np.expm1(-lam * step)
        inter = alive & (rng.uniform(size=n) < pint)
        isa = inter & (rng.uniform(size=n) < sa / np.clip(sa + ss, 1e-12, None))
        absb |= isa; alive &= ~isa
        pos = pos + step * dhat * alive[:, None]
    return np.pi * R ** 2 * absb.mean() * 10.0


print(f"{'p':>5} {'oracle':>7} {'A(rho*sig)':>11} {'B(*vrel/vp)':>11}")
for p in [95, 155, 215, 275, 335, 395, 485]:
    o = np.interp(p, orc[:, 0], orc[:, 2])
    print(f"{p:5d} {o:7.1f} {run(float(p), rate='A'):11.1f} {run(float(p), rate='B'):11.1f}")
