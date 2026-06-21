"""Diagnose the residual variance of resonance+Vegas RES weights: which physics structure to map next.
Decompose N_eff/N into amplitude (a2) vs phase-space (J) vs flux, and locate the top-weight events in
Q^2, pion angle wrt q, hadronic W, and phase-space-edge proximity (min sqlam)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R
from adonis.xsec.flux import T2KFlux
from adonis.xsec.backend import flux_factor
from adonis.xsec.dcc_current import exclusive_amps2_batch

R.SAMPLER_3BODY = "resonance"
grid = R.warmup_vegas(n=100000, iters=6, nbins=50, alpha=1.5, progress=False)
def neffN(w): w = w[w > 0]; return w.sum() ** 2 / np.sum(w ** 2) / len(w)

rng = np.random.default_rng(0); flux = T2KFlux(); minE = flux.seed_min_GeV()
A2 = []; FL = []; J = []; W2 = []; Q2 = []; CPI = []; LAM = []
for (ipid, itiz, mNf, ppid, mpi, mstr) in R.CHANNELS:
    s = R._sample_channel(300000, rng, flux, minE, flux.max_energy, R._pi_kin_mass(mpi), mNf, grid=grid)
    v = s["valid"]; idx = np.where(v & (s["energy"] > 2.5) & (s["energy"] < 400) & (s["J"] > 0))[0]
    a2 = np.zeros(len(v))
    a2[idx] = exclusive_amps2_batch(s["k_nu"][idx], s["k_mu"][idx], s["p_struck"][idx],
                                    s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
    fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=mstr))
    iw = 6
    keep = v & (a2 > 0) & np.isfinite(s["J"])
    A2.append(a2[keep]); FL.append(fl[keep]); J.append(s["J"][keep] * iw)
    knu, kmu, pN, ppi = s["k_nu"][keep], s["k_mu"][keep], s["p_N"][keep], s["p_pi"][keep]
    had = pN + ppi; W2.append(np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1), 0, None)))
    q = knu - kmu; Q2.append((-(q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, 1))) / 1e6)
    # pion cos angle wrt q (lab)
    qv = q[:, 1:]; pv = ppi[:, 1:]
    CPI.append(np.sum(qv * pv, 1) / np.clip(np.linalg.norm(qv, axis=1) * np.linalg.norm(pv, axis=1), 1e-9, None))
    P = knu + s["p_struck"][keep]; ss = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, 1)
    sH = had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1)
    laA = R._sqlam(ss, sH, R.M_MU ** 2); laB = R._sqlam(sH, mNf ** 2, mpi ** 2)
    LAM.append(np.minimum(laA, laB))
a2 = np.concatenate(A2); fl = np.concatenate(FL); j = np.concatenate(J)
W = np.concatenate(W2); Q = np.concatenate(Q2); cpi = np.concatenate(CPI); lam = np.concatenate(LAM)
w = a2 * fl * 0.5 * j
print(f"full weight        N_eff/N = {neffN(w):.3f}")
print(f"a2 flat (fl*J)     N_eff/N = {neffN(fl * j):.3f}   <- residual from PHASE-SPACE J (+flux)")
print(f"J  flat (a2*fl)    N_eff/N = {neffN(a2 * fl):.3f}   <- residual from AMPLITUDE a2 (angular+Q2)")
print(f"a2 only            N_eff/N = {neffN(a2):.3f}")
print(f"J  only            N_eff/N = {neffN(j):.3f}")
print(f"fl only            N_eff/N = {neffN(fl):.3f}")
thr = np.percentile(w, 99.5); top = w >= thr
def med(x): return np.median(x)
print(f"\ntop-0.5% weight events vs all (medians):")
print(f"  Q2 [GeV^2]:      top={med(Q[top]):.3f}  all={med(Q):.3f}")
print(f"  |cos(pi,q)|:     top={med(np.abs(cpi[top])):.3f}  all={med(np.abs(cpi)):.3f}  (1=forward/back peak)")
print(f"  cos(pi,q):       top={med(cpi[top]):.3f}  all={med(cpi):.3f}")
print(f"  W [MeV]:         top={med(W[top]):.0f}  all={med(W):.0f}")
print(f"  min sqlam (edge):top={med(lam[top]):.2e}  all={med(lam):.2e}  (small=near edge)")
print(f"  top-0.5% carry {w[top].sum()/w.sum()*100:.1f}% of sigma")
