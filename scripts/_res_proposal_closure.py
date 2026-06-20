"""Closure: the resonance-importance RES proposal reproduces the tchannel proposal's PREDICTIONS
(sigma + binned distributions) while raising N_eff.  Pure proposal change -> same expectation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R


def obs(ev):
    knu, kmu, pN, ppi = ev["k_nu"], ev["k_mu"], ev["p_N"], ev["p_pi"]
    had = pN + ppi
    W = np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1), 0, None))
    q = knu - kmu
    Q2 = -(q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, 1)) / 1e6
    pip = np.linalg.norm(ppi[:, 1:], axis=1)
    pmu = np.linalg.norm(kmu[:, 1:], axis=1)
    return W, Q2, pip, pmu


def acc(samp, N=120000, seeds=range(6)):
    R.SAMPLER_3BODY = samp
    cols = [[], [], [], [], []]
    for sd in seeds:
        e = R.generate(N, seed=sd, return_events=True)["events"]
        a, b, c, d = obs(e)
        for col, v in zip(cols, (a, b, c, d, e["w"])):
            col.append(v)
    return [np.concatenate(x) for x in cols]


A = acc("tchannel")
B = acc("resonance")
defs = [("W_had", np.linspace(1080, 1800, 13)), ("Q2", np.linspace(0, 1.5, 11)),
        ("pi_p", np.linspace(0, 1000, 11)), ("p_mu", np.linspace(0, 2000, 11))]
print("resonance vs tchannel predictions (6 seeds x 120k each):")
for i, (nm, ed) in enumerate(defs):
    ha = np.histogram(A[i], bins=ed, weights=A[4])[0]
    hb = np.histogram(B[i], bins=ed, weights=B[4])[0]
    sa = np.sqrt(np.histogram(A[i], bins=ed, weights=A[4] ** 2)[0])
    sb = np.sqrt(np.histogram(B[i], bins=ed, weights=B[4] ** 2)[0])
    sel = ha > ha.max() * 0.02
    r = hb[sel] / np.clip(ha[sel], 1e-30, None)
    nsig = np.abs(hb - ha) / np.clip(np.sqrt(sa ** 2 + sb ** 2), 1e-30, None)
    print(f"  {nm:6s} ratio min/max={r.min():.3f}/{r.max():.3f}  worst |Δ|/σ={nsig[sel].max():.2f}")
print(f"sigma: tchannel={A[4].sum()/6:.4e}  resonance={B[4].sum()/6:.4e}  ratio={B[4].sum()/A[4].sum():.4f}")
