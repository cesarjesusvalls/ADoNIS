"""Vegas RES validation gates:
1. grid=None determinism + sigma == resonance baseline (refactor is behaviour-preserving).
2. warm-up + grid: sigma closure (same expectation) and N_eff gain (full bank).
3. distribution closure: W/Q2/pi_p/p_mu grid-vs-nogrid shapes within stats."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R


def neffN(w):
    w = np.asarray(w); w = w[w > 0]; return w.sum() ** 2 / np.sum(w ** 2) / len(w)


def obs(ev):
    knu, kmu, pN, ppi = ev["k_nu"], ev["k_mu"], ev["p_N"], ev["p_pi"]
    had = pN + ppi
    W = np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1), 0, None))
    q = knu - kmu; Q2 = -(q[:, 0] ** 2 - np.sum(q[:, 1:] ** 2, 1)) / 1e6
    pip = np.linalg.norm(ppi[:, 1:], axis=1); pmu = np.linalg.norm(kmu[:, 1:], axis=1)
    return W, Q2, pip, pmu


# --- Gate 1: determinism + baseline ---
a = R.generate(40000, seed=0, return_events=True, grid=None)
b = R.generate(40000, seed=0, return_events=True, grid=None)
det = np.array_equal(a["events"]["w"], b["events"]["w"]) and abs(a["sigma"] - b["sigma"]) == 0
print(f"[gate1] grid=None deterministic: {det}")
print(f"[gate1] grid=None sigma={a['sigma']:.5e} (resonance baseline ~1.66e-5)")

# --- build the grid ---
print("[warmup] building VegasGrid (6 iters x 100k)...", flush=True)
grid = R.warmup_vegas(n=100000, iters=6, nbins=50)

# --- Gate 2/3: sigma closure + N_eff gain + distributions, 6 seeds each ---
def acc(use_grid):
    cols = [[], [], [], [], []]
    sig = 0.0
    for sd in range(6):
        r = R.generate(120000, seed=sd, return_events=True, grid=(grid if use_grid else None))
        e = r["events"]; W, Q2, pip, pmu = obs(e)
        for c, v in zip(cols, (W, Q2, pip, pmu, e["w"])):
            c.append(v)
        sig += r["sigma"]
    return [np.concatenate(c) for c in cols], sig / 6

(A, sigA) = acc(False)   # no grid (resonance only)
(B, sigB) = acc(True)    # + vegas grid
print(f"[gate2] sigma  nogrid={sigA:.5e}  vegas={sigB:.5e}  ratio={sigB/sigA:.4f}")
print(f"[gate4] N_eff/N  nogrid={neffN(A[4]):.3f}  vegas={neffN(B[4]):.3f}  gain={neffN(B[4])/neffN(A[4]):.1f}x")
defs = [("W", np.linspace(1080, 1800, 13)), ("Q2", np.linspace(0, 1.5, 11)),
        ("pi_p", np.linspace(0, 1000, 11)), ("p_mu", np.linspace(0, 2000, 11))]
print("[gate3] distribution closure (vegas/nogrid):")
for i, (nm, ed) in enumerate(defs):
    ha = np.histogram(A[i], bins=ed, weights=A[4])[0]; hb = np.histogram(B[i], bins=ed, weights=B[4])[0]
    sa = np.sqrt(np.histogram(A[i], bins=ed, weights=A[4] ** 2)[0])
    sb = np.sqrt(np.histogram(B[i], bins=ed, weights=B[4] ** 2)[0])
    sel = ha > ha.max() * 0.02
    r = hb[sel] / np.clip(ha[sel], 1e-30, None)
    nsig = np.abs(hb - ha) / np.clip(np.sqrt(sa ** 2 + sb ** 2), 1e-30, None)
    print(f"   {nm:5s} ratio {r.min():.3f}/{r.max():.3f}  worst |d|/sigma={nsig[sel].max():.2f}")
