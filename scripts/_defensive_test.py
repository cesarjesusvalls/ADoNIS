"""Prototype the DEFENSIVE MIXTURE on the RES proposal (reuses the cached C grid res_vegasgrid_C12.npz).
Sweep the defensive fraction; report sigma (unbiased check), N_eff/N, the weight tail (max/median,
p99.9/median), and the worst per-bin single-event fraction (on the hadronic-W spectrum)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R
from adonis.xsec.vegas_grid import VegasGrid

grid = VegasGrid.load("data/oracle/res_vegasgrid_C12.npz")
def neffN(w): w = w[w > 0]; return w.sum() ** 2 / np.sum(w ** 2) / len(w)
edges = np.linspace(1080, 1800, 25)

def run(defv, seeds=3, n=120000):
    ws = []; Ws = []; sig = 0.0
    for sd in range(seeds):
        r = R.generate(n, seed=sd, return_events=True, grid=grid, defensive=defv)
        e = r["events"]; w = e["w"]; had = e["p_N"] + e["p_pi"]
        W = np.sqrt(np.clip(had[:, 0] ** 2 - np.sum(had[:, 1:] ** 2, 1), 0, None))
        ws.append(w); Ws.append(W); sig += r["sigma"]
    w = np.concatenate(ws); W = np.concatenate(Ws); sig /= seeds
    # worst per-bin single-event fraction on W
    idx = np.clip(np.digitize(W, edges) - 1, 0, len(edges) - 2)
    worst = 0.0
    for b in range(len(edges) - 1):
        m = idx == b; s = w[m].sum()
        if s > 0: worst = max(worst, w[m].max() / s)
    med = np.median(w[w > 0])
    return sig, neffN(w), w.max() / med, np.percentile(w, 99.9) / med, worst, len(w)

print(f"{'defensive':>9s} {'sigma':>11s} {'Neff/N':>7s} {'max/med':>8s} {'p99.9/med':>9s} {'worst bin%':>10s} {'Nsel':>7s}")
for defv in (0.0, 0.05, 0.1, 0.2):
    sig, ne, mx, p999, worst, nsel = run(defv)
    print(f"{defv:9.2f} {sig:11.4e} {ne:7.3f} {mx:8.0f} {p999:9.0f} {worst*100:9.1f}% {nsel:7d}", flush=True)
