"""Find the cheapest warm-up (events x iters) that keeps the Vegas N_eff gain.  amps2 is pure-numpy
and ~constant cost/event, so warm-up wall-time ~ warmup_n*iters*3; the grid (6 axes x 50 bins) is a
coarse marginal estimator that should converge well below 1.8M events."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R


def neffN(w):
    w = np.asarray(w); w = w[w > 0]; return w.sum() ** 2 / np.sum(w ** 2) / len(w)


def test_grid(grid, seeds=4):
    cols = []; sig = 0.0
    for sd in range(seeds):
        r = R.generate(120000, seed=sd, return_events=True, grid=grid)
        cols.append(r["events"]["w"]); sig += r["sigma"]
    w = np.concatenate(cols); return neffN(w), sig / seeds


for (wn, it) in [(100000, 6), (30000, 4), (15000, 4), (15000, 3)]:
    t0 = time.time()
    g = R.warmup_vegas(n=wn, iters=it, nbins=50, progress=False)
    twarm = time.time() - t0
    ne, sig = test_grid(g)
    print(f"warmup_n={wn:6d} iters={it}: warm-up {twarm:5.1f}s  -> N_eff/N={ne:.3f}  sigma={sig:.4e}", flush=True)
