"""Head-to-head: ACHILLES-mirrored (t-channel mapper) vs our (resonance/BW) RES proposal, BOTH with the
new Lepage-damped Vegas grid (alpha=1.5).  Build a grid for each, generate, compare sigma (must agree --
same physics), full-bank N_eff/N, and the weight tail (max/median, p99.9/median)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R


def neffN(w): w = w[w > 0]; return w.sum() ** 2 / np.sum(w ** 2) / len(w)

print(f"{'sampler':12s} {'warmup_s':>8s} {'sigma':>11s} {'Neff/N':>7s} {'max/med':>8s} {'p99.9/med':>9s}")
for samp in ("tchannel", "resonance"):
    R.SAMPLER_3BODY = samp
    t0 = time.time()
    grid = R.warmup_vegas(n=100000, iters=6, nbins=50, alpha=1.5, progress=False)
    tw = time.time() - t0
    ws = []; sig = 0.0
    for sd in range(3):
        r = R.generate(120000, seed=sd, return_events=True, grid=grid)
        ws.append(r["events"]["w"]); sig += r["sigma"]
    w = np.concatenate(ws); w = w[w > 0]; med = np.median(w)
    print(f"{samp:12s} {tw:8.0f} {sig/3:11.4e} {neffN(w):7.3f} {w.max()/med:8.0f} "
          f"{np.percentile(w,99.9)/med:9.0f}", flush=True)
