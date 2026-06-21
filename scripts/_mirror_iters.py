"""Does the ACHILLES t-channel mapper catch up to our resonance map with more Vegas iterations?
Sweep nitn for tchannel (ACHILLES budget is nitn=10), with resonance as the reference.  Both use the
ACHILLES-faithful Lepage grid (alpha=1.5).  Report N_eff/N, max/median, sigma."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R


def neffN(w): w = w[w > 0]; return w.sum() ** 2 / np.sum(w ** 2) / len(w)

def measure(samp, iters):
    R.SAMPLER_3BODY = samp
    t0 = time.time()
    grid = R.warmup_vegas(n=100000, iters=iters, nbins=50, alpha=1.5, progress=False)
    tw = time.time() - t0
    ws = []; sig = 0.0
    for sd in range(3):
        r = R.generate(120000, seed=sd, return_events=True, grid=grid)
        ws.append(r["events"]["w"]); sig += r["sigma"]
    w = np.concatenate(ws); w = w[w > 0]
    return tw, sig / 3, neffN(w), w.max() / np.median(w)

print(f"{'sampler':10s} {'nitn':>4s} {'warmup_s':>8s} {'sigma':>11s} {'Neff/N':>7s} {'max/med':>8s}")
for samp, iters in [("tchannel", 6), ("tchannel", 10), ("tchannel", 15),
                    ("resonance", 6), ("resonance", 10)]:
    tw, sig, ne, mx = measure(samp, iters)
    print(f"{samp:10s} {iters:4d} {tw:8.0f} {sig:11.4e} {ne:7.3f} {mx:8.0f}", flush=True)
