"""Test the Q^2 map: rotate split-A polar axis to the CM-neutrino direction so ctA ~ linear in Q^2 and
Vegas resolves the leptonic/Q^2 structure.  Compare resonance (baseline) / resonance_q2 (Q^2 on axis) /
resonance_q2qrel (Q^2 + q-relative pion angle).  sigma must agree (basis rotations); watch N_eff/N.
Progress printed per sampler."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from adonis.xsec import res_xsec as R
def neffN(w): w=w[w>0]; return w.sum()**2/np.sum(w**2)/len(w)
print(f"{'sampler':18s} {'Neff/N':>7s} {'max/med':>8s} {'sigma':>11s}")
for samp in ("resonance","resonance_q2","resonance_q2qrel"):
    R.SAMPLER_3BODY = samp; t0=time.time()
    print(f"  [{samp}] warming up...", flush=True)
    g = R.warmup_vegas(n=100000, iters=6, nbins=50, alpha=1.5, progress=False)
    ws=[]; sig=0.0
    for sd in range(4):
        r = R.generate(120000, seed=sd, return_events=True, grid=g); ws.append(r["events"]["w"]); sig+=r["sigma"]
    w=np.concatenate(ws); w=w[w>0]
    print(f"{samp:18s} {neffN(w):7.3f} {w.max()/np.median(w):8.0f} {sig/4:11.4e}  ({time.time()-t0:.0f}s)", flush=True)
