"""Generate N ADoNIS RES events and SAVE the full final-state momenta + weights to .npz,
so they can be combined with an ACHILLES hepmc and re-projected into any variable later.
Usage: python scripts/gen_res_events.py <N> <out.npz>"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import adonis.xsec.dcc_current as dcc
from adonis.xsec import res_xsec as M

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
out = sys.argv[2] if len(sys.argv) > 2 else "scripts/res_events_1M.npz"
dcc.BATCH_INTERP = sys.argv[3] if len(sys.argv) > 3 else "spline"   # spline = faithful (default); "bilinear" = fast diagnostic only (W-shape offender)
print(f"generating {N:,} RES events  (interp={dcc.BATCH_INTERP}) -> {out}", flush=True)
nchunks = 25; chunk = max(N // nchunks, 1)
cols = {k: [] for k in ("k_nu", "k_mu", "p_struck", "p_N", "p_pi", "w")}
sig = []; t0 = time.time()
for i in range(nchunks):
    ev = M.generate(chunk, seed=i, return_events=True)["events"]
    for k in cols: cols[k].append(ev[k])
    sig.append(float(ev["w"].sum()))
    print(f"  [gen] {100*(i+1)/nchunks:5.1f}%  ({(i+1)*chunk:,}/{nchunks*chunk:,})  "
          f"running sigma={np.mean(sig):.4e} nb  {time.time()-t0:.0f}s", flush=True)
D = {k: np.concatenate(v) for k, v in cols.items()}
D["w"] = D["w"] / nchunks                       # so total sums to MEAN sigma over chunks
q = D["k_nu"] - D["k_mu"]
D["Q2"] = (np.sum(q[:, 1:]**2, axis=1) - q[:, 0]**2) / 1e6
pcm = D["p_N"] + D["p_pi"]
D["W"] = np.sqrt(np.clip(pcm[:, 0]**2 - np.sum(pcm[:, 1:]**2, axis=1), 0, None))
np.savez_compressed(out, **D)
print(f"  saved {len(D['w'])} events -> {out}  (sigma={D['w'].sum():.4e} nb)", flush=True)
