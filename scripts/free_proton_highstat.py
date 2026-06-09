"""High-stat free-proton (mono 1 GeV) RES sigma, chunked with running-sigma progress, spline interp,
saving the (W,Q2,w) to npz.  w is per-event (sums to N*sigma); sigma = w.sum()/N.  Usage: python ... N nchunks"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import adonis.xsec.dcc_current as dcc
dcc.BATCH_INTERP = "spline"
from scripts.free_proton_gen import generate

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
nch = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out = sys.argv[3] if len(sys.argv) > 3 else "scripts/free_proton_2M.npz"
chunk = N // nch
print(f"generating {N:,} free-proton events (spline, _NORM={dcc._NORM:.6e}) in {nch} chunks", flush=True)
Wl, Ql, wl, sigs = [], [], [], []
t0 = time.time()
for i in range(nch):
    W, Q2, w = generate(chunk, seed=i)
    Wl.append(W); Ql.append(Q2); wl.append(w)
    sigs.append(w.sum() / chunk)               # per-chunk sigma estimate
    run = np.mean(sigs)
    print(f"  [{100*(i+1)//nch:3d}%] chunk {i+1}/{nch}  running sigma={run:.5e} nb  {time.time()-t0:.0f}s", flush=True)
W = np.concatenate(Wl); Q2 = np.concatenate(Ql); w = np.concatenate(wl)
np.savez_compressed(out, W=W, Q2=Q2, w=w)
sig = w.sum() / N; se = np.sqrt(((w / N)**2).sum())
print(f"FINAL ADoNIS free-p sigma = {sig:.6e} +/- {se:.3e} nb ({100*se/sig:.3f}%)  N={N:,}  -> {out}", flush=True)
