"""Generate model (diffpi faithful fold) events and SAVE them event-level, so the
comparison can be re-binned at any resolution without re-folding.

Runs dcc_fold_full in chunks, keeps the physical (weight>0) events, concatenates and
saves (W, Q2, weight) to model_nu_events.npz. Mirrors the oracle's event-level storage
(oracle_nu_events.npz) so both sides can be histogrammed identically downstream.

Run:  python make_model_events.py            # default size below
"""
import numpy as np
import jax
import jax.numpy as jnp

from diffpi.dcc import DCCKnobs
from diffpi.hadron_xsec import HadronStructure
from diffpi.dcc_fold_full import fold_full_events

jax.config.update("jax_enable_x64", True)

N_CHUNK, N_CHUNKS = 250_000, 10            # 2.5M (spline gather ~1GB/chunk); scale up freely
OUT = "model_nu_events.npz"

hs = HadronStructure(n_theta=16, n_phi=16)
Ws, Q2s, ws = [], [], []
for c in range(N_CHUNKS):
    W, Q2, w = fold_full_events(DCCKnobs(), jax.random.PRNGKey(1000 + c), n=N_CHUNK, hs=hs)
    W, Q2, w = np.asarray(W), np.asarray(Q2), np.asarray(w)
    keep = w != 0.0
    Ws.append(W[keep]); Q2s.append(Q2[keep]); ws.append(w[keep])
    print(f"  chunk {c+1}/{N_CHUNKS} done ({keep.sum()/len(w)*100:.0f}% kept)", end="\r")

W = np.concatenate(Ws); Q2 = np.concatenate(Q2s); w = np.concatenate(ws)
np.savez(OUT, W=W, Q2=Q2, w=w, n_generated=N_CHUNK * N_CHUNKS)
print(f"\ngenerated {N_CHUNK*N_CHUNKS:,}, kept {len(w):,} physical events -> {OUT}")
