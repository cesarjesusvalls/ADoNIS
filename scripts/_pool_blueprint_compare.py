"""Stage 3 gate: the pool-backed blueprint reproduces the legacy CC0pi dpt prediction within stats.
Same frozen proposal -> legacy replicas (DiscreteCascadeFSI+DiscreteNucleonFSI, brec/srec) vs pool
replicas (cascade_carbon_v2 pool, joint pool_fsi_reweight).  Compare nominal histogram + a short fit."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from tqdm import tqdm
import scripts.cc0pi_tune_adonis as T
T.NQE = T.NRES = 30000                                          # smaller for a fast gate
NREP = 3

print("building proposal (first jit compile ~1-2 min)...", flush=True)
qe, qw, res, rw = T.build_proposal()
leg = [T.build_replica(jax.random.PRNGKey(50 + i), qe, qw, res, rw)
       for i in tqdm(range(NREP), desc="legacy replicas (1st ~compile)", file=sys.stdout)]
pool = [T.build_replica_pool(jax.random.PRNGKey(50 + i), qe, qw, res, rw)
        for i in tqdm(range(NREP), desc="pool replicas (1st ~compile)", file=sys.stdout)]

th = jnp.array([1.0, 1.0])
hl = np.mean([np.asarray(T.model_hist(th, R)) for R in leg], axis=0)
hp = np.mean([np.asarray(T.model_hist_pool(th, R)) for R in pool], axis=0)
DATA = np.asarray(T.DATA); COVINV = np.asarray(T.COVINV)
def chi2(h):
    A = float((h @ COVINV @ DATA) / (h @ COVINV @ h)); r = A * h - DATA
    return float(r @ COVINV @ r) / (8 - 2), A
c_l, A_l = chi2(hl); c_p, A_p = chi2(hp)
ctr = 0.5 * (T.EDGES[1:] + T.EDGES[:-1])
print("\nbin        legacy        pool      pool/legacy")
for i in range(len(hl)):
    print(f"{ctr[i]:7.0f}  {hl[i]:11.4e}  {hp[i]:11.4e}   {hp[i]/hl[i] if hl[i] else float('nan'):.3f}")
print(f"\nnominal chi2/ndf  legacy={c_l:.2f} (A={A_l:.3f})   pool={c_p:.2f} (A={A_p:.3f})")
print(f"shape ratio pool/legacy: mean={np.mean(hp/hl):.3f} max|dev|={np.max(np.abs(hp/hl-1)):.3f}")
# autodiff sanity on the pool model
g = jax.grad(lambda t: float(jnp.sum(T.model_hist_pool(t, pool[0]))) if False else jnp.sum(T.model_hist_pool(t, pool[0])))(th)
print(f"pool model grad @nominal: {np.asarray(g)}")
print("STAGE3 COMPARE: done")
