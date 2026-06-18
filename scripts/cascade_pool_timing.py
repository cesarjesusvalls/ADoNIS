"""Wall-clock pool-vs-BFS for the QE cascade (post-JIT: run twice, report the 2nd call).

The pool's whole motivation is speed; this is the make-or-break measurement.  Same primaries, same
P/max_steps, nn_inelastic=True (production)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction

NEV = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
P = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MS = int(sys.argv[3]) if len(sys.argv) > 3 else 683
tg = resolve_targets("Ar")[0][0]
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", NEV, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
pN = jnp.asarray(a["p_N"][s]); ipid = jnp.asarray(a["ipid"][s]); Npid = jnp.asarray(a["Npid"][s])
print(f"[gen] {m} live QE events, P={P} ms={MS}", flush=True)


def run(engine):
    cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                                pauli=True, early_exit=True, nucleus=tg.density_p,
                                density_n=tg.density_n, configs=tg.configs, engine=engine)
    def go():
        out = CF.cascade_carbon_v2(pN, pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32),
                                   Npid.astype(jnp.int32), cfg, jax.random.PRNGKey(11),
                                   P=P, max_gen=6, channel="qe")
        jax.block_until_ready(out[1][0]["p4"]); return out
    t0 = time.time(); go(); t_jit = time.time() - t0      # 1st: compile + run
    t0 = time.time(); go(); t_run = time.time() - t0      # 2nd: run only
    return t_jit, t_run


for eng in ("bfs", "pool"):
    tj, tr = run(eng)
    print(f"[{eng:4s}] jit+run={tj:6.1f}s   run-only={tr:6.1f}s   ({1000*tr/m:.2f} ms/event)", flush=True)
