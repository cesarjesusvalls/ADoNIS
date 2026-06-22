"""Decisive test for 'why was the old pool fast?': production ran the pool with cylinder=True (cheap
where() hit), the all-Gaussian path uses exp() every step.  Time pool cylinder True vs False at the
same N/max_steps/with_rec=False (the unchanged production engine).  Also legacy as the reference."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction

tg = resolve_targets("C")[0][0]
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", 5000, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
g = lambda k: jnp.asarray(a[k][s])
print(f"[gen] {m} live RES events\n=== pool cylinder vs gaussian, max_steps=260, with_rec=False, N={m} ===", flush=True)

def time_pool(cyl):
    c = DiscreteCascadeConfig(cylinder=cyl, step=0.04, max_steps=260, seed=1, nn_inelastic=True, pauli=True,
                              early_exit=True, nucleus=tg.density_p, density_n=tg.density_n,
                              configs=tg.configs, engine="pool")
    def run(key):
        r = CF.cascade_carbon_v2(g("p_pi"), g("p_N"), g("ppid").astype(jnp.int32), g("ipid").astype(jnp.int32),
                                 jnp.full(m, 2212, jnp.int32), c, key, P=12, max_gen=6, channel="res")
        return jax.block_until_ready(r[1][0]["p4"])
    t0 = time.time(); run(jax.random.PRNGKey(11)); tc = time.time() - t0
    t0 = time.time(); run(jax.random.PRNGKey(12)); tr = time.time() - t0
    print(f"  pool cylinder={str(cyl):5s}: compile={tc:6.1f}s  run={tr:6.1f}s", flush=True)

time_pool(True)
time_pool(False)
print("\nDONE", flush=True)
