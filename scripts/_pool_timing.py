"""Attribute the pool tuning-path cost: time RES cascade (compile=1st call, run=2nd call) for
pool {with_rec off/on} x {max_steps 260/600} vs the legacy DiscreteCascadeFSI+DiscreteNucleonFSI,
all at the same N.  Tells us what actually makes the differentiable pool path slow."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, DiscreteCascadeFSI, DiscreteNucleonFSI
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction
from adonis.core.event import EventRecord

N = 5000
tg = resolve_targets("C")[0][0]
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", N, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
g = lambda k: jnp.asarray(a[k][s])
print(f"[gen] {m} live RES events\n", flush=True)

def cfg(ms, eng="pool"):
    return DiscreteCascadeConfig(cylinder=False, step=0.04, max_steps=ms, seed=1, nn_inelastic=True,
                                 pauli=True, early_exit=True, nucleus=tg.density_p, density_n=tg.density_n,
                                 configs=tg.configs, engine=eng)

def time_pool(ms, rec):
    c = cfg(ms); k = jax.random.PRNGKey(11); caps = (32, 64) if rec else None
    def run(key):
        r = CF.cascade_carbon_v2(g("p_pi"), g("p_N"), g("ppid").astype(jnp.int32), g("ipid").astype(jnp.int32),
                                 jnp.full(m, 2212, jnp.int32), c, key, P=12, max_gen=6, channel="res", rec_caps=caps)
        return jax.block_until_ready(r[1][0]["p4"])
    t0 = time.time(); run(k); t_compile = time.time() - t0
    t0 = time.time(); run(jax.random.PRNGKey(12)); t_run = time.time() - t0
    print(f"  pool  max_steps={ms} with_rec={str(rec):5s}: compile={t_compile:6.1f}s  run={t_run:6.1f}s", flush=True)

def time_legacy():
    ev = EventRecord(k=g("k_nu"), kp=g("k_mu"), p_struck=g("p_struck"), p_pi=g("p_pi"), p_N=g("p_N"),
                     w=jnp.ones(m), channel=jnp.zeros(m, jnp.int32), pid_pi=g("ppid").astype(jnp.int32),
                     pid_N=jnp.full(m, 2212, jnp.int32), pid_Ni=g("ipid").astype(jnp.int32),
                     W=jnp.zeros(m), Q2_adj=jnp.zeros(m))
    C = lambda: DiscreteCascadeConfig(cylinder=False, step=0.04, max_steps=260, nucleus=tg.density_p,
                                      density_n=tg.density_n, configs=tg.configs)
    def run(seedoff):
        pion = DiscreteCascadeFSI(C()); e2 = pion.apply(None, ev, key=jax.random.PRNGKey(1 + seedoff), sabs=1.0, sscat=1.0)
        nf = DiscreteNucleonFSI(C()); e3 = nf.apply(None, e2, key=jax.random.PRNGKey(2 + seedoff), sscat=1.0)
        return jax.block_until_ready(e3.p_N)
    t0 = time.time(); run(0); t_compile = time.time() - t0
    t0 = time.time(); run(10); t_run = time.time() - t0
    print(f"  legacy(seg) max_steps=260           : compile={t_compile:6.1f}s  run={t_run:6.1f}s", flush=True)

print("=== timing (compile=1st call, run=2nd call), N={} ===".format(m), flush=True)
time_legacy()
time_pool(260, False)
time_pool(260, True)
time_pool(600, False)
time_pool(600, True)
print("\nDONE timing", flush=True)
