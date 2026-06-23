"""Stage 3a: cascade_nucleus(rec_caps=...) returns the joint per-event FSI record; nominal==1; the
4-tuple default is unchanged."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction

tg = resolve_targets("C")[0][0]; P = 12; MG = 6
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", 5000, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
g = lambda k: jnp.asarray(a[k][s])
cfg = DiscreteCascadeConfig(step=0.04, max_steps=600, seed=1, nn_inelastic=True, pauli=True,
                            early_exit=True, nucleus=tg.density_p, density_n=tg.density_n,
                            configs=tg.configs, engine="pool")
key = jax.random.PRNGKey(11)
# 4-tuple default unchanged
r4 = CF.cascade_nucleus(g("p_pi"), g("p_N"), g("ppid").astype(jnp.int32), g("ipid").astype(jnp.int32),
                          g("Npid").astype(jnp.int32), cfg, key, P=P, max_gen=MG, channel="res")
print(f"[4-tuple default] len={len(r4)}  (expect 4)")
# 5-tuple with record
r5 = CF.cascade_nucleus(g("p_pi"), g("p_N"), g("ppid").astype(jnp.int32), g("ipid").astype(jnp.int32),
                          g("Npid").astype(jnp.int32), cfg, key, P=P, max_gen=MG, channel="res",
                          rec_caps=(32, 256))
pterm, nterms, ofl, created, rec = r5
print(f"[5-tuple with rec] len={len(r5)}  rec keys={sorted(rec.keys())}")
wnom = np.asarray(CF.pool_fsi_reweight(rec, 1.0, 1.0))
print(f"[engine record nominal] n={len(wnom)} (events={m})  maxdev|w-1|={np.abs(wnom-1).max():.2e}"
      f"  nh_max={int(rec['nh'].max())} ns_max={int(rec['ns'].max())}")
# forward outputs identical between 4-tuple and 5-tuple paths (same key)
same = np.array_equal(np.asarray(r4[0]["pid"]), np.asarray(pterm["pid"])) and \
       np.array_equal(np.asarray(r4[1][0]["p4"]), np.asarray(nterms[0]["p4"]))
print(f"[forward identical 4-tuple vs 5-tuple] {same}")
print("STAGE3a GATE:", "PASS" if (len(r4) == 4 and len(r5) == 5 and len(wnom) == m
                                  and np.abs(wnom-1).max() < 1e-12 and same) else "FAIL")
