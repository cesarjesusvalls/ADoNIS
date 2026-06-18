"""Is the pool-engine overflow on DEAD (w=0) events (harmless, zero weight) or LIVE events (biases the
bank)?  Run the pool QE cascade on the FULL unfiltered 30000-event seed and report overflow split by
w>0 vs w==0.  Mirrors run_one_seed (no pre-filter)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "2"
import numpy as np
import jax, jax.numpy as jnp
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction

tg = resolve_targets("Ar")[0][0]
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", 30000, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); m = len(w); live = w > 0
print(f"[gen] {m} events, {int(live.sum())} live ({100*live.mean():.1f}%)", flush=True)
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=683, seed=1, nn_inelastic=True,
                            pauli=True, early_exit=True, nucleus=tg.density_p,
                            density_n=tg.density_n, configs=tg.configs, engine="pool")
pN = jnp.asarray(a["p_N"]); ipid = jnp.asarray(a["ipid"]); Npid = jnp.asarray(a["Npid"])
su = CF.setup_carbon(pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
g0 = CF.empty_batch(m, 1)
g0["alive"] = jnp.ones((m, 1), bool); g0["species"] = jnp.full((m, 1), CF.NUCLEON, jnp.int32)
g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]; g0["p4"] = pN[:, None, :]
g0["pos"] = su["pos0"][:, None, :]
stepper = CF.make_pool_stepper(su, cfg)
_, knuc, _ = jax.random.split(su["kp"], 3)
# per-event overflow: re-run run_cascade_pool but we need per-event counts; instead count terminals vs
# input live-proton avalanche.  Simpler: M_out generous (64) and M generous (32) -> if overflow vanishes
# it was capacity; compare to M=16 to localize.  Here just report total at M=16 split is not per-event,
# so approximate: events whose proton-terminal count is clipped.  Use the out buffer alive-count.
for M, Mout in ((16, 24), (32, 64)):
    out, sofl, oofl = CF.run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=M, max_steps=683, M_out=Mout)
    nterm = np.asarray(out["alive"]).sum(axis=1)               # terminals collected per event
    print(f"[M={M},Mout={Mout}] stack_ofl={int(sofl)} out_ofl={int(oofl)}  "
          f"<nterm live>={nterm[live].mean():.3f}  <nterm dead>={nterm[~live].mean():.3f}  "
          f"max nterm live={nterm[live].max()} dead={nterm[~live].max()}", flush=True)
