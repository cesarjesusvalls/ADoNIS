"""Pool-vs-BFS QE nucleon-elastic avalanche multiplicity check.

Runs the SAME QE primary bank through (a) the BFS nucleon cascade (cascade_nucleus engine=bfs) and
(b) the pooled engine (make_pool_stepper + run_cascade_pool) on the gen-0 QE stack, and compares
proton(>250 MeV)/event.  Pool is NOT bit-identical to bfs (true step-order consumption) so we expect
ratio ~1.0, not exact.  nn_inelastic toggled via argv[1] (default False = pure elastic avalanche)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax, jax.numpy as jnp
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction

NN = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
M = int(sys.argv[3]) if len(sys.argv) > 3 else 12
MS = int(sys.argv[4]) if len(sys.argv) > 4 else 683
print(f"[cfg] nn_inelastic={NN} nev={NEV} M={M} max_steps={MS}", flush=True)

tg = resolve_targets("Ar")[0][0]
cfg = DiscreteCascadeConfig(step=0.04, max_steps=MS, seed=1, nn_inelastic=NN,
                            pauli=True, early_exit=True, nucleus=tg.density_p,
                            density_n=tg.density_n, configs=tg.configs)
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", NEV, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
pN = jnp.asarray(a["p_N"][s]); ipid = jnp.asarray(a["ipid"][s]); Npid = jnp.asarray(a["Npid"][s])
ppid = jnp.zeros(m, jnp.int32)
print(f"[gen] {m} live QE events", flush=True)

# ---- BFS ----
key = jax.random.PRNGKey(11)
pterm, nterms, ofl, created = CF.cascade_nucleus(
    pN, pN, ppid, ipid.astype(jnp.int32), Npid.astype(jnp.int32), cfg, key,
    P=M, max_gen=6, channel="qe")
bfs_n = np.zeros(m)
for g in nterms:
    sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
    good = (sp == CF.NUCLEON) & (pid == 2212) & al
    mom = np.linalg.norm(p4[:, :, 1:], axis=2)
    bfs_n += ((mom > 0.25) & good).sum(axis=1)
print(f"[BFS]  proton(>250)/ev = {bfs_n.mean():.4f}  overflow={int(ofl)}", flush=True)

# ---- POOL ----
su = CF.setup_nucleus(pN, ppid, ipid.astype(jnp.int32), cfg, key)
g0 = CF.empty_batch(m, 1)
g0["alive"] = jnp.ones((m, 1), bool)
g0["species"] = jnp.full((m, 1), CF.NUCLEON, jnp.int32)
g0["charge"] = (Npid == 2212).astype(jnp.int32)[:, None]
g0["p4"] = pN[:, None, :]
g0["pos"] = su["pos0"][:, None, :]
g0["fz"] = jnp.zeros((m, 1))
stepper = CF.make_pool_stepper(su, cfg)
kpi, knuc, kpi2 = jax.random.split(su["kp"], 3)
out, sofl, oofl, _ = CF.run_cascade_pool(g0, stepper, knuc, su["consumed0"], M, MS, M_out=24)
sp = np.asarray(out["species"]); chg = np.asarray(out["charge"]); p4 = np.asarray(out["p4"]); al = np.asarray(out["alive"])
mom = np.linalg.norm(p4[:, :, 1:], axis=2)
good = (sp == CF.NUCLEON) & (chg == 1) & al
pool_n = ((mom > 0.25) & good).sum(axis=1)
print(f"[POOL] proton(>250)/ev = {pool_n.mean():.4f}  stack_ofl={int(sofl)} out_ofl={int(oofl)}", flush=True)
print(f"[RATIO] pool/bfs = {pool_n.mean()/max(bfs_n.mean(),1e-9):.4f}", flush=True)
