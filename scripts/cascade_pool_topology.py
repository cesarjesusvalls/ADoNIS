"""End-to-end pool-vs-BFS QE CC0pi proton TOPOLOGY (0p/1p/2p) on identical primaries.

Confirms the S2b-integrate pool path in cascade_nucleus runs and quantifies the engine difference on
the observable that matters (proton multiplicity per event), weighted, with binomial uncertainties."""
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

NEV = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
P = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MS = int(sys.argv[3]) if len(sys.argv) > 3 else 683
PCUT = 0.25
tg = resolve_targets("Ar")[0][0]
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", NEV, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
pN = jnp.asarray(a["p_N"][s]); ipid = jnp.asarray(a["ipid"][s]); Npid = jnp.asarray(a["Npid"][s])
ww = w[s]
print(f"[gen] {m} live QE events, P={P} ms={MS}", flush=True)


def topo(engine):
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                                pauli=True, early_exit=True, nucleus=tg.density_p,
                                density_n=tg.density_n, configs=tg.configs, engine=engine)
    pterm, nterms, ofl, created = CF.cascade_nucleus(
        pN, pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32), Npid.astype(jnp.int32),
        cfg, jax.random.PRNGKey(11), P=P, max_gen=6, channel="qe")
    npr = np.zeros(m)
    for g in nterms:
        sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
        mom = np.linalg.norm(p4[:, :, 1:], axis=2)
        npr += ((mom > PCUT) & (sp == CF.NUCLEON) & (pid == 2212) & al).sum(axis=1)
    pisurv = np.asarray(created["alive"]) & (np.asarray(created["pid"]) != 0)
    return npr, int(ofl), pisurv


for eng in ("bfs", "pool"):
    npr, ofl, pisurv = topo(eng)
    tot = ww.sum()
    def frac(mask):
        f = ww[mask].sum() / tot
        return f, np.sqrt(f * (1 - f) / m)
    f0, e0 = frac(npr == 0); f1, e1 = frac(npr == 1); f2, e2 = frac(npr >= 2)
    print(f"[{eng:4s}] <Np>={np.average(npr, weights=ww):.4f}  "
          f"0p={f0:.4f}±{e0:.4f}  1p={f1:.4f}±{e1:.4f}  2p+={f2:.4f}±{e2:.4f}  "
          f"pi_surv={ww[pisurv].sum()/tot:.4f}  ofl={ofl}", flush=True)
