"""Pool-vs-BFS RES (CC1pi) sanity check on identical primaries: surviving-pi+ fraction (the CC1pi
signal proxy), primary-pion fate split (escape/absorb/convert), and proton multiplicity.  Run before
the 20-seed RES production to confirm the pool RES path is sane vs BFS."""
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

MAT = sys.argv[1] if len(sys.argv) > 1 else "Ar"
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
P = int(sys.argv[3]) if len(sys.argv) > 3 else 10
tg = resolve_targets(MAT)[0][0]
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", NEV, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
pPi = jnp.asarray(a["p_pi"][s]); pN = jnp.asarray(a["p_N"][s])
ppid = jnp.asarray(a["ppid"][s]); ipid = jnp.asarray(a["ipid"][s]); Npid = jnp.asarray(a["Npid"][s])
ww = w[s]
print(f"[gen] {MAT}: {m} live RES events, P={P}", flush=True)


def run(engine, MS):
    cfg = DiscreteCascadeConfig(step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                                pauli=True, early_exit=True, nucleus=tg.density_p,
                                density_n=tg.density_n, configs=tg.configs, engine=engine)
    pterm, nterms, ofl, created = CF.cascade_nucleus(
        pPi, pN, ppid.astype(jnp.int32), ipid.astype(jnp.int32), Npid.astype(jnp.int32),
        cfg, jax.random.PRNGKey(11), P=P, channel="res")
    pp = np.asarray(pterm["pid"]); cp = np.asarray(created["pid"])
    # CC1pi proxy: exactly one pi+ across {primary, created}, no other meson
    PIP, OTHER = 211, (111, -211, -1)
    n_pip = (pp == PIP).astype(int) + (cp == PIP).astype(int)
    n_oth = np.isin(pp, OTHER).astype(int) + np.isin(cp, OTHER).astype(int)
    cc1pi = (n_pip == 1) & (n_oth == 0)
    npr = np.zeros(m)
    for g in nterms:
        sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
        npr += ((np.linalg.norm(p4[:, :, 1:], axis=2) > 0.25) & (sp == CF.NUCLEON) & (pid == 2212) & al).sum(1)
    tot = ww.sum()
    print(f"[{engine:4s}] prim_pi+ surv={ww[pp==PIP].sum()/tot:.4f}  prim_abs={ww[pp==0].sum()/tot:.4f}  "
          f"prim_conv={ww[pp==-1].sum()/tot:.4f}  created_pi+={ww[cp==PIP].sum()/tot:.4f}  "
          f"CC1pi_frac={ww[cc1pi].sum()/tot:.4f}  <Np>={np.average(npr,weights=ww):.4f}  ofl={int(ofl)}", flush=True)


MS = 683 if MAT == "Ar" else 491
run("bfs", MS)
run("pool", MS)
