"""ADoNIS per-NN-scatter vertex dump (radius + outgoing leading & recoil |p|), single-pass carbon
protons, to overlay on the ACHILLES NSCATR distribution.  Saves /tmp/ado_nscatr_C.npz."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _nucleon_step, _load_density
from adonis.fsi.cascade_full import setup_carbon
from adonis.xsec.spectral import SpectralFunction
from adonis.fsi import oset_xsec as ox
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
M_N = float(ox.M_N); MS = 600

tg = resolve_targets("C")[0][0]
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                            pauli=True, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", 40000, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum()); w = w[s]
pN = jnp.asarray(a["p_N"][s]); ipid = jnp.asarray(a["ipid"][s])
su = setup_carbon(pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
rgrid, rhoP, rhoN, radius = _load_density(tg.density_p, tg.density_n)
keys = jax.random.split(jax.random.PRNGKey(99), MS)
p4 = pN; pos = su["pos0"]; d3 = p4[:, 1:]
dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
fz = jnp.zeros(m); alive = jnp.ones(m, bool); consumed = su["consumed0"]; is_p = jnp.ones(m, bool)
R, PL, PR, W = [], [], [], []                                    # radius, lead-out |p|, recoil |p|, weight
for i in range(MS):
    pos_pre = pos
    (p4, pos, dhat, fz, alive), esc, recap, do, ko, pin, consumed, _ = _nucleon_step(
        p4, pos, dhat, fz, is_p, alive, su["npos"], su["nmom"], su["nisp"], consumed,
        rgrid, rhoP, rhoN, radius, cfg, keys[i])
    do = np.asarray(do).astype(bool)
    if do.any():
        r = np.linalg.norm(np.asarray(pos_pre)[do], axis=1)
        pl = np.linalg.norm(np.asarray(p4)[do, 1:], axis=1)       # leading after scatter
        pr = np.linalg.norm(np.asarray(ko[0])[do, 1:], axis=1)    # recoil knockout
        R.append(r); PL.append(pl); PR.append(pr); W.append(w[do])
R = np.concatenate(R); PL = np.concatenate(PL); PR = np.concatenate(PR); W = np.concatenate(W)
np.savez("/tmp/ado_nscatr_C.npz", r=R, pl=PL, pr=PR, w=W, radius=radius)
print(f"ADoNIS NN scatters: {len(R)}  <r>={np.average(R,weights=W):.2f}  radius={radius:.2f}")
print(f"frac scatters beyond r=3.14 (k_F<137): {np.average(R>3.14,weights=W):.3f}")
print("wrote /tmp/ado_nscatr_C.npz")
