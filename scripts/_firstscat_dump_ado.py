"""ADoNIS single-pass: per PRIMARY proton, the radius of its FIRST elastic scatter (where, in
ACHILLES's model, it would be 'consumed') + initial |p| + whether it scattered.  Matched to the
ACHILLES FATE 'posr' (the consumed/terminal radius).  Saves /tmp/ado_firstscat_C.npz."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _nucleon_step, _load_density
from adonis.fsi.cascade_full import setup_nucleus
from adonis.xsec.spectral import SpectralFunction
from adonis.fsi import oset_xsec as ox
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
M_N = float(ox.M_N); MS = 600
tg = resolve_targets("C")[0][0]
cfg = DiscreteCascadeConfig(step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                            pauli=True, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
sf_n = SpectralFunction(tg.spectral_n)
a = gen_events("qe", 40000, 0, sf_n=sf_n, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum()); w = w[s]
pN = jnp.asarray(a["p_N"][s]); ipid = jnp.asarray(a["ipid"][s])
su = setup_nucleus(pN, jnp.zeros(m, jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
rgrid, rhoP, rhoN, radius = _load_density(tg.density_p, tg.density_n)
keys = jax.random.split(jax.random.PRNGKey(99), MS)
p4 = pN; pos = su["pos0"]; d3 = p4[:, 1:]
dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
fz = jnp.zeros(m); alive = jnp.ones(m, bool); consumed = su["consumed0"]; is_p = jnp.ones(m, bool)
first_r = jnp.full(m, -1.0)                                       # radius of first scatter (-1 = never)
for i in range(MS):
    pos_pre = pos
    (p4, pos, dhat, fz, alive, _qcx), esc, recap, do, ko, pin, consumed, _ = _nucleon_step(
        p4, pos, dhat, fz, is_p, alive, su["npos"], su["nmom"], su["nisp"], consumed,
        rgrid, rhoP, rhoN, radius, cfg, keys[i])
    r = jnp.linalg.norm(pos_pre, axis=1)
    newly = (do > 0) & (first_r < 0)
    first_r = jnp.where(newly, r, first_r)
first_r = np.asarray(first_r)
p_init = np.linalg.norm(np.asarray(pN)[:, 1:], axis=1)
np.savez("/tmp/ado_firstscat_C.npz", first_r=first_r, p_init=p_init, w=w, radius=float(radius))
scat = first_r >= 0
print(f"ADoNIS scattered primaries: {np.average(scat,weights=w):.4f}  <first-scat r>={np.average(first_r[scat],weights=w[scat]):.2f} fm")
print("wrote /tmp/ado_firstscat_C.npz")
