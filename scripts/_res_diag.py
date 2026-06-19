"""Isolate the pool RES over-absorption: compare primary-pion absorption for
  (a) BFS pion_segment (validated),
  (b) pool with g0 = primary pion ONLY (no recoil nucleon),
  (c) pool with g0 = pion + recoil (full RES).
If (b)~(a): the recoil/consumed-sharing causes it.  If (b)>>(a): the pion physics in the pool loop
differs (RNG/bug)."""
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

tg = resolve_targets("Ar")[0][0]; MS = 683; P = 10
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", 8000, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
pPi = jnp.asarray(a["p_pi"][s]); pN = jnp.asarray(a["p_N"][s])
ppid = jnp.asarray(a["ppid"][s]); ipid = jnp.asarray(a["ipid"][s]); Npid = jnp.asarray(a["Npid"][s])
ww = w[s]; tot = ww.sum()
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                            pauli=True, early_exit=True, nucleus=tg.density_p,
                            density_n=tg.density_n, configs=tg.configs, engine="pool")
su = CF.setup_carbon(pPi, ppid.astype(jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
_, knuc, _ = jax.random.split(su["kp"], 3)
print(f"[gen] {m} live RES events", flush=True)

# (a) BFS pion_segment
out = CF._propagate_discrete  # noqa
from adonis.fsi.cascade_full import pion_segment
pt, _, _ = pion_segment(pPi, su["pos0"], su["ch0"], su["consumed0"], su["npos"], su["nmom"], su["nisp"],
                        cfg, jax.random.split(su["kp"], 3)[0], 1.0, 1.0)
pp = np.asarray(pt["pid"]); bnsc = np.asarray(pt["nsc"])
esc_b = pp > 0
print(f"[a BFS pion_segment] abs={ww[pp==0].sum()/tot:.4f} surv_pip={ww[pp==211].sum()/tot:.4f} "
      f"<nsc all>={np.average(bnsc,weights=ww):.3f} <nsc esc>={np.average(bnsc[esc_b],weights=ww[esc_b]):.3f}", flush=True)

# (b) pool, pion-only g0
g0 = CF.empty_batch(m, 1)
g0["alive"] = jnp.ones((m, 1), bool); g0["species"] = jnp.full((m, 1), CF.PION, jnp.int32)
g0["charge"] = su["ch0"][:, None]; g0["p4"] = pPi[:, None, :]; g0["pos"] = su["pos0"][:, None, :]
g0["origin"] = jnp.full((m, 1), CF._ORIG_PRIM_PI, jnp.int32)
stepper = CF.make_pool_stepper(su, cfg)
o, sof, oof, pf = CF.run_cascade_pool(g0, stepper, knuc, su["consumed0"], M=P, max_steps=MS, M_out=24,
                                      prim_origin=CF._ORIG_PRIM_PI)
pf = np.asarray(pf)
# escaped primary pion nsc from out (origin-tagged surviving pion)
spo = np.asarray(o["species"]); alo = np.asarray(o["alive"]); oro = np.asarray(o["origin"]); nsco = np.asarray(o["nsc"])
isp = (spo == CF.PION) & alo & (oro == CF._ORIG_PRIM_PI)
arr = np.arange(m); jp = np.argmax(isp, axis=1); has = np.any(isp, axis=1)
pnsc = nsco[arr, jp]
print(f"[b pool pion-only] abs(fate)={ww[pf==CF.FATE_ABSORB].sum()/tot:.4f} "
      f"esc(fate)={ww[pf==CF.FATE_ESCAPE].sum()/tot:.4f} conv={ww[pf==CF.FATE_CONVERT].sum()/tot:.4f} "
      f"none={ww[pf==CF.FATE_NONE].sum()/tot:.4f} <nsc esc>={np.average(pnsc[has],weights=ww[has]):.3f} "
      f"ofl={int(sof)}", flush=True)
