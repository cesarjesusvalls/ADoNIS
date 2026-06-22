"""Stage 1 gate: pool emits kind-1 FSI records; pool_fsi_reweight(record, 1,1) == 1 bit-exact, and
the forward output is byte-identical with vs without record accumulation (records don't touch the walk)."""
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

tg = resolve_targets("C")[0][0]; MS = 600; P = 12; KP, KN = 32, 256
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", 6000, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); s = w > 0; m = int(s.sum())
pPi = jnp.asarray(a["p_pi"][s]); pN = jnp.asarray(a["p_N"][s])
ppid = jnp.asarray(a["ppid"][s]); ipid = jnp.asarray(a["ipid"][s]); Npid = jnp.asarray(a["Npid"][s])
cfg = DiscreteCascadeConfig(step=0.04, max_steps=MS, seed=1, nn_inelastic=True,
                            pauli=True, early_exit=True, nucleus=tg.density_p,
                            density_n=tg.density_n, configs=tg.configs, engine="pool")
su = CF.setup_carbon(pPi, ppid.astype(jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
_, knuc, _ = jax.random.split(su["kp"], 3)
print(f"[gen] {m} live RES events", flush=True)

# full RES g0 = primary pion + recoil nucleon
g0 = CF.empty_batch(m, 2)
g0["alive"] = jnp.ones((m, 2), bool)
g0["species"] = jnp.array([CF.PION, CF.NUCLEON], jnp.int32)[None, :] * jnp.ones((m, 1), jnp.int32)
g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
g0["p4"] = jnp.stack([pPi, pN], axis=1)
g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (m, 2, 3))
g0["origin"] = jnp.array([CF._ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((m, 1), jnp.int32)
# (1) without records (baseline forward) -- stepper built with_rec=False (forward path, no overhead)
o0, sof0, oof0, pf0 = CF.run_cascade_pool(g0, CF.make_pool_stepper(su, cfg), knuc, su["consumed0"],
                                          M=P, max_steps=MS, M_out=24, prim_origin=CF._ORIG_PRIM_PI)
# (2) with records -- stepper built with_rec=True
o1, sof1, oof1, pf1, (rec, rofl) = CF.run_cascade_pool(
    g0, CF.make_pool_stepper(su, cfg, with_rec=True), knuc, su["consumed0"], M=P, max_steps=MS, M_out=24,
    prim_origin=CF._ORIG_PRIM_PI, rec_caps=(KP, KN))

# forward identical?
same = all(np.array_equal(np.asarray(o0[k]), np.asarray(o1[k])) for k in o0)
print(f"[forward identical with/without records] {same}  (sof {int(sof0)}={int(sof1)}, oof {int(oof0)}={int(oof1)}, pf {np.array_equal(np.asarray(pf0),np.asarray(pf1))})")

# nominal reweight == 1?
wnom = np.asarray(CF.pool_fsi_reweight(rec, 1.0, 1.0))
print(f"[record overflow] {int(rofl)}  pion nh: max={int(rec['nh'].max())} mean={float(rec['nh'].mean()):.2f}"
      f"   nucleon ns: max={int(rec['ns'].max())} mean={float(rec['ns'].mean()):.2f}")
print(f"[pool_fsi_reweight(1,1)] min={wnom.min():.3e} max={wnom.max():.3e} maxdev|w-1|={np.abs(wnom-1).max():.3e}")
print("STAGE1 GATE:", "PASS" if (same and np.abs(wnom-1).max() < 1e-12 and int(rofl) == 0) else "FAIL")
# show off-nominal spread (sanity: reweight actually moves)
for th in [(1.3, 1.0), (1.0, 0.7), (1.3, 0.7)]:
    wt = np.asarray(CF.pool_fsi_reweight(rec, th[0], th[1]))
    print(f"  reweight{th}: mean={wt.mean():.4f} min={wt.min():.3f} max={wt.max():.3f}")

# save the real pool record (Stage-2 grad closure + fast reuse)
np.savez("/tmp/pool_fsi_record.npz", **{k: np.asarray(v) for k, v in rec.items()})

# STAGE 2: autodiff == finite-difference closure on pool_fsi_reweight (the record is theta-independent;
# theta enters only via the pure reweight -> gradients must be exact, mirroring the legacy grad test).
def loss(theta):
    return jnp.sum(CF.pool_fsi_reweight(rec, theta[0], theta[1]))
th0 = jnp.array([1.15, 0.85])
g_ad = np.asarray(jax.grad(loss)(th0))
eps = 1e-4
g_fd = np.array([float((loss(th0.at[d].add(eps)) - loss(th0.at[d].add(-eps))) / (2 * eps)) for d in (0, 1)])
print(f"[autodiff==FD @theta=(1.15,0.85)] grad_ad={g_ad}  grad_fd={g_fd}  maxreldiff={np.abs((g_ad-g_fd)/g_fd).max():.2e}")
print("STAGE2 GATE:", "PASS" if np.allclose(g_ad, g_fd, rtol=1e-4) else "FAIL")
