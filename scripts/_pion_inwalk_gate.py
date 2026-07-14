"""DIAGNOSTIC (not a code path): does the PION FSI reweight reproduce an IN-WALK sigma scale?

The nucleon reweight carries the interaction-probability term (p0=exp(-a), pk=exp(-a/g), hit/no-hit
ratio).  The pion reweight (fsi_pion_reweight) is per = s_realized * D0/D -- a pure BRANCH-ratio
likelihood ratio with no such term, and the record does not even store the impact parameter.

But the walk's pion interaction probability DOES depend on the cross sections:
    _pion_step:  sig = sa + ss + si ;  prob = exp(-pi * perp^2 / (sig * MB_TO_FM2))

So scale all four pion sigmas by a COMMON factor s:
    walk     -> prob changes (shorter mean free path for s>1) => MORE interactions.
                branch ratios sa/sig are UNCHANGED (common factor cancels).
    reweight -> per = s*D0/(s*D0) = 1 EXACTLY => predicts NO change at all.

If the scaled walk interacts more than the nominal walk, the reweight is INCOMPLETE and the
"pion-FSI flat direction" is a record artifact, not physics.

Same generator, same seeds, same config on both sides; the ONLY difference is the sigma scale.
"""
import os, sys
sys.path.insert(0, "/Users/homelab/Lab/Playground/projects/DIFFGEN/ADoNIS")
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp

import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.fsi import oset_xsec as ox
from adonis.fsi.mb import cascade_mb
from adonis.xsec.spectral import SpectralFunction

S = float(os.environ.get("SCALE", "1.20"))          # common scale on ALL FOUR pion sigmas
NEV = int(os.environ.get("NEV", "6000"))
MS, P, KP, KN = 600, 12, 256, 256              # KP: slots are now in-slab CANDIDATE steps

tg = resolve_targets("C")[0][0]
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
a = gen_events("res", NEV, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
w = np.asarray(a["w"]); sel = w > 0; m = int(sel.sum())
pPi = jnp.asarray(a["p_pi"][sel]); pN = jnp.asarray(a["p_N"][sel])
ppid = jnp.asarray(a["ppid"][sel]); ipid = jnp.asarray(a["ipid"][sel]); Npid = jnp.asarray(a["Npid"][sel])
print(f"[gen] {m} live RES events | common pion-sigma scale s = {S}", flush=True)

cfg = DiscreteCascadeConfig(step=0.04, max_steps=MS, seed=1, nn_inelastic=True, pauli=True,
                            early_exit=True, nucleus=tg.density_p, density_n=tg.density_n,
                            configs=tg.configs, engine="pool")


def build_g0(su):
    g0 = CF.empty_batch(m, 2)
    g0["alive"] = jnp.ones((m, 2), bool)
    g0["species"] = jnp.array([CF.PION, CF.NUCLEON], jnp.int32)[None, :] * jnp.ones((m, 1), jnp.int32)
    g0["charge"] = jnp.stack([su["ch0"], (Npid == 2212).astype(jnp.int32)], axis=1)
    g0["p4"] = jnp.stack([pPi, pN], axis=1)
    g0["pos"] = jnp.broadcast_to(su["pos0"][:, None, :], (m, 2, 3))
    g0["origin"] = jnp.array([CF._ORIG_PRIM_PI, 0], jnp.int32)[None, :] * jnp.ones((m, 1), jnp.int32)
    return g0


def run(tag):
    """Run the walk with whatever sigma functions are currently installed; return the kind-1 record."""
    su = CF.setup_nucleus(pPi, ppid.astype(jnp.int32), ipid.astype(jnp.int32), cfg, jax.random.PRNGKey(11))
    _, knuc, _ = jax.random.split(su["kp"], 3)
    _, _, _, _, (rec, rofl) = CF.run_cascade_pool(
        build_g0(su), CF.make_pool_stepper(su, cfg, with_rec=True), knuc, su["consumed0"],
        M=P, max_steps=MS, M_out=24, prim_origin=CF._ORIG_PRIM_PI, rec_caps=(KP, KN))
    assert int(rofl) == 0, "record overflow"
    print(f"[{tag}] done", flush=True)
    return {k: np.asarray(v) for k, v in rec.items()}


def stats(rec, label):
    """nh is now the count of IN-SLAB CANDIDATE STEP slots; the INTERACTIONS are the pi_hh flags."""
    nh = rec["nh"]; K = rec["sa"].shape[1]
    valid = np.arange(K)[None, :] < nh[:, None]
    hit = rec["pi_hh"] & valid
    code = rec["bc"]
    n_hit = int(hit.sum())
    per_ev = hit.sum(axis=1)
    out = dict(mean_nh=float(per_ev.mean()), hits=n_hit, per_ev=per_ev,
               f_abs=float(((code == 2) & hit).sum()) / max(n_hit, 1),
               n_abs=int(((code == 2) & hit).sum()),
               ev_any=float((per_ev > 0).mean()), slots=float(nh.mean()))
    print(f"  {label:>26}: mean interactions = {out['mean_nh']:.4f} | events w/ >=1 hit = {out['ev_any']:.4f} "
          f"| hits = {out['hits']} | absorbed = {out['n_abs']} (branch frac {out['f_abs']:.4f}) "
          f"| rec slots/ev {out['slots']:.1f}")
    return out


# ---------- (1) NOMINAL walk ------------------------------------------------------------------------ #
print("\n== walks ==", flush=True)
rec0 = run("nominal walk")

# ---------- (2) SCALED walk: all four pion sigmas x S (monkeypatch, DIAGNOSTIC ONLY) ---------------- #
_abs, _chan, _conv = ox.abs_cross_section, cascade_mb.jax_channel_sigmas_resolved, cascade_mb.jax_conversion_sigma
ox.abs_cross_section = lambda *A, **K: S * _abs(*A, **K)
cascade_mb.jax_channel_sigmas_resolved = lambda *A, **K: S * _chan(*A, **K)
cascade_mb.jax_conversion_sigma = lambda *A, **K: S * _conv(*A, **K)
recS = run(f"scaled walk (x{S})")
ox.abs_cross_section, cascade_mb.jax_channel_sigmas_resolved, cascade_mb.jax_conversion_sigma = _abs, _chan, _conv

print("\n== in-walk effect of a common pion-sigma rescale ==")
s0 = stats(rec0, "NOMINAL walk")
sS = stats(recS, f"IN-WALK sigma x{S}")

# ---------- (3) what the REWEIGHT predicts for the same scale ---------------------------------------- #
wr = np.asarray(CF.pool_fsi_reweight({k: jnp.asarray(v) for k, v in rec0.items()},
                                     S, 1.0, s_piN_elastic=S, s_piN_cex=S, s_conv=S,
                                     s_NN_elastic=(1., 1., 1.), s_NN_inelastic=(1., 1., 1.), f_NN_cex=0.5))
print(f"\n== reweight of the NOMINAL record at the SAME scale (all four pion sigmas = {S}) ==")
print(f"  per-event weight: mean = {wr.mean():.8f}  min = {wr.min():.8f}  max = {wr.max():.8f}"
      f"  max|w-1| = {np.abs(wr - 1).max():.3e}")

# reweighted prediction of the interaction rate vs what the walk actually did
pred_nh = float((wr * s0["per_ev"]).sum() / wr.sum())
print(f"\n== VERDICT ==")
print(f"  mean pion interactions/event   nominal walk : {s0['mean_nh']:.4f}")
print(f"                                 REWEIGHT pred: {pred_nh:.4f}   (w == 1 => identical to nominal)")
print(f"                                 IN-WALK x{S} : {sS['mean_nh']:.4f}")
d = 100 * (sS["mean_nh"] - pred_nh) / max(pred_nh, 1e-12)
# bootstrap the DISCREPANCY over events (both estimators share the same event set + seeds, so resample
# the event index jointly -- that keeps the positive correlation and gives the SE of the DIFFERENCE).
rng = np.random.default_rng(0)
n0, nS_, ww = s0["per_ev"].astype(float), sS["per_ev"].astype(float), np.asarray(wr, float)
bs = []
for _ in range(400):
    k = rng.integers(0, len(n0), len(n0))
    p = float((ww[k] * n0[k]).sum() / ww[k].sum())
    bs.append(100 * (nS_[k].mean() - p) / max(p, 1e-12))
se = float(np.std(bs))
print(f"  reweight vs in-walk discrepancy: {d:+.2f}%  +/- {se:.2f}% (bootstrap, 400x)  "
      f"-> {abs(d)/max(se,1e-9):.1f} sigma")
print(f"  absorbed hits   nominal {s0['n_abs']}  ->  in-walk x{S}: {sS['n_abs']}  "
      f"({100*(sS['n_abs']/max(s0['n_abs'],1)-1):+.1f}%)")
ok = abs(d) < max(2.0 * se, 1.0)
print("\n  GATE:", "PASS (reweight reproduces the in-walk scale within MC error)" if ok else
      f"FAIL -- residual {d:+.2f}% is {abs(d)/max(se,1e-9):.1f} sigma: a SYSTEMATIC, not noise")
