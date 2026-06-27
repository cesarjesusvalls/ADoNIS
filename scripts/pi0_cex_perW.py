"""pi0 charge-exchange residual: per-W diagnostic (logbook cascade_transport_residual.md sec 8/10).

Fires a pi0 BEAM at uniform lab |p| into 12C and, at each pion's FIRST interaction, records
(lab |p| at scatter, W = invariant mass of pion+struck nucleon, channel: elastic/cex/abs) by driving
the validated `_pion_step` directly (so W is available -- the engine segment logger does not emit it).

Outputs to /tmp/pi0_cex_ado.npz:  for SCATTERS (elastic+cex):  plab, W, is_cex.
Then bin the CEX fraction by W (must match the cross-section f_cex(W)) and by lab |p| (the sec-8 axis),
and the W-distribution at fixed lab |p|.  Compare to ACHILLES achilles:vertex pi0 beam (separate run).

Usage: python -u scripts/pi0_cex_perW.py [n] [pmax]
"""
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig, _load_density, sample_nucleons, _pion_step, _CH_MASS


def run(n=300_000, p_lo=80.0, p_hi=900.0, seed=0, target="C", max_steps=900):
    tg = resolve_targets(target)[0][0]
    cfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                step=0.04, pauli=True, nn_inelastic=True)
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    z0 = -1.05 * float(radius); R_DISK = 8.0   # >nuclear radius (6.55); CEX fraction is R-independent
    key = jax.random.PRNGKey(seed)
    km, kd, kn = jax.random.split(key, 3)
    mom = jax.random.uniform(km, (n,), minval=p_lo, maxval=p_hi)
    u = jax.random.uniform(kd, (n, 2)); br = R_DISK * jnp.sqrt(u[:, 0]); th = 2 * jnp.pi * u[:, 1]
    m_pi = _CH_MASS[1]                                  # pi0 kinematic mass
    E = jnp.sqrt(mom ** 2 + m_pi ** 2)
    p4 = jnp.stack([E, jnp.zeros(n), jnp.zeros(n), mom], axis=1)
    pos = jnp.stack([br * jnp.cos(th), br * jnp.sin(th), jnp.full((n,), z0)], axis=1)
    dhat = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
    ch = jnp.ones(n, jnp.int32)                         # pi0 = charge index 1
    nsc = jnp.zeros(n, jnp.int32)
    alive = jnp.ones(n, bool)
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    consumed = jnp.zeros(nisp.shape, bool)

    # records (filled at first interaction)
    rec_plab = np.full(n, -1.0); rec_W = np.full(n, -1.0); rec_kind = np.full(n, -1, np.int32)  # 0 el,1 cex,2 abs
    done = np.zeros(n, bool)

    step_fn = jax.jit(_pion_step, static_argnums=(14,))     # cfg (frozen dataclass) is static
    for st in range(max_steps):
        kP = jax.random.split(jax.random.fold_in(key, st), n)   # per-event keys (n,2) as _pion_step expects
        plab_pre = np.asarray(jnp.linalg.norm(p4[:, 1:], axis=1))
        ch_pre = np.asarray(ch)
        (p4n, posn, dhn, chn, nscn, aln), esc, is_abs, is_conv, s1, s2, consumedn, _srec = step_fn(
            p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed,
            rgrid, rhoP, rhoN, radius, cfg, kP)
        nscn_np = np.asarray(nscn); is_abs_np = np.asarray(is_abs)
        scattered = (nscn_np > np.asarray(nsc)) & ~done
        absorbed = is_abs_np & ~done
        # W at a scatter = invariant mass of continuing pion (p4n) + recoil nucleon (s1[0])
        if scattered.any():
            P = np.asarray(p4n) + np.asarray(s1[0])
            W = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1), 0, None))
            chn_np = np.asarray(chn)
            sidx = np.where(scattered)[0]
            rec_plab[sidx] = plab_pre[sidx]; rec_W[sidx] = W[sidx]
            rec_kind[sidx] = (chn_np[sidx] != ch_pre[sidx]).astype(np.int32)   # 0 elastic, 1 cex
            done[sidx] = True
        if absorbed.any():
            aidx = np.where(absorbed)[0]
            rec_plab[aidx] = plab_pre[aidx]; rec_kind[aidx] = 2; done[aidx] = True
        p4, pos, dhat, ch, nsc, alive, consumed = p4n, posn, dhn, chn, nscn, aln, consumedn
        alive = alive & ~jnp.asarray(done)                 # freeze finished pions
        if bool(jnp.all(~alive)):
            break
        if st % 100 == 0:
            print(f"  step {st}: done {int(done.sum())}/{n}", flush=True)
    sc = rec_kind >= 0
    print(f"interacted {int(sc.sum())}/{n}  (scatter {int((rec_kind==0).sum()+(rec_kind==1).sum())}, "
          f"abs {int((rec_kind==2).sum())})", flush=True)
    np.savez("/tmp/pi0_cex_ado.npz", plab=rec_plab, W=rec_W, kind=rec_kind)
    return rec_plab, rec_W, rec_kind


def report():
    d = np.load("/tmp/pi0_cex_ado.npz"); plab, W, kind = d["plab"], d["W"], d["kind"]
    scat = (kind == 0) | (kind == 1); cex = kind == 1
    print("\n=== ADoNIS pi0 CEX fraction by W (must match cross-section f_cex) ===")
    Wed = np.arange(1150, 1601, 50.0)
    for lo, hi in zip(Wed[:-1], Wed[1:]):
        m = scat & (W >= lo) & (W < hi); nb = int(m.sum())
        if nb > 50:
            print(f"  W [{lo:.0f},{hi:.0f}): cex_frac = {cex[m].sum()/nb:.3f}  (n={nb})")
    print("\n=== ADoNIS pi0 CEX fraction by lab |p| (the sec-8 axis) ===")
    Ped = np.array([0, 200, 400, 500, 600, 700, 800, 900.0])
    for lo, hi in zip(Ped[:-1], Ped[1:]):
        m = scat & (plab >= lo) & (plab < hi); nb = int(m.sum())
        if nb > 50:
            mc = m & cex
            print(f"  |p| [{lo:.0f},{hi:.0f}): cex_frac = {cex[m].sum()/nb:.3f}  (n={nb})  "
                  f"<W>={W[m].mean():.0f}  W in [{np.percentile(W[m],16):.0f},{np.percentile(W[m],84):.0f}]")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    if n > 0:
        run(n=n)
    report()
