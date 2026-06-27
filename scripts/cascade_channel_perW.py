"""Per-W cascade channel diagnostic for ANY beam hadron (pi+/pi0/pi-/p/n) on 12C, ADoNIS vs ACHILLES.

Generalizes scripts/pi0_cex_perW.py.  Fires a BEAM at uniform lab |p| into 12C and records, at each
particle's FIRST interaction, (lab |p|, W, channel) by driving the validated `_pion_step` (pions) or
`_nucleon_step` (nucleons) directly.  The "channel-2 fraction" = chan2/(chan1+chan2):
  pions    -> CEX / (elastic + CEX)         (out pion charge != in)
  nucleons -> inelastic / (elastic + inel)  (NN -> NN pi produced a pion)
ACHILLES VERTEXDUMP uses the SAME codes (1 elastic, 2 cex|inel) so the comparison is uniform.

ADoNIS -> /tmp/chan_ado_<pid>.npz (plab, W, kind: 0 chan1, 1 chan2, 2 abs, 3 conv, -1 none).
Usage: python -u scripts/cascade_channel_perW.py <pid> [n]      (pid in 211 111 -211 2212 2112)
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adonis.workflow.materials import resolve_targets
from adonis.fsi.cascade_discrete import (DiscreteCascadeConfig, _load_density, sample_nucleons,
                                         _pion_step, _nucleon_step, _CH_MASS, M_N)

_CH = {211: 0, 111: 1, -211: 2}            # pion charge index


def run(pid, n=400_000, seed=0, target="C", max_steps=1400, pauli=True):
    is_pion = pid in _CH
    tg = resolve_targets(target)[0][0]
    cfg = DiscreteCascadeConfig(nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs,
                                step=0.04, pauli=pauli, nn_inelastic=True)
    tag = "" if pauli else "_nopauli"
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    z0 = -1.05 * float(radius); R_DISK = 8.0
    p_lo, p_hi = (80.0, 900.0) if is_pion else (200.0, 1700.0)     # nucleon inelastic opens ~800 MeV
    key = jax.random.PRNGKey(seed); km, kd, kn = jax.random.split(key, 3)
    mom = jax.random.uniform(km, (n,), minval=p_lo, maxval=p_hi)
    u = jax.random.uniform(kd, (n, 2)); br = R_DISK * jnp.sqrt(u[:, 0]); th = 2 * jnp.pi * u[:, 1]
    m0 = _CH_MASS[_CH[pid]] if is_pion else M_N
    E = jnp.sqrt(mom ** 2 + m0 ** 2)
    p4 = jnp.stack([E, jnp.zeros(n), jnp.zeros(n), mom], axis=1)
    pos = jnp.stack([br * jnp.cos(th), br * jnp.sin(th), jnp.full((n,), z0)], axis=1)
    dhat = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
    npos, nmom, nisp = sample_nucleons(kn, n, cfg); consumed = jnp.zeros(nisp.shape, bool)
    alive = jnp.ones(n, bool)
    rec_plab = np.full(n, -1.0); rec_W = np.full(n, -1.0); rec_kind = np.full(n, -1, np.int32)
    done = np.zeros(n, bool)

    if is_pion:
        ch = jnp.full(n, _CH[pid], jnp.int32); nsc = jnp.zeros(n, jnp.int32)
        step = jax.jit(_pion_step, static_argnums=(14,))
        for st in range(max_steps):
            kP = jax.random.split(jax.random.fold_in(key, st), n)
            plab_pre = np.asarray(jnp.linalg.norm(p4[:, 1:], axis=1)); ch_pre = np.asarray(ch)
            (p4n, posn, dhn, chn, nscn, aln), esc, is_abs, is_conv, s1, s2, consn, _ = step(
                p4, pos, dhat, ch, nsc, alive, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kP)
            scat = (np.asarray(nscn) > np.asarray(nsc)) & ~done
            absb = np.asarray(is_abs) & ~done; conv = np.asarray(is_conv) & ~done
            if scat.any():
                P = np.asarray(p4n) + np.asarray(s1[0]); W = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, 1), 0, None))
                idx = np.where(scat)[0]; rec_plab[idx] = plab_pre[idx]; rec_W[idx] = W[idx]
                rec_kind[idx] = (np.asarray(chn)[idx] != ch_pre[idx]).astype(np.int32); done[idx] = True
            for mask, kv in ((absb, 2), (conv, 3)):
                if mask.any():
                    idx = np.where(mask)[0]; rec_plab[idx] = plab_pre[idx]; rec_kind[idx] = kv; done[idx] = True
            p4, pos, dhat, ch, nsc, consumed = p4n, posn, dhn, chn, nscn, consn
            alive = aln & ~jnp.asarray(done)
            if bool(jnp.all(~alive)): break
            if st % 200 == 0: print(f"  step {st}: done {int(done.sum())}/{n}", flush=True)
    else:
        isp = jnp.full(n, pid == 2212); fz = jnp.zeros(n)
        step = jax.jit(_nucleon_step, static_argnums=(14,))
        for st in range(max_steps):
            kN = jax.random.split(jax.random.fold_in(key, st), n)
            plab_pre = np.asarray(jnp.linalg.norm(p4[:, 1:], axis=1))
            (p4n, posn, dhn, fzn, aln), term, recap, do, koN, pio, consn, _ = step(
                p4, pos, dhat, fz, isp, alive, npos, nmom, nisp, consumed,
                rgrid, rhoP, rhoN, radius, cfg, kN)
            # `do` is ELASTIC-only (NN->NN); inelastic (NN->NN pi) comes back as the created-pion spawn
            # pio[4]=pi_alive with do=False.  An ACTUAL interaction this step = do | inel.
            do_np = np.asarray(do).astype(bool); inel = np.asarray(pio[4]).astype(bool)
            scat = (do_np | inel) & ~done
            if scat.any():
                P = np.asarray(p4n) + np.asarray(koN[0]) + np.where(inel[:, None], np.asarray(pio[0]), 0.0)
                W = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, 1), 0, None))
                idx = np.where(scat)[0]; rec_plab[idx] = plab_pre[idx]; rec_W[idx] = W[idx]
                rec_kind[idx] = inel[idx].astype(np.int32); done[idx] = True
            p4, pos, dhat, fz, consumed = p4n, posn, dhn, fzn, consn
            alive = aln & ~jnp.asarray(done)
            if bool(jnp.all(~alive)): break
            if st % 200 == 0: print(f"  step {st}: done {int(done.sum())}/{n}", flush=True)
    ch12 = (rec_kind == 0) | (rec_kind == 1)
    print(f"pid {pid}: interacted {int((rec_kind>=0).sum())}/{n} (chan1+2 {int(ch12.sum())}, chan2 {int((rec_kind==1).sum())})", flush=True)
    np.savez(f"/tmp/chan_ado_{pid}{tag}.npz", plab=rec_plab, W=rec_W, kind=rec_kind)


if __name__ == "__main__":
    pid = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    pauli = not (len(sys.argv) > 3 and sys.argv[3] == "nopauli")
    run(pid, n=n, pauli=pauli)
