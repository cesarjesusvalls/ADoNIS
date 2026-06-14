"""Faithful multi-particle differentiable cascade engine (standalone, in development).

Design (docs/logbook/cascade_engine.md): BFS over a shared nucleus. A PARTICLE is one Markov walk
segment of fixed identity; it propagates (multi-step, elastic = same identity) until a terminal
(escape/absorb/convert) or a transmuting interaction (charge-exchange/inelastic), then emits its
spawned secondaries.  Generations are processed breadth-first with FIXED-shape buffers so the whole
thing is jit/vmap-able; per-particle kind-1 reweight records keep it differentiable.

This file builds the mechanism (problem #2: variable particle count in fixed shapes) FIRST, with a
PLUGGABLE single-particle kernel, then the physics kernels are dropped in (P1b+).  The kernel contract:

    kernel(part, nucleus, key) -> (term, spawn)
      part   : ParticleBatch (n, P, ...) of the current generation (alive mask says which are real)
      term   : ParticleBatch (n, P, ...) -- each input particle's TERMINAL state + fate (for observables)
      spawn  : ParticleBatch (n, P*SPAWN, ...) -- secondaries emitted (alive mask says which are real)

Fields per particle (all leading dim (n, P)):
    species : 0 = pion, 1 = nucleon
    charge  : pion charge idx (0:+,1:0,2:-) | nucleon isospin (1=p,0=n) -- interpretation by species
    p4      : (..,4) (E,px,py,pz) MeV
    pos     : (..,3) fm
    fz      : (..,) formation zone [fm]
    alive   : (..,) real-slot mask
    w       : (..,) particle weight (kind-1; 1.0 at nominal)
    fate    : (..,) terminal code -- 0 alive/none, 1 escaped, 2 absorbed, 3 converted (set by kernel)
"""
from __future__ import annotations
import jax, jax.numpy as jnp
from adonis.fsi.cascade_discrete import (_propagate_discrete, _CH_PID, sample_nucleons,
                                         DiscreteCascadeConfig)

PION, NUCLEON = 0, 1
FATE_NONE, FATE_ESCAPE, FATE_ABSORB, FATE_CONVERT = 0, 1, 2, 3


def pion_segment(p4, pos, ch, consumed, npos, nmom, nisp, cfg, key, sabs=1.0, sscat=1.0):
    """Propagate one pion (batch (n,)) through its FULL segment via the validated _propagate_discrete.
    Returns (term, sec) flat dicts: term = the pion's terminal state + fate; sec = its leading recoil
    proton (alive only where a proton recoil exists).  Charge oscillation is internal to the segment."""
    out = _propagate_discrete(pos, p4, ch, npos, nmom, nisp, cfg, key, consumed,
                              jnp.asarray(sabs, float), jnp.asarray(sscat, float))
    (p_pi, ch_out, absorbed, conv, nsc, best_abs, w_fsi, nseg, n_trunc, brec,
     (best_rec, best_rec_pos, best_rec_fz)) = out
    fate = jnp.where(absorbed, FATE_ABSORB, jnp.where(conv, FATE_CONVERT, FATE_ESCAPE))
    pid = jnp.where(absorbed, 0, jnp.where(conv, -1, _CH_PID[ch_out]))     # 0 abs, -1 conv, else pi pid
    n = p4.shape[0]
    term = dict(species=jnp.full((n,), PION, jnp.int32), charge=ch_out, pid=pid, p4=p_pi,
                fate=fate, w=w_fsi, nsc=nsc)
    has_rec = jnp.linalg.norm(best_rec[:, 1:], axis=1) > 1.0
    sec = dict(species=jnp.full((n,), NUCLEON, jnp.int32), charge=jnp.ones((n,), jnp.int32),  # proton
               p4=best_rec, pos=best_rec_pos, fz=best_rec_fz, alive=has_rec, w=w_fsi)
    return term, sec


def setup_carbon(p_pi, pid_pi, pid_Ni, cfg, key):
    """Sample the carbon background + struck vertex EXACTLY as DiscreteCascadeFSI.apply, so the engine's
    primary-pion segment reproduces the production chain bit-for-bit.  Returns the nucleus + pion init."""
    kn, kv, kp = jax.random.split(key, 3)
    n = p_pi.shape[0]
    npos, nmom, nisp = sample_nucleons(kn, n, cfg)
    A = nisp.shape[1]
    struck_isp = (pid_Ni == 2212)
    rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
    vtx = jnp.argmax(rsel, axis=1)
    pos0 = npos[jnp.arange(n), vtx]
    consumed0 = jax.nn.one_hot(vtx, A).astype(bool)
    ch0 = jnp.where(pid_pi == 211, 0, jnp.where(pid_pi == -211, 2, 1)).astype(jnp.int32)
    return dict(npos=npos, nmom=nmom, nisp=nisp, pos0=pos0, consumed0=consumed0, ch0=ch0, kp=kp)


def empty_batch(n, P):
    """An all-dead particle buffer of shape (n, P)."""
    return dict(species=jnp.zeros((n, P), jnp.int32), charge=jnp.zeros((n, P), jnp.int32),
                p4=jnp.zeros((n, P, 4)), pos=jnp.zeros((n, P, 3)), fz=jnp.zeros((n, P)),
                alive=jnp.zeros((n, P), bool), w=jnp.ones((n, P)), fate=jnp.zeros((n, P), jnp.int32))


def _take(b, idx):
    """Gather along the particle axis (n, P) -> (n, K) with idx (n, K)."""
    n = idx.shape[0]; ar = jnp.arange(n)[:, None]
    return dict(species=b["species"][ar, idx], charge=b["charge"][ar, idx], p4=b["p4"][ar, idx],
                pos=b["pos"][ar, idx], fz=b["fz"][ar, idx], alive=b["alive"][ar, idx],
                w=b["w"][ar, idx], fate=b["fate"][ar, idx])


def compact(b, P_out):
    """Compact live particles of a (n, M) batch to the front of a (n, P_out) buffer (BFS refill /
    overflow handling).  Returns (compacted_batch, n_overflow) where n_overflow counts live particles
    that did not fit in P_out (dropped, never silently)."""
    alive = b["alive"]; n, M = alive.shape
    # stable rank of each live particle within its event (0-based); dead get a large rank
    rank = jnp.cumsum(alive.astype(jnp.int32), axis=1) - 1
    rank = jnp.where(alive, rank, M + P_out)                      # dead -> out of range
    n_overflow = jnp.sum((alive & (rank >= P_out)).astype(jnp.int32))
    out = empty_batch(n, P_out)
    ar = jnp.broadcast_to(jnp.arange(n)[:, None], (n, M))
    dst = jnp.where(rank < P_out, rank, P_out)                    # P_out = scratch drop slot (will be sliced off)
    for k in ("species", "charge", "fate"):
        out[k] = jnp.zeros((n, P_out + 1), b[k].dtype).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    for k in ("fz", "w"):
        out[k] = jnp.zeros((n, P_out + 1)).at[ar, dst].set(b[k], mode="drop")[:, :P_out]
    out["alive"] = jnp.zeros((n, P_out + 1), bool).at[ar, dst].set(b["alive"], mode="drop")[:, :P_out]
    out["p4"] = jnp.zeros((n, P_out + 1, 4)).at[ar, dst].set(b["p4"], mode="drop")[:, :P_out]
    out["pos"] = jnp.zeros((n, P_out + 1, 3)).at[ar, dst].set(b["pos"], mode="drop")[:, :P_out]
    return out, n_overflow


def run_cascade(init, kernel, key, P=10, max_gen=6):
    """BFS over generations with fixed-shape buffers.

    init   : ParticleBatch (n, P0) of generation-0 particles (the pre-FSI interaction products).
    kernel : (part, key) -> (term, spawn)  (nucleus/background captured in the kernel closure).
    Returns (terminals_list, total_overflow): terminals_list is one ParticleBatch (n, P) per generation
    (terminal state of every particle processed), to be masked+histogrammed downstream.
    """
    n = init["alive"].shape[0]
    cur, _ = compact(init, P)                                     # normalize to width P
    terminals = []
    overflow = jnp.int32(0)
    gkeys = jax.random.split(key, max_gen)
    for g in range(max_gen):
        term, spawn = kernel(cur, gkeys[g])
        terminals.append(term)
        nxt, ofl = compact(spawn, P)                             # pack secondaries into the next generation
        overflow = overflow + ofl
        cur = nxt
    return terminals, overflow
