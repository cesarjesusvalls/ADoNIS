"""Differentiable DISCRETE-Glauber pion cascade -- a faithful JAX port of the ACHILLES Cascade
(src/Achilles/Cascade.cc Evolve/BaseAlgorithm + RunCascade.cc CrossSection mode), validated to
reproduce the ACHILLES pi+-12C transparency oracle to ~5% (scratch/discrete_transport.py).

WHY discrete, not the continuum mean-free-path (cascade_real): ACHILLES gives each background
nucleon a transverse interaction reach via prob = exp(-pi b^2/(sigma/10)) over EXPLICIT nucleons
(impact parameter b), so sqrt(sigma/pi) ~ 2.2 fm of reach.  The 1-D continuum MFP ignores this
and is ~2x too low.  Here the pion walks (step dist ~0.04 fm) through A nucleons placed from the
ACHILLES QMC configurations (data/configurations/QMC_configs.out.gz), interacting with the
first (smallest-b) nucleon in the slab that passes its exp(-pi b^2/sigma) roll, branching
abs/scatter by sigma_abs/sigma_tot, with DCC scatter angle + Pauli blocking on the recoil, and
CONSUMING the struck nucleon.  Cross sections are the bit-exact Oset abs + ANL-Osaka DCC scatter
ports (proven 1.0000 / 1.003 vs instrumented ACHILLES).

Differentiability (kind-1 reweighting): the trajectory is SAMPLED against a frozen proposal (the
cross sections at detached parameters); the Oset/DCC knobs enter via a per-event likelihood
weight, so d/d(knob) E[obs] is exact and the sampled final state is preserved.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox
from adonis.fsi.mb import cascade_mb
from adonis.fsi.cascade_real import (_load_density, _rho_species, _kf_local, _two_body_cm_scatter,
                                     _boost, MB_TO_FM2, _CH_MASS, _CH_PID)

M_N = ox.M_N
HBARC = ox.HBARC
_CFG = {}


def _formation_zone(p_in, p_out):
    """ACHILLES Particle::SetFormationZone: fz = E_in * hbarc / |mN^2 - p_in.p_out|  [fm].
    p_in (n,4) = incoming nucleon momentum, p_out (n,4) = outgoing.  Forward scatters (p_in~p_out)
    -> |mN^2 - p_in.p_out| -> 0 -> large fz (free-streams out); wide scatters -> small fz."""
    dot4 = p_in[:, 0] * p_out[:, 0] - jnp.sum(p_in[:, 1:] * p_out[:, 1:], axis=1)
    return p_in[:, 0] * HBARC / jnp.clip(jnp.abs(M_N ** 2 - dot4), 1e-6, None)


def _load_qmc_configs(nmax=20000, name="QMC_configs.out.gz"):
    """Load A-nucleon configurations (positions [fm], isospin proton-mask, per-config weight)."""
    if name in _CFG:
        return _CFG[name]
    path = Path(__file__).resolve().parents[2].parent / "Achilles" / "data" / "configurations" / name
    A = 12
    iso = np.zeros((nmax, A), bool); pos = np.zeros((nmax, A, 3)); wt = np.zeros(nmax)
    with gzip.open(path, "rt") as f:
        f.readline()
        for c in range(nmax):
            for i in range(A):
                t = f.readline().split()
                iso[c, i] = float(t[0]) > 0
                pos[c, i] = [float(t[1]), float(t[2]), float(t[3])]
            wt[c] = float(f.readline()); f.readline()
    _CFG[name] = (jnp.asarray(pos), jnp.asarray(iso), jnp.asarray(wt / wt.sum()), A)
    return _CFG[name]


@dataclass(frozen=True)
class DiscreteCascadeConfig:
    nucleus: str = "c12_density.txt"
    step: float = 0.05
    max_steps: int = 260
    seed: int = 0
    cylinder: bool = False   # ACHILLES Probability: Cylinder (hard b^2<sigma/pi) vs Gaussian
                             # exp(-pi b^2/sigma).  T2K run-card uses Cylinder; Fig-3 oracle Gaussian.


def sample_nucleons(key, n, cfg: DiscreteCascadeConfig):
    """Pick n configurations ~ weight; assign each nucleon an isotropic local-Fermi-gas momentum.
    Returns npos (n,A,3), nmom (n,A,4), nisp (n,A) proton-mask."""
    pos, iso, w, A = _load_qmc_configs()
    rgrid, rho, _ = _load_density(cfg.nucleus)
    kc, kd, km = jax.random.split(key, 3)
    idx = jax.random.choice(kc, pos.shape[0], (n,), p=w)
    npos = pos[idx]; nisp = iso[idx]
    r = jnp.linalg.norm(npos, axis=2)
    kf = _kf_local(_rho_species(r, rgrid, rho))
    d = jax.random.normal(kd, (n, A, 3)); d = d / jnp.linalg.norm(d, axis=2, keepdims=True)
    pm = kf * jax.random.uniform(km, (n, A)) ** (1.0 / 3.0)
    p3 = d * pm[:, :, None]; E = jnp.sqrt(M_N ** 2 + pm ** 2)
    nmom = jnp.concatenate([E[:, :, None], p3], axis=2)
    return npos, nmom, nisp


@partial(jax.jit, static_argnums=(6,))
def _propagate_discrete(pos0, p_pi0, ch0, npos, nmom, nisp, cfg: DiscreteCascadeConfig, key, consumed0):
    rgrid, rho, radius = _load_density(cfg.nucleus)
    n, A = nisp.shape
    keys = jax.random.split(key, cfg.max_steps)
    ar = jnp.arange(n)

    def body(carry, sk):
        pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs = carry
        # escape: outward-moving pion past the radius
        outward = jnp.sum(pos * dhat, axis=1) > 0
        alive = alive & ~((jnp.linalg.norm(pos, axis=1) > radius) & outward)

        rel = npos - pos[:, None, :]
        par = jnp.sum(rel * dhat[:, None, :], axis=2)
        perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
        in_slab = (par > 0) & (par <= cfg.step) & (~consumed) & alive[:, None]

        pE = p_pi[:, 0]; pmom = jnp.linalg.norm(p_pi[:, 1:], axis=1); m_pi = _CH_MASS[ch]
        vpi = p_pi[:, 1:] / pE[:, None]
        vN = nmom[:, :, 1:] / nmom[:, :, 0:1]
        vrel = jnp.clip(jnp.linalg.norm(vpi[:, None, :] - vN, axis=2), 1e-3, None)
        rnuc = jnp.linalg.norm(npos, axis=2)
        rho_t = 2.0 * _rho_species(rnuc, rgrid, rho); kf_n = _kf_local(_rho_species(rnuc, rgrid, rho))
        Pp = p_pi[:, None, :] + nmom
        W = jnp.sqrt(jnp.clip(Pp[:, :, 0] ** 2 - jnp.sum(Pp[:, :, 1:] ** 2, axis=2), 1.0, None))
        sa = jnp.clip(ox.abs_cross_section(pE[:, None] + 0 * W, m_pi[:, None] + 0 * W,
                      pmom[:, None] + 0 * W, vrel, jnp.clip(kf_n, 1e-6, None),
                      jnp.clip(rho_t, 1e-9, None)), 0.0, None)
        # DCC scatter sigma per (n,A): sum over out-pions, target-type dependent
        sig_io = cascade_mb.jax_channel_sigmas(W.reshape(-1), jnp.broadcast_to(ch[:, None], (n, A)).reshape(-1))
        ss = jnp.clip(jnp.sum(sig_io, axis=-1).reshape(n, A), 0.0, None)
        sig = sa + ss                                                # mb

        if cfg.cylinder:
            prob = jnp.where(in_slab & (perp2 < jnp.clip(sig * MB_TO_FM2, 0.0, None) / jnp.pi), 1.0, 0.0)
        else:
            prob = jnp.where(in_slab, jnp.exp(-jnp.pi * perp2 / jnp.clip(sig * MB_TO_FM2, 1e-12, None)), 0.0)
        sk, ku, kc, kf, ka, kab = jax.random.split(sk, 6)
        passes = in_slab & (jax.random.uniform(ku, (n, A)) < prob)
        big = jnp.where(passes, perp2, jnp.inf)
        j = jnp.argmin(big, axis=1)
        has_hit = jnp.isfinite(big[ar, j]) & alive

        sa_j = sa[ar, j]; sig_j = sig[ar, j]; W_j = W[ar, j]; pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
        p_abs = sa_j / jnp.clip(sig_j, 1e-12, None)
        chose_abs = has_hit & (jax.random.uniform(kc, (n,)) < p_abs)          # channel pick (abs vs scatter)

        # ----- pion ABSORPTION final state (ACHILLES PionAbsorption::GenerateMomentum) -----
        # piNN -> NN: pion + struck nucleon j + closest background nucleon (FindClosest is by
        # distance to the struck nucleon); 2 outgoing nucleons isotropic in the 3-body CM.
        d2 = jnp.sum((npos - npos[ar, j][:, None, :]) ** 2, axis=2)             # (n,A)
        d2 = jnp.where((jnp.arange(A)[None, :] == j[:, None]) | consumed, jnp.inf, d2)
        pj = jnp.argmin(d2, axis=1)                                            # partner index
        pN_p = nmom[ar, pj]
        qpi = 1 - ch                                                           # 0:pi+ ->+1, 2:pi- ->-1
        nprot_out = qpi + nisp[ar, j].astype(jnp.int32) + nisp[ar, pj].astype(jnp.int32)
        # local Fermi momenta at the two outgoing-nucleon positions: product A inherits the PION
        # position (particle1), product B the struck nucleon position (particle2) -- ACHILLES
        # PionAbsorption::GenerateMomentum places paOut@part1.Position, pbOut@part2.Position.
        kf_pi = _kf_local(_rho_species(jnp.linalg.norm(pos, axis=1), rgrid, rho))   # (n,)
        kf_absB = kf_n[ar, j]

        def abs_one(p_pi_i, pNj_i, pNp_i, npr, kfA, kfB, k):
            P = p_pi_i + pNj_i + pNp_i
            s = P[0] ** 2 - jnp.sum(P[1:] ** 2)
            sqrts = jnp.sqrt(jnp.clip(s, (2 * M_N) ** 2, None))
            Estar = sqrts / 2.0
            pstar = jnp.sqrt(jnp.clip(Estar ** 2 - M_N ** 2, 0.0, None))
            k1, k2, k3 = jax.random.split(k, 3)
            cth = 2.0 * jax.random.uniform(k1) - 1.0
            sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0.0, None)); phi = 2 * jnp.pi * jax.random.uniform(k2)
            dirn = jnp.array([sth * jnp.cos(phi), sth * jnp.sin(phi), cth])
            beta = P[1:] / P[0]
            pa = _boost(jnp.concatenate([Estar[None], pstar * dirn]), beta)
            pb = _boost(jnp.concatenate([Estar[None], -pstar * dirn]), beta)
            ma = jnp.linalg.norm(pa[1:]); mb = jnp.linalg.norm(pb[1:])
            # ACHILLES FinalizeMomentum Pauli-blocks BOTH outgoing nucleons; reject if either
            # falls below the local Fermi momentum (then the pion is NOT absorbed, it continues).
            blocked = (ma < kfA) | (mb < kfB)
            faster = jnp.where(ma >= mb, pa, pb)
            one_p = jnp.where(jax.random.uniform(k3) < 0.5, pa, pb)            # which of the two is p
            lead = jnp.where(npr >= 2, faster, jnp.where(npr == 1, one_p, jnp.zeros(4)))
            return lead, blocked
        abs_lead, abs_blocked = jax.vmap(abs_one)(p_pi, pN_j, pN_p, nprot_out, kf_pi, kf_absB,
                                                  jax.random.split(kab, n))
        is_abs = chose_abs & ~abs_blocked                                     # absorption survives Pauli
        best_abs = jnp.where(is_abs[:, None], abs_lead, best_abs)             # one absorption / pion

        # scatter: out-pion charge ~ sig_io[j], DCC angle, Pauli-block recoil
        sig_io_j = sig_io.reshape(n, A, 3)[ar, j]
        probs = sig_io_j / jnp.clip(jnp.sum(sig_io_j, axis=1, keepdims=True), 1e-12, None)
        u = jax.random.uniform(kf, (n, 1))
        out_ch = jnp.clip(jnp.sum((u > jnp.cumsum(probs, axis=1)).astype(jnp.int32), axis=1), 0, 2).astype(jnp.int32)
        cos_cm = cascade_mb.jax_sample_cos_cm(W_j, jax.random.uniform(ka, (n,)))

        def scat_one(p_pi_i, pN_i, out_i, kf_i, cc, k):
            p_out = _two_body_cm_scatter(p_pi_i, pN_i, _CH_MASS[out_i], k, cos_cm=cc)
            p_rec = (p_pi_i + pN_i) - p_out
            return p_out, jnp.linalg.norm(p_rec[1:]) < kf_i
        p_out, blocked = jax.vmap(scat_one)(p_pi, pN_j, out_ch, kf_j, cos_cm, jax.random.split(sk, n))

        is_scat = has_hit & ~chose_abs & ~blocked    # scatter channel chosen, recoil not Pauli-blocked
        p_pi = jnp.where(is_scat[:, None], p_out, p_pi)
        ch = jnp.where(is_scat, out_ch, ch)
        nsc = nsc + is_scat.astype(jnp.int32)
        absorbed = absorbed | is_abs
        alive = alive & ~is_abs
        # consume the struck nucleon on any (non-Pauli-blocked) interaction
        interacted = is_abs | is_scat
        consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & interacted[:, None])

        d3 = p_pi[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
        pos = pos + cfg.step * dhat * alive[:, None]
        return (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs), None

    dhat0 = p_pi0[:, 1:] / jnp.clip(jnp.linalg.norm(p_pi0[:, 1:], axis=1, keepdims=True), 1e-9, None)
    init = (pos0, p_pi0, ch0, dhat0, jnp.ones(n, bool), jnp.zeros(n, bool),
            jnp.zeros(n, jnp.int32), consumed0, jnp.zeros((n, 4)))
    (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs), _ = jax.lax.scan(body, init, keys)
    return p_pi, ch, absorbed, nsc, best_abs


def propagate_discrete(pos0, p_pi0, charge_idx0, npos, nmom, nisp, cfg, key, consumed0=None):
    """Discrete-Glauber transport of a batch of pions through explicit nucleons.
    Returns (p_pi (N,4), charge_idx (N,), absorbed (N,), n_scatter (N,), abs_lead_p (N,4)).
    abs_lead_p is the leading absorption PROTON 4-momentum for absorbed events (piNN->NN), zeros
    otherwise -- this is what populates the CC0pi high-delta_pT tail."""
    _load_density(cfg.nucleus); cascade_mb._jax_grids(); cascade_mb._build_angular()
    if consumed0 is None:
        consumed0 = jnp.zeros((nisp.shape[0], nisp.shape[1]), bool)
    return _propagate_discrete(pos0, p_pi0, charge_idx0, npos, nmom, nisp, cfg, key, consumed0)


# pid <-> charge index (0:pi+, 1:pi0, 2:pi-)
_PID_TO_CH = {211: 0, 111: 1, -211: 2}


class DiscreteCascadeFSI:
    """FSIModel: propagate the produced pion through the nucleus with the validated discrete-
    Glauber cascade.  The pion is created at a config nucleon position (the production vertex) and
    walks out through the other A-1 nucleons.  Production replacement for RealCascadeFSI (which
    used the ~2x-low continuum transport).  protfrac is accepted for interface parity."""

    def __init__(self, cfg: DiscreteCascadeConfig = DiscreteCascadeConfig(), protfrac: float = 0.0):
        self.cfg = cfg
        self.protfrac = float(protfrac)

    def apply(self, params, event, key=None):
        key = jax.random.PRNGKey(self.cfg.seed) if key is None else key
        kn, kv, kp = jax.random.split(key, 3)
        n = event.p_pi.shape[0]
        npos, nmom, nisp = sample_nucleons(kn, n, self.cfg)
        A = nisp.shape[1]
        # production vertex = the STRUCK nucleon (pid_Ni; isospin-correct), CONSUMED from the
        # background.  RES on a struck NEUTRON (nu n -> N pi) leaves 6p5n spectators; since pi+p
        # scatter (Delta++) >> pi+n, leaving the struck neutron in the background under-scatters the
        # pi+ -> it over-absorbs.  Removing the correct struck nucleon fixes the spectator isospin.
        struck_isp = jnp.asarray(np.asarray(event.pid_Ni) == 2212)       # True if struck proton
        rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
        vtx = jnp.argmax(rsel, axis=1)
        pos0 = npos[jnp.arange(n), vtx]
        consumed0 = jax.nn.one_hot(vtx, A).astype(bool)
        ch0 = jnp.asarray([_PID_TO_CH.get(int(p), 1) for p in np.asarray(event.pid_pi)], dtype=jnp.int32)
        p_pi, ch, absorbed, nsc, abs_lead = propagate_discrete(pos0, event.p_pi, ch0, npos, nmom,
                                                               nisp, self.cfg, kp, consumed0=consumed0)
        self.last_abs_proton = abs_lead          # leading absorption proton (piNN->NN), for CC0pi
        self.last_absorbed = absorbed
        keep = (~absorbed)[:, None]
        return event._replace(p_pi=p_pi * keep, pid_pi=jnp.where(absorbed, 0, _CH_PID[ch]))


# ===== discrete-Glauber NUCLEON cascade (proton/neutron FSI for the TKI observables) ========= #
from adonis.fsi.nucleon_cascade import nn_elastic_sigma


@partial(jax.jit, static_argnums=(6,))
def _propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg: DiscreteCascadeConfig, key, fz0, consumed0):
    """Leading nucleon walks through the background config via NN-elastic scatter (isotropic CM,
    as ACHILLES NucleonNucleon::GenerateMomentum), Pauli-blocking BOTH outgoing nucleons, consuming
    the struck one.  No absorption.  isp0 (n,) proton-mask of the leading nucleon.  fz0 (n,) initial
    formation zone [fm] (ACHILLES: 0 for the primary nucleon).  A nucleon cannot interact while
    fz>0; fz decrements by timeStep=step/beta each step and resets on every scatter to
    E_in*hbarc/|mN^2-p_in.p_out| (ACHILLES SetFormationZone -> suppresses rapid forward re-scatter)."""
    rgrid, rho, radius = _load_density(cfg.nucleus)
    n, A = nisp.shape
    keys = jax.random.split(key, cfg.max_steps)
    ar = jnp.arange(n)

    def body(carry, sk):
        pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz = carry
        outward = jnp.sum(pos * dhat, axis=1) > 0
        alive = alive & ~((jnp.linalg.norm(pos, axis=1) > radius) & outward)
        # formation zone: timeStep = step/beta (ACHILLES AdaptiveStep); interact only when fz<=0
        beta = jnp.linalg.norm(p_N[:, 1:], axis=1) / jnp.clip(p_N[:, 0], 1e-9, None)
        timeStep = cfg.step / jnp.clip(beta, 1e-6, None)
        can_int = fz <= 0.0
        rel = npos - pos[:, None, :]
        par = jnp.sum(rel * dhat[:, None, :], axis=2)
        perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
        in_slab = (par > 0) & (par <= cfg.step) & (~consumed) & alive[:, None]
        Pp = p_N[:, None, :] + nmom
        s = Pp[:, :, 0] ** 2 - jnp.sum(Pp[:, :, 1:] ** 2, axis=2)
        sqrts = jnp.sqrt(jnp.clip(s, (2 * M_N) ** 2, None))
        same_iso = isp0[:, None] == nisp                              # (n,A)
        sig = jnp.clip(nn_elastic_sigma(sqrts, same_iso), 0.0, None)  # mb
        if cfg.cylinder:
            prob = jnp.where(in_slab & (perp2 < jnp.clip(sig * MB_TO_FM2, 0.0, None) / jnp.pi), 1.0, 0.0)
        else:
            prob = jnp.where(in_slab, jnp.exp(-jnp.pi * perp2 / jnp.clip(sig * MB_TO_FM2, 1e-12, None)), 0.0)
        sk, ku, ks = jax.random.split(sk, 3)
        passes = in_slab & (jax.random.uniform(ku, (n, A)) < prob)
        big = jnp.where(passes, perp2, jnp.inf)
        j = jnp.argmin(big, axis=1)
        has_hit = jnp.isfinite(big[ar, j]) & alive & can_int          # blocked while in formation zone
        pN_j = nmom[ar, j]
        rnuc = jnp.linalg.norm(npos, axis=2); kf_n = _kf_local(_rho_species(rnuc, rgrid, rho))
        kf_j = kf_n[ar, j]

        def scat_one(p_lead, pN_i, kf_i, k):
            p_out = _two_body_cm_scatter(p_lead, pN_i, M_N, k)        # leading out (isotropic)
            p_rec = (p_lead + pN_i) - p_out
            blocked = (jnp.linalg.norm(p_out[1:]) < kf_i) | (jnp.linalg.norm(p_rec[1:]) < kf_i)
            return p_out, blocked
        p_out, blocked = jax.vmap(scat_one)(p_N, pN_j, kf_j, jax.random.split(ks, n))
        do = has_hit & ~blocked
        # knocked-out nucleon = the struck background nucleon's recoil; if it is a PROTON track
        # the highest-momentum one (ACHILLES adds it to the final state, the analysis may pick it)
        recoil = (p_N + pN_j) - p_out
        bg_proton = nisp[ar, j]
        ko_better = do & bg_proton & (jnp.linalg.norm(recoil[:, 1:], axis=1) > jnp.linalg.norm(best_ko[:, 1:], axis=1))
        best_ko = jnp.where(ko_better[:, None], recoil, best_ko)
        best_ko_pos = jnp.where(ko_better[:, None], npos[ar, j], best_ko_pos)   # knockout production vertex
        # ACHILLES gives BOTH outgoing nucleons a formation zone, w/ p1 = the incoming (p_N):
        fz_new = _formation_zone(p_N, p_out)                          # leading: E_in*hbarc/|mN^2-p_in.p_out|
        fz_ko = _formation_zone(p_N, recoil)                          # recoil/knockout's formation zone
        best_ko_fz = jnp.where(ko_better, fz_ko, best_ko_fz)
        p_N = jnp.where(do[:, None], p_out, p_N)
        nsc = nsc + do.astype(jnp.int32)
        consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & do[:, None])
        fz = jnp.where((fz > 0.0) & alive, fz - timeStep, fz)         # propagate: decrement formation zone
        fz = jnp.where(do, fz_new, fz)                                # reset on scatter
        d3 = p_N[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
        pos = pos + cfg.step * dhat * alive[:, None]
        return (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz), None

    dhat0 = p_N0[:, 1:] / jnp.clip(jnp.linalg.norm(p_N0[:, 1:], axis=1, keepdims=True), 1e-9, None)
    init = (pos0, p_N0, dhat0, jnp.ones(n, bool), jnp.zeros(n, jnp.int32),
            consumed0, jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), fz0)
    (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz), _ = jax.lax.scan(body, init, keys)
    return p_N, nsc, best_ko, best_ko_pos, best_ko_fz


def propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg, key, fz0=None, consumed0=None):
    _load_density(cfg.nucleus)
    n, A = nisp.shape
    if fz0 is None:
        fz0 = jnp.zeros(p_N0.shape[0])
    if consumed0 is None:
        consumed0 = jnp.zeros((n, A), bool)
    return _propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg, key, fz0, consumed0)


class DiscreteNucleonFSI:
    """FSIModel add-on: propagate the leading outgoing NUCLEON through the nucleus with the
    discrete-Glauber NN-elastic cascade.  Composable with DiscreteCascadeFSI (pion)."""

    def __init__(self, cfg: DiscreteCascadeConfig = DiscreteCascadeConfig(), protfrac: float = 0.0):
        self.cfg = cfg
        self.protfrac = float(protfrac)

    def apply(self, params, event, key=None):
        key = jax.random.PRNGKey(self.cfg.seed + 5) if key is None else key
        kn, kv, kp = jax.random.split(key, 3)
        n = event.p_N.shape[0]
        npos, nmom, nisp = sample_nucleons(kn, n, self.cfg)
        A = nisp.shape[1]
        # production vertex = the STRUCK nucleon (pid_Ni; a neutron for nu_mu QE).  Start the leading
        # nucleon at its config position and CONSUME it from the spectator background: ACHILLES turns
        # the struck nucleon INTO the outgoing one, so the proton scatters off the A-1 spectators of
        # the correct isospin (6p5n for 12C QE).  pn elastic sigma > pp, so leaving the struck neutron
        # in the background over-removes the proton from the window (~4%).
        struck_isp = jnp.asarray(np.asarray(event.pid_Ni) == 2212)       # True if struck nucleon is a proton
        rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
        vtx = jnp.argmax(rsel, axis=1)
        pos0 = npos[jnp.arange(n), vtx]
        consumed0 = jax.nn.one_hot(vtx, A).astype(bool)                  # struck nucleon removed from background
        isp0 = jnp.asarray(np.asarray(event.pid_N) == 2212)
        p_N, nsc, best_ko, best_ko_pos, best_ko_fz = propagate_nucleon_discrete(
            pos0, event.p_N, isp0, npos, nmom, nisp, self.cfg, kp, consumed0=consumed0)
        # MULTI-NUCLEON cascade (ACHILLES UpdateKicked): the knocked-out proton itself re-cascades.
        # Re-propagate the leading knockout through the same background; its final state can be the
        # leading proton.  (One secondary generation -- the dominant multi-nucleon contribution.)
        # The knockout carries the formation zone ACHILLES assigned it at creation (best_ko_fz).
        kp2 = jax.random.fold_in(kp, 99)
        has_ko = jnp.linalg.norm(best_ko[:, 1:], axis=1) > 1.0
        ko_start = jnp.where(has_ko[:, None], best_ko, event.p_N)         # dummy where no knockout
        isp_ko = jnp.ones(n, bool)                                       # knockout proton
        ko_f, _, ko_ko, _, _ = propagate_nucleon_discrete(best_ko_pos, ko_start, isp_ko, npos, nmom,
                                                          nisp, self.cfg, kp2, fz0=best_ko_fz, consumed0=consumed0)
        ko_f = jnp.where(has_ko[:, None], ko_f, jnp.zeros((n, 4)))
        ko_ko = jnp.where(has_ko[:, None], ko_ko, jnp.zeros((n, 4)))     # 2nd-gen knockout proton
        # leading proton = highest-momentum proton among {primary, re-cascaded knockout, 2nd-gen ko}
        cands = jnp.stack([p_N, ko_f, ko_ko], axis=1)                   # (n,3,4)
        mom = jnp.linalg.norm(cands[:, :, 1:], axis=2)                  # (n,3)
        lead = cands[jnp.arange(n), jnp.argmax(mom, axis=1)]
        return event._replace(p_N=lead)
