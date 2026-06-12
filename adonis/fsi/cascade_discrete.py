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


def _load_qmc_configs(nmax=36000, name="QMC_configs.out.gz"):
    """Load A-nucleon configurations (positions [fm], isospin proton-mask, per-config weight).
    nmax=36000 = ALL configs in QMC_configs.out.gz (ACHILLES uses the full set); verified the
    transparency is unchanged vs the old 20000 cap (first-20k and full-36k have identical rms radius
    and both sample ∝ weight)."""
    # Cache as NUMPY (not jnp): a jnp array first created inside a jit trace would leak the
    # tracer context to later traces (cf. cascade_mb._jax_grids_resolved); asarray per-call is free.
    if name not in _CFG:
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
        _CFG[name] = (pos, iso, wt / wt.sum(), A)
    pos, iso, w, A = _CFG[name]
    return jnp.asarray(pos), jnp.asarray(iso), jnp.asarray(w), A


_KSLAB = 3                   # # of nearest in-slab nucleons whose cross sections are evaluated per step
                             # (fast_xsec).  sigma for non-in-slab nucleons is never used (prob is masked
                             # to in_slab); >=K in one 0.04 fm slab is ~never, so this is bit-exact.
_MAX_SEG = 12                # interaction-driven kernel: max interaction ATTEMPTS per particle.  Measured
                             # pion tail over 622k cascades: max=11, P(>11)=0.  A particle still
                             # propagating after _MAX_SEG segments is force-escaped (logged, not silent).
_K_BR = 16                   # compressed-record slots for the pion branch reweight (hits/event <= _MAX_SEG
                             # measured; nh in the records lets callers verify no overflow).
_K_SLAB_REC = 48             # compressed-record slots for the nucleon sigma_scatter reweight (steps with
                             # an in-slab candidate; ns in the records verifies no overflow).


def pion_branch_reweight(brec, sabs, sscat):
    """Kind-1 abs/scatter branching reweight from compressed walk records brec =
    (chose_abs (n,K), sa (n,K), sig (n,K), n_hits (n,)).  Pure in (sabs, sscat) and bit-exact
    equal to the in-propagation w_fsi -- the walk is theta-independent, so a fit can run the
    cascade ONCE (records out) and re-evaluate/differentiate this cheaply per theta."""
    ca, sa, sig, nh = brec
    valid = jnp.arange(sa.shape[1])[None, :] < nh[:, None]
    ss = jnp.clip(sig - sa, 1e-6, None)
    p_d = jnp.clip(sa / sig, 1e-6, 1.0 - 1e-6)
    p_k = jnp.clip((sabs * sa) / (sabs * sa + sscat * ss), 1e-6, 1.0 - 1e-6)
    br = jnp.where(valid, jnp.where(ca, p_k / p_d, (1.0 - p_k) / (1.0 - p_d)), 1.0)
    return jnp.prod(br, axis=1)


def nucleon_scat_reweight(srec, sscat):
    """Kind-1 sigma_scatter reweight from compressed nucleon-walk records srec =
    (hit (n,K), a_nom (n,K), n_slab (n,)) with a_nom = pi b^2/(sigma fm^2) of the closest
    in-slab nucleon at each candidate step.  Pure in sscat; bit-exact equal to the
    in-propagation w_scat."""
    hh, a_nom, ns = srec
    valid = jnp.arange(a_nom.shape[1])[None, :] < ns[:, None]
    p_nom = jnp.clip(jnp.exp(-a_nom), 1e-6, 1.0 - 1e-6)
    p_knb = jnp.clip(jnp.exp(-a_nom / sscat), 1e-6, 1.0 - 1e-6)
    br = jnp.where(valid, jnp.where(hh, p_knb / p_nom, (1.0 - p_knb) / (1.0 - p_nom)), 1.0)
    return jnp.prod(br, axis=1)


@dataclass(frozen=True)
class DiscreteCascadeConfig:
    nucleus: str = "c12_density.txt"
    step: float = 0.05
    max_steps: int = 260
    seed: int = 0
    cylinder: bool = False   # ACHILLES Probability: Cylinder (hard b^2<sigma/pi) vs Gaussian
                             # exp(-pi b^2/sigma).  T2K run-card uses Cylinder; Fig-3 oracle Gaussian.
    prob: str = ""           # interaction-probability model: "gaussian"/"cylinder"/"pion".  Empty ->
                             # back-compat: cylinder bool picks Cylinder vs Gaussian.  "pion" =
                             # ACHILLES Probability: Pion, exp(-sqrt(2 pi/sigma) b) (Cascade.cc:47-52).
                             # WARNING: "pion" is NOT reliable -- ACHILLES itself flags it with
                             # "TODO: This does not work, we need to rethink this" (Cascade.cc:49) and
                             # no ACHILLES run-card uses it.  Provided for parity only; prefer
                             # gaussian (Fig-3 oracle) or cylinder (T2K run-card).
    fast_xsec: bool = True   # evaluate Oset/DCC cross sections only for the K nearest in-slab nucleons
                             # (scatter back into the (n,A) grid); bit-exact, ~4x cheaper per step.
    pauli: bool = True       # Pauli-block the outgoing nucleon(s) of scatter/absorption (ACHILLES
                             # FinalizeMomentum); set False for ablation (no-blocking) studies.
    algo: str = "step"       # "step": fixed-step Glauber march (reference).  "interaction": jump
                             # directly to the next interaction (same probability model, ~20x fewer
                             # iterations).  Statistically equivalent; validated against "step".


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
def _propagate_discrete(pos0, p_pi0, ch0, npos, nmom, nisp, cfg: DiscreteCascadeConfig, key, consumed0,
                        sabs, sscat):
    """sabs, sscat scale the Oset absorption / MB-scatter cross sections.  The absorb-vs-scatter
    BRANCHING is kind-1 reweighted (sampled against the nominal p_abs, theta enters via the
    likelihood ratio in w_fsi); scales=1 -> w_fsi=1 (the nominal forward result), so d/d(scale)
    E[absorbed obs] is exact and differentiable."""
    rgrid, rho, radius = _load_density(cfg.nucleus)
    n, A = nisp.shape
    nsteps = cfg.max_steps if cfg.algo == "step" else _MAX_SEG
    keys = jax.random.split(key, nsteps)
    ar = jnp.arange(n)

    def body(carry, sk):
        pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs = carry
        if cfg.algo == "step":
            # Escape (ACHILLES Cascade.cc:532-553).  The un-scattered BEAM pion is external_test:
            # it escapes at the z>=radius PLANE (continues while Z<radius), so it traverses the whole
            # nucleus regardless of impact parameter.  Once it interacts (nsc>0 -> internal) it escapes
            # at the |pos|>radius SPHERE.  Using the sphere for the beam pion (the old code) cut off
            # off-axis pions early at z=sqrt(R^2-b^2), missing exit-side nucleons (~2% fewer reactions).
            ext = nsc == 0                                            # external_test: not yet scattered
            outward = jnp.sum(pos * dhat, axis=1) > 0
            esc_plane = ext & (pos[:, 2] >= radius)                   # beam pion: z>=radius plane
            esc_sphere = (~ext) & (jnp.linalg.norm(pos, axis=1) > radius) & outward
            alive = alive & ~(esc_plane | esc_sphere)

        rel = npos - pos[:, None, :]
        par = jnp.sum(rel * dhat[:, None, :], axis=2)
        perp2 = jnp.sum(rel ** 2, axis=2) - par ** 2
        # candidate nucleons: ahead (par>0), not consumed, alive.  "step" additionally requires being
        # in the current 0.04 fm slab (par<=step); "interaction" considers ALL nucleons ahead at once.
        cand = (par > 0) & (~consumed) & alive[:, None]
        if cfg.algo == "step":
            cand = cand & (par <= cfg.step)
        else:
            # only nucleons reached BEFORE the pion exits the nuclear radius -- replicates the "step"
            # escape (|pos|>radius & outward).  t_exit = outward ray-sphere crossing distance.
            b = jnp.sum(pos * dhat, axis=1); c = jnp.sum(pos * pos, axis=1) - radius ** 2
            t_exit = -b + jnp.sqrt(jnp.clip(b * b - c, 0.0, None))
            cand = cand & (par < t_exit[:, None])

        pE = p_pi[:, 0]; pmom = jnp.linalg.norm(p_pi[:, 1:], axis=1); m_pi = _CH_MASS[ch]
        vpi = p_pi[:, 1:] / pE[:, None]
        rnuc = jnp.linalg.norm(npos, axis=2)
        kf_n = _kf_local(_rho_species(rnuc, rgrid, rho))             # (n,A) cheap; used at the hit index j

        def _xsec(nm, npo, nip):
            """Oset abs + DCC scatter cross sections for nucleon states nm (..,4), npo (..,3),
            proton-mask nip (..) vs the current pion (pE,pmom,m_pi,vpi,ch).  Returns sa, ss,
            sig_io(...,3), W -- same leading shape.  Scatter sigma is CHARGE-RESOLVED by the struck
            nucleon (ACHILLES GetCchannel(pion,baryon)): sigma(pi+ p) ~ 3x sigma(pi+ n) at the Delta."""
            vN = nm[..., 1:] / nm[..., 0:1]
            vrel = jnp.clip(jnp.linalg.norm(vpi[:, None, :] - vN, axis=-1), 1e-3, None)
            rho_t = 2.0 * _rho_species(jnp.linalg.norm(npo, axis=-1), rgrid, rho)
            kf = _kf_local(_rho_species(jnp.linalg.norm(npo, axis=-1), rgrid, rho))
            Pp = p_pi[:, None, :] + nm
            Wl = jnp.sqrt(jnp.clip(Pp[..., 0] ** 2 - jnp.sum(Pp[..., 1:] ** 2, axis=-1), 1.0, None))
            sal = jnp.clip(ox.abs_cross_section(pE[:, None] + 0 * Wl, m_pi[:, None] + 0 * Wl,
                           pmom[:, None] + 0 * Wl, vrel, jnp.clip(kf, 1e-6, None),
                           jnp.clip(rho_t, 1e-9, None)), 0.0, None)
            K_ = Wl.shape[1]
            nuc_i = jnp.where(nip, 0, 1).astype(jnp.int32)           # 0=proton 1=neutron
            sio = cascade_mb.jax_channel_sigmas_resolved(Wl.reshape(-1),
                      jnp.broadcast_to(ch[:, None], (n, K_)).reshape(-1),
                      nuc_i.reshape(-1)).reshape(n, K_, 3)
            ssl = jnp.clip(jnp.sum(sio, axis=-1), 0.0, None)
            return sal, ssl, sio, Wl

        if cfg.fast_xsec and cfg.algo == "step":
            # evaluate the cross sections ONLY for the K nearest in-slab nucleons, scatter back into
            # the (n,A) grid (sigma off-slab is unused: prob is masked to in_slab).  Bit-exact <=K/slab.
            # (Only valid for "step": "interaction" needs sigma for ALL nucleons ahead -> dense path.)
            score = jnp.where(cand, -perp2, -jnp.inf)
            _, idx = jax.lax.top_k(score, _KSLAB)                    # (n,K) nearest-impact in-slab nucleons
            gi = (ar[:, None], idx)
            sa_k, ss_k, sio_k, W_k = _xsec(nmom[ar[:, None], idx], npos[ar[:, None], idx], nisp[ar[:, None], idx])
            sa = jnp.zeros((n, A)).at[gi].set(sa_k)
            ss = jnp.zeros((n, A)).at[gi].set(ss_k)
            sig_io = jnp.zeros((n, A, 3)).at[gi].set(sio_k).reshape(n * A, 3)
            W = jnp.zeros((n, A)).at[gi].set(W_k)
        else:
            sa, ss, sio, W = _xsec(nmom, npos, nisp)
            sig_io = sio.reshape(n * A, 3)
        # ACHILLES PionAbsorption isospin partition (Nucl.Phys.A568 Table 1, PionAbsorption.cc:85-136):
        # the absorption xsec entering the cascade competition is split by partner-channel isospin.
        # For LIKE-CHARGE pairs (pi+ p, pi- n) charge conservation forces identical outgoing nucleons
        # (p+p / n+n) -> only the opposite-isospin partner channel survives -> abs = (5/6)*oset_abs.
        # All other (pi,N) pairs keep the full oset_abs (the 3 modes sum back to it).  ADoNIS used the
        # full oset for all, over-absorbing pi+ on protons (the dominant Delta++ channel).
        like_charge = ((ch == 0)[:, None] & nisp) | ((ch == 2)[:, None] & (~nisp))   # (n,A)
        sa = sa * jnp.where(like_charge, 5.0 / 6.0, 1.0)
        sig = sa + ss                                                # mb

        _mode = cfg.prob if cfg.prob else ("cylinder" if cfg.cylinder else "gaussian")
        _sfm = jnp.clip(sig * MB_TO_FM2, 1e-12, None)                 # sigma in fm^2
        if _mode == "cylinder":
            prob = jnp.where(cand & (perp2 < jnp.clip(sig * MB_TO_FM2, 0.0, None) / jnp.pi), 1.0, 0.0)
        elif _mode == "pion":
            # ACHILLES Probability: Pion -- exp(-sqrt(2 pi/sigma) b), b=sqrt(perp2) (Cascade.cc:47-52).
            # Integrates over the impact-parameter plane to sigma, like Gaussian/Cylinder.
            # WARNING: NOT reliable -- ACHILLES marks this form "TODO: This does not work, we need to
            # rethink this" (Cascade.cc:49); no run-card uses it.  Here for parity, not for production.
            prob = jnp.where(cand, jnp.exp(-jnp.sqrt(2.0 * jnp.pi * perp2 / _sfm)), 0.0)
        else:
            prob = jnp.where(cand, jnp.exp(-jnp.pi * perp2 / _sfm), 0.0)
        sk, ku, kc, kf, ka, kab = jax.random.split(sk, 6)
        passes = cand & (jax.random.uniform(ku, (n, A)) < prob)
        # interacting nucleon: "step" = smallest impact parameter within the slab; "interaction" =
        # the FIRST one reached along the track (smallest par).  Same physical pick (nearest passer).
        metric = jnp.where(passes, perp2 if cfg.algo == "step" else par, jnp.inf)
        j = jnp.argmin(metric, axis=1)
        has_hit = jnp.isfinite(metric[ar, j]) & alive

        sa_j = sa[ar, j]; sig_j = sig[ar, j]; W_j = W[ar, j]; pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
        p_abs = sa_j / jnp.clip(sig_j, 1e-12, None)                           # NOMINAL branch prob
        chose_abs = has_hit & (jax.random.uniform(kc, (n,)) < p_abs)          # channel pick (abs vs scatter)

        # ----- pion ABSORPTION final state (ACHILLES PionAbsorption::GenerateMomentum) -----
        # piNN -> NN: pion + struck nucleon j + closest background nucleon; 2 outgoing nucleons
        # isotropic in the 3-body CM.  ACHILLES FindClosest picks the partner of the charge REQUIRED
        # by the channel (charge conservation): pi+ p forces a neutron partner, pi- n forces a proton.
        qpi = 1 - ch                                                           # 0:pi+ ->+1, 2:pi- ->-1
        struck_p = nisp[ar, j].astype(jnp.int32)                              # struck nucleon proton(1)/neutron(0)
        forced_n = (qpi + struck_p) > 1                                        # only a neutron partner conserves charge
        forced_p = (qpi + struck_p) < 0                                        # only a proton partner conserves charge
        bad_chg = (forced_n[:, None] & nisp) | (forced_p[:, None] & ~nisp)     # (n,A) charge-forbidden partners
        d2 = jnp.sum((npos - npos[ar, j][:, None, :]) ** 2, axis=2)             # (n,A)
        d2 = jnp.where((jnp.arange(A)[None, :] == j[:, None]) | consumed | bad_chg, jnp.inf, d2)
        pj = jnp.argmin(d2, axis=1)                                            # partner index (closest of allowed charge)
        pN_p = nmom[ar, pj]
        nprot_out = qpi + nisp[ar, j].astype(jnp.int32) + nisp[ar, pj].astype(jnp.int32)
        # local Fermi momenta at the two outgoing-nucleon positions: product A inherits the PION
        # position (particle1), product B the struck nucleon position (particle2) -- ACHILLES
        # PionAbsorption::GenerateMomentum places paOut@part1.Position, pbOut@part2.Position.
        pos_hit = pos if cfg.algo == "step" else npos[ar, j]    # interaction-mode vertex = hit nucleon
        kf_pi = _kf_local(_rho_species(jnp.linalg.norm(pos_hit, axis=1), rgrid, rho))   # (n,)
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
        if not cfg.pauli:
            abs_blocked = abs_blocked & False
        is_abs = chose_abs & ~abs_blocked                                     # absorption survives Pauli
        best_abs = jnp.where(is_abs[:, None], abs_lead, best_abs)             # one absorption / pion

        # scatter: out-pion charge ~ sig_io[j], DCC angle, Pauli-block recoil
        sig_io_j = sig_io.reshape(n, A, 3)[ar, j]
        probs = sig_io_j / jnp.clip(jnp.sum(sig_io_j, axis=1, keepdims=True), 1e-12, None)
        u = jax.random.uniform(kf, (n, 1))
        out_ch = jnp.clip(jnp.sum((u > jnp.cumsum(probs, axis=1)).astype(jnp.int32), axis=1), 0, 2).astype(jnp.int32)
        # channel-specific scatter angle: chan = pi_in*6 + nuc*3 + pi_out (nuc 0=p,1=n) -- matches
        # ACHILLES sampling cos_CMS from the channel-specific partial-wave dsigma/dOmega.
        nuc_idx = jnp.where(nisp[ar, j], 0, 1)
        chan_idx = ch * 6 + nuc_idx * 3 + out_ch
        cos_cm = cascade_mb.jax_sample_cos_cm(W_j, jax.random.uniform(ka, (n,)), chan_idx)

        def scat_one(p_pi_i, pN_i, out_i, kf_i, cc, k):
            p_out = _two_body_cm_scatter(p_pi_i, pN_i, _CH_MASS[out_i], k, cos_cm=cc)
            p_rec = (p_pi_i + pN_i) - p_out
            return p_out, jnp.linalg.norm(p_rec[1:]) < kf_i
        p_out, blocked = jax.vmap(scat_one)(p_pi, pN_j, out_ch, kf_j, cos_cm, jax.random.split(sk, n))
        if not cfg.pauli:
            blocked = blocked & False

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
        if cfg.algo == "step":
            pos = pos + cfg.step * dhat * alive[:, None]
        else:
            # jump to the hit nucleon (scatter continues from there; a Pauli-blocked passer advances
            # past it without consuming; NO passer -> the particle has left the nucleus -> escape).
            pos = jnp.where((has_hit & alive)[:, None], npos[ar, j], pos)
            alive = alive & has_hit
        # record the per-step branching decision (DETACHED) for the kind-1 reweight done OUTSIDE the scan
        rec = jax.lax.stop_gradient((has_hit, chose_abs, sa_j, sig_j))
        return (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs), rec

    dhat0 = p_pi0[:, 1:] / jnp.clip(jnp.linalg.norm(p_pi0[:, 1:], axis=1, keepdims=True), 1e-9, None)
    init = (pos0, p_pi0, ch0, dhat0, jnp.ones(n, bool), jnp.zeros(n, bool),
            jnp.zeros(n, jnp.int32), consumed0, jnp.zeros((n, 4)))
    (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs), recs = jax.lax.scan(body, init, keys)
    # kind-1 branching reweight, OUTSIDE the scan: the records are detached, so ONLY the (sabs, sscat)
    # knobs carry gradient -- no NaN VJPs from the cascade's final-state sampling can reach it.
    hh, ca, saj, sigj = recs                                      # each (max_steps, n)
    # compress the branch records to the <=_K_BR hit slots (in step order): the walk is
    # theta-INDEPENDENT, so the (sabs, sscat) reweight is a pure function of these records --
    # a fit can precompute the walk once and reweight cheaply (pion_branch_reweight).
    slot = jnp.cumsum(hh.astype(jnp.int32), axis=0) - 1
    slot = jnp.where(hh, slot, _K_BR)                              # _K_BR = out of range -> dropped
    arn = jnp.broadcast_to(jnp.arange(hh.shape[1])[None, :], slot.shape)
    ca_c = jnp.zeros((_K_BR, hh.shape[1]), bool).at[slot, arn].set(ca, mode="drop")
    sa_c = jnp.ones((_K_BR, hh.shape[1])).at[slot, arn].set(saj, mode="drop")
    sig_c = jnp.full((_K_BR, hh.shape[1]), 2.0).at[slot, arn].set(sigj, mode="drop")
    nh = jnp.sum(hh.astype(jnp.int32), axis=0)
    brec = (ca_c.T, sa_c.T, sig_c.T, nh)                           # (n,K)x3 + (n,)
    w_fsi = pion_branch_reweight(brec, sabs, sscat)
    nseg = nh                                       # per-event interaction-ATTEMPT count (segments); for MAX_SEG sizing
    n_trunc = jnp.sum((alive & ~absorbed).astype(jnp.int32))   # still propagating at the cap (interaction-mode truncations)
    return p_pi, ch, absorbed, nsc, best_abs, w_fsi, nseg, n_trunc, brec


def propagate_discrete(pos0, p_pi0, charge_idx0, npos, nmom, nisp, cfg, key, consumed0=None,
                       sabs=1.0, sscat=1.0):
    """Discrete-Glauber transport of a batch of pions through explicit nucleons.
    Returns (p_pi, charge_idx, absorbed, n_scatter, abs_lead_p, w_fsi, nseg, n_trunc).  w_fsi is the
    kind-1 branching reweight for the (sabs, sscat) scales (1 at nominal); nseg is the per-event
    interaction-attempt count; n_trunc is the # of particles still propagating at the MAX_SEG cap."""
    _load_density(cfg.nucleus); cascade_mb._jax_grids(); cascade_mb._build_angular()
    if consumed0 is None:
        consumed0 = jnp.zeros((nisp.shape[0], nisp.shape[1]), bool)
    return _propagate_discrete(pos0, p_pi0, charge_idx0, npos, nmom, nisp, cfg, key, consumed0,
                               jnp.asarray(sabs, float), jnp.asarray(sscat, float))


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

    def apply(self, params, event, key=None, sabs=1.0, sscat=1.0):
        key = jax.random.PRNGKey(self.cfg.seed) if key is None else key
        kn, kv, kp = jax.random.split(key, 3)
        n = event.p_pi.shape[0]
        npos, nmom, nisp = sample_nucleons(kn, n, self.cfg)
        A = nisp.shape[1]
        # production vertex = the STRUCK nucleon (pid_Ni; isospin-correct), CONSUMED from the
        # background.  RES on a struck NEUTRON (nu n -> N pi) leaves 6p5n spectators; since pi+p
        # scatter (Delta++) >> pi+n, leaving the struck neutron in the background under-scatters the
        # pi+ -> it over-absorbs.  Removing the correct struck nucleon fixes the spectator isospin.
        struck_isp = (event.pid_Ni == 2212)                              # True if struck proton (trace-safe)
        rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
        vtx = jnp.argmax(rsel, axis=1)
        pos0 = npos[jnp.arange(n), vtx]
        consumed0 = jax.nn.one_hot(vtx, A).astype(bool)
        # pion pid -> channel index (211->0, 111->1, -211->2, default->1), trace-safe
        ch0 = jnp.where(event.pid_pi == 211, 0, jnp.where(event.pid_pi == -211, 2, 1)).astype(jnp.int32)
        p_pi, ch, absorbed, nsc, abs_lead, w_fsi, nseg, n_trunc, brec = propagate_discrete(
            pos0, event.p_pi, ch0, npos, nmom, nisp, self.cfg, kp, consumed0=consumed0, sabs=sabs, sscat=sscat)
        self.last_abs_proton = abs_lead          # leading absorption proton (piNN->NN), for CC0pi
        self.last_absorbed = absorbed
        self.last_w_fsi = w_fsi                   # kind-1 branching reweight (1 at nominal scales)
        self.last_brec = brec                     # compressed walk records for pion_branch_reweight
        self.last_nseg = nseg                     # per-event interaction-attempt count (segments)
        self.last_nsc = nsc                       # per-event scatter count
        self.last_n_trunc = n_trunc               # # pions truncated at MAX_SEG (should be 0)
        keep = (~absorbed)[:, None]
        return event._replace(p_pi=p_pi * keep, pid_pi=jnp.where(absorbed, 0, _CH_PID[ch]),
                              w=event.w * w_fsi)


# ===== discrete-Glauber NUCLEON cascade (proton/neutron FSI for the TKI observables) ========= #
from adonis.fsi.nucleon_cascade import nn_elastic_sigma


@partial(jax.jit, static_argnums=(6,))
def _propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg: DiscreteCascadeConfig, key, fz0, consumed0, sscat):
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
        # record the closest in-slab nucleon (perp2, sigma) for the kind-1 sigma_scatter reweight;
        # clamp perp2 to a finite dummy when there is NO in-slab nucleon (else exp(-inf/sscat) NaNs the grad)
        perp2_is = jnp.where(in_slab, perp2, jnp.inf); cidx = jnp.argmin(perp2_is, axis=1)
        has_slab = jnp.any(in_slab, axis=1)
        perp2_c = jnp.where(has_slab, perp2_is[ar, cidx], 1e6); sig_c = sig[ar, cidx]
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
        rec = jax.lax.stop_gradient((has_hit, perp2_c, sig_c))
        return (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz), rec

    dhat0 = p_N0[:, 1:] / jnp.clip(jnp.linalg.norm(p_N0[:, 1:], axis=1, keepdims=True), 1e-9, None)
    init = (pos0, p_N0, dhat0, jnp.ones(n, bool), jnp.zeros(n, jnp.int32),
            consumed0, jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), fz0)
    (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz), recs = jax.lax.scan(body, init, keys)
    # kind-1 sigma_scatter reweight (closest-in-slab Bernoulli approximation), OUTSIDE the scan.
    # Compress to the <=_K_SLAB_REC steps with an in-slab candidate (perp2_c dummy = 1e6 marks
    # "no slab"; its br is exactly 1) -- theta-independent walk records for nucleon_scat_reweight.
    hh, perp2_c, sig_c = recs                                      # (max_steps, n)
    a_all = jnp.pi * perp2_c / jnp.clip(sig_c * MB_TO_FM2, 1e-12, None)
    m = perp2_c < 1e5
    slot = jnp.cumsum(m.astype(jnp.int32), axis=0) - 1
    slot = jnp.where(m, slot, _K_SLAB_REC)
    arn = jnp.broadcast_to(jnp.arange(n)[None, :], slot.shape)
    hh_c = jnp.zeros((_K_SLAB_REC, n), bool).at[slot, arn].set(hh, mode="drop")
    a_c = jnp.full((_K_SLAB_REC, n), 50.0).at[slot, arn].set(a_all, mode="drop")
    ns = jnp.sum(m.astype(jnp.int32), axis=0)
    srec = (hh_c.T, a_c.T, ns)                                     # (n,K)x2 + (n,)
    w_scat = nucleon_scat_reweight(srec, sscat)
    return p_N, nsc, best_ko, best_ko_pos, best_ko_fz, w_scat, srec


def propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg, key, fz0=None, consumed0=None, sscat=1.0):
    _load_density(cfg.nucleus)
    n, A = nisp.shape
    if fz0 is None:
        fz0 = jnp.zeros(p_N0.shape[0])
    if consumed0 is None:
        consumed0 = jnp.zeros((n, A), bool)
    return _propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg, key, fz0, consumed0,
                                       jnp.asarray(sscat, float))


class DiscreteNucleonFSI:
    """FSIModel add-on: propagate the leading outgoing NUCLEON through the nucleus with the
    discrete-Glauber NN-elastic cascade.  Composable with DiscreteCascadeFSI (pion)."""

    def __init__(self, cfg: DiscreteCascadeConfig = DiscreteCascadeConfig(), protfrac: float = 0.0):
        self.cfg = cfg
        self.protfrac = float(protfrac)

    def apply(self, params, event, key=None, sscat=1.0):
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
        struck_isp = (event.pid_Ni == 2212)                             # True if struck nucleon is a proton (trace-safe)
        rsel = jnp.where(nisp == struck_isp[:, None], jax.random.uniform(kv, (n, A)), -1.0)
        vtx = jnp.argmax(rsel, axis=1)
        pos0 = npos[jnp.arange(n), vtx]
        consumed0 = jax.nn.one_hot(vtx, A).astype(bool)                  # struck nucleon removed from background
        isp0 = (event.pid_N == 2212)
        p_N, nsc, best_ko, best_ko_pos, best_ko_fz, w_sc1, srec1 = propagate_nucleon_discrete(
            pos0, event.p_N, isp0, npos, nmom, nisp, self.cfg, kp, consumed0=consumed0, sscat=sscat)
        # MULTI-NUCLEON cascade (ACHILLES UpdateKicked): the knocked-out proton itself re-cascades.
        # Re-propagate the leading knockout through the same background; its final state can be the
        # leading proton.  (One secondary generation -- the dominant multi-nucleon contribution.)
        # The knockout carries the formation zone ACHILLES assigned it at creation (best_ko_fz).
        kp2 = jax.random.fold_in(kp, 99)
        has_ko = jnp.linalg.norm(best_ko[:, 1:], axis=1) > 1.0
        ko_start = jnp.where(has_ko[:, None], best_ko, event.p_N)         # dummy where no knockout
        isp_ko = jnp.ones(n, bool)                                       # knockout proton
        ko_f, _, ko_ko, _, _, w_sc2, srec2 = propagate_nucleon_discrete(best_ko_pos, ko_start, isp_ko, npos, nmom,
                                                                        nisp, self.cfg, kp2, fz0=best_ko_fz,
                                                                        consumed0=consumed0, sscat=sscat)
        ko_f = jnp.where(has_ko[:, None], ko_f, jnp.zeros((n, 4)))
        ko_ko = jnp.where(has_ko[:, None], ko_ko, jnp.zeros((n, 4)))     # 2nd-gen knockout proton
        # leading proton = highest-momentum proton among {primary, re-cascaded knockout, 2nd-gen ko}
        cands = jnp.stack([p_N, ko_f, ko_ko], axis=1)                   # (n,3,4)
        mom = jnp.linalg.norm(cands[:, :, 1:], axis=2)                  # (n,3)
        lead = cands[jnp.arange(n), jnp.argmax(mom, axis=1)]
        self.last_w_scat = w_sc1 * jnp.where(has_ko, w_sc2, 1.0)        # kind-1 sigma_scatter reweight (1 at nominal)
        self.last_srec = (srec1, srec2, has_ko)   # compressed walk records for nucleon_scat_reweight
        return event._replace(p_N=lead, w=event.w * self.last_w_scat)
