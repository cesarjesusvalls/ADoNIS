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
import os
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
        with gzip.open(path, "rt") as f:
            # header: [A Nconfigs maxWgt minWgt] (ACHILLES Configuration.cc:30-37).  A is read HERE,
            # not hardcoded -> QMC (A=12, C) and RMF (A=40, Ar) share this parser unchanged.
            hdr = f.readline().split()
            A = int(hdr[0]); ncfg = int(hdr[1]); nread = min(nmax, ncfg)
            iso = np.zeros((nread, A), bool); pos = np.zeros((nread, A, 3)); wt = np.zeros(nread)
            for c in range(nread):
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
_N_RECOIL = int(os.environ.get("ADONIS_N_RECOIL", "4"))   # top-K proton recoils tracked per particle (was
                             # 1, leading-only).  The LEADING (max-momentum) slot reproduces the old single
                             # best_rec bit-exactly -> production (which uses only the leading) is unaffected;
                             # the full top-K is exposed for the faithful multi-particle cascade engine
                             # (cascade_full).  Env-overridable (ADONIS_N_RECOIL) to shrink the spawn buffer.


def pion_branch_reweight(brec, sabs, sscat):
    """Kind-1 branching reweight from compressed walk records brec =
    (branch (n,K) int {0 scatter, 1 abs, 2 conversion}, sa, ss, si (n,K), n_hits (n,)).
    Per-hit likelihood ratio p_branch(theta)/p_branch(nominal) with
    p_abs = sabs*sa/D, p_scat = sscat*ss/D, p_conv = si/D, D = sabs*sa + sscat*ss + si
    (conversion sigma unscaled).  Pure in (sabs, sscat); == in-propagation w_fsi; reduces to
    the previous two-branch formula where si = 0."""
    bc, sa, ss, si, nh = brec
    valid = jnp.arange(sa.shape[1])[None, :] < nh[:, None]
    ss = jnp.clip(ss, 1e-6, None)
    D0 = sa + ss + si
    Dk = sabs * sa + sscat * ss + si
    num = jnp.where(bc == 1, sabs * sa, jnp.where(bc == 2, si, sscat * ss))
    den = jnp.where(bc == 1, sa, jnp.where(bc == 2, si, ss))
    br = (num / Dk) / jnp.clip(den / D0, 1e-12, None)
    return jnp.prod(jnp.where(valid, br, 1.0), axis=1)


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
    nucleus: str = "c12_density.txt"       # proton density file (data/nuclear/)
    density_n: str = "c12_density.txt"     # neutron density file (= nucleus for N=Z nuclei, e.g. C)
    configs: str = "QMC_configs.out.gz"    # nucleon configuration file (QMC/RMF; A read from header)
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
    nn_inelastic: bool = True  # NN -> N Delta -> N N pi in the NUCLEON cascade (ACHILLES
                             # NucleonNucleon GiBUU ResonanceMode: Decay; adonis/fsi/nn_inelastic).
                             # Degrades fast nucleons and CREATES a pion (meson-veto relevant).
                             # False = the previous elastic-only walk (bit-exact).
    early_exit: bool = True  # while_loop walk that stops once NO particle can interact again
                             # (dead, or outside the radius moving outward = inert).  BIT-EXACT in
                             # every returned output (same per-step keys; skipped steps are
                             # identity), gated in tests/test_cascade_early_exit.py.  max_steps
                             # then acts as a pure safety bound.  False = the reference lax.scan.
    track_steps: bool = False  # MC-truth: also stack the per-step (pos, p4, alive) trajectory from the
                             # scan -> returns traj for viz/diagnostics (adonis/fsi/tracking).  Forces the
                             # reference scan (not early_exit).  DEFAULT OFF -> production path unchanged.


def sample_nucleons(key, n, cfg: DiscreteCascadeConfig):
    """Pick n configurations ~ weight; assign each nucleon an isotropic local-Fermi-gas momentum.
    Returns npos (n,A,3), nmom (n,A,4), nisp (n,A) proton-mask."""
    pos, iso, w, A = _load_qmc_configs(name=cfg.configs)
    rgrid, rhoP, rhoN, _ = _load_density(cfg.nucleus, cfg.density_n)
    kc, kd, km = jax.random.split(key, 3)
    idx = jax.random.choice(kc, pos.shape[0], (n,), p=w)
    npos = pos[idx]; nisp = iso[idx]
    r = jnp.linalg.norm(npos, axis=2)
    # ACHILLES Local FG: PER-SPECIES k_F -- protons from rho_p, neutrons from rho_n (Nucleus.cc:212-238).
    # For N=Z (carbon) rho_p == rho_n bitwise -> unchanged.
    kf = _kf_local(jnp.where(nisp, _rho_species(r, rgrid, rhoP), _rho_species(r, rgrid, rhoN)))
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
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    n, A = nisp.shape
    nsteps = cfg.max_steps if cfg.algo == "step" else _MAX_SEG
    keys = jax.random.split(key, nsteps)
    ar = jnp.arange(n)

    def body(carry, sk):
        (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs,
         best_rec, best_rec_pos, best_rec_fz, conv, best_abs2) = carry
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
        kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP),   # (n,A) per-species k_F at the
                                   _rho_species(rnuc, rgrid, rhoN)))        # hit nucleon (proton/neutron)

        def _xsec(nm, npo, nip):
            """Oset abs + DCC scatter cross sections for nucleon states nm (..,4), npo (..,3),
            proton-mask nip (..) vs the current pion (pE,pmom,m_pi,vpi,ch).  Returns sa, ss,
            sig_io(...,3), W -- same leading shape.  Scatter sigma is CHARGE-RESOLVED by the struck
            nucleon (ACHILLES GetCchannel(pion,baryon)): sigma(pi+ p) ~ 3x sigma(pi+ n) at the Delta."""
            vN = nm[..., 1:] / nm[..., 0:1]
            vrel = jnp.clip(jnp.linalg.norm(vpi[:, None, :] - vN, axis=-1), 1e-3, None)
            rnpo = jnp.linalg.norm(npo, axis=-1)
            rho_t = _rho_species(rnpo, rgrid, rhoP) + _rho_species(rnpo, rgrid, rhoN)  # TOTAL density
            kf = _kf_local(jnp.where(nip, _rho_species(rnpo, rgrid, rhoP),   # per-species k_F of the
                                     _rho_species(rnpo, rgrid, rhoN)))       # candidate nucleon
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
            # piN -> {etaN, KLambda, KSigma} conversion (ACHILLES MesonBaryonAmplitudes open
            # channels; W > ~1.49 GeV) -- removes the pion from the pi system.
            sil = jnp.clip(cascade_mb.jax_conversion_sigma(Wl.reshape(-1),
                      jnp.broadcast_to(ch[:, None], (n, K_)).reshape(-1),
                      nuc_i.reshape(-1)).reshape(n, K_), 0.0, None)
            return sal, ssl, sil, sio, Wl

        if cfg.fast_xsec and cfg.algo == "step":
            # evaluate the cross sections ONLY for the K nearest in-slab nucleons, scatter back into
            # the (n,A) grid (sigma off-slab is unused: prob is masked to in_slab).  Bit-exact <=K/slab.
            # (Only valid for "step": "interaction" needs sigma for ALL nucleons ahead -> dense path.)
            score = jnp.where(cand, -perp2, -jnp.inf)
            _, idx = jax.lax.top_k(score, _KSLAB)                    # (n,K) nearest-impact in-slab nucleons
            gi = (ar[:, None], idx)
            sa_k, ss_k, si_k, sio_k, W_k = _xsec(nmom[ar[:, None], idx], npos[ar[:, None], idx], nisp[ar[:, None], idx])
            sa = jnp.zeros((n, A)).at[gi].set(sa_k)
            ss = jnp.zeros((n, A)).at[gi].set(ss_k)
            si = jnp.zeros((n, A)).at[gi].set(si_k)
            sig_io = jnp.zeros((n, A, 3)).at[gi].set(sio_k).reshape(n * A, 3)
            W = jnp.zeros((n, A)).at[gi].set(W_k)
        else:
            sa, ss, si, sio, W = _xsec(nmom, npos, nisp)
            sig_io = sio.reshape(n * A, 3)
        # ACHILLES PionAbsorption isospin partition (Nucl.Phys.A568 Table 1, PionAbsorption.cc:85-136):
        # the absorption xsec entering the cascade competition is split by partner-channel isospin.
        # For LIKE-CHARGE pairs (pi+ p, pi- n) charge conservation forces identical outgoing nucleons
        # (p+p / n+n) -> only the opposite-isospin partner channel survives -> abs = (5/6)*oset_abs.
        # All other (pi,N) pairs keep the full oset_abs (the 3 modes sum back to it).  ADoNIS used the
        # full oset for all, over-absorbing pi+ on protons (the dominant Delta++ channel).
        like_charge = ((ch == 0)[:, None] & nisp) | ((ch == 2)[:, None] & (~nisp))   # (n,A)
        sa = sa * jnp.where(like_charge, 5.0 / 6.0, 1.0)
        sig = sa + ss + si                                           # mb (total interaction reach)

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

        sa_j = sa[ar, j]; si_j = si[ar, j]; sig_j = sig[ar, j]
        W_j = W[ar, j]; pN_j = nmom[ar, j]; kf_j = kf_n[ar, j]
        p_abs = sa_j / jnp.clip(sig_j, 1e-12, None)                           # NOMINAL branch probs
        p_conv = si_j / jnp.clip(sig_j, 1e-12, None)
        u_br = jax.random.uniform(kc, (n,))                                   # ONE roll, 3-way split
        chose_abs = has_hit & (u_br < p_abs)                                  # abs | conversion | scatter
        chose_conv = has_hit & ~chose_abs & (u_br < p_abs + p_conv)           # (si=0 -> never fires)

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
        # absorption product-A Fermi momentum at the pion vertex.  The outgoing-nucleon species is
        # channel-dependent (nprot_out); use the proton density (== neutron for N=Z carbon, bit-exact).
        # NOTE (Ar): this is a per-species approximation for the absorption Pauli block -- validate.
        kf_pi = _kf_local(_rho_species(jnp.linalg.norm(pos_hit, axis=1), rgrid, rhoP))   # (n,)
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
            one_p = jnp.where(jax.random.uniform(k3) < 0.5, pa, pb)            # which of the two is p (when 1)
            # FIX: piNN->NN has TWO outgoing nucleons; feed BOTH protons (npr==2 -> pa & pb; npr==1 ->
            # one_p only; npr==0 -> none).  Previously only the leading was kept -> exactly-2p overshoot.
            protA = jnp.where(npr >= 2, pa, jnp.where(npr == 1, one_p, jnp.zeros(4)))
            protB = jnp.where(npr >= 2, pb, jnp.zeros(4))
            return protA, protB, blocked
        abs_protA, abs_protB, abs_blocked = jax.vmap(abs_one)(p_pi, pN_j, pN_p, nprot_out, kf_pi, kf_absB,
                                                             jax.random.split(kab, n))
        if not cfg.pauli:
            abs_blocked = abs_blocked & False
        is_abs = chose_abs & ~abs_blocked                                     # absorption survives Pauli
        best_abs = jnp.where(is_abs[:, None], abs_protA, best_abs)            # 1st absorption proton
        best_abs2 = jnp.where(is_abs[:, None], abs_protB, best_abs2)          # 2nd absorption proton (piNN->NN)

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

        # pion CONVERSION (piN -> etaN/KLambda/KSigma): the pion leaves the pi system.  The
        # eta/K + baryon products are NOT propagated (declared approximation; both CC0pi and
        # CC1pi signal definitions veto these events regardless).  No Pauli check on the
        # conversion products (declared; their nucleons are far above kF near the thresholds).
        is_conv = chose_conv
        conv = conv | is_conv
        is_scat = has_hit & ~chose_abs & ~chose_conv & ~blocked    # scatter chosen, recoil not Pauli-blocked
        # pion-scatter RECOIL nucleon (ACHILLES FinalizeMomentum emits it as a propagating
        # particle with fz = SetFormationZone(p_pi_in, p_rec); pid from the charge-resolved
        # channel: q_rec = q_struck + q_pi_in - q_pi_out).  Track the LEADING PROTON recoil
        # per event (the leading-proton observables' dominant contribution; neutron recoils'
        # secondary knockouts are neglected -- declared approximation).
        p_rec = (p_pi + pN_j) - p_out
        q_rec = struck_p + out_ch - ch                                # +1 = proton recoil
        fz_rec = _formation_zone(p_pi, p_rec)
        # TOP-K proton recoils: insert p_rec into the K-slot buffer if it beats the slot of smallest
        # momentum (K=1 -> identical to the old running-max single best_rec; the max slot is always the
        # leading recoil bit-exactly).  Tracks the K highest-momentum proton recoils per pion.
        rec_is_p = is_scat & (q_rec == 1)
        cand_mom = jnp.linalg.norm(p_rec[:, 1:], axis=1)              # (n,)
        slot_mom = jnp.linalg.norm(best_rec[:, :, 1:], axis=2)        # (n,K)
        minslot = jnp.argmin(slot_mom, axis=1)                        # (n,) smallest-momentum slot
        do_ins = rec_is_p & (cand_mom > slot_mom[ar, minslot])
        sel = jax.nn.one_hot(minslot, _N_RECOIL, dtype=bool) & do_ins[:, None]   # (n,K)
        best_rec = jnp.where(sel[:, :, None], p_rec[:, None, :], best_rec)
        best_rec_pos = jnp.where(sel[:, :, None], npos[ar, j][:, None, :], best_rec_pos)
        best_rec_fz = jnp.where(sel, fz_rec[:, None], best_rec_fz)
        p_pi = jnp.where(is_scat[:, None], p_out, p_pi)
        ch = jnp.where(is_scat, out_ch, ch)
        nsc = nsc + is_scat.astype(jnp.int32)
        absorbed = absorbed | is_abs
        alive = alive & ~is_abs & ~is_conv
        # consume the struck nucleon on any (non-Pauli-blocked) interaction
        interacted = is_abs | is_scat | is_conv
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
        # branch code per hit: 0 = scatter picked (incl. Pauli-blocked), 1 = abs, 2 = conversion
        bcode = jnp.where(chose_abs, 1, jnp.where(chose_conv, 2, 0)).astype(jnp.int32)
        ss_j = sig_j - sa_j - si_j                                            # elastic sigma at the hit
        rec = jax.lax.stop_gradient((has_hit, bcode, sa_j, ss_j, si_j))
        return (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs,
                best_rec, best_rec_pos, best_rec_fz, conv, best_abs2), rec

    dhat0 = p_pi0[:, 1:] / jnp.clip(jnp.linalg.norm(p_pi0[:, 1:], axis=1, keepdims=True), 1e-9, None)
    init = (pos0, p_pi0, ch0, dhat0, jnp.ones(n, bool), jnp.zeros(n, bool),
            jnp.zeros(n, jnp.int32), consumed0, jnp.zeros((n, 4)),
            jnp.zeros((n, _N_RECOIL, 4)), jnp.zeros((n, _N_RECOIL, 3)), jnp.zeros((n, _N_RECOIL)), jnp.zeros(n, bool),
            jnp.zeros((n, 4)))                                            # best_abs2 (2nd absorption proton)
    traj = None
    if cfg.early_exit and not cfg.track_steps:
        # EARLY-EXIT walk (bit-exact): while_loop over the SAME per-step keys, stopping once no
        # particle can interact again.  A pion outside the nuclear radius moving OUTWARD is INERT:
        # it can never re-enter (straight line, rho=0 outside), so its remaining march to the
        # escape boundary only advances `pos` (not returned) -- skipping those steps changes no
        # output.  Branch records are compressed ON THE FLY (same step order as the scan cumsum).
        bufs0 = (jnp.zeros((_K_BR, n), jnp.int32), jnp.ones((_K_BR, n)), jnp.ones((_K_BR, n)),
                 jnp.zeros((_K_BR, n)), jnp.zeros(n, jnp.int32))

        def wcond(st):
            i, carry, _ = st
            pos, _, _, dhat, alive = carry[0], carry[1], carry[2], carry[3], carry[4]
            outward = jnp.sum(pos * dhat, axis=1) > 0
            inert = (jnp.linalg.norm(pos, axis=1) > radius) & outward
            return (i < nsteps) & jnp.any(alive & ~inert)

        def wbody(st):
            i, carry, bufs = st
            carry2, rec = body(carry, keys[i])
            hh, bcj, saj, ssj, sij = rec
            bc_c, sa_c, ss_c, si_c, nh = bufs
            slot = jnp.where(hh, nh, _K_BR)                        # _K_BR = out of range -> dropped
            bc_c = bc_c.at[slot, ar].set(bcj, mode="drop")
            sa_c = sa_c.at[slot, ar].set(saj, mode="drop")
            ss_c = ss_c.at[slot, ar].set(ssj, mode="drop")
            si_c = si_c.at[slot, ar].set(sij, mode="drop")
            return i + 1, carry2, (bc_c, sa_c, ss_c, si_c, nh + hh.astype(jnp.int32))

        _, carry, bufs = jax.lax.while_loop(wcond, wbody, (jnp.int32(0), init, bufs0))
        (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs,
         best_rec, best_rec_pos, best_rec_fz, conv, best_abs2) = carry
        bc_c, sa_c, ss_c, si_c, nh = bufs
        # truly truncated = still able to interact at the cap (inert walkers excluded)
        outward = jnp.sum(pos * dhat, axis=1) > 0
        inert = (jnp.linalg.norm(pos, axis=1) > radius) & outward
        n_trunc = jnp.sum((alive & ~absorbed & ~inert).astype(jnp.int32))
    elif cfg.track_steps:
        # MC-truth trajectory: reuse `body` verbatim, additionally stack (pos, p4, alive) per step.
        def body_t(carry, sk):
            c2, rec = body(carry, sk)
            return c2, (rec, (c2[0], c2[1], c2[4]))                   # pos, p_pi, alive AFTER the step
        (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs,
         best_rec, best_rec_pos, best_rec_fz, conv, best_abs2), (recs, tr) = jax.lax.scan(body_t, init, keys)
        traj = (tr[0], tr[1], tr[2])                                  # (nsteps,n,3),(nsteps,n,4),(nsteps,n)
        hh, bcj, saj, ssj, sij = recs
        slot = jnp.cumsum(hh.astype(jnp.int32), axis=0) - 1
        slot = jnp.where(hh, slot, _K_BR)
        arn = jnp.broadcast_to(jnp.arange(hh.shape[1])[None, :], slot.shape)
        bc_c = jnp.zeros((_K_BR, hh.shape[1]), jnp.int32).at[slot, arn].set(bcj, mode="drop")
        sa_c = jnp.ones((_K_BR, hh.shape[1])).at[slot, arn].set(saj, mode="drop")
        ss_c = jnp.ones((_K_BR, hh.shape[1])).at[slot, arn].set(ssj, mode="drop")
        si_c = jnp.zeros((_K_BR, hh.shape[1])).at[slot, arn].set(sij, mode="drop")
        nh = jnp.sum(hh.astype(jnp.int32), axis=0)
        n_trunc = jnp.sum((alive & ~absorbed).astype(jnp.int32))
    else:
        (pos, p_pi, ch, dhat, alive, absorbed, nsc, consumed, best_abs,
         best_rec, best_rec_pos, best_rec_fz, conv, best_abs2), recs = jax.lax.scan(body, init, keys)
        # kind-1 branching reweight, OUTSIDE the scan: the records are detached, so ONLY the
        # (sabs, sscat) knobs carry gradient -- no NaN VJPs from the cascade's final-state
        # sampling can reach it.  Compress to the <=_K_BR hit slots (in step order): the walk is
        # theta-INDEPENDENT, so the reweight is a pure function of these records.
        hh, bcj, saj, ssj, sij = recs                             # each (max_steps, n)
        slot = jnp.cumsum(hh.astype(jnp.int32), axis=0) - 1
        slot = jnp.where(hh, slot, _K_BR)                          # _K_BR = out of range -> dropped
        arn = jnp.broadcast_to(jnp.arange(hh.shape[1])[None, :], slot.shape)
        bc_c = jnp.zeros((_K_BR, hh.shape[1]), jnp.int32).at[slot, arn].set(bcj, mode="drop")
        sa_c = jnp.ones((_K_BR, hh.shape[1])).at[slot, arn].set(saj, mode="drop")
        ss_c = jnp.ones((_K_BR, hh.shape[1])).at[slot, arn].set(ssj, mode="drop")
        si_c = jnp.zeros((_K_BR, hh.shape[1])).at[slot, arn].set(sij, mode="drop")
        nh = jnp.sum(hh.astype(jnp.int32), axis=0)
        n_trunc = jnp.sum((alive & ~absorbed).astype(jnp.int32))   # still propagating at the cap
    brec = (bc_c.T, sa_c.T, ss_c.T, si_c.T, nh)                    # (n,K)x4 + (n,)
    w_fsi = pion_branch_reweight(brec, sabs, sscat)
    nseg = nh                                       # per-event interaction-ATTEMPT count (segments); for MAX_SEG sizing
    # leading proton recoil = the max-momentum top-K slot -- BIT-EXACT to the old single best_rec, so the
    # production scat_ko (which uses only the leading) is unchanged; scat_ko_all exposes all top-K recoils.
    _arn = jnp.arange(best_rec.shape[0])
    _lead = jnp.argmax(jnp.linalg.norm(best_rec[:, :, 1:], axis=2), axis=1)
    scat_ko_lead = (best_rec[_arn, _lead], best_rec_pos[_arn, _lead], best_rec_fz[_arn, _lead])
    return (p_pi, ch, absorbed, conv, nsc, best_abs, w_fsi, nseg, n_trunc, brec,
            scat_ko_lead, (best_rec, best_rec_pos, best_rec_fz), traj,   # traj None unless cfg.track_steps
            pos, best_abs2)                                               # pion terminal pos; 2nd absorption proton


def propagate_discrete(pos0, p_pi0, charge_idx0, npos, nmom, nisp, cfg, key, consumed0=None,
                       sabs=1.0, sscat=1.0):
    """Discrete-Glauber transport of a batch of pions through explicit nucleons.
    Returns (p_pi, charge_idx, absorbed, n_scatter, abs_lead_p, w_fsi, nseg, n_trunc).  w_fsi is the
    kind-1 branching reweight for the (sabs, sscat) scales (1 at nominal); nseg is the per-event
    interaction-attempt count; n_trunc is the # of particles still propagating at the MAX_SEG cap."""
    _load_density(cfg.nucleus, cfg.density_n); cascade_mb._jax_grids(); cascade_mb._build_angular()
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
        (p_pi, ch, absorbed, conv, nsc, abs_lead, w_fsi, nseg, n_trunc, brec, scat_ko, scat_ko_all, *_) = propagate_discrete(
            pos0, event.p_pi, ch0, npos, nmom, nisp, self.cfg, kp, consumed0=consumed0, sabs=sabs, sscat=sscat)
        self.last_scat_ko_all = scat_ko_all       # all top-K proton recoils (n,K,4),(n,K,3),(n,K) -- engine
        self.last_abs_proton = abs_lead          # leading absorption proton (piNN->NN), for CC0pi
        self.last_absorbed = absorbed
        self.last_w_fsi = w_fsi                   # kind-1 branching reweight (1 at nominal scales)
        self.last_brec = brec                     # compressed walk records for pion_branch_reweight
        self.last_scat_ko = scat_ko               # leading PROTON recoil of pion scatters:
                                                  # (p (n,4), vertex pos (n,3), formation zone (n,))
                                                  # -- zero rows where no proton recoil was emitted
        self.last_nseg = nseg                     # per-event interaction-attempt count (segments)
        self.last_nsc = nsc                       # per-event scatter count
        self.last_n_trunc = n_trunc               # # pions truncated at MAX_SEG (should be 0)
        self.last_converted = conv               # pion converted to etaN/KLambda/KSigma (not a pion)
        gone = absorbed | conv
        keep = (~gone)[:, None]
        return event._replace(p_pi=p_pi * keep,
                              pid_pi=jnp.where(absorbed, 0, jnp.where(conv, -1, _CH_PID[ch])),
                              w=event.w * w_fsi)


# ===== discrete-Glauber NUCLEON cascade (proton/neutron FSI for the TKI observables) ========= #
from adonis.fsi.nucleon_cascade import nn_elastic_sigma
from adonis.fsi import nn_inelastic as nni


@partial(jax.jit, static_argnums=(6,))
def _propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg: DiscreteCascadeConfig, key, fz0, consumed0, sscat):
    """Leading nucleon walks through the background config via NN-elastic scatter (isotropic CM,
    as ACHILLES NucleonNucleon::GenerateMomentum), Pauli-blocking BOTH outgoing nucleons, consuming
    the struck one.  No absorption.  isp0 (n,) proton-mask of the leading nucleon.  fz0 (n,) initial
    formation zone [fm] (ACHILLES: 0 for the primary nucleon).  A nucleon cannot interact while
    fz>0; fz decrements by timeStep=step/beta each step and resets on every scatter to
    E_in*hbarc/|mN^2-p_in.p_out| (ACHILLES SetFormationZone -> suppresses rapid forward re-scatter)."""
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    n, A = nisp.shape
    keys = jax.random.split(key, cfg.max_steps)
    ar = jnp.arange(n)

    def body(carry, sk):
        (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz, made_pi,
         best_pi, best_pi_pos, best_pi_fz, best_pi_ch) = carry
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
        sig_el = jnp.clip(nn_elastic_sigma(sqrts, same_iso), 0.0, None)   # mb
        if cfg.nn_inelastic:
            # NN -> N Delta (ACHILLES GiBUU; ExpSup=0 in the T2K card -> no density suppression)
            pcm = jnp.sqrt(jnp.clip(s / 4.0 - M_N ** 2, 1e-6, None)) / 1000.0   # GeV
            sig_in = jnp.clip(nni.sigma_nn_ndelta(sqrts / 1000.0, pcm, same_iso), 0.0, None)
        else:
            sig_in = jnp.zeros_like(sig_el)
        sig = sig_el + sig_in                                         # total interaction reach
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
        rnuc = jnp.linalg.norm(npos, axis=2)
        kf_n = _kf_local(jnp.where(nisp, _rho_species(rnuc, rgrid, rhoP), _rho_species(rnuc, rgrid, rhoN)))
        kf_j = kf_n[ar, j]

        def scat_one(p_lead, pN_i, kf_i, k):
            p_out = _two_body_cm_scatter(p_lead, pN_i, M_N, k)        # leading out (isotropic)
            p_rec = (p_lead + pN_i) - p_out
            blocked = (jnp.linalg.norm(p_out[1:]) < kf_i) | (jnp.linalg.norm(p_rec[1:]) < kf_i)
            return p_out, blocked
        p_out, blocked = jax.vmap(scat_one)(p_N, pN_j, kf_j, jax.random.split(ks, n))
        # ---- NN -> N Delta -> N N pi (inelastic) branch: fold_in keys leave the elastic
        # stream untouched (bit-exact when nn_inelastic=False) ------------------------------
        sig_in_j = sig_in[ar, j]; sig_el_j = sig_el[ar, j]
        u_br = jax.random.uniform(jax.random.fold_in(sk, 101), (n,))
        chose_inel = has_hit & (u_br < sig_in_j / jnp.clip(sig_el_j + sig_in_j, 1e-12, None))
        Pj = p_N + pN_j
        rs_j = jnp.sqrt(jnp.clip(Pj[:, 0] ** 2 - jnp.sum(Pj[:, 1:] ** 2, axis=1), (2 * M_N) ** 2, None))
        u_m = jax.random.uniform(jax.random.fold_in(sk, 102), (n,))
        m_d = jnp.clip(nni.sample_delta_mass(rs_j / 1000.0, u_m) * 1000.0,
                       M_N + 135.0, rs_j - M_N - 1.0)                  # MeV, kinematic clamp
        cth1 = 2 * jax.random.uniform(jax.random.fold_in(sk, 103), (n,)) - 1.0
        phi1 = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 104), (n,))
        cth2 = 2 * jax.random.uniform(jax.random.fold_in(sk, 105), (n,)) - 1.0
        phi2 = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(sk, 106), (n,))

        def _split2(P4, mA, mB, cth_, phi_):
            ss = jnp.clip(P4[:, 0] ** 2 - jnp.sum(P4[:, 1:] ** 2, axis=1), (mA + mB) ** 2 * 1.0001, None)
            rss = jnp.sqrt(ss)
            EA = (ss + mA ** 2 - mB ** 2) / (2 * rss)
            pf = jnp.sqrt(jnp.clip(EA ** 2 - mA ** 2, 0.0, None))
            sth_ = jnp.sqrt(jnp.clip(1 - cth_ ** 2, 0, None))
            d_ = jnp.stack([sth_ * jnp.cos(phi_), sth_ * jnp.sin(phi_), cth_], axis=1)
            pa = jnp.concatenate([EA[:, None], pf[:, None] * d_], axis=1)
            pb = jnp.concatenate([(rss - EA)[:, None], -pf[:, None] * d_], axis=1)
            beta_ = P4[:, 1:] / P4[:, [0]]
            b2_ = jnp.sum(beta_ ** 2, axis=1); g_ = 1 / jnp.sqrt(jnp.clip(1 - b2_, 1e-12, None))
            def lab(p4):
                bp_ = jnp.sum(beta_ * p4[:, 1:], axis=1)
                E = g_ * (p4[:, 0] + bp_)
                p3 = p4[:, 1:] + ((g_ - 1) * bp_ / jnp.clip(b2_, 1e-30, None) + g_ * p4[:, 0])[:, None] * beta_
                return jnp.concatenate([E[:, None], p3], axis=1)
            return lab(pa), lab(pb)

        pN1, pD = _split2(Pj, jnp.full((n,), M_N), m_d, cth1, phi1)    # N + Delta
        pN2, _pPiX = _split2(pD, jnp.full((n,), M_N), jnp.full((n,), 138.04), cth2, phi2)  # Delta -> N' pi
        in_blocked = ((jnp.linalg.norm(pN1[:, 1:], axis=1) < kf_j)
                      | (jnp.linalg.norm(pN2[:, 1:], axis=1) < kf_j))
        if not cfg.pauli:
            in_blocked = in_blocked & False
        is_inel = chose_inel & ~in_blocked
        lead_in = jnp.where((jnp.linalg.norm(pN1[:, 1:], axis=1)
                             >= jnp.linalg.norm(pN2[:, 1:], axis=1))[:, None], pN1, pN2)
        made_pi = made_pi | is_inel
        # CREATED-PION CHARGE (ACHILLES AllowedResonanceStates -> Delta -> N pi, data/decays.yml).
        # NEW fold_in keys (107/108) -> the elastic + existing-inelastic streams and made_pi stay bit-exact.
        q_pair = isp0.astype(jnp.int32) + nisp[ar, j].astype(jnp.int32)        # 0=nn 1=pn 2=pp
        u107 = jax.random.uniform(jax.random.fold_in(sk, 107), (n,))
        u108 = jax.random.uniform(jax.random.fold_in(sk, 108), (n,))
        dch = jnp.where(q_pair == 2, jnp.where(u107 < 0.75, 2, 1),             # pp: ++ 3/4 | + 1/4
                jnp.where(q_pair == 1, jnp.where(u107 < 0.5, 1, 0),            # pn: + 1/2 | 0 1/2
                                       jnp.where(u107 < 0.25, 0, -1)))         # nn: 0 1/4 | - 3/4
        pi_q = jnp.where(dch == 2, 1,                                          # ++ -> pi+
                jnp.where(dch == 1, jnp.where(u108 < 1.0/3.0, 1, 0),           # + -> pi+ 1/3 | pi0 2/3
                jnp.where(dch == 0, jnp.where(u108 < 2.0/3.0, 0, -1), -1)))    # 0 -> pi0 2/3 | pi- 1/3 ; - -> pi-
        pi_chidx = (1 - pi_q).astype(jnp.int32)                               # +1->0(pi+),0->1(pi0),-1->2(pi-)
        # track the LEADING created pion (4-vec _pPiX + charge + vertex + formation zone)
        pi_better = is_inel & (jnp.linalg.norm(_pPiX[:, 1:], axis=1) > jnp.linalg.norm(best_pi[:, 1:], axis=1))
        fz_pi = _formation_zone(p_N, _pPiX)
        best_pi = jnp.where(pi_better[:, None], _pPiX, best_pi)
        best_pi_pos = jnp.where(pi_better[:, None], npos[ar, j], best_pi_pos)
        best_pi_fz = jnp.where(pi_better, fz_pi, best_pi_fz)
        best_pi_ch = jnp.where(pi_better, pi_chidx, best_pi_ch)
        do = has_hit & ~chose_inel & ~blocked
        # knocked-out nucleon = the struck background nucleon's recoil; if it is a PROTON track
        # the highest-momentum one (ACHILLES adds it to the final state, the analysis may pick it)
        recoil = (p_N + pN_j) - p_out
        bg_proton = nisp[ar, j]
        fz_new = _formation_zone(p_N, p_out)                          # leading: E_in*hbarc/|mN^2-p_in.p_out|
        # FIX: the NN-inelastic (NN->NDelta->NN'pi) SECOND nucleon was dropped (best_ko was elastic-only).
        # Register the non-leading inelastic nucleon as a knockout IF it is a proton.  Charges from the
        # channel: q(pN1)=q_pair-dch (NN->N Delta), q(pN2)=dch-pi_q (Delta->N' pi).
        nl_is1 = jnp.linalg.norm(pN1[:, 1:], axis=1) >= jnp.linalg.norm(pN2[:, 1:], axis=1)  # leading=pN1?
        inel_nl = jnp.where(nl_is1[:, None], pN2, pN1)                # the NON-leading inelastic nucleon
        inel_nl_q = jnp.where(nl_is1, dch - pi_q, q_pair - dch)       # its charge (1 = proton)
        # combined knockout candidate (elastic recoil OR inelastic 2nd nucleon -- mutually exclusive per event)
        ko_cand = jnp.where(is_inel[:, None], inel_nl, recoil)
        ko_is_p = (do & bg_proton) | (is_inel & (inel_nl_q == 1))
        fz_ko = jnp.where(is_inel, _formation_zone(p_N, inel_nl), _formation_zone(p_N, recoil))
        # TOP-K proton knockouts (K=1 -> old single best_ko bit-exactly; max slot = leading knockout):
        cand_mom = jnp.linalg.norm(ko_cand[:, 1:], axis=1)
        slot_mom = jnp.linalg.norm(best_ko[:, :, 1:], axis=2)         # (n,K)
        minslot = jnp.argmin(slot_mom, axis=1)
        do_ins = ko_is_p & (cand_mom > slot_mom[ar, minslot])
        sel = jax.nn.one_hot(minslot, _N_RECOIL, dtype=bool) & do_ins[:, None]
        best_ko = jnp.where(sel[:, :, None], ko_cand[:, None, :], best_ko)
        best_ko_pos = jnp.where(sel[:, :, None], npos[ar, j][:, None, :], best_ko_pos)
        best_ko_fz = jnp.where(sel, fz_ko[:, None], best_ko_fz)
        p_N = jnp.where(do[:, None], p_out, jnp.where(is_inel[:, None], lead_in, p_N))
        nsc = nsc + do.astype(jnp.int32)
        consumed = consumed | (jax.nn.one_hot(j, A, dtype=bool) & (do | is_inel)[:, None])
        fz = jnp.where((fz > 0.0) & alive, fz - timeStep, fz)         # propagate: decrement formation zone
        fz = jnp.where(do, fz_new, jnp.where(is_inel, _formation_zone(p_N, lead_in), fz))                                # reset on scatter
        d3 = p_N[:, 1:]; dhat = d3 / jnp.clip(jnp.linalg.norm(d3, axis=1, keepdims=True), 1e-9, None)
        pos = pos + cfg.step * dhat * alive[:, None]
        rec = jax.lax.stop_gradient((has_hit, perp2_c, sig_c))
        return (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz, made_pi,
                best_pi, best_pi_pos, best_pi_fz, best_pi_ch), rec

    dhat0 = p_N0[:, 1:] / jnp.clip(jnp.linalg.norm(p_N0[:, 1:], axis=1, keepdims=True), 1e-9, None)
    init = (pos0, p_N0, dhat0, jnp.ones(n, bool), jnp.zeros(n, jnp.int32),
            consumed0, jnp.zeros((n, _N_RECOIL, 4)), jnp.zeros((n, _N_RECOIL, 3)), jnp.zeros((n, _N_RECOIL)), fz0, jnp.zeros(n, bool),
            jnp.zeros((n, 4)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n, jnp.int32))
    traj = None
    if cfg.early_exit and not cfg.track_steps:
        # EARLY-EXIT walk (bit-exact): nucleons escape on the sphere (alive -> False), so the
        # loop ends when every nucleon has escaped -- measured ~step 150-180 vs the 260/325 cap.
        # sigma_scatter records (closest-in-slab a = pi b^2/sigma) compressed on the fly.
        bufs0 = (jnp.zeros((_K_SLAB_REC, n), bool), jnp.full((_K_SLAB_REC, n), 50.0),
                 jnp.zeros(n, jnp.int32))

        def wcond(st):
            i, carry, _ = st
            return (i < cfg.max_steps) & jnp.any(carry[3])

        def wbody(st):
            i, carry, bufs = st
            carry2, rec = body(carry, keys[i])
            hh, perp2_c, sig_c = rec
            a_all = jnp.pi * perp2_c / jnp.clip(sig_c * MB_TO_FM2, 1e-12, None)
            m = perp2_c < 1e5
            hh_c, a_c, ns = bufs
            slot = jnp.where(m, ns, _K_SLAB_REC)
            hh_c = hh_c.at[slot, ar].set(hh, mode="drop")
            a_c = a_c.at[slot, ar].set(a_all, mode="drop")
            return i + 1, carry2, (hh_c, a_c, ns + m.astype(jnp.int32))

        _, carry, bufs = jax.lax.while_loop(wcond, wbody, (jnp.int32(0), init, bufs0))
        (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz, made_pi,
         best_pi, best_pi_pos, best_pi_fz, best_pi_ch) = carry
        hh_c, a_c, ns = bufs
    elif cfg.track_steps:
        def body_t(carry, sk):
            c2, rec = body(carry, sk)
            return c2, (rec, (c2[0], c2[1], c2[3]))                   # pos, p_N, alive AFTER the step
        (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz,
         made_pi, best_pi, best_pi_pos, best_pi_fz, best_pi_ch), (recs, tr) = jax.lax.scan(body_t, init, keys)
        traj = (tr[0], tr[1], tr[2])
        hh, perp2_c, sig_c = recs
        a_all = jnp.pi * perp2_c / jnp.clip(sig_c * MB_TO_FM2, 1e-12, None)
        m = perp2_c < 1e5
        slot = jnp.cumsum(m.astype(jnp.int32), axis=0) - 1
        slot = jnp.where(m, slot, _K_SLAB_REC)
        arn = jnp.broadcast_to(jnp.arange(n)[None, :], slot.shape)
        hh_c = jnp.zeros((_K_SLAB_REC, n), bool).at[slot, arn].set(hh, mode="drop")
        a_c = jnp.full((_K_SLAB_REC, n), 50.0).at[slot, arn].set(a_all, mode="drop")
        ns = jnp.sum(m.astype(jnp.int32), axis=0)
    else:
        (pos, p_N, dhat, alive, nsc, consumed, best_ko, best_ko_pos, best_ko_fz, fz,
         made_pi, best_pi, best_pi_pos, best_pi_fz, best_pi_ch), recs = jax.lax.scan(body, init, keys)
        # kind-1 sigma_scatter reweight (closest-in-slab Bernoulli approximation), OUTSIDE the scan.
        # Compress to the <=_K_SLAB_REC steps with an in-slab candidate (perp2_c dummy = 1e6 marks
        # "no slab"; its br is exactly 1) -- theta-independent walk records for nucleon_scat_reweight.
        hh, perp2_c, sig_c = recs                                  # (max_steps, n)
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
    # leading knockout = max-momentum top-K slot (bit-exact to the old single best_ko for the callers);
    # (best_ko, best_ko_pos, best_ko_fz) full top-K appended for the engine.
    _arn = jnp.arange(best_ko.shape[0])
    _lead = jnp.argmax(jnp.linalg.norm(best_ko[:, :, 1:], axis=2), axis=1)
    n_trunc = jnp.sum(alive.astype(jnp.int32))   # nucleons still propagating at the cap (no absorption -> still inside)
    return (p_N, nsc, best_ko[_arn, _lead], best_ko_pos[_arn, _lead], best_ko_fz[_arn, _lead],
            w_scat, srec, made_pi, (best_ko, best_ko_pos, best_ko_fz),
            (best_pi, best_pi_ch, best_pi_pos, best_pi_fz),   # leading CREATED pion (4-vec, charge idx, vertex, fz)
            n_trunc, nsc, traj)                               # diagnostics: #still-propagating @cap, scatter count, traj


def propagate_nucleon_discrete(pos0, p_N0, isp0, npos, nmom, nisp, cfg, key, fz0=None, consumed0=None, sscat=1.0):
    _load_density(cfg.nucleus, cfg.density_n)
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
        p_N, nsc, best_ko, best_ko_pos, best_ko_fz, w_sc1, srec1, made_pi1, best_ko_all, _, n_trunc1, nseg1, _ = propagate_nucleon_discrete(
            pos0, event.p_N, isp0, npos, nmom, nisp, self.cfg, kp, consumed0=consumed0, sscat=sscat)
        self.last_ko_all = best_ko_all            # all top-K knockout protons (n,K,4),(n,K,3),(n,K) -- engine
        self.last_n_trunc = n_trunc1              # nucleons still propagating at MAX_SEG (diagnostic; should be ~0)
        self.last_nseg = nseg1                    # per-event NN-scatter count (primary nucleon)
        # MULTI-NUCLEON cascade (ACHILLES UpdateKicked): the knocked-out proton itself re-cascades.
        # Re-propagate the leading knockout through the same background; its final state can be the
        # leading proton.  (One secondary generation -- the dominant multi-nucleon contribution.)
        # The knockout carries the formation zone ACHILLES assigned it at creation (best_ko_fz).
        kp2 = jax.random.fold_in(kp, 99)
        has_ko = jnp.linalg.norm(best_ko[:, 1:], axis=1) > 1.0
        ko_start = jnp.where(has_ko[:, None], best_ko, event.p_N)         # dummy where no knockout
        isp_ko = jnp.ones(n, bool)                                       # knockout proton
        ko_f, _, ko_ko, _, _, w_sc2, srec2, made_pi2, _, _, _, _, _ = propagate_nucleon_discrete(best_ko_pos, ko_start, isp_ko, npos, nmom,
                                                                        nisp, self.cfg, kp2, fz0=best_ko_fz,
                                                                        consumed0=consumed0, sscat=sscat)
        ko_f = jnp.where(has_ko[:, None], ko_f, jnp.zeros((n, 4)))
        ko_ko = jnp.where(has_ko[:, None], ko_ko, jnp.zeros((n, 4)))     # 2nd-gen knockout proton
        # leading proton = highest-momentum proton among {primary, re-cascaded knockout, 2nd-gen ko}
        cands = jnp.stack([p_N, ko_f, ko_ko], axis=1)                   # (n,3,4)
        mom = jnp.linalg.norm(cands[:, :, 1:], axis=2)                  # (n,3)
        lead = cands[jnp.arange(n), jnp.argmax(mom, axis=1)]
        # leading PROTON among the candidates (species threaded, not re-asserted downstream): the primary
        # nucleon is a proton iff isp0 (= input pid_N); the NN knockouts are protons by construction
        # (best_ko tracks only proton recoils, _propagate_nucleon_discrete bg_proton).  This is the only
        # way an n->n pi+ (neutron recoil) acquires a signal proton, exactly as in ACHILLES.
        prot_mask = jnp.stack([isp0, jnp.ones(n, bool), jnp.ones(n, bool)], axis=1)   # (n,3)
        prot_mom = mom * prot_mask
        has_prot = jnp.any(prot_mask & (mom > 0.0), axis=1)
        lead_prot = cands[jnp.arange(n), jnp.argmax(prot_mom, axis=1)]
        self.last_lead_prot = jnp.where(has_prot[:, None], lead_prot, jnp.zeros((n, 4)))
        # rich-bank exposures: separate post-cascade candidates so arbitrary signal defs are re-binnable
        self.last_primary = p_N                    # post-cascade primary nucleon (4-vec)
        self.last_primary_isp = isp0               # primary nucleon is a proton?
        ko_cands = jnp.stack([ko_f, ko_ko], axis=1)
        ko_mom = jnp.linalg.norm(ko_cands[:, :, 1:], axis=2)
        self.last_nuc_ko = ko_cands[jnp.arange(n), jnp.argmax(ko_mom, axis=1)]   # leading NN knockout proton (0 if none)
        self.last_w_scat = w_sc1 * jnp.where(has_ko, w_sc2, 1.0)        # kind-1 sigma_scatter reweight (1 at nominal)
        self.last_srec = (srec1, srec2, has_ko)   # compressed walk records for nucleon_scat_reweight
        self.last_made_pion = made_pi1 | (has_ko & made_pi2)   # NN->NDelta->NNpi created a pion
                                                  # (meson-veto relevant for CC0pi/CC1pi signals)
        return event._replace(p_N=lead, w=event.w * self.last_w_scat)
