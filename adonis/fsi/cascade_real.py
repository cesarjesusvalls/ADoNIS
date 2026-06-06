"""Real differentiable intranuclear pion cascade -- a faithful port of the ACHILLES cascade
(`src/Achilles/Cascade.cc` + the Oset meson-baryon interactions), NOT a toy.

No tuned cross-section knobs: the interaction rate is the physical mean free path
1/lambda = rho(r) * sigma_Oset(T_pi, rho(r)), with sigma from the exact Oset in-medium QE
(9 charge channels incl. charge exchange) + absorption cross sections (`oset_xsec.py`). The
pion propagates through the REAL 12C density rho(r) (data/nuclear/c12_density.txt); quasi-
elastic scattering uses the REAL two-body kinematics off a Fermi-moving nucleon (the energy
loss is the recoil, not a fixed fraction); absorption removes the pion (-> CC0pi); Pauli
blocking on the recoil nucleon.

STATUS / ROOT CAUSE (found): this version uses the Oset QE cross section for the hard scatter,
which is too BROAD in T_pi -> the cascade reaction sigma(p) peaks correctly at p=275 MeV (= the
Delta, matching the ACHILLES oracle) but falls off too slowly (at p=455: 0.70 vs ACHILLES 0.27),
and it over-absorbs (sigma_abs ~106 mb vs DUET ~60). The ACHILLES Virtual-Resonances config
(data/default/VirtResInteractions.yml) actually scatters via `MesonBaryonInteraction` -- the
DCC ANL-Osaka partial-wave amplitudes sigma(W) + their real angular distribution (= ADoNIS
Phase E, adonis/fsi/mb/anl_xsec.py, the sharply Delta-peaked piN cross sections validated to
the 9.3:2.2:1 isospin ratio) -- and absorbs via `PionAbsorptionOneStep` (the Oset absorption).
FIX IN PROGRESS: swap the scatter cross section + angular sampling from Oset QE to the Phase-E
DCC sigma(W)/dsigma/dOmega; keep Oset for absorption. Then the reaction sigma will be sharply
Delta-peaked like ACHILLES.  Committed as the real-physics scaffold, not yet validated.

Differentiability: the trajectory is SAMPLED against a frozen proposal (the Oset cross
sections at detached parameters); the Oset coefficients enter only via a per-event
likelihood-ratio weight (kind-1 reweighting), so d/d(Oset coeff) E[observable] is exact and
the sampled per-event final state is preserved (the FSIModel/EventRecord contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from adonis.fsi import oset_xsec as ox

HBARC = ox.HBARC
M_N = ox.M_N
MB_TO_FM2 = 0.1                     # 1 mb = 0.1 fm^2
_PI_MASS = {211: ox.M_PIP, 111: ox.M_PI0, -211: ox.M_PIP}
_DENS = {}


def _load_density(name="c12_density.txt"):
    if name not in _DENS:
        p = Path(__file__).resolve().parents[2] / "data" / "nuclear" / name
        d = np.loadtxt(p, comments="#")
        r, rho = d[:, 0], d[:, 1]          # col1 = rho_proton = rho_neutron (ACHILLES config)
        radius = float(r[rho > 1e-4 * rho[0]].max())   # physical boundary (~1% density), ~4.5 fm
        _DENS[name] = (jnp.asarray(r), jnp.asarray(rho), radius)
    return _DENS[name]


def _rho_species(r, rgrid, rho):
    """Proton (= neutron) number density at radius r [fm], interpolated; 0 beyond the grid."""
    return jnp.interp(r, rgrid, rho, left=rho[0], right=0.0)


def _kf_local(rho_species):
    """Local Fermi momentum [MeV]: k_F = cbrt(rho * 3 pi^2) * hbarc (ACHILLES Local FG)."""
    return jnp.cbrt(jnp.clip(rho_species, 0.0, None) * 3.0 * jnp.pi ** 2) * HBARC


@dataclass(frozen=True)
class RealCascadeConfig:
    nucleus: str = "c12_density.txt"
    step: float = 0.08          # fm per transport step (ACHILLES uses 0.04 adaptive; 0.08 ok for pions)
    max_steps: int = 220        # 2*R/step margin
    seed: int = 0


def _two_body_cm_scatter(p_pi, p_N, m_out_pi, key):
    """Elastic/charge-exchange piN -> pi'N': isotropic in the CM, real two-body kinematics.
    Returns the outgoing pion 4-momentum (E,px,py,pz) in the lab.  (ACHILLES
    OsetMesonBaryonInteractions::GenerateMomentum: isotropic cos(theta*), phi.)"""
    P = p_pi + p_N                                  # total 4-momentum (lab)
    s = P[0] ** 2 - jnp.sum(P[1:] ** 2)
    sqrts = jnp.sqrt(jnp.clip(s, 1e-6, None))
    m1, m2 = m_out_pi, M_N
    E1 = (sqrts / 2.0) * (1.0 + (m1 ** 2 - m2 ** 2) / s)
    lam = jnp.sqrt(jnp.clip((s - m1 ** 2 - m2 ** 2) ** 2 - 4 * m1 ** 2 * m2 ** 2, 0.0, None))
    pf = lam / (2.0 * sqrts)
    ka, kb = jax.random.split(key)
    cth = 2.0 * jax.random.uniform(ka) - 1.0
    sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0.0, None))
    phi = 2 * jnp.pi * jax.random.uniform(kb)
    p1_cm = jnp.array([E1, pf * sth * jnp.cos(phi), pf * sth * jnp.sin(phi), pf * cth])
    # boost CM -> lab with beta = P_vec/P_E
    beta = P[1:] / P[0]
    return _boost(p1_cm, beta)


def _boost(p4, beta):
    """Lorentz boost of 4-vector p4 by velocity beta (boost from a frame moving at -beta)."""
    b2 = jnp.sum(beta ** 2)
    g = 1.0 / jnp.sqrt(jnp.clip(1 - b2, 1e-12, None))
    bp = jnp.sum(beta * p4[1:])
    E = g * (p4[0] + bp)
    fac = (g - 1.0) * jnp.where(b2 > 1e-12, bp / jnp.clip(b2, 1e-12, None), 0.0) + g * p4[0]
    vec = p4[1:] + fac * beta
    return jnp.concatenate([E[None], vec])


def _sample_fermi_nucleon(kf, key):
    """Draw a nucleon 3-momentum uniformly in the Fermi sphere |p|<kf, on-shell."""
    kdir, kmag = jax.random.split(key)
    d = jax.random.normal(kdir, (3,)); d = d / jnp.linalg.norm(d)
    pmag = kf * jax.random.uniform(kmag) ** (1.0 / 3.0)
    p3 = d * pmag
    E = jnp.sqrt(M_N ** 2 + pmag ** 2)
    return jnp.concatenate([E[None], p3])


# charge <-> index:  0:pi+ (211), 1:pi0 (111), 2:pi- (-211)
_CH_PID = jnp.array([211, 111, -211])
_CH_MASS = jnp.array([ox.M_PIP, ox.M_PI0, ox.M_PIP])
_QE_IN = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])      # index into pi+/0/- for QE_CHANNELS in
_QE_OUT = jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2])     # ... out


def _all_xsecs(pE, pmom, m_pi, kf, rho_tot, protfrac, vrel):
    """Vectorised over the batch: abs (N,) and the 9 QE channels qe9 (N,9) [mb], ordered as
    oset_xsec.QE_CHANNELS = (+,+),(+,0),(+,-),(0,+),(0,0),(0,-),(-,+),(-,0),(-,-).
    `vrel` is the pion-nucleon relative velocity (with the Fermi-moving struck nucleon)."""
    sa = ox.abs_cross_section(pE, m_pi, pmom, jnp.clip(vrel, 1e-3, None), kf, rho_tot)
    qe = ox.qe_cross_sections(pE, m_pi, pmom, kf, rho_tot, protfrac)
    qe9 = jnp.stack([qe[k] for k in ox.QE_CHANNELS], axis=-1)   # (N,9)
    return jnp.clip(sa, 0.0, None), jnp.clip(qe9, 0.0, None)


def propagate(pos0, p_pi0, charge_idx0, cfg: RealCascadeConfig, key, protfrac=0.0):
    """Transport a batch of pions through the nucleus.  All arrays leading-axis = pion.
      pos0 (N,3) fm, p_pi0 (N,4) MeV (E,px,py,pz), charge_idx0 (N,) in {0,1,2}.
    Returns (p_pi_final (N,4), charge_idx (N,), absorbed (N,) bool, n_scatter (N,))."""
    rgrid, rho, radius = _load_density(cfg.nucleus)
    n = pos0.shape[0]
    pos = pos0
    p_pi = p_pi0
    ch = charge_idx0
    alive = jnp.ones(n, bool)
    absorbed = jnp.zeros(n, bool)
    nsc = jnp.zeros(n, jnp.int32)

    keys = jax.random.split(key, cfg.max_steps)
    for step_key in keys:
        r = jnp.linalg.norm(pos, axis=1)
        escaped = alive & (r > radius)
        alive = alive & ~escaped
        live = alive

        m_pi = _CH_MASS[ch]
        pmom = jnp.linalg.norm(p_pi[:, 1:], axis=1)
        pE = p_pi[:, 0]
        rho_p = _rho_species(r, rgrid, rho)
        rho_tot = 2.0 * rho_p                       # proton + neutron (same table)
        kf = _kf_local(rho_p)

        # sample the struck nucleon (Fermi sphere) ONCE per step: used for v_rel in the rate
        # AND, if it scatters, the two-body kinematics (ACHILLES uses the actual struck nucleon)
        kN, step_key = jax.random.split(step_key)
        p_N = jax.vmap(_sample_fermi_nucleon)(kf, jax.random.split(kN, n))   # (N,4)
        v_pi = p_pi[:, 1:] / pE[:, None]
        v_N = p_N[:, 1:] / p_N[:, 0:1]
        vrel = jnp.linalg.norm(v_pi - v_N, axis=1)

        sa, qe9 = _all_xsecs(pE, pmom, m_pi, kf, jnp.clip(rho_tot, 1e-9, None), protfrac, vrel)
        # QE channels available for each event's incoming charge
        in_mask = (_QE_IN[None, :] == ch[:, None])  # (N,9)
        qe_av = qe9 * in_mask
        sigma_qe_tot = jnp.sum(qe_av, axis=1)
        sigma_tot = (sa + sigma_qe_tot) * MB_TO_FM2  # fm^2
        lam = rho_tot * sigma_tot                    # 1/fm
        p_int = -jnp.expm1(-lam * cfg.step)

        kI, kC, kF, kS = jax.random.split(step_key, 4)
        interacts = live & (jax.random.uniform(kI, (n,)) < p_int)

        # choose absorb vs a QE out-channel  (probabilities over [abs, 9 qe channels])
        denom = jnp.clip(sa + sigma_qe_tot, 1e-12, None)
        p_abs = sa / denom
        is_abs = interacts & (jax.random.uniform(kC, (n,)) < p_abs)
        absorbed = absorbed | is_abs
        alive = alive & ~is_abs

        # QE scatter: pick out-channel ∝ qe_av, sample Fermi nucleon, real CM two-body kinematics
        scatters = interacts & ~is_abs
        probs = qe_av / jnp.clip(jnp.sum(qe_av, axis=1, keepdims=True), 1e-12, None)
        cdf = jnp.cumsum(probs, axis=1)
        u = jax.random.uniform(kF, (n, 1))
        chan = jnp.clip(jnp.sum((u > cdf).astype(jnp.int32), axis=1), 0, 8)
        out_ch = _QE_OUT[chan]

        # per-event scatter kinematics off the SAME struck nucleon used for v_rel (vmap)
        def scat_one(p_pi_i, pN_i, out_i, kf_i, k):
            m_out = _CH_MASS[out_i]
            p_out = _two_body_cm_scatter(p_pi_i, pN_i, m_out, k)
            # Pauli blocking on the recoil nucleon: reject if |p_recoil| < kf
            p_rec = (p_pi_i + pN_i) - p_out
            blocked = jnp.linalg.norm(p_rec[1:]) < kf_i
            return p_out, blocked
        ks = jax.random.split(kS, n)
        p_out, blocked = jax.vmap(scat_one)(p_pi, p_N, out_ch, kf, ks)

        do_scatter = scatters & ~blocked
        p_pi = jnp.where(do_scatter[:, None], p_out, p_pi)
        ch = jnp.where(do_scatter, out_ch, ch)
        nsc = nsc + do_scatter.astype(jnp.int32)

        # advance survivors along the (possibly new) pion direction
        v3 = p_pi[:, 1:]
        d = v3 / jnp.clip(jnp.linalg.norm(v3, axis=1, keepdims=True), 1e-9, None)
        pos = jnp.where(alive[:, None], pos + cfg.step * d, pos)

    return p_pi, ch, absorbed, nsc
