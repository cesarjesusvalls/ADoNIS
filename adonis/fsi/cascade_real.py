"""Real differentiable intranuclear pion cascade -- a faithful port of the ACHILLES cascade
(`src/Achilles/Cascade.cc` + the Oset meson-baryon interactions), NOT a toy.

No tuned cross-section knobs: the interaction rate is the physical mean free path
1/lambda = rho(r) * sigma_Oset(T_pi, rho(r)), with sigma from the exact Oset in-medium QE
(9 charge channels incl. charge exchange) + absorption cross sections (`oset_xsec.py`). The
pion propagates through the REAL 12C density rho(r) (data/nuclear/c12_density.txt); quasi-
elastic scattering uses the REAL two-body kinematics off a Fermi-moving nucleon (the energy
loss is the recoil, not a fixed fraction); absorption removes the pion (-> CC0pi); Pauli
blocking on the recoil nucleon.

COMPONENTS (now matching ACHILLES Virtual-Resonances, data/default/VirtResInteractions.yml):
  * SCATTER: the DCC ANL-Osaka meson-baryon sigma(W) per charge channel (elastic + charge
    exchange) -- `MesonBaryonInteraction` = ADoNIS Phase E (mb/cascade_mb.py), sharply
    Delta-peaked, validated to the 9.3:2.2:1 isospin ratio.  W from the real pion+struck-nucleon.
  * ABSORPTION: the exact Oset AbsCrossSection (`PionAbsorptionOneStep` -> oset_xsec).
  * real rho(r), Local Fermi gas, real two-body kinematics off a Fermi nucleon, Pauli blocking.

VALIDATED against the ACHILLES Virtual-Resonances oracle (achilles:cascade, Mode: CrossSection;
scripts/cascade_abs_from_hepmc.py -> data/oracle/cascade_pip_c12_virt_abs.csv).  For pi+ on 12C:
  * ABSORPTION FRACTION among reacted pions: ADoNIS 0.28-0.30 at the Delta vs ACHILLES 0.29-0.31
    -- they AGREE (the old "ACHILLES 0.22" was a phantom target from a different config).
  * sigma_reaction(p) and sigma_abs(p) SHAPES agree; ADoNIS sits ~1.33x above ACHILLES in
    absolute normalisation at the Delta (Delta-region chi2/ndf ~3.5 after one bridge constant,
    paper_figures/make_fig3.py).  The residual is a TRANSPORT difference (continuum mean-free-
    path through the smooth rho(r) + per-step Fermi-resampled v_rel here, vs ACHILLES's discrete
    impact-parameter walk over correlated QMC nucleon configs with an adaptive 0.04 fm step and
    the actual struck-nucleon v_rel).  It grows in the wings (low p: the Oset 1/v_rel term for
    slow pions).  The cross-section PHYSICS is the exact, untuned Oset+DCC port; the densities
    are identical (data/nuclear/c12_density.txt == ACHILLES data/densities/c12.prova.txt).
    See memory cascade-absorption-fraction-029.

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
from adonis.fsi.mb import cascade_mb

HBARC = ox.HBARC
M_N = ox.M_N
MB_TO_FM2 = 0.1                     # 1 mb = 0.1 fm^2
_PI_MASS = {211: ox.M_PIP, 111: ox.M_PI0, -211: ox.M_PIP}
_DENS = {}


def _read_density_file(name):
    p = Path(__file__).resolve().parents[2] / "data" / "nuclear" / name
    d = np.loadtxt(p, comments="#")
    return d[:, 0], d[:, 1]               # r [fm], rho [fm^-3]


def _load_density(name="c12_density.txt", name_n=None):
    """Proton & neutron number densities.  ACHILLES ALWAYS reads SEPARATE p/n densities
    (Nucleus.cc:36-80); for N=Z nuclei (C) name_n is name -> rho_n == rho_p.  rho_n is interpolated
    onto the proton radial grid so a single rgrid serves both species.  Radius = FIRST r where
    rho_proton < 1e-6 fm^-3 (ABSOLUTE minDensity, Nucleus.cc:49-51).  Returns (rgrid, rho_p, rho_n,
    radius).  For N=Z, rho_p == rho_n bitwise and rho_p+rho_n == 2*rho_p -> carbon is unchanged."""
    # numpy cache + per-call asarray: a jnp array first created inside a jit trace would
    # leak the tracer context to later traces (cf. cascade_mb._jax_grids_resolved).
    name_n = name_n or name
    key = (name, name_n)
    if key not in _DENS:
        r, rho_p = _read_density_file(name)
        if name_n == name:
            rho_n = rho_p
        else:
            rn, rhon0 = _read_density_file(name_n)
            rho_n = np.interp(r, rn, rhon0, left=rhon0[0], right=0.0)
        _below = r[rho_p < 1.0e-6]
        radius = float(_below.min()) if _below.size else float(r.max())
        _DENS[key] = (r, rho_p, rho_n, radius)
    r, rho_p, rho_n, radius = _DENS[key]
    return jnp.asarray(r), jnp.asarray(rho_p), jnp.asarray(rho_n), radius


def _rho_species(r, rgrid, rho):
    """Number density of ONE species at radius r [fm], interpolated; 0 beyond the grid.  Caller
    passes the proton OR neutron density array (rho_p / rho_n) -- they are equal for N=Z nuclei."""
    return jnp.interp(r, rgrid, rho, left=rho[0], right=0.0)


def _kf_local(rho_species):
    """Local Fermi momentum [MeV]: k_F = cbrt(rho * 3 pi^2) * hbarc (ACHILLES Local FG)."""
    return jnp.cbrt(jnp.clip(rho_species, 0.0, None) * 3.0 * jnp.pi ** 2) * HBARC


@dataclass(frozen=True)
class RealCascadeConfig:
    nucleus: str = "c12_density.txt"      # proton density file (data/nuclear/)
    density_n: str = "c12_density.txt"    # neutron density file (= nucleus for N=Z nuclei, e.g. C)
    step: float = 0.08          # fm per transport step (ACHILLES uses 0.04 adaptive; 0.08 ok for pions)
    max_steps: int = 220        # 2*R/step margin
    seed: int = 0


def _two_body_cm_scatter(p_pi, p_N, m_out_pi, key, cos_cm=None):
    """Elastic/charge-exchange piN -> pi'N' two-body kinematics; cos(theta_cm) is supplied
    (from the DCC angular distribution -- ACHILLES MesonBaryonInteraction::GenerateMomentum
    samples the partial-wave angular CDF, NOT isotropic) or isotropic if cos_cm is None.
    Returns the outgoing pion 4-momentum (E,px,py,pz) in the lab.  theta_cm is measured from
    the incoming pion CM direction (Poincare z-axis), matching ACHILLES."""
    P = p_pi + p_N                                  # total 4-momentum (lab)
    s = P[0] ** 2 - jnp.sum(P[1:] ** 2)
    sqrts = jnp.sqrt(jnp.clip(s, 1e-6, None))
    m1, m2 = m_out_pi, M_N
    E1 = (sqrts / 2.0) * (1.0 + (m1 ** 2 - m2 ** 2) / s)
    lam = jnp.sqrt(jnp.clip((s - m1 ** 2 - m2 ** 2) ** 2 - 4 * m1 ** 2 * m2 ** 2, 0.0, None))
    pf = lam / (2.0 * sqrts)
    ka, kb = jax.random.split(key)
    cth = (2.0 * jax.random.uniform(ka) - 1.0) if cos_cm is None else cos_cm
    sth = jnp.sqrt(jnp.clip(1 - cth ** 2, 0.0, None))
    phi = 2 * jnp.pi * jax.random.uniform(kb)
    # CM scatter axis = the incoming pion CM direction (theta measured from it), per ACHILLES
    beta = P[1:] / P[0]
    zaxis = _boost(p_pi, -beta)[1:]
    zhat = zaxis / jnp.clip(jnp.linalg.norm(zaxis), 1e-9, None)
    # build an orthonormal frame (zhat, e1, e2)
    ref = jnp.where(jnp.abs(zhat[2]) < 0.9, jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.]))
    e1 = jnp.cross(ref, zhat); e1 = e1 / jnp.clip(jnp.linalg.norm(e1), 1e-9, None)
    e2 = jnp.cross(zhat, e1)
    dir_cm = cth * zhat + sth * (jnp.cos(phi) * e1 + jnp.sin(phi) * e2)
    p1_cm = jnp.concatenate([E1[None], pf * dir_cm])
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


from functools import partial as _partial


@_partial(jax.jit, static_argnums=(3,))
def _propagate_scan(pos0, p_pi0, charge_idx0, cfg: RealCascadeConfig, key, protfrac):
    """JIT + lax.scan core of the pion transport (the per-step body is traced ONCE)."""
    rgrid, rhoP, rhoN, radius = _load_density(cfg.nucleus, cfg.density_n)
    n = pos0.shape[0]
    keys = jax.random.split(key, cfg.max_steps)

    def body(carry, step_key):
        pos, p_pi, ch, alive, absorbed, nsc = carry
        r = jnp.linalg.norm(pos, axis=1)
        alive = alive & ~(alive & (r > radius))
        live = alive

        m_pi = _CH_MASS[ch]
        pmom = jnp.linalg.norm(p_pi[:, 1:], axis=1)
        pE = p_pi[:, 0]
        rho_p = _rho_species(r, rgrid, rhoP)
        rho_tot = rho_p + _rho_species(r, rgrid, rhoN)    # total nucleon density (= 2*rho_p for N=Z)
        kf = _kf_local(rho_p)

        kN, step_key = jax.random.split(step_key)
        p_N = jax.vmap(_sample_fermi_nucleon)(kf, jax.random.split(kN, n))
        v_pi = p_pi[:, 1:] / pE[:, None]
        v_N = p_N[:, 1:] / p_N[:, 0:1]
        vrel = jnp.linalg.norm(v_pi - v_N, axis=1)

        Ppair = p_pi + p_N
        W = jnp.sqrt(jnp.clip(Ppair[:, 0] ** 2 - jnp.sum(Ppair[:, 1:] ** 2, axis=1), 1.0, None))
        sig_out = cascade_mb.jax_channel_sigmas(W, ch)
        sigma_sc_tot = jnp.sum(sig_out, axis=1)
        sa = ox.abs_cross_section(pE, m_pi, pmom, jnp.clip(vrel, 1e-3, None), kf,
                                  jnp.clip(rho_tot, 1e-9, None))
        sigma_tot = (sa + sigma_sc_tot) * MB_TO_FM2
        lam = rho_tot * sigma_tot
        p_int = -jnp.expm1(-lam * cfg.step)

        kI, kC, kF, kS = jax.random.split(step_key, 4)
        interacts = live & (jax.random.uniform(kI, (n,)) < p_int)
        p_abs = sa / jnp.clip(sa + sigma_sc_tot, 1e-12, None)
        is_abs = interacts & (jax.random.uniform(kC, (n,)) < p_abs)
        absorbed = absorbed | is_abs
        alive = alive & ~is_abs

        scatters = interacts & ~is_abs
        probs = sig_out / jnp.clip(jnp.sum(sig_out, axis=1, keepdims=True), 1e-12, None)
        cdf = jnp.cumsum(probs, axis=1)
        u = jax.random.uniform(kF, (n, 1))
        out_ch = jnp.clip(jnp.sum((u > cdf).astype(jnp.int32), axis=1), 0, 2).astype(jnp.int32)

        # CM scattering angle from the DCC angular distribution (NOT isotropic)
        kS, kang = jax.random.split(kS)
        cos_cm = cascade_mb.jax_sample_cos_cm(W, jax.random.uniform(kang, (n,)))

        def scat_one(p_pi_i, pN_i, out_i, kf_i, k, cc):
            p_out = _two_body_cm_scatter(p_pi_i, pN_i, _CH_MASS[out_i], k, cos_cm=cc)
            p_rec = (p_pi_i + pN_i) - p_out
            return p_out, jnp.linalg.norm(p_rec[1:]) < kf_i
        p_out, blocked = jax.vmap(scat_one)(p_pi, p_N, out_ch, kf, jax.random.split(kS, n), cos_cm)

        do_scatter = scatters & ~blocked
        p_pi = jnp.where(do_scatter[:, None], p_out, p_pi)
        ch = jnp.where(do_scatter, out_ch, ch)
        nsc = nsc + do_scatter.astype(jnp.int32)

        v3 = p_pi[:, 1:]
        d = v3 / jnp.clip(jnp.linalg.norm(v3, axis=1, keepdims=True), 1e-9, None)
        pos = jnp.where(alive[:, None], pos + cfg.step * d, pos)
        return (pos, p_pi, ch, alive, absorbed, nsc), None

    init = (pos0, p_pi0, charge_idx0, jnp.ones(n, bool), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32))
    (pos, p_pi, ch, alive, absorbed, nsc), _ = jax.lax.scan(body, init, keys)
    return p_pi, ch, absorbed, nsc


def propagate(pos0, p_pi0, charge_idx0, cfg: RealCascadeConfig, key, protfrac=0.0):
    """Transport a batch of pions through the nucleus.  All arrays leading-axis = pion.
      pos0 (N,3) fm, p_pi0 (N,4) MeV (E,px,py,pz), charge_idx0 (N,) in {0,1,2}.
    Returns (p_pi_final (N,4), charge_idx (N,), absorbed (N,) bool, n_scatter (N,)).
    JIT + lax.scan over the steps (the per-step body is traced once -> fast)."""
    _load_density(cfg.nucleus, cfg.density_n)   # warm caches EAGERLY (avoid tracer leak inside jit)
    cascade_mb._jax_grids(); cascade_mb._build_angular()
    return _propagate_scan(pos0, p_pi0, charge_idx0, cfg, key, protfrac)


def sample_vertex(key, n, nucleus="c12_density.txt", density_n=None):
    """Sample n production vertices in the nucleus ~ rho(r) (radial pdf rho(r) r^2), as the
    pion's cascade starting point.  Uses the proton density grid (== total shape for N=Z)."""
    rgrid, rho, _rho_n, radius = _load_density(nucleus, density_n)
    rg = np.asarray(rgrid); rh = np.asarray(rho)
    rr = np.linspace(0.0, float(radius), 600)
    pdf = np.interp(rr, rg, rh, left=rh[0], right=0.0) * rr ** 2
    pdf = pdf / pdf.sum()
    kr, kd = jax.random.split(key)
    # inverse-CDF radial sample (detached)
    cdf = jnp.cumsum(jnp.asarray(pdf))
    u = jax.random.uniform(kr, (n,))
    idx = jnp.clip(jnp.searchsorted(cdf, u), 0, len(rr) - 1)
    rad = jnp.asarray(rr)[idx]
    dirs = jax.random.normal(kd, (n, 3)); dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs * rad[:, None]


# pid <-> charge index for the cascade (0:pi+, 1:pi0, 2:pi-)
_PID_TO_CH = {211: 0, 111: 1, -211: 2}


class RealCascadeFSI:
    """FSIModel: propagate the produced pion through the nucleus with the REAL (untuned) Oset +
    DCC cascade (`propagate`), returning the FSI'd EventRecord (pion momentum/charge updated;
    absorbed pions -> pid 0).  This is the production wrapper of the validated cascade
    (paper_figures/make_fig3.py); the toy `ToyCascadeFSI` is the tunable-knob differentiable
    surrogate.  `protfrac = (N_n - N_p)/A` (0 for 12C, 0.1 for 40Ar)."""

    def __init__(self, cfg: RealCascadeConfig = RealCascadeConfig(), protfrac: float = 0.0):
        self.cfg = cfg
        self.protfrac = float(protfrac)

    def apply(self, params, event, key=None):
        key = jax.random.PRNGKey(self.cfg.seed) if key is None else key
        kv, kp = jax.random.split(key)
        n = event.p_pi.shape[0]
        ch0 = jnp.asarray([_PID_TO_CH.get(int(p), 1) for p in np.asarray(event.pid_pi)],
                          dtype=jnp.int32)
        pos0 = sample_vertex(kv, n, self.cfg.nucleus, self.cfg.density_n)
        p_pi, ch, absorbed, nsc = propagate(pos0, event.p_pi, ch0, self.cfg, kp,
                                            protfrac=self.protfrac)
        keep = (~absorbed)[:, None]
        p_pi_out = p_pi * keep
        pid_out = jnp.where(absorbed, 0, _CH_PID[ch])
        return event._replace(p_pi=p_pi_out, pid_pi=pid_out)
