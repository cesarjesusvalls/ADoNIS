"""Absolute ACHILLES CC RES (single-pion) cross section by flat-MC, VECTORISED.

sigma_RES = sum over the 3 CC channels {n->n pi+, n->p pi0, p->p pi+} of E_u[ amps2 * flux *
initwgt * spinavg * event.Weight() ], using the ported mappers (Beam/QESpectral/ThreeBody) + the
batched exclusive DCC matrix element (dcc_current.exclusive_amps2_batch).  The mu+N+pi final state
is sampled with two isotropic 2-body splits (same integral as ACHILLES's t-channel sampler, higher
variance -> needs high N, now affordable via the vectorised current).  10 random dims: [0:4] struck
nucleon, [4] beam, [5] s23, [6:8] split-A angles, [8:10] split-B angles.  nu_mu + 12C, T2K flux.
ACHILLES target sigma_RES ~ 1.698e-5 nb.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.channels import constants as C
from adonis.flux.spectrum import SpectrumFlux
from adonis.nuclear.spectral import SpectralFunction
from adonis.channels.currents.matrix_element import flux_factor, MASS_PDG_NEUTRON, MASS_PDG_PROTON
from adonis.channels.dcc.current import exclusive_amps2_batch
from adonis.nuclear.spectral import SpectralImportanceSampler

_MN = C.mN
_SF_CACHE = {}


def _default_sf(which):
    key = f"pke12{which}_tot.data"
    if key not in _SF_CACHE:
        _SF_CACHE[key] = SpectralFunction(f"data/Spectral_Functions/{key}")
    return _SF_CACHE[key]


_IMP_CACHE = {}
def _imp_for(sf_n):
    k = id(sf_n)
    if k not in _IMP_CACHE:
        _IMP_CACHE[k] = SpectralImportanceSampler(sf_n)
    return _IMP_CACHE[k]
from adonis.constants import MASS_PDG_MUON as M_MU
_TWO_PI = 2 * np.pi
N_NUC = 6
M_PIP = C.mpip; M_PI0 = C.mpi0
M_P = MASS_PDG_PROTON; M_N = MASS_PDG_NEUTRON
SPIN_AVG = 0.5

from adonis.channels.dcc import conventions as _conv
MATCH_ACHILLES_PION_MASS = _conv.MATCH_ACHILLES
def _pi_kin_mass(physical_mpi):
    """Kinematic pion mass for the 3-body phase space (delegates to conventions.kin_m_pi)."""
    return _conv.kin_m_pi(physical_mpi)

CHANNELS = [
    (2112, -1, M_N, 211, M_PIP, MASS_PDG_NEUTRON),
    (2112, -1, M_P, 111, M_PI0, MASS_PDG_NEUTRON),
    (2212, +1, M_P, 211, M_PIP, MASS_PDG_PROTON),
]


from adonis.kinematics import kallen


def _sqlam(s, s1, s2):
    a = kallen(s, s1, s2)
    return np.where(a > 0, np.sqrt(np.clip(a, 0, None)) / s, 0.0)


def _boost_to_lab(p4cm, P):
    """Boost CM 4-vec (N,4) to lab where parent P (N,4) (Boost lflag=0)."""
    rsq = np.sqrt(np.clip(P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1), 1e-9, None))
    E = (P[:, 0] * p4cm[:, 0] + np.sum(P[:, 1:] * p4cm[:, 1:], axis=1)) / rsq
    c1 = (p4cm[:, 0] + E) / (rsq + P[:, 0])
    return np.concatenate([E[:, None], p4cm[:, 1:] + c1[:, None] * P[:, 1:]], axis=1)


def _sample_3body(k_nu, p_struck, m_pi, m_Nf, u):
    """Shared 3-body final state (mu + N + pi): two isotropic 2-body splits from the lab beam k_nu
    and struck nucleon p_struck.  u is (n,5) = [s23, ctA, phA, ctB, phB] in [0,1).  Returns the lab
    final momenta, the 3-body phase-space Jacobian J_3body, the invariants s/s23, and the 3-body
    validity mask.  ONE core -- used by both the 12C importance estimator (_sample_channel) and the
    free-proton wrapper (scripts/free_proton_gen)."""
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - m_pi) ** 2; s23min = max((M_MU + m_Nf) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 0]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    EmuN = (s + s23 - m_pi ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, s23, m_pi ** 2) / 2
    ctA = 2 * u[:, 1] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 2]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(m_pi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = _boost_to_lab(muN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, m_pi ** 2), 1e-12, None)
    Emu = (s23 + M_MU ** 2 - m_Nf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, M_MU ** 2, m_Nf ** 2) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(m_Nf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_lep = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, M_MU ** 2, m_Nf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J_3body = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid3 = ((s23max > s23min) & (_sqlam(s, s23, m_pi ** 2) > 0)
              & (_sqlam(s23, M_MU ** 2, m_Nf ** 2) > 0))
    return dict(k_lep=k_lep, p_N=p_N, p_pi=p_pi, J_3body=J_3body, s=s, s23=s23, valid3=valid3)


_TBM_ALPHA, _TBM_CTMAX, _TBM_CTMIN, _TBM_AMCT = 0.9, 1.0, -1.0, 1.0
SAMPLER_3BODY = "resonance"


def _m2(p):
    return p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)


def _boost_to_rest(ph, q):
    """ThreeBodyMapper::Boost lflag=1: lab vector ph -> rest frame of q."""
    rsq = np.sqrt(np.clip(_m2(q), 1e-12, None))
    dot = np.sum(q[:, 1:] * ph[:, 1:], axis=1)
    p0 = (q[:, 0] * ph[:, 0] - dot) / rsq
    c1 = (p0 + ph[:, 0]) / (rsq + q[:, 0])
    return np.concatenate([p0[:, None], ph[:, 1:] - c1[:, None] * q[:, 1:]], axis=1)


def _tj1(cn, amcxm, amcxp, ran):
    ce = 1.0 - cn
    return (ran * amcxm ** ce + (1.0 - ran) * amcxp ** ce) ** (1.0 / ce)


def _hj1(cn, amcxm, amcxp):
    ce = 1.0 - cn
    return (amcxp ** ce - amcxm ** ce) / ce


def _basis_from(nhat):
    """Orthonormal basis (e1,e2,n) with 3rd axis n = nhat (N,3)."""
    n = nhat / np.linalg.norm(nhat, axis=1, keepdims=True)
    ref = np.tile([1.0, 0.0, 0.0], (len(n), 1)); alt = np.tile([0.0, 1.0, 0.0], (len(n), 1))
    use_alt = np.abs(np.sum(n * ref, axis=1)) > 0.9
    ref = np.where(use_alt[:, None], alt, ref)
    e1 = ref - np.sum(ref * n, axis=1)[:, None] * n
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(n, e1), n


def _sample_3body_tchannel(k_nu, p_struck, m_pi, m_Nf, u):
    """ACHILLES ThreeBodyMapper proposal: TChannelMomenta (pion split, t-channel along nu) +
    Isotropic2Momenta (mu/N split).  Same signature/return as _sample_3body; only the pion-split
    proposal+weight differ (the mu/N split stays isotropic, identical to _sample_3body)."""
    s2, s3, s4 = M_MU ** 2, m_Nf ** 2, m_pi ** 2
    P = k_nu + p_struck
    s = _m2(P); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - m_pi) ** 2; s23min = max((M_MU + m_Nf) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 0]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    s1in = _m2(p_struck); s2in = _m2(k_nu)
    p1inhE = (s + s1in - s2in) / (2 * sqrts); p1inmass = sqrts * _sqlam(s, s1in, s2in) / 2
    p1outhE = (s + s23 - s4) / (2 * sqrts); p1outmass = sqrts * _sqlam(s, s23, s4) / 2
    a = (0.0 - s1in - s23 + 2 * p1outhE * p1inhE) / (2 * np.clip(p1inmass * p1outmass, 1e-30, None))
    a = np.where(a <= 1.0 + 1e-6, 1.0 + 1e-6, a)
    a = np.where(a < _TBM_AMCT, _TBM_AMCT, a)
    a = np.where(np.abs(a - _TBM_CTMAX) < 1e-14, _TBM_CTMAX, a)
    aminct = _tj1(_TBM_ALPHA, a - _TBM_CTMIN, a - _TBM_CTMAX, u[:, 1]); ct = a - aminct
    st = np.sqrt(np.clip(1 - ct ** 2, 0, None)); phi = _TWO_PI * u[:, 2]
    ref_cm = _boost_to_rest(p_struck, P)
    e1, e2, nhat = _basis_from(ref_cm[:, 1:])
    dirv = (st * np.cos(phi))[:, None] * e1 + (st * np.sin(phi))[:, None] * e2 + ct[:, None] * nhat
    p1out_cm = np.concatenate([p1outhE[:, None], p1outmass[:, None] * dirv], axis=1)
    p_muN = _boost_to_lab(p1out_cm, P); p_pi = P - p_muN
    tcw = 2.0 * sqrts / (-(a - ct) ** _TBM_ALPHA * _hj1(_TBM_ALPHA, a - _TBM_CTMIN, a - _TBM_CTMAX)
                         * np.clip(p1outmass, 1e-30, None) * np.pi)
    Emu = (s23 + s2 - s3) / (2 * rs23); pB = rs23 * _sqlam(s23, s2, s3) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(s3 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_lep = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, s2, s3), 1e-12, None)
    gw = (2 * np.pi) ** 5 * tcw * I2W_B / (s23max - s23min)
    J_3body = np.where(np.isfinite(gw) & (gw > 0), 1.0 / np.clip(gw, 1e-300, None), 0.0)
    valid3 = ((s23max > s23min) & (_sqlam(s, s23, s4) > 0) & (_sqlam(s23, s2, s3) > 0)
              & (p1outmass > 0) & np.isfinite(gw) & (gw > 0))
    return dict(k_lep=k_lep, p_N=p_N, p_pi=p_pi, J_3body=J_3body, s=s, s23=s23, valid3=valid3)


_BW_M0, _BW_GAMMA = 1232.0, 350.0


def _sample_3body_resonance(k_nu, p_struck, m_pi, m_Nf, u):
    """Lepton-first 3-body proposal with a Breit-Wigner-importance hadronic (N pi) mass -- the DEFAULT
    RES proposal.  Same signature/return as _sample_3body.  u is (n,5) = [m_H (BW), ctA, phA, ctB, phB].
    (q-relative angle / Q^2-axis variants were tested and gave <=+3% i.e. noise -- the resonance mass is
    the only sharp off-axis structure worth mapping; see docs and the invariant/correlation scans.)"""
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    mHmin = m_Nf + m_pi; mHmax = np.clip(sqrts - M_MU, mHmin + 1e-6, None)
    hg = _BW_GAMMA / 2.0
    a = (mHmin - _BW_M0) / hg; b = (mHmax - _BW_M0) / hg
    ata, atb = np.arctan(a), np.arctan(b)
    theta = ata + (atb - ata) * u[:, 0]
    m_H = _BW_M0 + hg * np.tan(theta)
    sH = m_H ** 2
    Z = (atb - ata) / hg
    pdf_m = 1.0 / np.clip(Z * ((m_H - _BW_M0) ** 2 + hg ** 2), 1e-300, None)
    pdf_sH = pdf_m / np.clip(2.0 * m_H, 1e-12, None)
    EHad = (s + sH - M_MU ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, sH, M_MU ** 2) / 2
    ctA = 2 * u[:, 1] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 2]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    Had_cm = np.concatenate([EHad[:, None], pA[:, None] * dA], axis=1)
    mu_cm = np.concatenate([np.sqrt(M_MU ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_Had = _boost_to_lab(Had_cm, P); k_lep = _boost_to_lab(mu_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, sH, M_MU ** 2), 1e-12, None)
    EN = (sH + m_Nf ** 2 - m_pi ** 2) / (2 * m_H); pB = m_H * _sqlam(sH, m_Nf ** 2, m_pi ** 2) / 2
    ctB = 2 * u[:, 3] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 4]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    N_cm = np.concatenate([EN[:, None], pB[:, None] * dB], axis=1)
    pi_cm = np.concatenate([np.sqrt(m_pi ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    p_N = _boost_to_lab(N_cm, p_Had); p_pi = _boost_to_lab(pi_cm, p_Had)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(sH, m_Nf ** 2, m_pi ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B * pdf_sH
    J_3body = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid3 = ((mHmax > mHmin) & (_sqlam(s, sH, M_MU ** 2) > 0) & (_sqlam(sH, m_Nf ** 2, m_pi ** 2) > 0)
              & np.isfinite(J_3body) & (J_3body > 0))
    muN = k_lep + p_N
    s23 = muN[:, 0] ** 2 - np.sum(muN[:, 1:] ** 2, axis=1)
    return dict(k_lep=k_lep, p_N=p_N, p_pi=p_pi, J_3body=J_3body, s=s, s23=s23, valid3=valid3)


def _sample_3body_dispatch(k_nu, p_struck, m_pi, m_Nf, u):
    if SAMPLER_3BODY == "resonance":
        return _sample_3body_resonance(k_nu, p_struck, m_pi, m_Nf, u)
    if SAMPLER_3BODY == "tchannel":
        return _sample_3body_tchannel(k_nu, p_struck, m_pi, m_Nf, u)
    return _sample_3body(k_nu, p_struck, m_pi, m_Nf, u)


def free_nucleon_weights(k_nu, itiz, m_Nf, pi_pid, m_pi_phys, had_mass, u, chunk=50_000):
    """Per-event FREE-nucleon single-pion RES weight w = amps2 * flux_factor * SPIN_AVG * J_3body
    (NO beam/flux J_beam factor), for a nucleon AT REST.  The one primitive behind BOTH the T2K-flux
    free-proton generator (free_proton.generate_H) and the monochromatic sigma(E_nu) scan (analysis Fig 2,
    anl_bnl) -- so the free-nucleon weight algebra lives in exactly one place.

    k_nu (n,4): neutrino 4-momentum per event.  (itiz, pi_pid) select the amps2 channel.  The struck nucleon
    is placed AT REST with mass `had_mass` (the INCOMING nucleon); the 3-body phase space is sampled toward the
    OUTGOING nucleon mass `m_Nf`.  These DIFFER for n -> p pi0 (struck neutron, outgoing proton) -- keep them
    two distinct params.  u (n,>=5): uniforms for the 3-body sampler.  Returns (w, dict(k_lep, p_N, p_pi, valid));
    w is already filtered finite & > 0 (but NOT divided by n and NOT multiplied by any flux Jacobian)."""
    n = len(k_nu)
    m_pi = _pi_kin_mass(m_pi_phys)
    p_struck = np.tile([had_mass, 0.0, 0.0, 0.0], (n, 1)).astype(float)
    tb = _sample_3body_dispatch(k_nu, p_struck, m_pi, m_Nf, u)
    k_lep, p_N, p_pi, J3, valid = tb["k_lep"], tb["p_N"], tb["p_pi"], tb["J_3body"], tb["valid3"]
    a2 = np.zeros(n)
    idx = np.where(valid & (J3 > 0))[0]
    for i in range(0, len(idx), chunk):
        sl = idx[i:i + chunk]
        a2[sl] = np.asarray(exclusive_amps2_batch(k_nu[sl], k_lep[sl], p_struck[sl], p_N[sl], p_pi[sl],
                                                  int(itiz), int(pi_pid)))
    fl = np.asarray(flux_factor(k_nu, p_struck, had_mass=had_mass))
    w = np.where(valid, a2 * fl * SPIN_AVG * J3, 0.0)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return w, dict(k_lep=k_lep, p_N=p_N, p_pi=p_pi, valid=valid)


def sigma_free_nucleon(Enu_MeV, channel, n=80_000, seed=0):
    """Monochromatic free-nucleon single-pion sigma(E_nu) [nb] + standard error, nucleon at rest (J_beam=1)
    -> sigma = <w>.  `channel` is a row of res.CHANNELS: (pdg_in, itiz, m_Nf, pi_pid, m_pi_phys, had_mass)."""
    _pdg_in, itiz, m_Nf, pi_pid, m_pi_phys, had_mass = channel
    rng = np.random.default_rng(seed)
    u = rng.random((n, 6))
    E = float(Enu_MeV); k_nu = np.stack([np.full(n, E), np.zeros(n), np.zeros(n), np.full(n, E)], axis=1)
    w, _ = free_nucleon_weights(k_nu, itiz, m_Nf, pi_pid, m_pi_phys, had_mass, u[:, 1:6])
    return float(w.mean()), float(w.std() / np.sqrt(n))


def _sample_channel(n, rng, flux, minE, maxE, m_pi, m_Nf, imp=None, grid=None, defensive=0.0):
    """One RES channel for the importance estimator: spectrum beam + importance struck nucleon
    (imp = the nucleus's |p|^2 S_n sampler; defaults to carbon) + the shared 3-body core.

    Optional frozen VegasGrid remaps the 6 hypercube dims [beam u[4] + 3-body u[5:10]] (the struck-
    nucleon |p|^2 S sampler stays OUTSIDE the grid); its Jacobian is folded into J and the mapped grid
    coords are returned as `x_grid` (for warm-up accumulation).  grid=None -> bit-identical sampling.

    defensive in (0,1): DEFENSIVE MIXTURE q = (1-defensive)*p_vegas + defensive*uniform.  A fraction
    `defensive` of events is drawn uniformly (covering the regions the grid under-samples); EVERY event
    is weighted by the mixture density 1/q -> the grid Jacobian is bounded by 1/defensive, capping the
    grid-induced weight tail.  Unbiased (same expectation), 0.0 -> pure Vegas."""
    imp = imp or _imp_for(_default_sf('n'))
    u = rng.random((n, 10))
    jac_grid = 1.0; x_grid = u[:, 4:10]
    if grid is not None and defensive > 0.0:
        a = 1.0 - defensive
        xv, _ = grid.map(u[:, 4:10])
        ub = rng.random(n) >= a
        x_grid = np.where(ub[:, None], u[:, 4:10], xv)
        q = a * grid.density(x_grid) + defensive
        jac_grid = 1.0 / np.clip(q, 1e-300, None)
        u = u.copy(); u[:, 4:10] = x_grid
    elif grid is not None:
        x_grid, jac_grid = grid.map(u[:, 4:10]); u = u.copy(); u[:, 4:10] = x_grid
    Smin = (M_MU + m_Nf + m_pi) ** 2
    minE = max((Smin - m_Nf ** 2) / (2 * m_Nf) / 1000.0, flux.min_energy)
    E_GeV, J_beam = flux.sample_beam(u[:, 4], minE); Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    pvec, energy = imp.sample(n, rng)
    mom = np.linalg.norm(pvec, axis=1)
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    J_had = np.ones(n)
    tb = _sample_3body_dispatch(k_nu, p_struck, m_pi, m_Nf, u[:, 5:10])
    det_e = Enu ** 2 + mom ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = _MN + Enu - np.sqrt(np.clip(det_e, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom), 400.0)
    valid = ((tb["s"] > Smin) & tb["valid3"] & (energy < emax))
    return dict(k_nu=k_nu, p_struck=p_struck, k_lep=tb["k_lep"], p_N=tb["p_N"], p_pi=tb["p_pi"],
                J=J_beam * J_had * tb["J_3body"] * jac_grid, mom=mom, energy=energy, Enu=Enu,
                E_GeV=E_GeV, valid=valid, x_grid=x_grid)


_M_SHARED_NF, _M_SHARED_PI = M_P, M_PI0
_SMIN0 = (M_MU + M_P + M_PI0) ** 2


def _sample_shared(n, rng, flux, maxE):
    """ONE shared phase-space point per draw, ACHILLES process[0]=(m_p,m_pi0) masses & Smin0,
    struck nucleon sampled FLAT (QESpectralMapper) with explicit J_had (no importance)."""
    u = rng.random((n, 10))
    mpi, mNf, Smin = _M_SHARED_PI, _M_SHARED_NF, _SMIN0
    minE = max((Smin - mNf ** 2) / (2 * mNf) / 1000.0, flux.min_energy)
    E_GeV, J_beam = flux.sample_beam(u[:, 4], minE); Enu = E_GeV * 1000.0
    k_nu = np.stack([Enu, np.zeros(n), np.zeros(n), Enu], axis=1)
    radical = np.clip(Enu ** 2 + 2 * Enu * _MN + _MN ** 2 - Smin, 0, None)
    pmin = np.clip(Enu - np.sqrt(radical), 0, None); pmax = np.clip(Enu + np.sqrt(radical), None, 800.0)
    dp = pmax - pmin; mom = dp * u[:, 0] + pmin
    cosTm = np.clip((2 * Enu * _MN + _MN ** 2 - mom ** 2 - Smin) / (2 * Enu * np.clip(mom, 1e-9, None)), -1, 1)
    cosT = (cosTm + 1) * u[:, 1] - 1; sinT = np.sqrt(np.clip(1 - cosT ** 2, 0, None)); phi = _TWO_PI * u[:, 2]
    pvec = np.stack([mom * sinT * np.cos(phi), mom * sinT * np.sin(phi), mom * cosT], axis=1)
    det = Enu ** 2 + mom ** 2 + 2 * pvec[:, 2] * Enu + Smin
    emax = _MN + Enu - np.sqrt(np.clip(det, 0, None))
    emax = np.minimum(np.minimum(emax, _MN - mom), 400.0)
    energy = emax * u[:, 3] - 1e-8
    p_struck = np.concatenate([(_MN - energy)[:, None], pvec], axis=1)
    J_had = mom ** 2 * dp * (cosTm + 1) * _TWO_PI * emax
    P = k_nu + p_struck
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1); sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s23max = (sqrts - mpi) ** 2; s23min = max((M_MU + mNf) ** 2, 1e-8)
    s23 = s23min + (s23max - s23min) * u[:, 5]; rs23 = np.sqrt(np.clip(s23, 1e-9, None))
    EmuN = (s + s23 - mpi ** 2) / (2 * sqrts); pA = sqrts * _sqlam(s, s23, mpi ** 2) / 2
    ctA = 2 * u[:, 6] - 1; stA = np.sqrt(np.clip(1 - ctA ** 2, 0, None)); phA = _TWO_PI * u[:, 7]
    dA = np.stack([stA * np.cos(phA), stA * np.sin(phA), ctA], axis=1)
    muN_cm = np.concatenate([EmuN[:, None], pA[:, None] * dA], axis=1)
    pi_cm = np.concatenate([np.sqrt(mpi ** 2 + pA ** 2)[:, None], -pA[:, None] * dA], axis=1)
    p_muN = _boost_to_lab(muN_cm, P); p_pi = _boost_to_lab(pi_cm, P)
    I2W_A = 2.0 / np.pi / np.clip(_sqlam(s, s23, mpi ** 2), 1e-12, None)
    Emu = (s23 + M_MU ** 2 - mNf ** 2) / (2 * rs23); pB = rs23 * _sqlam(s23, M_MU ** 2, mNf ** 2) / 2
    ctB = 2 * u[:, 8] - 1; stB = np.sqrt(np.clip(1 - ctB ** 2, 0, None)); phB = _TWO_PI * u[:, 9]
    dB = np.stack([stB * np.cos(phB), stB * np.sin(phB), ctB], axis=1)
    mu_cm = np.concatenate([Emu[:, None], pB[:, None] * dB], axis=1)
    N_cm = np.concatenate([np.sqrt(mNf ** 2 + pB ** 2)[:, None], -pB[:, None] * dB], axis=1)
    k_lep = _boost_to_lab(mu_cm, p_muN); p_N = _boost_to_lab(N_cm, p_muN)
    I2W_B = 2.0 / np.pi / np.clip(_sqlam(s23, M_MU ** 2, mNf ** 2), 1e-12, None)
    density = (2 * np.pi) ** 5 * I2W_A * I2W_B / (s23max - s23min)
    J_3body = np.where(density > 0, 1.0 / np.clip(density, 1e-300, None), 0.0)
    valid = ((dp > 0) & (emax > 0) & (s > Smin) & (s23max > s23min) & (energy < emax)
             & (energy > imp.energy[0])
             & (_sqlam(s, s23, mpi ** 2) > 0) & (_sqlam(s23, M_MU ** 2, mNf ** 2) > 0))
    return dict(k_nu=k_nu, p_struck=p_struck, k_lep=k_lep, p_N=p_N, p_pi=p_pi,
                J=J_beam * J_had * J_3body, mom=mom, energy=energy, valid=valid)


def generate_faithful(n=20000, seed=0, return_events=False, sf_n=None, sf_p=None,
                      n_neutron=N_NUC, n_proton=N_NUC):
    """ACHILLES-faithful: ONE shared (process[0]) point per draw; sum the 3 channels' amps2 with
    EXPLICIT per-channel initwgt = N*S_channel and per-channel flux, on the shared momenta.
    sf_n/sf_p = the nucleus's neutron/proton SpectralFunction (default = carbon);
    n_neutron/n_proton = target species counts (A-Z / Z) for the N*S scaling."""
    sf_n = sf_n or _default_sf('n'); sf_p = sf_p or _default_sf('p')
    group = [(2112, -1, 111, sf_n, n_neutron, MASS_PDG_NEUTRON),
             (2112, -1, 211, sf_n, n_neutron, MASS_PDG_NEUTRON),
             (2212, +1, 211, sf_p, n_proton, MASS_PDG_PROTON)]
    rng = np.random.default_rng(seed)
    flux = SpectrumFlux(); maxE = flux.max_energy
    s = _sample_shared(n, rng, flux, maxE)
    v = s["valid"]; idx = np.where(v & (s["J"] > 0))[0]
    out = {}; w_tot = np.zeros(n)
    for (ipid, itiz, ppid, sf, ncount, hadmass) in group:
        a2 = np.zeros(n)
        if len(idx):
            a2[idx] = exclusive_amps2_batch(s["k_nu"][idx], s["k_lep"][idx], s["p_struck"][idx],
                                            s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
        initwgt = ncount * sf.batch(s["mom"], s["energy"])
        fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=hadmass))
        w_c = np.where(v, a2 * fl * initwgt * SPIN_AVG * s["J"], 0.0)
        w_c = np.where(np.isfinite(w_c) & (a2 > 0), w_c, 0.0)
        out[(ipid, ppid)] = w_c.mean(); w_tot += w_c
    out["sigma"] = w_tot.mean()
    if return_events:
        keep = w_tot > 0
        out["events"] = dict(k_nu=s["k_nu"][keep], k_lep=s["k_lep"][keep], p_struck=s["p_struck"][keep],
                             p_N=s["p_N"][keep], p_pi=s["p_pi"][keep], w=w_tot[keep] / n)
    return out


RES_METHOD = "importance"


def _channel_weight(s, ipid, itiz, ppid, mstr, sf_n, sf_p, n_neutron, n_proton):
    """Per-event RES weight for one channel from a sampled-channel dict `s` (importance estimator):
    w = a2 * flux * (N species) * SPIN_AVG * J * (S_p/S_n reweight on the proton channel).  Shared by
    generate_importance and warmup_vegas (single source of truth for the weight assembly)."""
    n = len(s["valid"]); v = s["valid"]
    iw = n_neutron if ipid == 2112 else n_proton
    a2 = np.zeros(n)
    idx = np.where(v & (s["energy"] > sf_n.energy[0]) & (s["energy"] < 400) & (s["J"] > 0))[0]
    if len(idx):
        a2[idx] = exclusive_amps2_batch(s["k_nu"][idx], s["k_lep"][idx], s["p_struck"][idx],
                                        s["p_N"][idx], s["p_pi"][idx], itiz, ppid)
    fl = np.asarray(flux_factor(s["k_nu"], s["p_struck"], had_mass=mstr))
    if ipid == 2212:
        sn = sf_n.batch(s["mom"], s["energy"]); sp = sf_p.batch(s["mom"], s["energy"])
        reweight = np.where(sn > 0, sp / np.clip(sn, 1e-300, None), 0.0)
    else:
        reweight = 1.0
    w = np.where(v, a2 * fl * iw * SPIN_AVG * s["J"] * reweight, 0.0)
    return np.where(np.isfinite(w) & (a2 > 0), w, 0.0)


def _warmup_pbar(total, desc):
    """tqdm progress bar if available, else a minimal stderr fallback with elapsed/ETA (same .update/
    .close interface).  The warm-up (integration) phase has a heavy first step (JAX amps2 compile), so
    the ETA settles after a couple of channels."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)
    except Exception:
        import sys, time
        class _Bar:
            def __init__(s): s.t0 = time.time(); s.n = 0
            def update(s, k=1):
                s.n += k; el = time.time() - s.t0; rate = s.n / el if el > 0 else 0
                eta = (total - s.n) / rate if rate > 0 else float("inf")
                sys.stderr.write(f"\r{desc}: {s.n}/{total}  elapsed {el:5.0f}s  ETA {eta:5.0f}s   ")
                sys.stderr.flush()
            def close(s): sys.stderr.write("\n"); sys.stderr.flush()
        return _Bar()


def warmup_vegas(n=100000, iters=6, nbins=50, alpha=1.5, seed=987654321, sf_n=None, sf_p=None,
                 n_neutron=N_NUC, n_proton=N_NUC, progress=True):
    """Build + FREEZE a 6-dim VegasGrid for the RES importance estimator by warming up on the summed
    3-channel weight at nominal knobs.  Grid dims = [beam, hadronic-mass(BW), ctA, phA, ctB, phB].
    One shared grid across the 3 (kinematically similar) channels.  Returns the frozen VegasGrid.
    `progress` shows a tqdm-style bar (iters x channels steps) with ETA for the integration phase."""
    from adonis.vegas_grid import VegasGrid
    sf_n = sf_n or _default_sf('n'); sf_p = sf_p or _default_sf('p'); imp = _imp_for(sf_n)
    flux = SpectrumFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy
    grid = VegasGrid(6, nbins)
    pbar = _warmup_pbar(iters * len(CHANNELS), "vegas warm-up") if progress else None
    for it in range(iters):
        rng = np.random.default_rng(seed + it)
        xs, ws = [], []
        for (ipid, itiz, mNf, ppid, mpi, mstr) in CHANNELS:
            s = _sample_channel(n, rng, flux, minE, maxE, _pi_kin_mass(mpi), mNf, imp=imp, grid=grid)
            w = _channel_weight(s, ipid, itiz, ppid, mstr, sf_n, sf_p, n_neutron, n_proton)
            xs.append(s["x_grid"]); ws.append(w)
            if pbar is not None:
                pbar.update(1)
        grid.accumulate(np.concatenate(xs), np.concatenate(ws)); grid.refine(alpha=alpha)
    if pbar is not None:
        pbar.close()
    return grid.freeze()


def generate(n=20000, seed=0, return_events=False, method=None, sf_n=None, sf_p=None,
             n_neutron=N_NUC, n_proton=N_NUC, grid=None, defensive=0.0):
    """Dispatch to the faithful (transliteration) or importance RES estimator.  Both estimate the
    same sigma; faithful mirrors ACHILLES operation-for-operation, importance is lower variance.
    sf_n/sf_p = the nucleus's neutron/proton SpectralFunction (default = carbon).
    n_neutron/n_proton = # of target neutrons (A-Z) / protons (Z) for the initwgt = N*S scaling
    (default 6 = carbon; n-initiated channels use n_neutron, the p-initiated channel n_proton).
    grid = optional frozen VegasGrid (importance method only) over the 6 final-state hypercube dims."""
    m = method or RES_METHOD
    if m == "importance":
        return generate_importance(n, seed=seed, return_events=return_events, sf_n=sf_n, sf_p=sf_p,
                                   n_neutron=n_neutron, n_proton=n_proton, grid=grid, defensive=defensive)
    return generate_faithful(n, seed=seed, return_events=return_events, sf_n=sf_n, sf_p=sf_p,
                             n_neutron=n_neutron, n_proton=n_proton)


def generate_importance(n=20000, seed=0, return_events=False, sf_n=None, sf_p=None,
                        n_neutron=N_NUC, n_proton=N_NUC, grid=None, defensive=0.0):
    sf_n = sf_n or _default_sf('n'); sf_p = sf_p or _default_sf('p'); imp = _imp_for(sf_n)
    rng = np.random.default_rng(seed)
    flux = SpectrumFlux(); minE = flux.seed_min_GeV(); maxE = flux.max_energy
    out = {}; sig = 0.0
    ev = {k: [] for k in ("k_nu", "k_lep", "p_struck", "p_N", "p_pi", "w", "ppid", "Npid", "ipid")}
    nch = len(CHANNELS); m = max(1, n // nch)
    for (ipid, itiz, mNf, ppid, mpi, mstr) in CHANNELS:
        Npid = 2212 if mNf == M_P else 2112
        s = _sample_channel(m, rng, flux, minE, maxE, _pi_kin_mass(mpi), mNf, imp=imp, grid=grid,
                            defensive=defensive)
        w = _channel_weight(s, ipid, itiz, ppid, mstr, sf_n, sf_p, n_neutron, n_proton)
        sc = w.mean(); out[(ipid, ppid)] = sc; sig += sc
        if return_events:
            keep = w > 0
            ev["k_nu"].append(s["k_nu"][keep]); ev["k_lep"].append(s["k_lep"][keep])
            ev["p_struck"].append(s["p_struck"][keep]); ev["p_N"].append(s["p_N"][keep])
            ev["p_pi"].append(s["p_pi"][keep])
            ev["w"].append(w[keep] / m)
            ev["ppid"].append(np.full(keep.sum(), ppid)); ev["Npid"].append(np.full(keep.sum(), Npid))
            ev["ipid"].append(np.full(keep.sum(), ipid))
    out["sigma"] = sig
    if return_events:
        out["events"] = {k: np.concatenate(ev[k]) if ev[k] else np.empty((0,)) for k in ev}
    return out


from adonis.core.sample import Sampler


class RESChannel(Sampler):
    """CC single-pion RES production (ANL-Osaka DCC) as a detached kind-1 SAMPLER.  `propose(seed, n)`
    draws the frozen proposal (beam + struck nucleon + 3-body mu/pi/N) with the NOMINAL weight; the
    knob-differentiable WEIGHT is the production reweight (adonis.reweight.bank_reweight).  Kernel =
    the verbatim `generate(..., return_events=True)` -- bit-for-bit."""

    def __init__(self, sf_n=None, sf_p=None, n_neutron=6, n_proton=6, grid=None):
        self.sf_n = sf_n; self.sf_p = sf_p
        self.n_neutron = n_neutron; self.n_proton = n_proton; self.grid = grid

    def propose(self, key, n):
        return generate(n, seed=int(key), return_events=True, sf_n=self.sf_n, sf_p=self.sf_p,
                        n_neutron=self.n_neutron, n_proton=self.n_proton, grid=self.grid)["events"]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    import numpy as _np
    sigs = []
    for sd in range(nseed):
        r = generate(n, seed=sd); sigs.append(r["sigma"])
        if sd == 0:
            for k, v in r.items():
                if k != "sigma":
                    print(f"  init {k[0]} -> pi {k[1]}: {v:.4e} nb")
    sigs = _np.array(sigs)
    print(f"sigma_RES = {sigs.mean():.4e} nb  (N={n} x{nseed} seeds, sem {sigs.std()/max(_np.sqrt(nseed),1):.2e})"
          f"  ACHILLES ~1.698e-5  ratio={sigs.mean()/1.698e-5:.3f}")
