"""Exact JAX port of ACHILLES `OsetCrossSections.cc` -- the in-medium pion-nucleus QE and
absorption cross sections (Oset et al., Nucl. Phys. A468 (1987) 631; A484 (1988) 557).

These are the REAL cross sections the ACHILLES cascade uses for pion FSI (the Virtual
Resonances mode) -- absolute, in millibarn, no tuned knobs.  Every line maps 1:1 to the C++
(`src/Achilles/OsetCrossSections.cc`, `include/.../OsetCrossSections.hh`); all arithmetic is
JAX so the cross sections (and anything downstream) are differentiable in the pion/nucleon
kinematics and the Oset coefficients.

QECrossSection returns the 9 charge channels (pi+/0/- in -> pi+/0/- out) including charge
exchange; AbsCrossSection returns the pi-absorption (pi N N -> N N) cross section.  Inputs are
in MeV / fm^-3; outputs in mb.
"""
from __future__ import annotations

import jax.numpy as jnp

# --- constants (Achilles/Constants.hh) --------------------------------------- #
HBARC = 197.3269804          # MeV fm
M_N = (938.27208816 + 939.56542054) / 2.0   # 938.918754 MeV (Constant::mN)
M_DELTA = 1232.25            # MeV
M_PIP = 139.57018
M_PI0 = 134.9764

# --- Oset parametrisation constants (OsetCrossSections.hh) -------------------- #
CONST_FACTOR = 1.0 / 12.0 / jnp.pi          # fConstFactor
COUPLING_CONSTANT = 0.36 * 4.0 * jnp.pi     # fCouplingConstant
NORMAL_DENSITY = 0.17                        # fm^-3
NORM_FACTOR = 197.327 * 197.327 * 10.0       # fNormFactor (-> mb)
IM_B0 = 0.035

C_Q = jnp.array([-5.19, 15.35, 2.06])        # fCoefCQ  (QE self-energy)
C_A2 = jnp.array([1.06, -6.64, 22.66])       # fCoefCA2 (2N absorption)
C_A3 = jnp.array([-13.46, 46.17, -20.34])    # fCoefCA3 (3N absorption)
C_ALPHA = jnp.array([0.382, -1.322, 1.466])  # fCoefAlpha (QE density exponent)
C_BETA = jnp.array([-0.038, 0.204, 0.613])   # fCoefBeta  (abs density exponent)
C_SIGMA = jnp.array([-0.01334, 0.06889, 0.19753])  # fCoefSigma (s-wave QE)
C_B = jnp.array([-0.01866, 0.06602, 0.21972])      # fCoefB
C_D = jnp.array([-0.08229, 0.37062, -0.03130])     # fCoefD


def _quad(x, a):
    return a[0] * x * x + a[1] * x + a[2]


def _coupling(m_pi):
    return COUPLING_CONSTANT / m_pi / m_pi


def _self_energy_abs_NN(pion_KE, m_pi, density):
    x = pion_KE / m_pi
    beta = _quad(x, C_BETA)
    return _quad(x, C_A2) * (density / NORMAL_DENSITY) ** beta


def _self_energy_abs_NNN(pion_KE, m_pi, density):
    x = pion_KE / m_pi
    beta = _quad(x, C_BETA)
    raw = _quad(x, C_A3)
    # C++: if(absNNN<0) absNNN=0; else absNNN *= (rho/rho0)^(2 beta)   (<0 for T_k < ~50 MeV)
    return jnp.where(raw < 0.0, 0.0, raw * (density / NORMAL_DENSITY) ** (2.0 * beta))


def _self_energy_qe(pion_KE, m_pi, density):
    x = pion_KE / m_pi
    alpha = _quad(x, C_ALPHA)
    return _quad(x, C_Q) * (density / NORMAL_DENSITY) ** alpha


def _reduced_half_width(pionE, m_pi, cms_mom, fermimom, sqrts):
    fermi_energy = jnp.sqrt(fermimom ** 2 + M_N ** 2)
    coupling = _coupling(m_pi)
    deltaE = pionE + jnp.sqrt(0.6 * fermimom ** 2 + M_N ** 2)
    pion_mom = jnp.sqrt(pionE * pionE - m_pi * m_pi)
    deltamom = jnp.sqrt(pion_mom ** 2 + 0.6 * fermimom ** 2)
    n_energy_cms = jnp.sqrt(cms_mom ** 2 + M_N ** 2)
    mu0 = (deltaE * n_energy_cms - fermi_energy * sqrts) / deltamom / cms_mom
    # delta_reduction: 0 for mu0<-1, 1 for mu0>1, else (mu0^3+mu0+2)/4
    dr = (mu0 ** 3 + mu0 + 2.0) / 4.0
    delta_reduction = jnp.where(mu0 < -1.0, 0.0, jnp.where(mu0 > 1.0, 1.0, dr))
    return (CONST_FACTOR * coupling * M_N * cms_mom ** 3 / sqrts * delta_reduction)


def _pxsec_common(pionE, m_pi, cms_mom, fermimom, sqrts, density):
    pion_KE = pionE - m_pi
    rhw = _reduced_half_width(pionE, m_pi, cms_mom, fermimom, sqrts)
    re_delta = sqrts - M_DELTA
    im_delta = (rhw + _self_energy_abs_NN(pion_KE, m_pi, density)
                + _self_energy_abs_NNN(pion_KE, m_pi, density)
                + _self_energy_qe(pion_KE, m_pi, density))
    delta_prop2 = 1.0 / (re_delta ** 2 + im_delta ** 2)
    return NORM_FACTOR * _coupling(m_pi) * delta_prop2 * cms_mom ** 2 / pionE


def _kinematics(pionE, m_pi, pion_mom, fermimom):
    """Average-nucleon Mandelstam s and CM pion momentum (C++ lines 32-45)."""
    deltamomsq = pion_mom ** 2 + 0.6 * fermimom ** 2
    deltaE = pionE + jnp.sqrt(0.6 * fermimom ** 2 + M_N ** 2)
    s = deltaE * deltaE - deltamomsq
    sqrts = jnp.sqrt(s)
    wcm = (s - M_N ** 2 + m_pi ** 2) / (2.0 * sqrts)
    cms_mom = jnp.sqrt(jnp.clip(wcm * wcm - m_pi ** 2, 1e-12, None))
    return s, sqrts, cms_mom


def abs_cross_section(pionE, m_pi, pion_mom, vrel, fermimom, total_density):
    """Pion absorption cross section [mb] (pi N N -> N N), s- + p-wave (C++ AbsCrossSection)."""
    s, sqrts, cms_mom = _kinematics(pionE, m_pi, pion_mom, fermimom)
    pion_KE = pionE - m_pi
    pxsec_common = _pxsec_common(pionE, m_pi, cms_mom, fermimom, sqrts, total_density)
    self_abs = (_self_energy_abs_NN(pion_KE, m_pi, total_density)
                + _self_energy_abs_NNN(pion_KE, m_pi, total_density))
    p_abs = (4.0 / 9.0) * pxsec_common * self_abs / vrel
    s_fac = 4.0 * jnp.pi * HBARC * 10.0 * IM_B0
    s_abs = (s_fac / pionE * total_density * (1.0 + pionE / 2.0 / M_N)
             / (m_pi / HBARC) ** 4 / vrel)
    return p_abs + s_abs


# pion-in/pion-out channel keys (PID): 211 pi+, 111 pi0, -211 pi-
QE_CHANNELS = [(211, 211), (211, 111), (211, -211),
               (111, 211), (111, 111), (111, -211),
               (-211, 211), (-211, 111), (-211, -211)]


def qe_cross_sections(pionE, m_pi, pion_mom, fermimom, density, protfrac):
    """The 9 QE charge channels [mb] (incl. charge exchange), C++ QECrossSection.

    `protfrac = (N_n - N_p)/A` is the nuclear isospin asymmetry (0 for N=Z like 12C)."""
    s, sqrts, cms_mom = _kinematics(pionE, m_pi, pion_mom, fermimom)
    p_qel = (4.0 / 9.0) * _pxsec_common(pionE, m_pi, cms_mom, fermimom, sqrts, density) \
        * _reduced_half_width(pionE, m_pi, cms_mom, fermimom, sqrts)
    ksi = (sqrts - M_N - m_pi) / m_pi
    s_qel = _quad(ksi, C_SIGMA) / m_pi ** 2 * NORM_FACTOR
    B = _quad(ksi, C_B); D = _quad(ksi, C_D)
    A = 0.5 + 0.5 * D
    C = 1.0 - A
    pf = protfrac
    out = {
        (211, 211): s_qel * (A - pf * B) + p_qel * (5.0 - 4 * pf) / 6.0,
        (211, 111): s_qel * (1.0 + pf) * C + p_qel * (1 + pf) / 6.0,
        (211, -211): s_qel * 0.0,
        (111, 211): s_qel * (1.0 - pf) * C + p_qel * (1.0 - pf) / 6.0,
        (111, 111): s_qel * D + p_qel * 4.0 / 6.0,
        (111, -211): s_qel * (1.0 + pf) * C + p_qel * (1.0 + pf) / 6.0,
        (-211, 211): s_qel * 0.0,
        (-211, 111): s_qel * (1.0 - pf) * C + p_qel * (1.0 - pf) / 6.0,
        (-211, -211): s_qel * (A + pf * B) + p_qel * (5.0 + 4 * pf) / 6.0,
    }
    return out
