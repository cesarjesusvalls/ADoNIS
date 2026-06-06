"""Oset pion self-energy in nuclear matter -- absorption + quasi-elastic (Phase F).

Faithful transcription of ACHILLES `OsetCrossSections.cc` / `.hh`: the imaginary parts of
the Delta self-energy that drive pion absorption (2N + 3N) and quasi-elastic scattering in
the medium, as quadratics in x = T_pi/m_pi times a density power.  These coefficients are
the paper's tunable FSI absorption knobs (C_Q, C_A2, C_A3 and the exponents alpha, beta).

    abs_NN (x, rho)  = q(x; C_A2) * (rho/rho0)^beta(x)
    abs_NNN(x, rho)  = max(q(x; C_A3), 0) * (rho/rho0)^{2 beta(x)}
    qe    (x, rho)  = q(x; C_Q) * (rho/rho0)^alpha(x)
with q(x;a) = a0 x^2 + a1 x + a2, beta(x)=q(x;C_beta), alpha(x)=q(x;C_alpha).

Everything is JAX and differentiable in the coefficient knobs -- the F-phase closure handle.
"""
from __future__ import annotations

import jax.numpy as jnp

# coefficients verbatim from include/Achilles/OsetCrossSections.hh
C_Q = (-5.19, 15.35, 2.06)
C_A2 = (1.06, -6.64, 22.66)
C_A3 = (-13.46, 46.17, -20.34)
C_ALPHA = (0.382, -1.322, 1.466)
C_BETA = (-0.038, 0.204, 0.613)
IM_B0 = 0.035
M_PI = 139.0          # MeV (pion mass scale for x = T_pi/m_pi)


def _quad(x, a):
    """a0 x^2 + a1 x + a2 (OsetCrossSections.hh::QuadraticFunction)."""
    return a[0] * x ** 2 + a[1] * x + a[2]


def self_energy_abs_NN(T_pi, rho_frac=1.0, c_a2=C_A2, c_beta=C_BETA, m_pi=M_PI):
    """2N absorption self-energy [MeV]. Differentiable in c_a2."""
    x = T_pi / m_pi
    beta = _quad(x, c_beta)
    return _quad(x, c_a2) * rho_frac ** beta


def self_energy_abs_NNN(T_pi, rho_frac=1.0, c_a3=C_A3, c_beta=C_BETA, m_pi=M_PI):
    """3N absorption self-energy [MeV]; clamped >=0 (may go negative for T_pi<50 MeV)."""
    x = T_pi / m_pi
    beta = _quad(x, c_beta)
    raw = _quad(x, c_a3)
    return jnp.where(raw < 0.0, 0.0, raw * rho_frac ** (2.0 * beta))


def self_energy_qe(T_pi, rho_frac=1.0, c_q=C_Q, c_alpha=C_ALPHA, m_pi=M_PI):
    """Quasi-elastic self-energy [MeV]. Differentiable in c_q."""
    x = T_pi / m_pi
    alpha = _quad(x, c_alpha)
    return _quad(x, c_q) * rho_frac ** alpha


def absorption_self_energy(T_pi, rho_frac=1.0, c_a2=C_A2, c_a3=C_A3, c_beta=C_BETA, m_pi=M_PI):
    """Total pion absorption self-energy Im Sigma_abs = abs_NN + abs_NNN [MeV]."""
    return (self_energy_abs_NN(T_pi, rho_frac, c_a2, c_beta, m_pi)
            + self_energy_abs_NNN(T_pi, rho_frac, c_a3, c_beta, m_pi))


T_PI_REF = 180.0          # MeV, near the Delta absorption peak (the shape normalisation point)


def absorption_rate_shape(T_pi, c_a2=C_A2, c_a3=C_A3, c_beta=C_BETA, m_pi=M_PI, t_ref=T_PI_REF):
    """Delta-peaked absorption SHAPE (>=0), normalised to 1 at `t_ref`, for the cascade's
    MOMENTUM-DEPENDENT sigma_abs: sigma_abs(T_pi) = fsi_sigma_abs * absorption_rate_shape(T_pi),
    so `fsi_sigma_abs` is the absorption rate at the Delta peak and the shape carries the
    Oset T_pi-dependence (low-T_pi s-wave + Delta resonance).  Differentiable in c_a2/c_a3."""
    s = jnp.clip(absorption_self_energy(T_pi, 1.0, c_a2, c_a3, c_beta, m_pi), 0.0, None)
    s_ref = absorption_self_energy(t_ref, 1.0, c_a2, c_a3, c_beta, m_pi)
    return s / (s_ref + 1e-30)
