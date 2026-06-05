"""Differentiable nucleon form factors (Phase-2 step 6) -- ported from ACHILLES
src/Achilles/FormFactor.cc + FormFactors.yml.

The headline tunable physics parameter is the AXIAL MASS M_A.  The DCC amplitude
table already bakes in a nominal Q^2 dependence, so M_A (and the z-expansion
coefficients) enter the cross section as a differentiable Q^2-dependent REWEIGHT
of the axial block:

    axial_reweight(Q^2) = [ F_A(Q^2; theta) / F_A(Q^2; theta_nominal) ]

applied to the axial amplitude (so |A|^2 picks up the square).  This is exactly
the standard axial-mass reweighting used to tune neutrino generators to data, and
it is smooth in theta -> exact gradients (kind-1).

All Q^2 here are in GeV^2 (the FormFactor.cc convention); the cross-section code
works in MeV^2 and converts.
"""
from __future__ import annotations

import jax.numpy as jnp

# --- constants from FormFactors.yml ----------------------------------------- #
MA_NOMINAL = 1.000          # axial mass [GeV]
GAN1 = 1.2694               # g_A (axial coupling at Q^2=0)
GANS = 0.08                 # strange axial coupling
TCUT = 0.1753180641         # 9 m_pi^2 [GeV^2]
T0 = -0.28                  # [GeV^2]
CC_PARAMS = (-0.759, 2.30, -0.6, -3.8, 2.3, 2.16, -0.896, -1.58, 0.823)
M_PI_GEV = 0.13957
M_N_GEV = 0.938272


# --- axial form factors ----------------------------------------------------- #
def axial_dipole(Q2_GeV2, MA=MA_NOMINAL, gan1=GAN1):
    """F_A(Q^2) = -g_A / (1 + Q^2/M_A^2)^2   (FormFactor.cc::AxialDipole)."""
    return -gan1 / (1.0 + Q2_GeV2 / (MA * MA)) ** 2


def _z_of_Q2(Q2_GeV2, tcut=TCUT, t0=T0):
    s = jnp.sqrt(tcut + Q2_GeV2)
    r = jnp.sqrt(tcut - t0)
    return (s - r) / (s + r)


def axial_zexpansion(Q2_GeV2, params=CC_PARAMS, tcut=TCUT, t0=T0):
    """F_A(Q^2) = sum_k a_k z(Q^2)^k   (FormFactor.cc::AxialZExpansion)."""
    z = _z_of_Q2(Q2_GeV2, tcut, t0)
    p = jnp.asarray(params)
    # Horner in z
    k = jnp.arange(p.shape[0])
    return jnp.sum(p * z ** k)


def axial_zexpansion_vec(Q2_GeV2, params=CC_PARAMS, tcut=TCUT, t0=T0):
    """Vectorised z-expansion over an array of Q^2."""
    z = _z_of_Q2(Q2_GeV2, tcut, t0)
    p = jnp.asarray(params)
    k = jnp.arange(p.shape[0])
    return jnp.sum(p[None, :] * z[:, None] ** k[None, :], axis=1)


# --- the tunable reweight feeding the DCC axial block ----------------------- #
def axial_reweight_dipole(Q2_MeV2, MA):
    """Dipole axial-mass reweight ratio at Q^2 [MeV^2], relative to M_A nominal.

    r(Q^2) = F_A(Q^2; MA) / F_A(Q^2; MA_nominal).  At MA=MA_nominal -> 1 exactly."""
    Q2 = Q2_MeV2 / 1.0e6
    return axial_dipole(Q2, MA) / axial_dipole(Q2, MA_NOMINAL)


def axial_reweight_zexp(Q2_MeV2, params):
    """Z-expansion reweight ratio relative to the nominal CC params."""
    Q2 = Q2_MeV2 / 1.0e6
    num = axial_zexpansion(Q2, params)
    den = axial_zexpansion(Q2, CC_PARAMS)
    return num / den
