"""Bit-exact JAX port of ACHILLES DefaultBackend::CrossSection (XSecBackend.cc:57-138):

  xsec = amps2 * FluxFactor * InitialStateWeight * SpinAvg * event.Weight()

This module assembles the FIRST FOUR factors (the per-kinematics matrix-element cross section,
i.e. everything except the phase-space Jacobian event.Weight(), which lives in phase_space.py).
amps2 = sum over the 4 lepton spin-combos and 4 hadron spin-combos of |L.H|^2, L.H the Minkowski
contraction.  FluxFactor = HBARC2/(2 E_lep * 2 sqrt(p_had^2+m^2)) * 1e6 (nb).  SpinAvg = 1/2 for a
neutrino QE (1 nu helicity x 2 nucleon spins).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.xsec import constants as C
from adonis.xsec.leptonic import lepton_current
from adonis.xsec.dirac import hadron_current_qe_dirac

_METRIC = jnp.array([1.0, -1.0, -1.0, -1.0])
# initial-nucleon mass used by FluxFactor = ParticleInfo(<had_in>).Mass() -- the rounded
# Particles.yml values (neutron 939.57, proton 938.27), NOT the precise Constant::mn/mp.
from adonis.constants import MASS_PDG_NEUTRON, MASS_PDG_PROTON  # noqa: E402
_MASS_BY_PID = {2112: MASS_PDG_NEUTRON, 2212: MASS_PDG_PROTON}


def _contract(L, H):
    """L (...,a,4), H (...,b,4) -> amps2 = sum_a sum_b |L_a . H_b|^2 (Minkowski dot)."""
    Lm = L * _METRIC                                              # lower the index on L
    LH = jnp.einsum('...am,...bm->...ab', Lm, H)                  # (...,a,b) complex
    return jnp.sum(jnp.abs(LH) ** 2, axis=(-2, -1))


def flux_factor(p_in_lep, p_in_nuc, had_mass=MASS_PDG_NEUTRON):
    """HBARC2 / (2 E_lep * 2 sqrt(p_had^2 + m_had^2)) * 1e6  [nb]  (XSecBackend.cc:35-46).
    m_had is ParticleInfo(had_in).Mass() (Particles.yml rounded: neutron 939.57, proton 938.27)."""
    Ehad = jnp.sqrt(jnp.sum(p_in_nuc[..., 1:] ** 2, axis=-1) + had_mass ** 2)
    flux = 2.0 * p_in_lep[..., 0] * 2.0 * Ehad
    return C.HBARC2 / flux * C.TO_NB


def me_cross_section(p_in_lep, p_out_lep, p_in_nuc, p_out_nuc, spin_avg=0.5, had_mass=MASS_PDG_NEUTRON,
                     axial_scale=1.0, vector_scale=1.0, ff_scale=None):
    """amps2 * FluxFactor * SpinAvg  [nb] -- the matrix-element cross section WITHOUT the
    InitialStateWeight (spectral function) and WITHOUT the phase-space Jacobian.  had_mass is the
    actual struck-nucleon mass (mn neutron / mp proton) used by FluxFactor.  axial_scale/vector_scale
    are the QE axial/vector reweight hooks (amps2 quadratic in each)."""
    L = lepton_current(p_in_lep, p_out_lep, kind="CC_nu", anti=False)
    H = hadron_current_qe_dirac(p_in_lep, p_out_lep, p_in_nuc, p_out_nuc,
                                axial_scale=axial_scale, vector_scale=vector_scale, ff_scale=ff_scale)
    amps2 = _contract(L, H)
    flux = flux_factor(p_in_lep, p_in_nuc, had_mass=had_mass)
    return dict(amps2=amps2, flux=flux, spin_avg=spin_avg,
                me_xsec=amps2 * flux * spin_avg)
