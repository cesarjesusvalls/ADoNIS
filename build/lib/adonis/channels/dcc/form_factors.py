"""Differentiable AXIAL form factor for the DCC axial-block reweight -- ported from ACHILLES
src/Achilles/FormFactor.cc + FormFactors.yml.

The DCC amplitude table already bakes in a nominal Q^2 dependence, so the axial mass M_A enters the
cross section as a differentiable Q^2-dependent reweight of the axial block (see
`axial_reweight_dipole`), applied to the axial amplitude (so |A|^2 picks up the square) -- smooth
in the knob, giving exact gradients.  Q^2 here is GeV^2 (FormFactor.cc convention); the
cross-section code works in MeV^2 and converts.

NOTE: the Kelly VECTOR form factors + the axial z-expansion live in adonis/channels/currents/
form_factors.py -- the single home for those.  This module is only the axial-dipole reweight.
"""
from __future__ import annotations

MA_NOMINAL = 1.000
GAN1 = 1.2694
M_PI_GEV = 0.13957


def axial_dipole(Q2_GeV2, MA=MA_NOMINAL, gan1=GAN1):
    """Axial dipole form factor (FormFactor.cc::AxialDipole)."""
    return -gan1 / (1.0 + Q2_GeV2 / (MA * MA)) ** 2


def axial_reweight_dipole(Q2_MeV2, MA):
    """Dipole axial-mass reweight ratio at Q^2 [MeV^2], relative to M_A nominal.

    r(Q^2) = F_A(Q^2; MA) / F_A(Q^2; MA_nominal).  At MA=MA_nominal -> 1 exactly."""
    Q2 = Q2_MeV2 / 1.0e6
    return axial_dipole(Q2, MA) / axial_dipole(Q2, MA_NOMINAL)
