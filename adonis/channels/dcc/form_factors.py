"""Differentiable AXIAL form factor for the DCC axial-block reweight -- ported from ACHILLES
src/Achilles/FormFactor.cc + FormFactors.yml.

The DCC amplitude table already bakes in a nominal Q^2 dependence, so the axial mass M_A enters the
cross section as a differentiable Q^2-dependent REWEIGHT of the axial block:

    axial_reweight(Q^2) = F_A(Q^2; M_A) / F_A(Q^2; M_A_nominal)

applied to the axial amplitude (so |A|^2 picks up the square) -- the standard axial-mass reweight,
smooth in the knob -> exact kind-1 gradients.  Q^2 here is GeV^2 (FormFactor.cc convention); the
cross-section code works in MeV^2 and converts.

NOTE: the Kelly VECTOR form factors + the axial z-expansion live in adonis/channels/currents/
form_factors.py -- the single home for those.  This module is only the axial-dipole reweight the DCC
stack actually uses (`axial_reweight_dipole`, consumed by dcc/channel.py, dcc/structure.py,
reweight/amps2_records.py; `M_PI_GEV` by fsi/interactions/meson_baryon_amplitudes.py).
"""
from __future__ import annotations

MA_NOMINAL = 1.000
GAN1 = 1.2694
M_PI_GEV = 0.13957


def axial_dipole(Q2_GeV2, MA=MA_NOMINAL, gan1=GAN1):
    """F_A(Q^2) = -g_A / (1 + Q^2/M_A^2)^2   (FormFactor.cc::AxialDipole)."""
    return -gan1 / (1.0 + Q2_GeV2 / (MA * MA)) ** 2


def axial_reweight_dipole(Q2_MeV2, MA):
    """Dipole axial-mass reweight ratio at Q^2 [MeV^2], relative to M_A nominal.

    r(Q^2) = F_A(Q^2; MA) / F_A(Q^2; MA_nominal).  At MA=MA_nominal -> 1 exactly."""
    Q2 = Q2_MeV2 / 1.0e6
    return axial_dipole(Q2, MA) / axial_dipole(Q2, MA_NOMINAL)
