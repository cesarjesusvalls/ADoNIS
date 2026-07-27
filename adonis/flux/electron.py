"""Monochromatic electron beam for (e,e') -- a fixed-energy e- probe fired along +z, J_beam = 1
(no flux weight).  The incoming-particle description for the electron-scattering studies.
(Extracted 2026-07-26 from adonis/channels/ee_xsec.py.)
"""
import numpy as np

M_E = 0.51099895               # electron mass [MeV] (ACHILLES Particles.yml)
E_BEAM_JLAB = 2222.0           # JLab Murphy:2019wed point [MeV]


def electron_k(E_beam, n):
    """Incoming e- 4-momentum [E, 0, 0, kz] (kz = sqrt(E^2 - m_e^2)) along +z, replicated to (n,4).
    Monochromatic -> J_beam = 1."""
    kz = np.sqrt(E_beam ** 2 - M_E ** 2)
    return np.tile(np.array([E_beam, 0.0, 0.0, kz]), (n, 1))
