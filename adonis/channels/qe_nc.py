"""Neutral-current QUASI-ELASTIC scattering: nu N -> nu N, on a free nucleon.

The NC twin of qe.py, and deliberately NOT a `probe=` argument on it: `qe.sample_importance`
hardcodes `s2 = M_MU**2` in the two-body split and hardcodes the `n -> mu- p` channel, so "reuse it
with m_lep = 0" is a rewrite wearing a flag's clothing.

NC elastic differs from CC QE in more than the coupling:

  * BOTH nucleons participate.  CC QE is `nu n -> mu- p` only; NC has `nu p -> nu p` AND
    `nu n -> nu n`, and D3 flags the neutron path as having no CC analogue anywhere in this repo.
  * The final nucleon is the SAME species as the initial one, so there is no mass step.
  * The outgoing lepton is massless.

Verified against ACHILLES (`configs/achilles/run_freenucleon_nc_qe_{H,N}.yml`, 20 k events,
E_nu = 1.5 GeV, free nucleon, no cascade), proc IDs 250 (`nu n -> nu n`) and 251 (`nu p -> nu p`):

    sigma(1H) = 1.4362e-06 nb        sigma(1N) = 2.0569e-06 nb        n/p = 1.432

That n/p ratio is the sharpest available check on the isospin structure and it needs NO
normalisation convention to be right: the proton's NC vector coupling carries (1/2 - 2 sin^2 th_W)
~ 0.037 while the neutron's carries -1/2, so the proton's NC elastic is axial-dominated and the
neutron's is not.  An implementation with the couplings the wrong way round gets ~1/1.43 instead.

Two-body phase space, nucleon at rest, sampled isotropically in the CM:

    sigma = <me_xsec>_{Omega* uniform} * Phi_2 ,   Phi_2 = |p*| / (4 pi sqrt(s))

with me_xsec = amps2 * flux_factor * spin_avg from matrix_element.me_cross_section (i.e. everything
except the phase-space Jacobian -- see that module's docstring).
"""
from __future__ import annotations

import numpy as np

from adonis.channels import constants as C
from adonis.channels.currents.matrix_element import (me_cross_section, MASS_PDG_PROTON,
                                                     MASS_PDG_NEUTRON)

SPIN_AVG_NC = 0.5               # 1 neutrino helicity x 2 nucleon spins


def _two_body_cm(k_nu, m_N, u):
    """nu(k) + N(at rest) -> nu(k') + N(p'), isotropic in the CM.  Returns (k_lep, p_out, Phi2)."""
    n = len(k_nu)
    E = k_nu[:, 0]
    s = m_N ** 2 + 2.0 * E * m_N                       # (k + p)^2 with p at rest, massless nu
    sqs = np.sqrt(s)
    # massless lepton, elastic nucleon: |p*| is the same before and after
    pstar = (s - m_N ** 2) / (2.0 * sqs)
    ct = 2.0 * u[:, 0] - 1.0                           # isotropic in cos(theta*)
    st = np.sqrt(np.clip(1.0 - ct ** 2, 0.0, None))
    ph = 2.0 * np.pi * u[:, 1]
    kx = pstar * st * np.cos(ph); ky = pstar * st * np.sin(ph); kz = pstar * ct
    k_cm = np.column_stack([pstar, kx, ky, kz])
    p_cm = np.column_stack([np.sqrt(pstar ** 2 + m_N ** 2), -kx, -ky, -kz])
    # boost CM -> lab along +z (the nucleon is at rest in the lab, so beta = E/(E+m_N))
    beta = E / (E + m_N)
    g = 1.0 / np.sqrt(1.0 - beta ** 2)

    def boost(v):
        e = g * (v[:, 0] + beta * v[:, 3])
        z = g * (v[:, 3] + beta * v[:, 0])
        return np.column_stack([e, v[:, 1], v[:, 2], z])

    phi2 = pstar / (4.0 * np.pi * sqs)
    return boost(k_cm), boost(p_cm), phi2


def sigma_free_nucleon_nc_qe(Enu_MeV, is_proton, n=200_000, seed=0, quirk=False, chunk=100_000):
    """Free-nucleon NC elastic sigma(E_nu) [nb] + standard error, nucleon at rest.

    `quirk` selects the D1 convention: False = correct physics (the SM coupling), True = ACHILLES's
    coupl1 verbatim.  The ACHILLES comparison above must be made with quirk=True to be like-for-like;
    the difference between the two is the measured 1.0396 on both nucleons' F1/F2.
    """
    m_N = MASS_PDG_PROTON if is_proton else MASS_PDG_NEUTRON
    rng = np.random.default_rng(seed)
    E = float(Enu_MeV)
    k_nu = np.column_stack([np.full(n, E), np.zeros(n), np.zeros(n), np.full(n, E)])
    p_in = np.column_stack([np.full(n, m_N), np.zeros(n), np.zeros(n), np.zeros(n)])
    u = rng.random((n, 2))
    k_lep, p_out, phi2 = _two_body_cm(k_nu, m_N, u)
    isp = np.full(n, bool(is_proton))
    w = np.zeros(n)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        d = me_cross_section(k_nu[sl], k_lep[sl], p_in[sl], p_out[sl], spin_avg=SPIN_AVG_NC,
                             had_mass=m_N, probe="NC", is_proton=isp[sl], coupl1_quirk=quirk)
        w[sl] = np.asarray(d["me_xsec"]) * phi2[sl]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return float(w.mean()), float(w.std() / np.sqrt(n))
