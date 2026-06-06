"""Observables computed from an EventRecord -- pure functions (EventRecord) -> (N,).

Covers the leptonic and hadronic single-pion variables and single-transverse kinematic
(TKI) imbalances.  Each is a plain function of the lab 4-momenta so any variable, for any
channel or aggregated set of channels, can be histogrammed identically to the oracle.

The neutrino defines +z in the lab (the EventRecord is built that way before the optional
uniform azimuthal rotation, which leaves all of these invariant except the per-particle
lab azimuth).
"""
from __future__ import annotations

import jax.numpy as jnp

from adonis.primary.dcc.lepton import boost_to_rest, _mink_dot


def _mag(v3):
    return jnp.sqrt(jnp.sum(v3 ** 2, axis=-1))


def _costheta(a3, b3, eps=1e-12):
    return jnp.sum(a3 * b3, axis=-1) / (_mag(a3) * _mag(b3) + eps)


# ---- leptonic ------------------------------------------------------------- #
def enu(ev):
    return ev.k[:, 0]


def q4(ev):
    return ev.k - ev.kp


def Q2(ev):
    """Observable leptonic Q^2 = -(k-k')^2 [MeV^2]."""
    return -_mink_dot(q4(ev), q4(ev))


def W(ev):
    """Hadronic invariant mass from the final pion+nucleon [MeV] (= the decay W)."""
    P = ev.p_pi + ev.p_N
    return jnp.sqrt(jnp.clip(_mink_dot(P, P), 0.0, None))


def lepton_energy(ev):
    return ev.kp[:, 0]


def lepton_costheta(ev):
    """cos of the outgoing-lepton polar angle wrt the neutrino (lab)."""
    return _costheta(ev.kp[:, 1:], ev.k[:, 1:])


# ---- hadronic ------------------------------------------------------------- #
def ppi_mag(ev):
    return _mag(ev.p_pi[:, 1:])


def ppi_costheta_lab(ev):
    """cos angle of the pion wrt q in the lab."""
    return _costheta(ev.p_pi[:, 1:], q4(ev)[:, 1:])


def nucleon_mom(ev):
    return _mag(ev.p_N[:, 1:])


def cos_theta_star(ev):
    """Pion polar cosine in the piN-CM relative to q (the direct un-integration test).
    Boost the pion and q into the rest frame of P = q + p_struck, take their cosine."""
    P = q4(ev) + ev.p_struck
    pi_r = boost_to_rest(P, ev.p_pi)
    q_r = boost_to_rest(P, q4(ev))
    return _costheta(pi_r[:, 1:], q_r[:, 1:])


def phi_star(ev):
    """Pion azimuth in the piN-CM relative to the lepton scattering plane [-pi,pi].
    e3 = q dir, e1 = lepton transverse dir (matches the fold's CM basis)."""
    P = q4(ev) + ev.p_struck
    pi_r = boost_to_rest(P, ev.p_pi)[:, 1:]
    q_r = boost_to_rest(P, q4(ev))[:, 1:]
    kp_r = boost_to_rest(P, ev.kp)[:, 1:]
    e3 = q_r / (_mag(q_r)[:, None] + 1e-12)
    kp_perp = kp_r - jnp.sum(kp_r * e3, axis=-1, keepdims=True) * e3
    e1 = kp_perp / (_mag(kp_perp)[:, None] + 1e-12)
    e2 = jnp.cross(e3, e1)
    x = jnp.sum(pi_r * e1, axis=-1)
    y = jnp.sum(pi_r * e2, axis=-1)
    return jnp.arctan2(y, x)


# ---- single-transverse kinematic imbalance (TKI) -------------------------- #
# Defined wrt the neutrino (+z); transverse = (x,y) components.  The hadronic transverse
# momentum here is pion + recoil nucleon (full visible hadronic system).  For a specific
# experiment's signal (e.g. lepton+pion only) drop p_N from `had` below.
def _pT(p4):
    return p4[..., 1:3]


def delta_pT(ev):
    """|vec p_T^lep + vec p_T^had| [MeV]; probes initial-nucleon transverse momentum."""
    had = ev.p_pi + ev.p_N
    return _mag(_pT(ev.kp) + _pT(had))


def delta_phiT(ev):
    """delta phi_T = arccos( -p_T^lep . p_T^had / (|p_T^lep||p_T^had|) ) [rad]."""
    had = ev.p_pi + ev.p_N
    lt, ht = _pT(ev.kp), _pT(had)
    c = -jnp.sum(lt * ht, axis=-1) / (_mag(lt) * _mag(ht) + 1e-12)
    return jnp.arccos(jnp.clip(c, -1.0, 1.0))


def delta_alphaT(ev):
    """delta alpha_T = arccos( -p_T^lep . vec dpT / (|p_T^lep||dpT|) ) [rad]."""
    had = ev.p_pi + ev.p_N
    lt = _pT(ev.kp)
    dpt = lt + _pT(had)
    c = -jnp.sum(lt * dpt, axis=-1) / (_mag(lt) * _mag(dpt) + 1e-12)
    return jnp.arccos(jnp.clip(c, -1.0, 1.0))


def delta_pTT(ev):
    """T2K CC1pi+ double-transverse momentum imbalance delta_pTT [MeV] (arXiv:2102.03346):
    the hadronic (pion+proton) momentum along the axis perpendicular to the lepton-scattering
    plane,  zhat_TT = (zhat_nu x p_mu)/|...|  (zhat_nu = beam = +z).  zhat_TT lies in the plane
    transverse to the beam, so this does NOT need the (unknown) neutrino energy -- the canonical
    double-transverse imbalance; zero for a stationary nucleon w/o FSI."""
    beam = jnp.array([0.0, 0.0, 1.0])
    mu3 = ev.kp[:, 1:]
    nrm = jnp.cross(jnp.broadcast_to(beam, mu3.shape), mu3)
    zhat = nrm / (jnp.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9)
    had = ev.p_pi[:, 1:] + ev.p_N[:, 1:]
    return jnp.sum(had * zhat, axis=-1)


# carbon target masses for the TKI longitudinal inference [MeV]
_M_A = 11174.862        # 12C nuclear mass
_M_A1 = 10252.547       # 11B residual


def p_N_tki(ev):
    """Inferred initial nucleon momentum |p_N| [MeV] (Furmanski-Sobczyk reconstruction,
    generalised to CC1pi+ with the visible hadron = pion+proton).  Transverse imbalance
    delta_pT plus a longitudinal component delta_pL inferred from energy-momentum conservation
    with the residual nucleus (12C->11B), eliminating the unknown neutrino energy."""
    lt = _pT(ev.kp); had = ev.p_pi + ev.p_N
    dpt_vec = lt + _pT(had)
    dpt2 = jnp.sum(dpt_vec ** 2, axis=-1)
    # visible longitudinal momentum and energy (muon + pion + proton)
    pL = ev.kp[:, 3] + had[:, 3]
    Evis = ev.kp[:, 0] + had[:, 0]
    R = _M_A + pL - Evis
    dpL = 0.5 * R - (_M_A1 ** 2 + dpt2) / (2.0 * jnp.clip(R, 1.0, None))
    return jnp.sqrt(jnp.clip(dpt2 + dpL ** 2, 0.0, None))


# registry for convenient batch histogramming / validation
OBSERVABLES = {
    "enu": enu, "Q2": Q2, "W": W, "lepton_energy": lepton_energy,
    "lepton_costheta": lepton_costheta, "ppi_mag": ppi_mag,
    "ppi_costheta_lab": ppi_costheta_lab, "nucleon_mom": nucleon_mom,
    "cos_theta_star": cos_theta_star, "phi_star": phi_star,
    "delta_pT": delta_pT, "delta_phiT": delta_phiT, "delta_alphaT": delta_alphaT,
    "delta_pTT": delta_pTT, "p_N_tki": p_N_tki,
}
