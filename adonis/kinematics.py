"""4-vector algebra and physics observables (JAX).

Two layers:
  * PRIMITIVES  : mink_dot, mass2, kallen, boost (CM->lab), boost_to_rest, boost_from_rest.
  * OBSERVABLES : leptonic / hadronic / single-transverse-kinematic-imbalance (TKI) formulas, each
                  written once on raw lab 4-vectors, exposed two ways --
                    - EventRecord wrappers + the OBSERVABLES registry (differentiable reweight, DCC),
                    - raw-4-vector functions (vertex_W_Q2 / tki / muon_obs / cc0pi_obs) for the
                      selection + info-content analysis paths.

Metric (+,-,-,-); 4-vectors (...,4) = (E, px, py, pz) [MeV].  JAX throughout -- numpy inputs promote
fine.
"""
from __future__ import annotations

import jax.numpy as jnp

from adonis.constants import M_12C as _M_A, M_11B as _M_A1


def mink_dot(a, b):
    """Minkowski dot a.b = a0 b0 - vec a . vec b, batched over a leading axis."""
    return a[..., 0] * b[..., 0] - jnp.sum(a[..., 1:] * b[..., 1:], axis=-1)


def mass2(p):
    """Invariant p.p = E^2 - |p|^2."""
    return mink_dot(p, p)


def kallen(s, s1, s2):
    """Kallen triangle lambda(s,s1,s2) = (s - s1 - s2)^2 - 4 s1 s2."""
    return (s - s1 - s2) ** 2 - 4 * s1 * s2


def boost(p4, beta):
    """CM->lab Lorentz boost of p4 (...,4) by beta (...,3) (= P/E of the boosting frame)."""
    b2 = jnp.sum(beta ** 2, axis=-1, keepdims=True)
    g = 1.0 / jnp.sqrt(jnp.clip(1 - b2, 1e-15, None))
    bp = jnp.sum(beta * p4[..., 1:], axis=-1, keepdims=True)
    E = g[..., 0] * (p4[..., 0] + bp[..., 0])
    fac = (g - 1.0) * jnp.where(b2 > 1e-15, bp / jnp.clip(b2, 1e-15, None), 0.0) + g * p4[..., :1]
    vec = p4[..., 1:] + fac * beta
    return jnp.concatenate([E[..., None], vec], axis=-1)


def boost_to_rest(P, a):
    """Boost 4-vector(s) `a` into the rest frame of 4-vector(s) `P` (batched on a leading axis)."""
    M = jnp.sqrt(jnp.clip(mink_dot(P, P), 1e-9, None))
    gamma = P[..., 0] / M
    beta = P[..., 1:] / P[..., 0:1]
    b2 = jnp.clip(jnp.sum(beta ** 2, axis=-1), 1e-30, None)
    bda = jnp.sum(beta * a[..., 1:], axis=-1)
    a0 = gamma * (a[..., 0] - bda)
    coef = ((gamma - 1.0) * bda / b2 - gamma * a[..., 0])[..., None]
    avec = a[..., 1:] + coef * beta
    return jnp.concatenate([a0[..., None], avec], axis=-1)


def boost_from_rest(P, a):
    """Inverse of `boost_to_rest`: `a` given in the rest frame of P -> the lab frame."""
    M = jnp.sqrt(jnp.clip(mink_dot(P, P), 1e-9, None))
    gamma = P[..., 0] / M
    beta = P[..., 1:] / P[..., 0:1]
    b2 = jnp.clip(jnp.sum(beta ** 2, axis=-1), 1e-30, None)
    bda = jnp.sum(beta * a[..., 1:], axis=-1)
    a0 = gamma * (a[..., 0] + bda)
    coef = ((gamma - 1.0) * bda / b2 + gamma * a[..., 0])[..., None]
    avec = a[..., 1:] + coef * beta
    return jnp.concatenate([a0[..., None], avec], axis=-1)


def _mag(v3):
    return jnp.sqrt(jnp.sum(v3 ** 2, axis=-1))


def _costheta(a3, b3, eps=1e-12):
    return jnp.sum(a3 * b3, axis=-1) / (_mag(a3) * _mag(b3) + eps)


def _pT(p4):
    return p4[..., 1:3]


def _q(nu, mu):
    return nu - mu


def _Q2_mev2(nu, mu):
    """Leptonic Q^2 = -(k-k')^2 [MeV^2]."""
    q = _q(nu, mu)
    return -mink_dot(q, q)


def _W_hadronic(p_pi, p_N):
    """Decay W from the final pion+nucleon [MeV]."""
    P = p_pi + p_N
    return jnp.sqrt(jnp.clip(mink_dot(P, P), 0.0, None))


def _W_vertex(nu, mu, struck):
    """Vertex W = |q + struck| [MeV]."""
    tot = _q(nu, mu) + struck
    return jnp.sqrt(jnp.clip(mink_dot(tot, tot), 0.0, None))


def _dpt(kp, had):
    """|vec p_T^lep + vec p_T^had| [MeV] (had = the visible hadronic system)."""
    return _mag(_pT(kp) + _pT(had))


def _dalphat(kp, had):
    """delta alpha_T = arccos(-p_T^lep . dpT / (|p_T^lep||dpT|)) [rad]."""
    lt = _pT(kp)
    dpt_vec = lt + _pT(had)
    c = -jnp.sum(lt * dpt_vec, axis=-1) / (_mag(lt) * _mag(dpt_vec) + 1e-12)
    return jnp.arccos(jnp.clip(c, -1.0, 1.0))


def _dptt(kp, p_pi, had_partner):
    """Double-transverse imbalance delta_pTT [MeV]: hadronic (pion + partner) momentum along
    zhat_TT = (beam x p_lep)/|...| (beam = +z).  `had_partner` = recoil nucleon (reweight) or lead
    proton (selection)."""
    beam = jnp.array([0.0, 0.0, 1.0])
    mu3 = kp[:, 1:]
    zhat = jnp.cross(jnp.broadcast_to(beam, mu3.shape), mu3)
    zhat = zhat / (jnp.linalg.norm(zhat, axis=-1, keepdims=True) + 1e-9)
    had3 = p_pi[:, 1:] + had_partner[:, 1:]
    return jnp.sum(had3 * zhat, axis=-1)


def _p_N_tki(kp, p_pi, had_partner):
    """Furmanski-Sobczyk inferred initial nucleon |p_N| [MeV] (visible hadron = pion + partner)."""
    had = p_pi + had_partner
    dpt_vec = _pT(kp) + _pT(had)
    dpt2 = jnp.sum(dpt_vec ** 2, axis=-1)
    pL = kp[:, 3] + had[:, 3]
    Evis = kp[:, 0] + had[:, 0]
    R = _M_A + pL - Evis
    dpL = 0.5 * R - (_M_A1 ** 2 + dpt2) / (2.0 * jnp.clip(R, 1.0, None))
    return jnp.sqrt(jnp.clip(dpt2 + dpL ** 2, 0.0, None))


def enu(ev):
    return ev.k[:, 0]


def q4(ev):
    return _q(ev.k, ev.kp)


def Q2(ev):
    """Observable leptonic Q^2 [MeV^2]."""
    return _Q2_mev2(ev.k, ev.kp)


def W(ev):
    """Hadronic invariant mass from final pion+nucleon [MeV]."""
    return _W_hadronic(ev.p_pi, ev.p_N)


def lepton_energy(ev):
    return ev.kp[:, 0]


def lepton_costheta(ev):
    """cos of the outgoing-lepton polar angle wrt the neutrino (lab)."""
    return _costheta(ev.kp[:, 1:], ev.k[:, 1:])


def ppi_mag(ev):
    return _mag(ev.p_pi[:, 1:])


def ppi_costheta_lab(ev):
    """cos angle of the pion wrt q in the lab."""
    return _costheta(ev.p_pi[:, 1:], q4(ev)[:, 1:])


def nucleon_mom(ev):
    return _mag(ev.p_N[:, 1:])


def cos_theta_star(ev):
    """Pion polar cosine in the piN-CM relative to q (rest frame of P = q + p_struck)."""
    P = q4(ev) + ev.p_struck
    pi_r = boost_to_rest(P, ev.p_pi)
    q_r = boost_to_rest(P, q4(ev))
    return _costheta(pi_r[:, 1:], q_r[:, 1:])


def phi_star(ev):
    """Pion azimuth in the piN-CM relative to the lepton scattering plane [-pi,pi]."""
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


def delta_pT(ev):
    """|vec p_T^lep + vec p_T^had| [MeV] (had = pion + recoil nucleon)."""
    return _dpt(ev.kp, ev.p_pi + ev.p_N)


def delta_phiT(ev):
    """delta phi_T = arccos(-p_T^lep . p_T^had / (|p_T^lep||p_T^had|)) [rad]."""
    had = ev.p_pi + ev.p_N
    lt, ht = _pT(ev.kp), _pT(had)
    c = -jnp.sum(lt * ht, axis=-1) / (_mag(lt) * _mag(ht) + 1e-12)
    return jnp.arccos(jnp.clip(c, -1.0, 1.0))


def delta_alphaT(ev):
    """delta alpha_T [rad] (had = pion + recoil nucleon)."""
    return _dalphat(ev.kp, ev.p_pi + ev.p_N)


def delta_pTT(ev):
    """T2K CC1pi+ double-transverse imbalance delta_pTT [MeV] (had partner = recoil nucleon)."""
    return _dptt(ev.kp, ev.p_pi, ev.p_N)


def p_N_tki(ev):
    """Inferred initial nucleon momentum |p_N| [MeV] (visible hadron = pion + recoil nucleon)."""
    return _p_N_tki(ev.kp, ev.p_pi, ev.p_N)


M_A_12C = _M_A
M_A_11B = _M_A1

OBSERVABLES = {
    "enu": enu, "Q2": Q2, "W": W, "lepton_energy": lepton_energy,
    "lepton_costheta": lepton_costheta, "ppi_mag": ppi_mag,
    "ppi_costheta_lab": ppi_costheta_lab, "nucleon_mom": nucleon_mom,
    "cos_theta_star": cos_theta_star, "phi_star": phi_star,
    "delta_pT": delta_pT, "delta_phiT": delta_phiT, "delta_alphaT": delta_alphaT,
    "delta_pTT": delta_pTT, "p_N_tki": p_N_tki,
}


def mom(p4):
    """|vec p| [MeV] of a 4-vector (batch)."""
    return _mag(jnp.atleast_2d(p4)[..., 1:])


def vertex_W_Q2(nu, mu, struck):
    """Vertex W [MeV] and leptonic Q2 [GeV^2] (note: GeV^2, the selection convention)."""
    return _W_vertex(nu, mu, struck), _Q2_mev2(nu, mu) / 1e6


def tki(kmu, ppi, lead):
    """CC1pi TKI (dptt, pN, dalphat, dpt) from raw 4-vectors -- deterministic (hydrogen overlay in caller)."""
    had = ppi + lead
    return _dptt(kmu, ppi, lead), _p_N_tki(kmu, ppi, lead), _dalphat(kmu, had), _dpt(kmu, had)


def muon_obs(mu, nu):
    """Muon-only observables (0-proton topology): leptonic Q2 [GeV^2], p_mu, cos_mu."""
    pmu = _mag(mu[:, 1:])
    return dict(Q2=_Q2_mev2(nu, mu) / 1e6, p_mu=pmu, cos_mu=mu[:, 3] / jnp.clip(pmu, 1e-9, None))


def cc0pi_obs(mu, lead, struck, nu):
    """CC0pi observables from muon + leading proton (no pion)."""
    lt = mu[:, 1:3]; pt = lead[:, 1:3]; dv = lt + pt
    dpt = jnp.linalg.norm(dv, axis=-1)
    c = -jnp.sum(lt * dv, axis=-1) / (jnp.linalg.norm(lt, axis=-1) * dpt + 1e-12)
    dat = jnp.arccos(jnp.clip(c, -1.0, 1.0))
    W_, Q2_ = vertex_W_Q2(nu, mu, struck)
    pmu = _mag(mu[:, 1:]); cmu = mu[:, 3] / jnp.clip(pmu, 1e-9, None)
    lpp = _mag(lead[:, 1:])
    return dict(dpt=dpt, dalphat=dat, Q2=Q2_, W=W_, p_mu=pmu, cos_mu=cmu, lp_p=lpp)
