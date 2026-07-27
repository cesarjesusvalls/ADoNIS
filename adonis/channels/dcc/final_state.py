"""Two-body piN decay -> lab-frame final state (pion + recoil nucleon), plus the
piN-CM basis shared with the lepton tensor.

Mirrors ACHILLES `FinalStateMapper.cc::TwoBodyMapper`: from the sampled pion CM angles
(cos theta*, phi*) build the on-shell pion/nucleon 4-momenta in the piN centre-of-mass
(q along +z), then boost back to the lab via the hadronic system P = q + p_struck.

The CM spatial basis (e1, e2, e3) is constructed to MATCH
`lepton_tensor.cm_lepton_momenta` exactly: e3 = q direction in the piN-CM, e1 = the
(common) lepton transverse direction (q has no transverse part, so k and k' share it),
e2 = e3 x e1.  Hence the pion azimuth phi* is measured from the lepton scattering plane,
and the differential hadron tensor (built with this same convention) contracts correctly
against the lepton tensor.
"""
from __future__ import annotations

import jax.numpy as jnp

from adonis.channels.dcc.lepton import boost_to_rest, _mink_dot


def boost_from_rest(P, a):
    """Inverse of `boost_to_rest`: boost a 4-vector `a` given in the rest frame of P
    back to the lab frame.  Batched on a leading axis; P, a shape (...,4)."""
    M = jnp.sqrt(jnp.clip(_mink_dot(P, P), 1e-9, None))
    gamma = P[..., 0] / M
    beta = P[..., 1:] / P[..., 0:1]
    b2 = jnp.clip(jnp.sum(beta ** 2, axis=-1), 1e-30, None)
    bda = jnp.sum(beta * a[..., 1:], axis=-1)
    a0 = gamma * (a[..., 0] + bda)
    coef = ((gamma - 1.0) * bda / b2 + gamma * a[..., 0])[..., None]
    avec = a[..., 1:] + coef * beta
    return jnp.concatenate([a0[..., None], avec], axis=-1)


def _unit(v, axis=-1, eps=1e-12):
    return v / jnp.clip(jnp.linalg.norm(v, axis=axis, keepdims=True), eps, None)


def cm_basis(k, kp, p_struck):
    """piN-CM spatial basis (e1, e2, e3) in lab axes, batched.  e3 = q direction in the
    piN-CM, e1 = lepton transverse direction (matches cm_lepton_momenta), e2 = e3 x e1.
    Returns (e1, e2, e3, P) with each e_i shape (...,3) and P the lab hadronic 4-vector."""
    q = k - kp
    P = q + p_struck
    q_r = boost_to_rest(P, q)
    kp_r = boost_to_rest(P, kp)
    e3 = _unit(q_r[..., 1:])
    # lepton transverse part (perp to e3); fallback handled by _unit's eps clip
    kp_perp = kp_r[..., 1:] - jnp.sum(kp_r[..., 1:] * e3, axis=-1, keepdims=True) * e3
    e1 = _unit(kp_perp)
    e2 = jnp.cross(e3, e1)
    return e1, e2, e3, P


def two_body_lab(P, e1, e2, e3, W, cos_t, phi, m_pi, m_N):
    """Two-body piN decay in the piN-CM, boosted to lab.

    P                 : (...,4) lab hadronic 4-vector (boost target).
    e1,e2,e3          : (...,3) CM spatial basis in lab axes.
    W                 : (...,) hadronic invariant mass [MeV].
    cos_t, phi        : (...,) sampled pion CM polar cosine and azimuth.
    m_pi, m_N         : final pion / nucleon masses [MeV].
    Returns (p_pi_lab, p_N_lab) each (...,4) lab 4-momenta (on-shell)."""
    E_pi = (W ** 2 + m_pi ** 2 - m_N ** 2) / (2.0 * W)
    E_N = (W ** 2 + m_N ** 2 - m_pi ** 2) / (2.0 * W)
    kpi = jnp.sqrt(jnp.clip(E_pi ** 2 - m_pi ** 2, 0.0, None))
    sin_t = jnp.sqrt(jnp.clip(1.0 - cos_t ** 2, 0.0, 1.0))
    # pion 3-momentum in the CM (lab axes) = kpi (sinT cosPhi e1 + sinT sinPhi e2 + cosT e3)
    dir_pi = (sin_t[..., None] * jnp.cos(phi)[..., None] * e1
              + sin_t[..., None] * jnp.sin(phi)[..., None] * e2
              + cos_t[..., None] * e3)
    p_pi_vec = kpi[..., None] * dir_pi
    p_pi_cm = jnp.concatenate([E_pi[..., None], p_pi_vec], axis=-1)
    p_N_cm = jnp.concatenate([E_N[..., None], -p_pi_vec], axis=-1)
    return boost_from_rest(P, p_pi_cm), boost_from_rest(P, p_N_cm)


def rotate_about_z(p4, phi):
    """Rotate the spatial part of 4-vector(s) p4 by angle phi about the lab z-axis
    (the neutrino direction).  Used to give the event a uniform lepton azimuth; leaves
    all invariants and the weight unchanged.  Batched; p4 (...,4), phi (...,)."""
    c, s = jnp.cos(phi), jnp.sin(phi)
    x, y = p4[..., 1], p4[..., 2]
    xr = c * x - s * y
    yr = s * x + c * y
    return jnp.stack([p4[..., 0], xr, yr, p4[..., 3]], axis=-1)
