"""Lepton tensor + contraction L_{mu,nu} W^{mu,nu} (Phase-2.5, milestone 6c option 1).

Replaces the factorized leptonic flux (Hand / flat W-propagator) with the faithful
contraction of the lepton tensor against the full hadron tensor W^{mu,nu} from
hadron_xsec.HadronStructure.tensor_at.  This restores, in one step:
  * the longitudinal response with its CORRECT kinematic weight (vs the EM-eps
    placeholder), and
  * the V-A interference (the antisymmetric -i eps^{mu nu a b} k_a k'_b term, which
    contracts with the antisymmetric Im part of the Hermitian hadron tensor).

Everything is done in the piN-CM frame with q along +z (the frame the hadron tensor
is built in): per event we boost the lab lepton momenta into the rest frame of
P = q + p_struck, then resolve them along q_cm.  Because q has no transverse part,
the incoming and outgoing leptons share the same transverse momentum, so both lie in
the x-z plane (e1 chosen along it) -- consistent with the azimuthally-symmetric (phi_pi
integrated) hadron tensor.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

ETA = jnp.array([1.0, -1.0, -1.0, -1.0])           # metric diag (+,-,-,-)

# Levi-Civita eps^{mu nu alpha beta} with eps^{0123} = +1
_EPS = np.zeros((4, 4, 4, 4))
for _p in __import__("itertools").permutations(range(4)):
    _i, _j, _k, _l = _p
    _sign = 1
    _pl = list(_p)
    for _a in range(4):
        for _b in range(_a + 1, 4):
            if _pl[_a] > _pl[_b]:
                _sign = -_sign
    _EPS[_i, _j, _k, _l] = _sign
_EPS = jnp.asarray(_EPS)


def _mink_dot(a, b):
    return a[..., 0] * b[..., 0] - jnp.sum(a[..., 1:] * b[..., 1:], axis=-1)


def boost_to_rest(P, a):
    """Boost 4-vector(s) `a` into the rest frame of 4-vector(s) `P`.  Batched on a
    leading event axis; P,a shape (...,4).  Returns a' shape (...,4)."""
    M = jnp.sqrt(jnp.clip(_mink_dot(P, P), 1e-9, None))
    gamma = P[..., 0] / M
    beta = P[..., 1:] / P[..., 0:1]                 # (...,3)
    b2 = jnp.clip(jnp.sum(beta ** 2, axis=-1), 1e-30, None)
    bda = jnp.sum(beta * a[..., 1:], axis=-1)        # beta . avec
    a0 = gamma * (a[..., 0] - bda)
    coef = ((gamma - 1.0) * bda / b2 - gamma * a[..., 0])[..., None]
    avec = a[..., 1:] + coef * beta
    return jnp.concatenate([a0[..., None], avec], axis=-1)


def cm_lepton_momenta(k, kp, p_struck):
    """Lab k (incoming), kp (outgoing lepton), p_struck -> the two lepton 4-momenta in
    the piN-CM frame resolved with q along +z and both leptons in the x-z plane.
    Returns (k_cm, kp_cm) each (...,4) = (E, pT, 0, pz)."""
    q = k - kp
    P = q + p_struck
    k_r = boost_to_rest(P, k)
    kp_r = boost_to_rest(P, kp)
    q_r = boost_to_rest(P, q)
    e3 = q_r[..., 1:] / jnp.linalg.norm(q_r[..., 1:], axis=-1, keepdims=True)
    def resolve(v):
        vz = jnp.sum(v[..., 1:] * e3, axis=-1)
        vT_vec = v[..., 1:] - vz[..., None] * e3
        vT = jnp.linalg.norm(vT_vec, axis=-1)
        return jnp.stack([v[..., 0], vT, jnp.zeros_like(vT), vz], axis=-1)
    return resolve(k_r), resolve(kp_r)


def lepton_tensor_cc(k, kp, anti=False):
    """CC (anti)neutrino lepton tensor L^{mu,nu} (massless leptons), batched (...,4).
    L = 8[k^mu k'^nu + k'^mu k^nu - g^{mu nu}(k.k') -/+ i eps^{mu nu a b} k_a k'_b]."""
    kk = _mink_dot(k, kp)[..., None, None]
    sym = (k[..., :, None] * kp[..., None, :] + kp[..., :, None] * k[..., None, :]
           - jnp.diag(ETA)[None] * kk)
    k_low = k * ETA
    kp_low = kp * ETA
    asym = jnp.einsum("mnab,...a,...b->...mn", _EPS, k_low, kp_low)
    sign = 1.0 if anti else -1.0
    return 8.0 * (sym + sign * 1j * asym)


def contract(L, W):
    """L_{mu,nu} W^{mu,nu} = sum_{mu,nu} eta_mu eta_nu L^{mu,nu} W^{mu,nu}; real part.
    L,W shape (...,4,4)."""
    g = ETA[:, None] * ETA[None, :]
    return jnp.real(jnp.sum(g[None] * L * W, axis=(-2, -1)))
