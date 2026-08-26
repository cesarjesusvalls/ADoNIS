"""Bit-exact ACHILLES nucleon form factors (FormFactor.cc): Kelly vector (F1p/F2p/F1n/F2n) +
AxialZExpansion axial (FA, FAP).  Q2 in GeV^2 (the backend passes -q.M2()/1e6).  Returns the
REAL Dirac/axial form factors; the complex electroweak coupling is applied separately in the
hadronic current (CouplingsFF).  Matches FormFactors.yml (Kelly + AxialZExpansion, MA=1.000)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels import constants as C

# Kelly (FormFactors.yml)
_LAMBDASQ = 0.7174
_MUP = 2.79278
_MUN = -1.91315
_EP = np.array([-0.24, 10.98, 12.82, 21.97])
_EN = np.array([1.70, 3.30])
_MP = np.array([0.12, 10.97, 18.86, 6.55])
_MN = np.array([2.33, 14.72, 24.20, 84.1])
# AxialZExpansion
_TCUT = 0.1753180641
_T0 = -0.28
_CC = np.array([-0.759, 2.30, -0.6, -3.8, 2.3, 2.16, -0.896, -1.58, 0.823])
_MP_GEV = C.mp / 1000.0


def _param(A, x):
    return (1 + A[0] * x) / (1 + A[1] * x + A[2] * x ** 2 + A[3] * x ** 3)


def _zexpand(A, z):
    res = jnp.zeros_like(z)
    for i in range(len(A) - 1, 0, -1):
        res = (res + A[i]) * z
    return res + A[0]


def nucleon_ff(Q2_GeV2, ff_scale=None):
    """Returns dict of F1p,F1n,F2p,F2n,FA,FAP at Q2 [GeV^2].  Vectorised over Q2.
    ff_scale (dict | None): per-Sachs-component MULTIPLIERS for the differentiable knobs --
    keys gep,gen,gmp,gmn (vector Sachs); each defaults to 1.0 (bit-exact no-op).  Scaling a single Sachs
    FF keeps amps2 QUADRATIC in that scale (the current is linear in F1/F2), so a 3-eval (s=0,1,-1)
    decomposition gives an exact reweight (mu_p == gmp scale, mu_n == gmn scale).  The axial FA stays on
    the dedicated axial_scale hook (dirac.py); ff_scale is vector-only to avoid double-scaling."""
    s = ff_scale or {}
    tau = Q2_GeV2 / 4.0 / _MP_GEV ** 2
    Gep = _param(_EP, tau) * s.get("gep", 1.0)
    Gen = (1.0 / (1 + Q2_GeV2 / _LAMBDASQ) ** 2 * _EN[0] * tau / (1 + _EN[1] * tau)) * s.get("gen", 1.0)
    Gmp = _MUP * _param(_MP, tau) * s.get("gmp", 1.0)
    Gmn = _MUN * _param(_MN, tau) * s.get("gmn", 1.0)
    F1p = (Gep + tau * Gmp) / (1 + tau); F1n = (Gen + tau * Gmn) / (1 + tau)
    F2p = (Gmp - Gep) / (1 + tau);        F2n = (Gmn - Gen) / (1 + tau)
    # GRADIENT PROTECTION: the z-expansion radicand TCUT+Q2 < 0 for unphysical Q2 < -TCUT (timelike q,
    # reachable from off-shell bound-nucleon kinematics).  sqrt of a negative NaNs FA and its gradient.
    # ACHILLES lets the NaN form and zeroes amps2 post-hoc (XSecBackend.cc:156, `if(isnan(amps2))
    # amps2=0`); here the sqrt argument is clamped instead, keeping forward+backward finite (bad events
    # are zeroed at the hadron-current level in dirac.py).  For physical Q2>=0, TCUT+Q2 >= TCUT > 0,
    # so this is an exact no-op.
    rad = _TCUT + Q2_GeV2
    sq = jnp.sqrt(jnp.where(rad > 0.0, rad, 1.0))                 # safe sqrt (no NaN fwd/bwd)
    z = (sq - np.sqrt(_TCUT - _T0)) / (sq + np.sqrt(_TCUT - _T0))
    FA = _zexpand(_CC, z)
    mpi = C.mpip
    fap_den = Q2_GeV2 * 1e6 + mpi ** 2                            # >= mpi^2 for all physical Q2>=0 (no pole)
    FAP = 2.0 * C.mN2 / jnp.where(jnp.abs(fap_den) > 1.0, fap_den, 1.0) * FA   # guard the pseudoscalar pole
    return dict(F1p=F1p, F1n=F1n, F2p=F2p, F2n=F2n, FA=FA, FAP=FAP)
