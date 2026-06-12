"""Bit-exact ACHILLES nucleon form factors (FormFactor.cc): Kelly vector (F1p/F2p/F1n/F2n) +
AxialZExpansion axial (FA, FAP).  Q2 in GeV^2 (the backend passes -q.M2()/1e6).  Returns the
REAL Dirac/axial form factors; the complex electroweak coupling is applied separately in the
hadronic current (CouplingsFF).  Matches FormFactors.yml (Kelly + AxialZExpansion, MA=1.000)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.xsec import constants as C

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


def nucleon_ff(Q2_GeV2):
    """Returns dict of F1p,F1n,F2p,F2n,FA,FAP at Q2 [GeV^2].  Vectorised over Q2."""
    tau = Q2_GeV2 / 4.0 / _MP_GEV ** 2
    Gep = _param(_EP, tau)
    Gen = 1.0 / (1 + Q2_GeV2 / _LAMBDASQ) ** 2 * _EN[0] * tau / (1 + _EN[1] * tau)
    Gmp = _MUP * _param(_MP, tau)
    Gmn = _MUN * _param(_MN, tau)
    F1p = (Gep + tau * Gmp) / (1 + tau); F1n = (Gen + tau * Gmn) / (1 + tau)
    F2p = (Gmp - Gep) / (1 + tau);        F2n = (Gmn - Gen) / (1 + tau)
    z = (jnp.sqrt(_TCUT + Q2_GeV2) - np.sqrt(_TCUT - _T0)) / (jnp.sqrt(_TCUT + Q2_GeV2) + np.sqrt(_TCUT - _T0))
    FA = _zexpand(_CC, z)
    mpi = C.mpip
    FAP = 2.0 * C.mN2 / (Q2_GeV2 * 1e6 + mpi ** 2) * FA          # Q2 GeV^2 -> MeV^2
    return dict(F1p=F1p, F1n=F1n, F2p=F2p, F2n=F2n, FA=FA, FAP=FAP)
