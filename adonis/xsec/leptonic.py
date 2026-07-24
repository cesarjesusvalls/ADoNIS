"""Bit-exact JAX port of ACHILLES LeptonicCurrent::CalcCurrents (LeptonicCurrent.cc:141-182).

Builds the leptonic current j^mu = ubar[i] (coupl_left gamma^mu P_L + coupl_right gamma^mu P_R)
u[j] * prop for the 4 spin combinations (i,j) = (00,01,10,11), helicities ubar/u in {-1,+1}.
The boson propagator prop = i/(q^2 - M^2 - i M Gamma) is FOLDED INTO the leptonic current (so the
hadronic side carries none).  Returns L of shape (..., 4_spincombo, 4_mu) complex.

Couplings (LeptonicCurrent.cc:24-45) for the standard probes:
  CC nu  : coupl_left = ee*i/(sw*sqrt2), coupl_right=0, M=MW, Gamma=GAMW
  EM e   : coupl_left = coupl_right = -ee*i, M=Gamma=0 -> prop = i/q^2 (the photon propagator)
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.xsec import constants as C
from adonis.xsec.spinor import GAMMA, GAMMA5, ubar, uspinor

_I = 1j
IDENT4 = np.eye(4, dtype=complex)
_PL = (IDENT4 - GAMMA5) / 2.0
_PR = (IDENT4 + GAMMA5) / 2.0


def _vertex_matrices(coupl_left, coupl_right):
    """M[mu] = coupl_left gamma^mu P_L + coupl_right gamma^mu P_R, shape (4,4,4) (mu,i,j)."""
    gPL = np.einsum('mij,jk->mik', GAMMA, _PL)
    gPR = np.einsum('mij,jk->mik', GAMMA, _PR)
    return coupl_left * gPL + coupl_right * gPR


def _couplings(kind):
    if kind == "CC_nu":
        return C.ee * _I / (C.sw * np.sqrt(2.0)), 0.0 + 0j, C.MW, C.GAMW, True
    if kind == "EM":
        c = -C.ee * _I
        return c, c, 0.0, 0.0, True                  # photon: prop = i/q^2 (M=Gamma=0 boson-propagator limit)
    raise ValueError(kind)


def lepton_current(p_in, p_out, kind="CC_nu", anti=False):
    """L (..., 4, 4) for incoming lepton p_in, outgoing p_out (..., 4) MeV.
    Spin-combo axis order (i,j)=(00,01,10,11); mu axis 0..3.  anti flips the spinor assignment."""
    cl, cr, M, G, has_prop = _couplings(kind)
    Mmu = jnp.asarray(_vertex_matrices(cl, cr))                  # (4,4,4)
    if anti:
        pUBar, pU = -p_in, p_out
    else:
        pU, pUBar = -p_in, p_out
    ub = [ubar(-1, pUBar), ubar(1, pUBar)]
    us = [uspinor(-1, pU), uspinor(1, pU)]
    q = p_in - p_out
    q2 = q[..., 0] ** 2 - jnp.sum(q[..., 1:] ** 2, axis=-1)
    prop = _I / (q2 - M * M - _I * M * G) if has_prop else jnp.ones_like(q2, complex)
    # subcur[mu]_(i,j) = ubar[i] . M[mu] . u[j] . prop, spin-combo order (i,j)=00,01,10,11
    out = []
    for i in range(2):
        for j in range(2):
            Muj = jnp.einsum('mij,...j->...mi', Mmu, us[j])      # (...,4mu,4comp)
            val = jnp.einsum('...i,...mi->...m', ub[i], Muj)     # (...,4mu)
            out.append(val * prop[..., None])
    return jnp.stack(out, axis=-2)                               # (...,4combo,4mu)
