"""Bit-exact JAX port of ACHILLES QESpectral::CalcCurrents/HadronicCurrent (NuclearModel.cc:545-643)
for CC nu n -> mu- p (pid=-24).  Ward=None (the paper run card), so no gauge correction.

J^mu = ubar(pOut) [ F1 g^mu + FA g^mu g5 + FAP g5 q^mu/mN + i F2 sigma^{mu nu} q_nu/(2 mN) ] u(-pIn)
with the de Forest energy shift: qVec.E() = omega + pIn.E() - sqrt(pIn^2+mN^2), pIn put on-shell.
F1,F2,FA,FAP carry the complex CC coupling coupl = Vud ee i/(sw sqrt2 * 2), via the isovector
combination F1 = coupl (F1p - F1n) etc. (CouplingsFF for the {neutron,-24} entry).
Returns H of shape (..., 4_spincombo, 4_mu), spin-combo order (i,j)=(00,01,10,11).

NOTE: currently UNUSED by the event chain -- the paper runs use the FORTRAN QE current
(FortranModel QE_Spectral_Func), ported in `adonis/channels/dirac.py`, which is what
`backend.me_cross_section` calls.  Retained as a working reference port of the
alternative C++ Weyl QESpectral path.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels import constants as C
from adonis.channels.currents.spinor import GAMMA, GAMMA5, SIGMA, ubar, uspinor
from adonis.channels.currents.form_factors import nucleon_ff

_I = 1j
_GG5 = np.einsum('mij,jk->mik', GAMMA, GAMMA5)            # gamma^mu gamma5  (4,4,4)
_COUPL_CC = C.Vud * C.ee * _I / (C.sw * np.sqrt(2.0) * 2.0)   # CC W coupling (neutron->proton)


def hadron_current_qe(p_in_lep, p_out_lep, p_in_nuc, p_out_nuc):
    """All args (...,4) MeV.  p_in_nuc is the OFF-SHELL struck nucleon (E = mN - removal); p_out_nuc
    the outgoing on-shell nucleon.  Returns H (...,4combo,4mu)."""
    q = p_in_lep - p_out_lep
    Q2_FF = -(q[..., 0] ** 2 - jnp.sum(q[..., 1:] ** 2, axis=-1)) / 1e6     # GeV^2, ORIGINAL q
    ff = nucleon_ff(Q2_FF)
    F1 = _COUPL_CC * (ff["F1p"] - ff["F1n"]); F2 = _COUPL_CC * (ff["F2p"] - ff["F2n"])
    FA = _COUPL_CC * ff["FA"];                FAP = _COUPL_CC * ff["FAP"]

    # de Forest energy shift on q; put initial nucleon on-shell
    p3 = p_in_nuc[..., 1:]
    free_E = jnp.sqrt(jnp.sum(p3 ** 2, axis=-1) + C.mN2)
    omega = q[..., 0]
    qE = omega + p_in_nuc[..., 0] - free_E
    qshift = jnp.concatenate([qE[..., None], q[..., 1:]], axis=-1)          # (...,4)
    pIn_on = jnp.concatenate([free_E[..., None], p3], axis=-1)
    qlow = qshift * jnp.array([1.0, -1.0, -1.0, -1.0])                       # q_nu (lower index)

    # vertex[mu] (...,4mu,4,4)
    G = jnp.asarray(GAMMA); GG5 = jnp.asarray(_GG5); G5 = jnp.asarray(GAMMA5); SIG = jnp.asarray(SIGMA)
    sterm = jnp.einsum('mnij,...n->...mij', SIG, qlow)                       # sigma^{mu nu} q_nu
    vtx = (F1[..., None, None, None] * G
           + FA[..., None, None, None] * GG5
           + FAP[..., None, None, None] * (qshift / C.mN)[..., :, None, None] * G5
           + (_I * F2 / (2.0 * C.mN))[..., None, None, None] * sterm)        # (...,4mu,4,4)

    ub = [ubar(-1, p_out_nuc), ubar(1, p_out_nuc)]
    us = [uspinor(-1, -pIn_on), uspinor(1, -pIn_on)]
    out = []
    for i in range(2):
        for j in range(2):
            Vu = jnp.einsum('...mij,...j->...mi', vtx, us[j])                # (...,4mu,4)
            out.append(jnp.einsum('...i,...mi->...m', ub[i], Vu))           # (...,4mu)
    return jnp.stack(out, axis=-2)                                          # (...,4combo,4mu)
