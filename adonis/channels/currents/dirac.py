"""Bit-exact JAX port of the ACHILLES Fortran QE Dirac current (currents_opt_v1.f90, module
dirac_matrices); used by the paper's FortranModel QE_Spectral_Func, not the C++ Weyl QESpectral.
Ports define_spinors / det_Ja / hadr_curr_matrix_el / current_init_had.

DIRAC basis: g0=diag(I,-I), g^i=offdiag(sigma,-sigma), g5=offdiag(I,I).  Spinors use a single
xmn=mN (constants%mqe) rather than the true on-shell mass: current_init_had puts the incoming
nucleon on-shell at mN, while the outgoing nucleon keeps its actual (mp-on-shell) energy but is
still normalised with xmn=mN, so ubar.u = 2 xmn.

Returns H (..., 4_spincombo, 4_mu); spin-combo order (i,j)=(00,01,10,11) matches ACHILLES's
cur(i+2*(j-1)), flattened here as (i1=initial, f1=final).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels import constants as C
from adonis.channels.currents.form_factors import nucleon_ff, _TCUT as _FF_TCUT

_I = 1j
_XMN = C.mN
_COUPL_CC = C.Vud * C.ee * _I / (C.sw * np.sqrt(2.0) * 2.0)
_COUPL_EM = _I * C.ee

_COUPL1_NC_QUIRK = (C.ee * _I / (4 * C.sin2w * C.cw)) * (0.5 - 2 * C.sin2w)
_COUPL1_NC_TRUE = (C.ee * _I / (2 * C.sw * C.cw)) * (0.5 - 2 * C.sin2w)
_COUPL2_NC = C.ee * _I / (4 * C.sw * C.cw)


def nc_coupl1(use_achilles_nc_coupling=False):
    """The NC QE vector coupling; `use_achilles_nc_coupling=True` reproduces the ACHILLES sin2w/sw discrepancy."""
    return _COUPL1_NC_QUIRK if use_achilles_nc_coupling else _COUPL1_NC_TRUE

_SIG = np.stack([
    np.array([[0, 1], [1, 0]], complex),
    np.array([[0, -_I], [_I, 0]], complex),
    np.array([[1, 0], [0, -1]], complex),
])
_ID2 = np.eye(2, dtype=complex)
_G = np.zeros((4, 4, 4), complex)
_G[0, :2, :2] = _ID2; _G[0, 2:, 2:] = -_ID2
for i in range(1, 4):
    _G[i, :2, 2:] = _SIG[i - 1]; _G[i, 2:, :2] = -_SIG[i - 1]
_G5 = np.zeros((4, 4), complex); _G5[:2, 2:] = _ID2; _G5[2:, :2] = _ID2
_GMETRIC = np.array([1.0, -1.0, -1.0, -1.0])
_SIGMN = np.zeros((4, 4, 4, 4), complex)
for mu in range(4):
    for nu in range(4):
        _SIGMN[mu, nu] = 0.5 * _I * (_G[mu] @ _G[nu] - _G[nu] @ _G[mu])
_GG5 = np.einsum('mij,jk->mik', _G, _G5)

_GJ = jnp.asarray(_G); _G5J = jnp.asarray(_G5); _GG5J = jnp.asarray(_GG5)
_SIGMNJ = jnp.asarray(_SIGMN); _SIGJ = jnp.asarray(_SIG)


def _spinors(p3, E):
    """define_spinors for one nucleon: returns u (2 spin, 4) and ubar (2 spin, 4), xmn=mN.
    p3 (...,3), E (...,) -- E the actual energy used (mN-on-shell for in, mp-on-shell for out)."""
    sigp = jnp.einsum('aij,...a->...ij', _SIGJ, p3)
    sden = E + _XMN
    denom = jnp.where(sden != 0.0, sden, 1.0)[..., None, None]
    cp = jnp.sqrt(jnp.where(sden > 0.0, sden, 1.0))[..., None]
    chi = jnp.asarray(_ID2)
    lower = jnp.einsum('...ij,jk->...ik', sigp / denom, chi)
    up = jnp.stack([chi[0, 0] + 0 * E, chi[1, 0] + 0 * E], axis=-1)
    def mk(col):
        u_up = jnp.broadcast_to(jnp.asarray(chi[:, col]), p3.shape[:-1] + (2,))
        u_lo = lower[..., col]
        return jnp.concatenate([u_up, u_lo], axis=-1) * cp
    u = jnp.stack([mk(0), mk(1)], axis=-2)
    lower_b = jnp.einsum('ij,...jk->...ik', chi, -sigp / denom)
    def mkb(row):
        ub_up = jnp.broadcast_to(jnp.asarray(chi[row, :]), p3.shape[:-1] + (2,))
        ub_lo = lower_b[..., row, :]
        return jnp.concatenate([ub_up, ub_lo], axis=-1) * cp
    ubar = jnp.stack([mkb(0), mkb(1)], axis=-2)
    return u, ubar


def _ubar_vtx_u(vtx, u_in, ubar_out):
    """H[...,combo,mu] = ubar_out[f1] . vtx[mu] . u_in[i1], combos ordered (i1 init outer, f1 final inner),
    matching ACHILLES cur(i+2*(j-1)).  Shared by the summed current and the per-structure
    (return_structures) unit currents, so both use the identical spinor contraction."""
    out = []
    for i1 in range(2):
        Vu = jnp.einsum('...mij,...j->...mi', vtx, u_in[..., i1, :])
        for f1 in range(2):
            out.append(jnp.einsum('...i,...mi->...m', ubar_out[..., f1, :], Vu))
    return jnp.stack(out, axis=-2)


def hadron_current_qe_dirac(p_in_lep, p_out_lep, p_in_nuc, p_out_nuc, axial_scale=1.0, vector_scale=1.0,
                            ff_scale=None, probe="CC", is_proton=None, use_achilles_nc_coupling=False,
                            return_structures=False):
    """All momenta (...,4) MeV.  p_in_nuc is the off-shell struck nucleon (E=mN-removal); p_out_nuc
    is outgoing (mp-on-shell energy).  Returns H (...,4combo,4mu) matching ACHILLES cur(i+2*(j-1)).
    axial_scale (scalar or (...,)) multiplies FA and FAP (FAP ~ FA), the M_A reweight hook;
    vector_scale multiplies F1 and F2, the vector-strength reweight hook.  amps2 is quadratic in
    each, so 3 evals give the exact per-event decomposition.  Default 1.0 = nominal.

    probe:
      "CC" (default) -- nu n -> mu- p.  Isovector combination with the W coupling, plus the axial
            current.
      "EM" -- e N -> e' N  (inclusive (e,e')).  ACHILLES LeptonicCurrent.cc:120 (pid == 22):
                coupl = i*ee
                proton : {F1p, coupl}, {F2p, coupl}    <- NO FA entry
                neutron: {F1n, coupl}, {F2n, coupl}    <- NO FA entry
            The photon couples to the struck nucleon's own form factors (not F1p - F1n), and
            CouplingsFF leaves FA = FAP = 0, so the current is purely vector.  Proton and neutron
            are both struck incoherently; `is_proton` (bool, broadcast over events) selects which.
            vector_scale still multiplies F1/F2, so vector-strength / mu_p / mu_n / gep / gen
            reweights work unchanged here."""
    q = p_in_lep - p_out_lep
    Q2_FF = -(q[..., 0] ** 2 - jnp.sum(q[..., 1:] ** 2, axis=-1)) / 1e6
    ff = nucleon_ff(Q2_FF, ff_scale=ff_scale)
    asc = jnp.asarray(axial_scale); vsc = jnp.asarray(vector_scale)
    if probe == "CC":
        F1 = _COUPL_CC * (ff["F1p"] - ff["F1n"]) * vsc; F2 = _COUPL_CC * (ff["F2p"] - ff["F2n"]) * vsc
        FA = _COUPL_CC * ff["FA"] * asc;                FAP = _COUPL_CC * ff["FAP"] * asc
    elif probe == "EM":
        if is_proton is None:
            raise ValueError("probe='EM' needs is_proton (the struck nucleon species, per event)")
        isp = jnp.asarray(is_proton)
        f1 = jnp.where(isp, ff["F1p"], ff["F1n"])
        f2 = jnp.where(isp, ff["F2p"], ff["F2n"])
        F1 = _COUPL_EM * f1 * vsc; F2 = _COUPL_EM * f2 * vsc
        FA = jnp.zeros_like(F1); FAP = jnp.zeros_like(F1)
    elif probe == "NC":
        if is_proton is None:
            raise ValueError("probe='NC' needs is_proton: the NC QE couplings are PER NUCLEON "
                             "(LeptonicCurrent.cc:97-115), unlike CC's single isovector combination")
        isp = jnp.asarray(is_proton)
        c1 = nc_coupl1(use_achilles_nc_coupling)
        f1_own = jnp.where(isp, ff["F1p"], ff["F1n"]); f1_oth = jnp.where(isp, ff["F1n"], ff["F1p"])
        f2_own = jnp.where(isp, ff["F2p"], ff["F2n"]); f2_oth = jnp.where(isp, ff["F2n"], ff["F2p"])
        F1 = (c1 * f1_own - _COUPL2_NC * f1_oth) * vsc
        F2 = (c1 * f2_own - _COUPL2_NC * f2_oth) * vsc
        ca = jnp.where(isp, _COUPL2_NC, -_COUPL2_NC)
        FA = ca * ff["FA"] * asc
        FAP = ca * ff["FAP"] * asc
    else:
        raise ValueError(f"probe must be 'CC', 'EM' or 'NC', got {probe!r}")

    p3_in = p_in_nuc[..., 1:]
    E_in_on = jnp.sqrt(jnp.sum(p3_in ** 2, axis=-1) + _XMN ** 2)
    qE = q[..., 0] + p_in_nuc[..., 0] - E_in_on
    qsh = jnp.concatenate([qE[..., None], q[..., 1:]], axis=-1)
    qlow = qsh * jnp.asarray(_GMETRIC)

    sterm = jnp.einsum('mnij,...n->...mij', _SIGMNJ, qlow)
    vtx = (F1[..., None, None, None] * _GJ
           + (_I * F2 / (2.0 * _XMN))[..., None, None, None] * sterm
           + FA[..., None, None, None] * _GG5J
           + FAP[..., None, None, None] * (qsh / _XMN)[..., :, None, None] * _G5J)

    u_in, _ = _spinors(p3_in, E_in_on)
    _, ubar_out = _spinors(p_out_nuc[..., 1:], p_out_nuc[..., 0])
    ok = (_FF_TCUT + Q2_FF) > 0.0

    if return_structures:
        cpl = {"CC": _COUPL_CC, "EM": _COUPL_EM}.get(probe)
        if cpl is None:
            raise NotImplementedError(f"return_structures supports probe CC/EM only, not {probe!r} "
                                      "(NC has per-structure couplings; add when NC QE is fit)")
        unit_vtx = (cpl * _GJ,
                    cpl * (_I / (2.0 * _XMN)) * sterm,
                    cpl * _GG5J,
                    cpl * (qsh / _XMN)[..., :, None, None] * _G5J)
        Hs = jnp.stack([_ubar_vtx_u(v, u_in, ubar_out) for v in unit_vtx], axis=-3)
        return jnp.where(ok[..., None, None, None], Hs, 0.0)

    H = _ubar_vtx_u(vtx, u_in, ubar_out)
    return jnp.where(ok[..., None, None], H, 0.0)
