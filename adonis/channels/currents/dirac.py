"""Bit-exact JAX port of the ACHILLES Fortran QE Dirac current (currents_opt_v1.f90 module
dirac_matrices) -- this is what the paper run uses (FortranModel QE_Spectral_Func), NOT the C++
Weyl QESpectral.  Faithful to define_spinors / det_Ja / hadr_curr_matrix_el / current_init_had.

DIRAC basis: g0=diag(I,-I), g^i=offdiag(sigma,-sigma), g5=offdiag(I,I).  Spinors are normalised
with a SINGLE xmn=mN (constants%mqe) regardless of the actual on-shell mass -- the incoming
nucleon is put mN-on-shell by current_init_had, the OUTGOING keeps its (mp-on-shell) energy but
still uses xmn=mN in the normalisation (the ACHILLES hybrid).  ubar.u = 2 xmn for that mass.

Returns the hadron current H (..., 4_spincombo, 4_mu), spin-combo order (i,j)=(00,01,10,11) i.e.
(f1=i//2... ) matching cur(i+2*(j-1)) -> we emit (i1=initial,f1=final) flattened as ACHILLES does.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels import constants as C
from adonis.channels.currents.form_factors import nucleon_ff, _TCUT as _FF_TCUT

_I = 1j
_XMN = C.mN                                  # constants%mqe = 0.5(mp+mn)
_COUPL_CC = C.Vud * C.ee * _I / (C.sw * np.sqrt(2.0) * 2.0)
_COUPL_EM = _I * C.ee          # photon (ACHILLES LeptonicCurrent.cc:121, pid == 22: coupl = i*ee)

# --------------------------------------------------------------------------- NC QE couplings (Z, pid 23)
# ACHILLES LeptonicCurrent.cc:93-96, read directly:
#     coupl1 = (ee*i / (4 * sin2w * cw)) * (0.5 - 2*sin2w)
#     coupl2 = (ee*i / (4 * sw    * cw))
# and the per-nucleon dictionaries at :97-115 attach them as
#     proton : F1p,F2p <- coupl1 ; F1n,F2n <- -coupl2 ; FA <- +coupl2
#     neutron: F1n,F2n <- coupl1 ; F1p,F2p <- -coupl2 ; FA <- -coupl2
# i.e. coupl1 always multiplies the STRUCK nucleon's own F1/F2 and -coupl2 the other's.  ACHILLES
# expresses the isoscalar/isovector recombination as this per-nucleon dictionary, so ADoNIS does the
# same rather than inventing an isospin decomposition.
#
# THE coupl1 QUIRK.  coupl1 has sin2w (= sin^2 theta_W) where coupl2, one line below, has sw
# (= sin theta_W).  coupl2's -1/4 * (g/c_W) isovector partner fixes the overall normalisation at
# g/(2 c_W) = ee/(2*sw*cw), so the Standard-Model coupling is
#     correct coupl1 = ee*i/(2*sw*cw) * (0.5 - 2*sin2w)
# and ACHILLES's is larger by  [1/(2*sin2w)] / [1/sw] = 1/(2*sw) ~ 1.0396.  That "it is a typo" is an
# INFERENCE from the adjacent line, not a fact -- so we do not bet on it.  Both branches exist and the
# bank records which one produced it (GenConfig.achilles_coupl1_quirk -> manifest).
#
#   quirk=False (default) = correct physics.  This is what ADoNIS IS.
#   quirk=True            = ACHILLES verbatim.  Set in the bank config that feeds the comparison
#                           figures, so that comparison is like-for-like.
#
# Written as two explicit branches, NOT as one expression with a substituted denominator: the
# single-expression form `ee*i/(4*X*cw)*(0.5-2*sin2w)` needs X = sw/2 for the correct branch, and
# writing X = sw there is an easy factor-2 error that would make the DEFAULT half the right coupling.
_COUPL1_NC_QUIRK = (C.ee * _I / (4 * C.sin2w * C.cw)) * (0.5 - 2 * C.sin2w)   # ACHILLES verbatim
_COUPL1_NC_TRUE = (C.ee * _I / (2 * C.sw * C.cw)) * (0.5 - 2 * C.sin2w)       # Standard Model
_COUPL2_NC = C.ee * _I / (4 * C.sw * C.cw)                                    # both branches


def nc_coupl1(quirk=False):
    """The NC QE vector coupling; `quirk=True` reproduces the ACHILLES sin2w/sw discrepancy."""
    return _COUPL1_NC_QUIRK if quirk else _COUPL1_NC_TRUE

# Pauli matrices
_SIG = np.stack([
    np.array([[0, 1], [1, 0]], complex),
    np.array([[0, -_I], [_I, 0]], complex),
    np.array([[1, 0], [0, -1]], complex),
])
_ID2 = np.eye(2, dtype=complex)
# Dirac gamma matrices (4x4), index mu=0..3 ; gamma5 separate
_G = np.zeros((4, 4, 4), complex)
_G[0, :2, :2] = _ID2; _G[0, 2:, 2:] = -_ID2                       # gamma^0
for i in range(1, 4):
    _G[i, :2, 2:] = _SIG[i - 1]; _G[i, 2:, :2] = -_SIG[i - 1]     # gamma^i
_G5 = np.zeros((4, 4), complex); _G5[:2, 2:] = _ID2; _G5[2:, :2] = _ID2
_GMETRIC = np.array([1.0, -1.0, -1.0, -1.0])
# sigma^{mu nu} = (i/2)[g^mu,g^nu]
_SIGMN = np.zeros((4, 4, 4, 4), complex)
for mu in range(4):
    for nu in range(4):
        _SIGMN[mu, nu] = 0.5 * _I * (_G[mu] @ _G[nu] - _G[nu] @ _G[mu])
_GG5 = np.einsum('mij,jk->mik', _G, _G5)                          # gamma^mu gamma5

_GJ = jnp.asarray(_G); _G5J = jnp.asarray(_G5); _GG5J = jnp.asarray(_GG5)
_SIGMNJ = jnp.asarray(_SIGMN); _SIGJ = jnp.asarray(_SIG)


def _spinors(p3, E):
    """define_spinors for one nucleon: returns u (2 spin, 4) and ubar (2 spin, 4), xmn=mN.
    p3 (...,3), E (...,) -- E the actual energy used (mN-on-shell for in, mp-on-shell for out)."""
    sigp = jnp.einsum('aij,...a->...ij', _SIGJ, p3)              # sigma.p (...,2,2)
    # GRADIENT PROTECTION: E+_XMN < 0 for unphysical (huge off-shell) outgoing nucleons -> sqrt NaN.
    # Safe sqrt / guarded division keep fwd+bwd finite; physical E>0 (E+_XMN ~ 2 GeV) -> exact no-op.
    sden = E + _XMN
    denom = jnp.where(sden != 0.0, sden, 1.0)[..., None, None]
    cp = jnp.sqrt(jnp.where(sden > 0.0, sden, 1.0))[..., None]
    chi = jnp.asarray(_ID2)                                       # columns = up,down
    # u: upper = chi, lower = sigma.p chi /(E+xmn)
    lower = jnp.einsum('...ij,jk->...ik', sigp / denom, chi)      # (...,2,2col)
    # build (...,2spin,4): spin index = column of chi
    up = jnp.stack([chi[0, 0] + 0 * E, chi[1, 0] + 0 * E], axis=-1)  # placeholder; do explicit below
    # explicit per spin
    def mk(col):
        u_up = jnp.broadcast_to(jnp.asarray(chi[:, col]), p3.shape[:-1] + (2,))
        u_lo = lower[..., col]                                    # (...,2)
        return jnp.concatenate([u_up, u_lo], axis=-1) * cp        # (...,4)
    u = jnp.stack([mk(0), mk(1)], axis=-2)                        # (...,2spin,4)
    # ubar: upper=chi, lower = -chi sigma.p/(E+xmn); built as row -> components, then *cp
    lower_b = jnp.einsum('ij,...jk->...ik', chi, -sigp / denom)   # chi.sigma.p (...,2row,2)
    def mkb(row):
        ub_up = jnp.broadcast_to(jnp.asarray(chi[row, :]), p3.shape[:-1] + (2,))
        ub_lo = lower_b[..., row, :]                              # (...,2)
        return jnp.concatenate([ub_up, ub_lo], axis=-1) * cp
    ubar = jnp.stack([mkb(0), mkb(1)], axis=-2)                   # (...,2spin,4)
    return u, ubar


def hadron_current_qe_dirac(p_in_lep, p_out_lep, p_in_nuc, p_out_nuc, axial_scale=1.0, vector_scale=1.0,
                            ff_scale=None, probe="CC", is_proton=None, coupl1_quirk=False):
    """All (...,4) MeV.  p_in_nuc OFF-SHELL struck nucleon (E=mN-removal); p_out_nuc outgoing
    (mp-on-shell energy).  Returns H (...,4combo,4mu) matching ACHILLES cur(i+2*(j-1)).
    axial_scale (scalar or (...,)) multiplies FA and FAP (FAP ~ FA): the M_A reweight hook;
    vector_scale multiplies F1 and F2 (the overall vector current): the vector-strength reweight hook.
    amps2 is QUADRATIC in EACH, so 3 evals give the exact per-event decomposition.  Default 1.0 = nominal.

    probe:
      "CC" (default) -- nu n -> mu- p.  The ISOVECTOR combination with the W coupling, and the axial
            current.  UNCHANGED, bit-for-bit.
      "EM" -- e N -> e' N  (inclusive (e,e')).  ACHILLES LeptonicCurrent.cc:120 (pid == 22):
                coupl = i*ee
                proton : {F1p, coupl}, {F2p, coupl}    <- NO FA entry
                neutron: {F1n, coupl}, {F2n, coupl}    <- NO FA entry
            so the photon couples to the struck nucleon's OWN form factors (NOT the isovector
            difference F1p - F1n), and CouplingsFF leaves FA = FAP = 0 -- the EM current is purely
            vector.  Protons AND neutrons are both struck (incoherently); `is_proton` (bool, broadcast
            over the event axis) selects which nucleon's form factors each event uses.
            vector_scale still multiplies F1/F2, so the vector-strength / mu_p / mu_n / gep / gen
            reweights work unchanged on an (e,e') sample -- which is exactly why it can break the
            axial-vs-vector degeneracy that a nu-only fit cannot."""
    q = p_in_lep - p_out_lep
    Q2_FF = -(q[..., 0] ** 2 - jnp.sum(q[..., 1:] ** 2, axis=-1)) / 1e6     # GeV^2, ORIGINAL q
    ff = nucleon_ff(Q2_FF, ff_scale=ff_scale)
    asc = jnp.asarray(axial_scale); vsc = jnp.asarray(vector_scale)
    if probe == "CC":
        F1 = _COUPL_CC * (ff["F1p"] - ff["F1n"]) * vsc; F2 = _COUPL_CC * (ff["F2p"] - ff["F2n"]) * vsc
        FA = _COUPL_CC * ff["FA"] * asc;                FAP = _COUPL_CC * ff["FAP"] * asc
    elif probe == "EM":
        if is_proton is None:
            raise ValueError("probe='EM' needs is_proton (the struck nucleon species, per event)")
        isp = jnp.asarray(is_proton)
        f1 = jnp.where(isp, ff["F1p"], ff["F1n"])          # the struck nucleon's OWN form factors
        f2 = jnp.where(isp, ff["F2p"], ff["F2n"])
        F1 = _COUPL_EM * f1 * vsc; F2 = _COUPL_EM * f2 * vsc
        FA = jnp.zeros_like(F1); FAP = jnp.zeros_like(F1)  # photon: no axial current
    elif probe == "NC":
        if is_proton is None:
            raise ValueError("probe='NC' needs is_proton: the NC QE couplings are PER NUCLEON "
                             "(LeptonicCurrent.cc:97-115), unlike CC's single isovector combination")
        isp = jnp.asarray(is_proton)
        c1 = nc_coupl1(coupl1_quirk)
        # struck nucleon's own F1/F2 carry coupl1; the OTHER nucleon's carry -coupl2
        f1_own = jnp.where(isp, ff["F1p"], ff["F1n"]); f1_oth = jnp.where(isp, ff["F1n"], ff["F1p"])
        f2_own = jnp.where(isp, ff["F2p"], ff["F2n"]); f2_oth = jnp.where(isp, ff["F2n"], ff["F2p"])
        F1 = (c1 * f1_own - _COUPL2_NC * f1_oth) * vsc
        F2 = (c1 * f2_own - _COUPL2_NC * f2_oth) * vsc
        # FA <- +coupl2 on a proton, -coupl2 on a neutron (:101 / :112)
        ca = jnp.where(isp, _COUPL2_NC, -_COUPL2_NC)
        FA = ca * ff["FA"] * asc
        # NO strange form factors.  ACHILLES computes FormFactors::FAs (FormFactor.cc:92,123) and
        # NEVER consumes it: FormFactorInfo::Type (FormFactor.hh:22-48) has no strange entry, so
        # CouplingsFF cannot dispatch on one.  Adding F1s/F2s/G_A^s would make ADoNIS more physically
        # complete and fail every gate here by construction, because every gate is an ACHILLES
        # comparison.  tests/test_nc_qe.py pins this omission with the enum citation.
        #
        # FAP (induced pseudoscalar): carried with the SAME +-coupl2 as FA, mirroring the validated CC
        # branch above, which pairs FAP with FA on one coupling.
        #
        # OPEN ITEM, recorded rather than guessed.  `FAP` appears NOWHERE in LeptonicCurrent.cc -- not
        # in the NC dictionary (:97-115) and not in the CC one (:82-92).  That is NOT evidence that
        # the term is absent from the physics: the paper runs use `FortranModel QE_Spectral_Func`
        # (currents_opt_v1.f90), which is what this module ports and which builds its own current;
        # LeptonicCurrent.cc's dictionary feeds the C++ backend.  So the Fortran, not this dictionary,
        # is the authority on whether NC QE carries an induced pseudoscalar -- and CC's FAP treatment
        # here is bit-validated, so mirroring it is the defensible default until the P6 free-nucleon
        # oracle gate settles it.  Flagged in docs/nc_implementation_plan.md.
        #
        # And note it DOES contribute: the term rides on `qsh`, the DE FOREST-SHIFTED transfer
        # (qsh[0] = q[0] + p_in_nuc[0] - E_in_on), not on the leptonic q.  The tempting argument
        # "a massless neutrino gives q.j_lep = 0, so the q^mu term drops" is therefore FALSE here --
        # measured, a x137 FAP moves NC amps2 by ~600x.  test_nc_qe.py pins that it matters, so the
        # open item cannot be quietly forgotten.
        FAP = ca * ff["FAP"] * asc
    else:
        raise ValueError(f"probe must be 'CC', 'EM' or 'NC', got {probe!r}")

    # current_init_had: q(1)=omega+E_in; p1 on-shell at xmn; q(1)-=p1(1)
    p3_in = p_in_nuc[..., 1:]
    E_in_on = jnp.sqrt(jnp.sum(p3_in ** 2, axis=-1) + _XMN ** 2)
    qE = q[..., 0] + p_in_nuc[..., 0] - E_in_on
    qsh = jnp.concatenate([qE[..., None], q[..., 1:]], axis=-1)
    qlow = qsh * jnp.asarray(_GMETRIC)                            # g(nu,nu) q(nu)

    # det_Ja vertex J_1[mu] (...,4mu,4,4)
    sterm = jnp.einsum('mnij,...n->...mij', _SIGMNJ, qlow)        # sigma^{mu nu} q_nu
    vtx = (F1[..., None, None, None] * _GJ
           + (_I * F2 / (2.0 * _XMN))[..., None, None, None] * sterm
           + FA[..., None, None, None] * _GG5J
           + FAP[..., None, None, None] * (qsh / _XMN)[..., :, None, None] * _G5J)

    u_in, _ = _spinors(p3_in, E_in_on)                           # incoming: mN-on-shell energy
    _, ubar_out = _spinors(p_out_nuc[..., 1:], p_out_nuc[..., 0])  # outgoing: actual (mp) energy
    # H[mu]_(i1 init, f1 final) = ubar_out[f1] . J_1[mu] . u_in[i1]
    out = []
    for i1 in range(2):          # initial spin
        Vu = jnp.einsum('...mij,...j->...mi', vtx, u_in[..., i1, :])     # (...,4mu,4)
        for f1 in range(2):      # final spin
            out.append(jnp.einsum('...i,...mi->...m', ubar_out[..., f1, :], Vu))
    # ACHILLES cur(i+2*(j-1),:) = J_mu(j=final, i=initial); order loop i1(init) outer, f1(final) inner
    H = jnp.stack(out, axis=-2)                                  # (...,4combo,4mu)
    # Zero the unphysical (Q2 < -TCUT) events so amps2 -> 0, matching ACHILLES's post-hoc
    # `if(isnan(amps2)) amps2=0` (XSecBackend.cc:156).  H is already finite (safe sqrts above), so this
    # mask is gradient-clean; for every physical Q2>=0 the condition is True -> exact no-op.
    ok = (_FF_TCUT + Q2_FF) > 0.0
    return jnp.where(ok[..., None, None], H, 0.0)
