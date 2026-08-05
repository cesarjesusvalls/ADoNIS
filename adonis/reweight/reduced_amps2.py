"""Flexible reduced-quadratic hard-vertex amps2 reweight (see docs/joint_amps2_plan.md).

amps2 is an EXACT quadratic form in the linear form-factor structures ("atoms"):

    amps2(F) = sum_ij F_i F_j M_ij ,   M_ij = Re[ sum_ab (L_a . H_i)* (L_a . H_j) ]

with H_i the per-event UNIT currents (kinematics only; coupling + intrinsic i/2m, q^mu/m folded in) and F_i the
REAL form factors times the dial scales.  M_ij is knob-INDEPENDENT -- assembled once from kinematics -- so the
reweight for ANY simultaneous knob setting is F(knobs)^T M F(knobs) / F0^T M F0: exact, and differentiable to any
order (the only nonlinearity is the smooth dial->F map).  This is the joint replacement for the per-knob product
in reweight_model._hv_qe, which kept only the diagonal M_ii and dropped every cross term.

QE atoms = {F1, F2, FA, FAP} (all four dial-touched -> store the full 4x4 M; no frozen atoms).  RES (larger atom
set, active subset chosen by the dial list) lands here later.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from adonis.channels.probes import probe_spec
from adonis.channels.currents.leptonic import lepton_current
from adonis.channels.currents.dirac import hadron_current_qe_dirac
from adonis.channels.currents.form_factors import nucleon_ff
from adonis.channels.dcc.form_factors import axial_reweight_dipole
from adonis.core.params import nominal_knobs

_METRIC = jnp.array([1.0, -1.0, -1.0, -1.0])


def _q2_mev2(k_nu, k_lep):
    q = jnp.asarray(k_nu) - jnp.asarray(k_lep)
    return jnp.sum(q[..., 1:] ** 2, axis=-1) - q[..., 0] ** 2          # MeV^2


def build_qe_reduced(k_nu, k_lep, p_struck, p_out, probe="CC", is_proton=None):
    """Per-event QE reduced-quadratic record: {M (...,4,4) real symmetric, Q2 (...,) MeV^2}.
    Rebuilt from the SAME kinematics the per-knob records used (k_nu,k_lep,p_struck,p_out) -- no bank regen."""
    kn, km, ps, po = (jnp.asarray(x) for x in (k_nu, k_lep, p_struck, p_out))
    spec = probe_spec(probe)
    L = lepton_current(kn, km, kind=spec.lep_kind)                     # (...,4a,4mu)
    Hs = hadron_current_qe_dirac(kn, km, ps, po, probe=probe, is_proton=is_proton,
                                 return_structures=True)               # (...,4s,4b,4mu)
    Lm = L * _METRIC                                                   # lower mu on L
    LH = jnp.einsum('...am,...sbm->...sab', Lm, Hs)                    # (...,4s,4a,4b) = L_a . H_s,b
    m = jnp.einsum('...sab,...tab->...st', jnp.conj(LH), LH)           # (...,4s,4t) Hermitian
    M = jnp.real(m)                                                    # amps2 = F^T M F  (F real -> Im cancels)
    return {"M": np.asarray(M), "Q2": np.asarray(_q2_mev2(kn, km))}


def qe_F(Q2_mev2, knobs, probe="CC", is_proton=None):
    """The four QE structure amplitudes F=(F1,F2,FA,FAP) at these knobs, from Q2 via nucleon_ff + the M_A dipole.
    Vector block <- vector_strength; Sachs knobs (gep,gen,mu_p->gmp,mu_n->gmn) <- ff_scale (vector-only); axial
    block <- axial_strength * dipole(Q2;M_A_qe).  EM: struck nucleon's own FFs, no axial."""
    Q2 = jnp.asarray(Q2_mev2)
    ff = nucleon_ff(Q2 / 1.0e6, ff_scale={"gep": knobs.gep, "gen": knobs.gen,
                                          "gmp": knobs.mu_p, "gmn": knobs.mu_n})
    dip = axial_reweight_dipole(Q2, knobs.M_A_qe)                      # =1 at M_A=1.0
    vs = jnp.asarray(knobs.vector_strength); asc = jnp.asarray(knobs.axial_strength) * dip
    if probe == "EM":
        isp = jnp.asarray(is_proton)
        f1 = jnp.where(isp, ff["F1p"], ff["F1n"]) * vs
        f2 = jnp.where(isp, ff["F2p"], ff["F2n"]) * vs
        fa = jnp.zeros_like(f1); fap = jnp.zeros_like(f1)
    else:                                                             # CC: isovector difference + axial
        f1 = (ff["F1p"] - ff["F1n"]) * vs
        f2 = (ff["F2p"] - ff["F2n"]) * vs
        fa = ff["FA"] * asc; fap = ff["FAP"] * asc
    return jnp.stack([f1, f2, fa, fap], axis=-1)                       # (...,4)


def qe_reduced_reweight(record, knobs, probe="CC", is_proton=None, nominal=None):
    """Exact per-event QE hard-vertex weight F^T M F / F0^T M F0; pure/differentiable in every QE vertex knob,
    ==1 at nominal, and exact for ARBITRARY simultaneous variation (all cross terms carried by M)."""
    if nominal is None:
        nominal = nominal_knobs()
    M = jnp.asarray(record["M"]); Q2 = jnp.asarray(record["Q2"])
    F = qe_F(Q2, knobs, probe, is_proton)
    F0 = qe_F(Q2, nominal, probe, is_proton)
    num = jnp.einsum('...i,...ij,...j->...', F, M, F)
    den = jnp.einsum('...i,...ij,...j->...', F0, M, F0)
    return jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), 1.0)
