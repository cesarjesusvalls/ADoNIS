"""Reduced-quadratic hard-vertex amps2 reweight (see docs/joint_amps2_plan.md).

amps2 is an EXACT quadratic form in the linear form-factor structures ("atoms"):

    amps2(F) = sum_ij F_i F_j M_ij ,   M_ij = Re[ sum_ab (L_a . H_i)* (L_a . H_j) ]

with H_i the per-event UNIT currents (kinematics only; coupling + intrinsic i/2m, q^mu/m folded in) and
F_i the REAL form factors times the dial scales.  M_ij is knob-INDEPENDENT -- assembled once from
kinematics -- so the reweight for ANY simultaneous knob setting is F(knobs)^T M F(knobs) / F0^T M F0:
exact, and differentiable to any order (the only nonlinearity is the smooth dial->F map).  This
supersedes a per-knob product that kept only the diagonal M_ii and dropped every cross term.

QE atoms = {F1, F2, FA, FAP} (all four dial-touched -> store the full 4x4 M; no frozen atoms).  RES
(larger atom set, active subset chosen by the dial list) below.
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
    return jnp.sum(q[..., 1:] ** 2, axis=-1) - q[..., 0] ** 2


def build_qe_reduced(k_nu, k_lep, p_struck, p_out, probe="CC", is_proton=None):
    """Per-event QE reduced-quadratic record: {M (...,4,4) real symmetric, Q2 (...,) MeV^2}.
    Rebuilt from the SAME kinematics the per-knob records used (k_nu,k_lep,p_struck,p_out) -- no bank regen."""
    kn, km, ps, po = (jnp.asarray(x) for x in (k_nu, k_lep, p_struck, p_out))
    spec = probe_spec(probe)
    L = lepton_current(kn, km, kind=spec.lep_kind)
    Hs = hadron_current_qe_dirac(kn, km, ps, po, probe=probe, is_proton=is_proton,
                                 return_structures=True)
    Lm = L * _METRIC
    LH = jnp.einsum('...am,...sbm->...sab', Lm, Hs)
    m = jnp.einsum('...sab,...tab->...st', jnp.conj(LH), LH)
    M = jnp.real(m)
    Q2 = _q2_mev2(kn, km)
    M = jnp.where((Q2 > 0)[..., None, None], M, 0.0)
    return {"M": np.asarray(M), "Q2": np.asarray(Q2)}


def qe_F(Q2_mev2, knobs, probe="CC", is_proton=None):
    """The four QE structure amplitudes F=(F1,F2,FA,FAP) at these knobs, from Q2 via nucleon_ff + the M_A dipole.
    Vector block <- vector_strength; Sachs knobs (gep,gen,mu_p->gmp,mu_n->gmn) <- ff_scale (vector-only); axial
    block <- axial_strength * dipole(Q2;M_A_qe).  EM: struck nucleon's own FFs, no axial."""
    Q2 = jnp.asarray(Q2_mev2)
    ff = nucleon_ff(Q2 / 1.0e6, ff_scale={"gep": knobs.gep, "gen": knobs.gen,
                                          "gmp": knobs.mu_p, "gmn": knobs.mu_n})
    dip = axial_reweight_dipole(Q2, knobs.M_A_qe)
    vs = jnp.asarray(knobs.vector_strength); asc = jnp.asarray(knobs.axial_strength) * dip
    if probe == "EM":
        isp = jnp.asarray(is_proton)
        f1 = jnp.where(isp, ff["F1p"], ff["F1n"]) * vs
        f2 = jnp.where(isp, ff["F2p"], ff["F2n"]) * vs
        fa = jnp.zeros_like(f1); fap = jnp.zeros_like(f1)
    else:
        f1 = (ff["F1p"] - ff["F1n"]) * vs
        f2 = (ff["F2p"] - ff["F2n"]) * vs
        fa = ff["FA"] * asc; fap = ff["FAP"] * asc
    return jnp.stack([f1, f2, fa, fap], axis=-1)


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


_RES_ATOMS = ("V_rest", "A_rest", "P_rest", "V_w5", "A_w5", "P_w5")
_RES_ITIZ = {(2112, 211): -1, (2112, 111): -1, (2212, 211): +1}


def build_res_reduced(k_nu, k_lep, p_struck, p_N, p_pi, ipid, ppid):
    """Per-event RES reduced record {M (...,6,6) real symmetric, Q2 (...,)} over the 6 atoms (channel loop)."""
    from adonis.channels.dcc import current as dcc
    metric = np.asarray(dcc._METRIC)
    n = len(np.asarray(k_nu))
    M = np.zeros((n, 6, 6)); Q2 = np.ones(n)
    ipid = np.asarray(ipid); ppid = np.asarray(ppid)
    for (ip, pp), itiz in _RES_ITIZ.items():
        mm = (ipid == ip) & (ppid == pp)
        if not mm.any():
            continue
        args = [np.asarray(x)[mm] for x in (k_nu, k_lep, p_struck, p_N, p_pi)]
        st = dcc.exclusive_amps2_batch(*args, itiz, pp, return_structures=True)
        zj, L, gate, norm = st["zj"], st["L"], st["gate"], st["norm"]
        LH = {s: np.einsum('ncm,nbm,m->ncb', L, zj[s], metric) for s in _RES_ATOMS}
        Mm = np.zeros((len(args[0]), 6, 6))
        for i, s in enumerate(_RES_ATOMS):
            for j, t in enumerate(_RES_ATOMS):
                Mm[:, i, j] = np.real(np.sum(np.conj(LH[s]) * LH[t], axis=(1, 2))) / norm
        M[mm] = np.where(gate[:, None, None], np.nan_to_num(Mm), 0.0)
        Q2[mm] = st["Q2"]
    return {"M": np.asarray(M, np.float64), "Q2": np.asarray(Q2, np.float32)}


def res_g(Q2_mev2, knobs):
    """The 6 atom coefficients g=(V,A,P)x(rest,w5).  r_axial = dipole(Q2;M_A_res)*res_axial_strength (scales the
    axial + pole blocks); pion_pole scales the pole; delta_strength scales wave 5.  g=1 everywhere at nominal."""
    r_ax = axial_reweight_dipole(jnp.asarray(Q2_mev2), knobs.M_A_res) * jnp.asarray(knobs.res_axial_strength)
    pp = jnp.asarray(knobs.pion_pole); dl = jnp.asarray(knobs.delta_strength)
    one = jnp.ones_like(r_ax)
    return jnp.stack([one, r_ax, r_ax * pp, dl * one, dl * r_ax, dl * r_ax * pp], axis=-1)


def res_reduced_reweight(record, knobs, nominal=None):
    """Exact per-event RES hard-vertex weight g^T M g / g0^T M g0; pure/differentiable in M_A_res,
    res_axial_strength, pion_pole, delta_strength, and exact for arbitrary simultaneous variation."""
    if nominal is None:
        nominal = nominal_knobs()
    M = jnp.asarray(record["M"]); Q2 = jnp.asarray(record["Q2"])
    g, g0 = res_g(Q2, knobs), res_g(Q2, nominal)
    num = jnp.einsum('...i,...ij,...j->...', g, M, g)
    den = jnp.einsum('...i,...ij,...j->...', g0, M, g0)
    return jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), 1.0)
