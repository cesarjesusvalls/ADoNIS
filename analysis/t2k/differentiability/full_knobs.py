"""End-to-end DIFFERENTIABLE T2K CC0pi dsigma/dx in ALL the exposed knobs (differentiable_knobs.md).

Every knob enters as an exact per-event REWEIGHT of a frozen pool walk + frozen primary sample:
  * hard vertex (amps2 quadratic): M_A, axial_strength (QE+RES), vector_strength, mu_p, mu_n, gep, gen (QE),
    res_axial_strength (C5A), pw_norm[14], pion_pole (RES) -> 3-eval (a,b,c) records + strength/ma reweight.
  * FSI (kind-1 record): s_pi_abs(=sabs), s_piN_elastic, s_piN_cex, s_conv, s_NN_elastic{pp,pn,nn},
    s_NN_inelastic{pp,pn,nn}, f_NN_cex -> pool_fsi_reweight granular keywords.
  * spectral function (density ratio): kF_sf, Eb_shift, sf_norm, src_tail -> sf_reweight on struck (|p|,E).
  * norms: qe_norm, res_norm (channel-level multipliers).

Per-knob reweights COMPOSE multiplicatively: EXACT to first order at nominal (so dH/dknob is exact -- the
Jacobian/sensitivity the fit uses); finite-pull joint accuracy across knobs needs a joint amps2 decomposition
(documented; not needed for gradients).  model_hist_full(knobs, R, HV, SF) is jit+grad-able in every knob.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from adonis.analysis.ma_records import (build_qe_ma_records, build_res_ma_records, build_qe_vector_records,
                                        build_qe_ff_records, build_res_pw_records, build_res_pionpole_records,
                                        ma_reweight, strength_reweight)
from adonis.analysis.sf_reweight import sf_grids, sf_reweight, removal_from_struck
import adonis.fsi.cascade_full as _CF

_NPW = 14
# Eb_shift nominal: a negligible epsilon (NOT 0).  The SF removal-energy reweight S(p,E-Eb)/S(p,E) has a
# coherent non-differentiable corner at Eb=0 (every event sits at the ratio=1 symmetric point, and the
# down-shift direction hits the clamped low-E rise of the heavy-tailed SF).  Anchoring the nominal at a
# small positive Eb puts the gradient on the smooth one-sided branch.  1e-2 MeV: forward bias vs ACHILLES
# (Eb=0) is 6e-5 (negligible), and the autodiff gradient closes vs FD (AD/FD~1.00); 1e-4 was too small
# (still inside the corner, AD/FD~1.14).  sf_reweight clamps Eb<0 -> 0 (one-sided knob).
_EB_EPS = 1e-2


def nominal_knobs():
    # M_A is split per channel: M_A_qe (QE dipole mass) and M_A_res (RES/DCC axial dipole mass) are
    # INDEPENDENT knobs (each reweights only its own channel's amps2 record); both nominal 1.0.
    return dict(M_A_qe=1.0, M_A_res=1.0, axial_strength=1.0, vector_strength=1.0, mu_p=1.0, mu_n=1.0, gep=1.0, gen=1.0,
                res_axial_strength=1.0, pw_norm=tuple([0.0] * _NPW), pion_pole=1.0,
                sabs=1.0, sscat=1.0, s_piN_elastic=1.0, s_piN_cex=1.0, s_conv=1.0,
                s_NN_elastic=(1., 1., 1.), s_NN_inelastic=(1., 1., 1.), f_NN_cex=0.5,
                kF_sf=1.0, Eb_shift=_EB_EPS, sf_norm=1.0, src_tail=1.0, qe_norm=1.0, res_norm=1.0)


def build_hv_sf(qe, res, sf, with_pw=True):
    """Build the per-channel hard-vertex amps2 records + SF grids/points ONCE (theta-independent).
    with_pw=False skips the 14 DCC partial-wave records (the dominant build cost) -> pw_norm has no effect."""
    qa = (qe["k_nu"], qe["k_mu"], qe["p_struck"], qe["p_out"])
    ra = (res["k_nu"], res["k_mu"], res["p_struck"], res["p_N"], res["p_pi"])
    ip, pp = np.asarray(res["ipid"]), np.asarray(res["ppid"])
    HV = dict(
        qe_ma=build_qe_ma_records(*qa), qe_vec=build_qe_vector_records(*qa),
        qe_gmp=build_qe_ff_records(*qa, "gmp"), qe_gmn=build_qe_ff_records(*qa, "gmn"),
        qe_gep=build_qe_ff_records(*qa, "gep"), qe_gen=build_qe_ff_records(*qa, "gen"),
        res_ma=build_res_ma_records(*ra, ip, pp), res_pp=build_res_pionpole_records(*ra, ip, pp),
        res_pw=[build_res_pw_records(*ra, ip, pp, w) for w in range(_NPW)] if with_pw else None)
    grids = sf_grids(sf)
    SF = dict(grids=grids,
              qe_pmag=removal_from_struck(qe["p_struck"])[0], qe_erem=removal_from_struck(qe["p_struck"])[1],
              res_pmag=removal_from_struck(res["p_struck"])[0], res_erem=removal_from_struck(res["p_struck"])[1])
    return HV, SF


def _hv_qe(k, HV):
    return (ma_reweight(HV["qe_ma"], k["M_A_qe"]) * strength_reweight(HV["qe_ma"], k["axial_strength"])
            * strength_reweight(HV["qe_vec"], k["vector_strength"]) * strength_reweight(HV["qe_gmp"], k["mu_p"])
            * strength_reweight(HV["qe_gmn"], k["mu_n"]) * strength_reweight(HV["qe_gep"], k["gep"])
            * strength_reweight(HV["qe_gen"], k["gen"]))


def _hv_res(k, HV):
    w = (ma_reweight(HV["res_ma"], k["M_A_res"]) * strength_reweight(HV["res_ma"], k["res_axial_strength"])
         * strength_reweight(HV["res_pp"], k["pion_pole"]))
    if HV.get("res_pw") is None:                              # pw records skipped (with_pw=False) -> no-op
        return w
    pw = jnp.asarray(k["pw_norm"])
    for i in range(_NPW):
        w = w * strength_reweight(HV["res_pw"][i], 1.0 + pw[i])
    return w


def _fsi(rec, k):
    return _CF.pool_fsi_reweight(rec, k["sabs"], k["sscat"], s_piN_elastic=k["s_piN_elastic"],
                                 s_piN_cex=k["s_piN_cex"], s_conv=k["s_conv"], s_NN_elastic=k["s_NN_elastic"],
                                 s_NN_inelastic=k["s_NN_inelastic"], f_NN_cex=k["f_NN_cex"])


def model_hist_full(knobs, R, HV, SF, edges, conv=1.0):
    """Differentiable CC0pi dsigma/dx histogram in ALL knobs.  R: build_replica output (FSI records +
    keep/idx + w0); HV/SF: build_hv_sf output; edges: bin edges."""
    k = knobs
    q_sf = sf_reweight(SF["grids"], SF["qe_pmag"], SF["qe_erem"], kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"],
                       sf_norm=k["sf_norm"], src_tail=k["src_tail"])
    r_sf = sf_reweight(SF["grids"], SF["res_pmag"], SF["res_erem"], kF_sf=k["kF_sf"], Eb_shift=k["Eb_shift"],
                       sf_norm=k["sf_norm"], src_tail=k["src_tail"])
    q_w = R["q_w0"] * k["qe_norm"] * _hv_qe(k, HV) * _fsi(R["q_rec"], k) * q_sf
    r_w = R["r_w0"] * k["res_norm"] * _hv_res(k, HV) * _fsi(R["r_rec"], k) * r_sf
    nb = len(edges) - 1
    h = (jax.ops.segment_sum(q_w * R["q_keep"], R["q_idx"], num_segments=nb)
         + jax.ops.segment_sum(r_w * R["r_keep"], R["r_idx"], num_segments=nb))
    return h / jnp.diff(jnp.asarray(edges)) * conv
