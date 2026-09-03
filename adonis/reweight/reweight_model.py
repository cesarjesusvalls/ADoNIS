"""End-to-end DIFFERENTIABLE T2K CC0pi dsigma/dx in ALL the exposed knobs.

Every knob enters as an exact per-event REWEIGHT of a frozen pool walk + frozen primary sample:
  * hard vertex (amps2 quadratic): M_A, axial_strength (QE+RES), vector_strength, mu_p, mu_n, gep, gen (QE),
    res_axial_strength (C5A), pw_norm[14], pion_pole (RES) -> 3-eval (a,b,c) records + strength/ma reweight.
  * FSI (kind-1 record): s_pi_abs(=sabs), s_piN_elastic, s_piN_cex, s_conv, s_NN_elastic{pp,pn,nn},
    s_NN_inelastic{pp,pn,nn}, f_NN_cex -> pool_fsi_reweight granular keywords.
  * spectral function (density ratio): kF_sf, Eb_shift, sf_norm, src_tail -> sf_reweight on struck (|p|,E).
  * norms: qe_norm, res_norm (channel-level multipliers).

Per-knob reweights COMPOSE multiplicatively: exact to first order at nominal, so dH/dknob is exact;
finite-pull joint accuracy across knobs needs the joint amps2 decomposition (reduced_amps2.py).
model_hist_full(knobs, R, HV, SF) is jit+grad-able in every knob.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from adonis.reweight.amps2_records import (build_qe_ma_records, build_res_ma_records, build_qe_vector_records,
                                        build_qe_ff_records, build_res_pw_records, build_res_pionpole_records,
                                        ma_reweight, strength_reweight)
from adonis.reweight.sf_reweight import sf_grids, sf_reweight, removal_from_struck
import adonis.fsi.cascade as _CF
from adonis.core.params import PhysicsParams, nominal_knobs, knob_specs, _NPW, _EB_EPS

_DELTA_WAVE = 5


def _ident_rec(n):
    """Identity (a,b,c,Q2) = (1,0,0,1) record: ma_reweight/strength_reweight == 1, gradient 0."""
    return (np.ones(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), np.ones(n, np.float32))


def _qe_hv(qa, **kw):
    """The six QE hard-vertex records (axial M_A, vector, and the four Sachs FFs) from the (k_in, k_out,
    p_struck, p_out) tuple `qa`.  kw carries {probe,is_proton} for the EM (e,e') photon records; empty
    for CC (record builders default to probe="CC")."""
    return dict(qe_ma=build_qe_ma_records(*qa, **kw), qe_vec=build_qe_vector_records(*qa, **kw),
                qe_gmp=build_qe_ff_records(*qa, "gmp", **kw), qe_gmn=build_qe_ff_records(*qa, "gmn", **kw),
                qe_gep=build_qe_ff_records(*qa, "gep", **kw), qe_gen=build_qe_ff_records(*qa, "gen", **kw))


def _qe_records(qa, probe, isp, qe_joint):
    """The QE slice of HV.  qe_joint=True (default): the exact reduced-quadratic record (one 4x4 M
    covering all cross terms; _hv_qe -> qe_reduced_reweight).  qe_joint=False: per-knob (a,b,c) records
    taken as a product, which is exact only to first order; it reads banks that carry no M."""
    if qe_joint:
        from adonis.reweight.reduced_amps2 import build_qe_reduced
        return {"qe_reduced": build_qe_reduced(*qa, probe=probe, is_proton=isp), "qe_probe": probe, "qe_isp": isp}
    return _qe_hv(qa, probe=probe, is_proton=isp)


def _res_records(ra, ip, pp, with_pw, res_joint):
    """The RES slice of HV.  res_joint=True (default): the exact reduced-quadratic record (6x6 M over
    the {V,A,P}x{rest,wave5} atoms; _hv_res -> res_reduced_reweight).  res_joint=False: per-knob records
    taken as a product; the dormant pw_norm loop lives on that branch, so the reduced form omits it."""
    if res_joint:
        from adonis.reweight.reduced_amps2 import build_res_reduced
        return {"res_reduced": build_res_reduced(*ra, ip, pp)}
    return dict(res_ma=build_res_ma_records(*ra, ip, pp), res_pp=build_res_pionpole_records(*ra, ip, pp),
                res_delta=build_res_pw_records(*ra, ip, pp, _DELTA_WAVE),
                res_pw=[build_res_pw_records(*ra, ip, pp, w) for w in range(_NPW)] if with_pw else None)


def build_hv_sf(qe, res, sf, with_pw=True, probe="CC", qe_joint=True, res_joint=True):
    """Build the per-channel hard-vertex amps2 records + SF grids/points ONCE (theta-independent).
    with_pw=False skips the 14 DCC partial-wave records -> pw_norm has no effect.

    probe="EM" builds the (e,e') photon records instead: the QE vector + Sachs-FF records carry real
    gradients (with per-event is_proton), the QE axial record auto-collapses to identity (no photon
    axial), and the RES hard-vertex records are identity (EM-Delta / delta_strength not yet
    implemented); the SF reweight still applies to both channels.  qe uses k_e/k_lep/p_out/is_p; res
    uses p_struck."""
    qe_pmag, qe_erem = removal_from_struck(qe["p_struck"])
    res_pmag, res_erem = removal_from_struck(res["p_struck"])
    SF = dict(grids=sf_grids(sf), qe_pmag=qe_pmag, qe_erem=qe_erem, res_pmag=res_pmag, res_erem=res_erem)

    if probe == "EM":
        qa = (qe["k_e"], qe["k_lep"], qe["p_struck"], qe["p_out"]); isp = np.asarray(qe["is_p"])
        nres = len(np.asarray(res["p_N"]))
        HV = dict(**_qe_records(qa, "EM", isp, qe_joint),
                  res_ma=_ident_rec(nres), res_pp=_ident_rec(nres), res_delta=_ident_rec(nres), res_pw=None)
        return HV, SF

    qa = (qe["k_nu"], qe["k_lep"], qe["p_struck"], qe["p_out"])
    ra = (res["k_nu"], res["k_lep"], res["p_struck"], res["p_N"], res["p_pi"])
    ip, pp = np.asarray(res["ipid"]), np.asarray(res["ppid"])
    HV = dict(**_qe_records(qa, "CC", None, qe_joint), **_res_records(ra, ip, pp, with_pw, res_joint))
    return HV, SF


def _hv_qe(k, HV):
    if "qe_reduced" in HV:
        from adonis.reweight.reduced_amps2 import qe_reduced_reweight
        return qe_reduced_reweight(HV["qe_reduced"], k, probe=HV.get("qe_probe", "CC"), is_proton=HV.get("qe_isp"))
    return (ma_reweight(HV["qe_ma"], k.M_A_qe) * strength_reweight(HV["qe_ma"], k.axial_strength)
            * strength_reweight(HV["qe_vec"], k.vector_strength) * strength_reweight(HV["qe_gmp"], k.mu_p)
            * strength_reweight(HV["qe_gmn"], k.mu_n) * strength_reweight(HV["qe_gep"], k.gep)
            * strength_reweight(HV["qe_gen"], k.gen))


def _hv_res(k, HV):
    if "res_reduced" in HV:
        from adonis.reweight.reduced_amps2 import res_reduced_reweight
        return res_reduced_reweight(HV["res_reduced"], k)
    w = (ma_reweight(HV["res_ma"], k.M_A_res) * strength_reweight(HV["res_ma"], k.res_axial_strength)
         * strength_reweight(HV["res_pp"], k.pion_pole)
         * strength_reweight(HV["res_delta"], k.delta_strength))
    if HV.get("res_pw") is None:
        return w
    pw = jnp.asarray(k.pw_norm)
    for i in range(_NPW):
        w = w * strength_reweight(HV["res_pw"][i], 1.0 + pw[i])
    return w


def _fsi(rec, k):
    return _CF.pool_fsi_reweight(rec, k.sabs, k.sscat, s_piN_elastic=k.s_piN_elastic,
                                 s_piN_cex=k.s_piN_cex, s_conv=k.s_conv, s_NN_elastic=k.s_NN_elastic,
                                 s_NN_inelastic=k.s_NN_inelastic, f_NN_cex=k.f_NN_cex)


def model_hist_full(knobs, R, HV, SF, edges, conv=1.0):
    """Differentiable CC0pi dsigma/dx histogram in ALL knobs.  R: build_replica output (FSI records +
    keep/idx + w0); HV/SF: build_hv_sf output; edges: bin edges."""
    k = knobs
    q_sf = sf_reweight(SF["grids"], SF["qe_pmag"], SF["qe_erem"], kF_sf=k.kF_sf, Eb_shift=k.Eb_shift,
                       sf_norm=k.sf_norm, src_tail=k.src_tail)
    r_sf = sf_reweight(SF["grids"], SF["res_pmag"], SF["res_erem"], kF_sf=k.kF_sf, Eb_shift=k.Eb_shift,
                       sf_norm=k.sf_norm, src_tail=k.src_tail)
    q_w = R["q_w0"] * k.qe_norm * _hv_qe(k, HV) * _fsi(R["q_rec"], k) * q_sf
    r_w = R["r_w0"] * k.res_norm * _hv_res(k, HV) * _fsi(R["r_rec"], k) * r_sf
    nb = len(edges) - 1
    h = (jax.ops.segment_sum(q_w * R["q_keep"], R["q_idx"], num_segments=nb)
         + jax.ops.segment_sum(r_w * R["r_keep"], R["r_idx"], num_segments=nb))
    return h / jnp.diff(jnp.asarray(edges)) * conv
