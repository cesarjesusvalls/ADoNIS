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

from adonis.reweight.sf_reweight import sf_grids, sf_reweight, removal_from_struck
import adonis.fsi.cascade as _CF
from adonis.core.params import PhysicsParams, nominal_knobs, knob_specs, _EB_EPS



def _ident_rec(n):
    """Identity (a,b,c,Q2) = (1,0,0,1) record: ma_reweight/strength_reweight == 1, gradient 0."""
    return (np.ones(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), np.ones(n, np.float32))


def _qe_records(qa, probe, isp):
    """The QE slice of HV: the exact reduced-quadratic record, one 4x4 M covering every cross term
    between the form-factor atoms (_hv_qe -> qe_reduced_reweight)."""
    from adonis.reweight.reduced_amps2 import build_qe_reduced
    return {"qe_reduced": build_qe_reduced(*qa, probe=probe, is_proton=isp), "qe_probe": probe, "qe_isp": isp}


def _res_records(ra, ip, pp):
    """The RES slice of HV: the exact reduced-quadratic record, a 6x6 M over the {V,A,P}x{rest,wave5}
    atoms (_hv_res -> res_reduced_reweight).  pw_norm has no atoms in this set; releasing it needs
    wave-split atoms here rather than a second reweight path."""
    from adonis.reweight.reduced_amps2 import build_res_reduced
    return {"res_reduced": build_res_reduced(*ra, ip, pp)}


def build_hv_sf(qe, res, sf, probe="CC"):
    """Build the per-channel hard-vertex amps2 records + SF grids/points ONCE (theta-independent).

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
        HV = dict(**_qe_records(qa, "EM", isp),
                  res_ma=_ident_rec(nres), res_pp=_ident_rec(nres), res_delta=_ident_rec(nres))
        return HV, SF

    qa = (qe["k_nu"], qe["k_lep"], qe["p_struck"], qe["p_out"])
    ra = (res["k_nu"], res["k_lep"], res["p_struck"], res["p_N"], res["p_pi"])
    ip, pp = np.asarray(res["ipid"]), np.asarray(res["ppid"])
    HV = dict(**_qe_records(qa, "CC", None), **_res_records(ra, ip, pp))
    return HV, SF


def _hv_qe(k, HV):
    from adonis.reweight.reduced_amps2 import qe_reduced_reweight
    return qe_reduced_reweight(HV["qe_reduced"], k, probe=HV.get("qe_probe", "CC"), is_proton=HV.get("qe_isp"))


def _hv_res(k, HV):
    from adonis.reweight.reduced_amps2 import res_reduced_reweight
    return res_reduced_reweight(HV["res_reduced"], k)


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
