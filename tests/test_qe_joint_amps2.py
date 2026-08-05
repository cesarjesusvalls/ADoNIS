"""Gates for the flexible reduced-quadratic QE hard-vertex reweight (adonis.reweight.reduced_amps2).

The point of the fix: amps2 is an EXACT quadratic form F^T M F over the four form-factor structures, so the joint
reweight is exact for ARBITRARY simultaneous knob variation -- unlike the per-knob product (reweight_model._hv_qe),
which drops the cross terms and has identically-zero mixed second derivatives.  Gates:
  1. exactness       -- F^T M F == direct me_cross_section amps2 for random (vector, axial, Sachs) scales;
  2. nominal identity-- reweight == 1 at nominal;
  3. joint match     -- multiple knobs moved TOGETHER == direct amps2 ratio (the product fails this);
  4. M_A match       -- M_A acts as the per-event dipole axial scale;
  5. mixed 2nd deriv -- d2 log w / d(vector) d(axial) is nonzero and equals the direct-amps2 value.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.channels import qe as qe_xsec
from adonis.channels.currents.matrix_element import me_cross_section
from adonis.channels.currents.form_factors import nucleon_ff
from adonis.channels.dcc.form_factors import axial_reweight_dipole
from adonis.core.params import nominal_knobs
from adonis.reweight.reduced_amps2 import build_qe_reduced, qe_F, qe_reduced_reweight

N = 2000
_QE = qe_xsec.sample_importance(N, seed=5)
_KN, _KM, _PS, _PO = (jnp.asarray(_QE[k]) for k in ("k_nu", "k_lep", "p_struck", "p_out"))
_REC = build_qe_reduced(_KN, _KM, _PS, _PO)
_M, _Q2 = np.asarray(_REC["M"]), np.asarray(_REC["Q2"])
_A_NOM = np.asarray(me_cross_section(_KN, _KM, _PS, _PO)["amps2"])
# shared validity: physical amps2 AND q2>0 (the legacy records' mask, which build_qe_reduced matches).
# The near-threshold q2<=0 events get hard-vertex reweight==1 in BOTH paths (placeholder), so exclude them.
_VALID = np.isfinite(_A_NOM) & (_A_NOM > 0) & (_Q2 > 0)


def _FtMF(F):
    return np.asarray(jnp.einsum('ni,nij,nj->n', F, jnp.asarray(_M), F))


def _F_from_scales(v, a, fs):
    ff = nucleon_ff(jnp.asarray(_Q2) / 1.0e6, ff_scale=fs)
    return jnp.stack([(ff["F1p"] - ff["F1n"]) * v, (ff["F2p"] - ff["F2n"]) * v,
                      ff["FA"] * a, ff["FAP"] * a], axis=-1)


def test_qe_mij_exact():
    """F^T M F reproduces the true amps2 for arbitrary (vector, axial, Sachs) -- M carries every cross term."""
    for (v, a, fs) in [(1.0, 1.0, {}), (1.3, 0.7, {}), (0.8, 1.2, {"gep": 1.1, "gmp": 0.9}),
                       (1.25, 0.85, {"gep": 1.15, "gen": 0.8, "gmp": 1.1, "gmn": 0.95})]:
        lhs = _FtMF(_F_from_scales(v, a, fs))
        rhs = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, vector_scale=v, axial_scale=a,
                                          ff_scale=fs or None)["amps2"])
        assert np.allclose(lhs[_VALID], rhs[_VALID], rtol=1e-9), (v, a, fs)


def test_qe_reduced_nominal_identity():
    w = np.asarray(qe_reduced_reweight(_REC, nominal_knobs()))
    assert np.allclose(w[_VALID], 1.0, atol=1e-12)


def test_qe_reduced_joint_matches_direct():
    """Several knobs moved TOGETHER: w * amps2(nom) == amps2(joint).  M_A=1 so it maps to me_cross_section."""
    nom = nominal_knobs()
    k = nom._replace(vector_strength=1.2, axial_strength=0.85, gep=1.1, mu_p=0.95, M_A_qe=1.0)
    w = np.asarray(qe_reduced_reweight(_REC, k))
    direct = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, vector_scale=1.2, axial_scale=0.85,
                                         ff_scale={"gep": 1.1, "gmp": 0.95})["amps2"])
    assert np.allclose((w * _A_NOM)[_VALID], direct[_VALID], rtol=1e-9)


def test_qe_reduced_MA_matches_direct():
    """M_A alone == direct amps2 at axial_scale = dipole(Q2; M_A) (per-event)."""
    nom = nominal_knobs()
    dip = np.asarray(axial_reweight_dipole(jnp.asarray(_Q2), 1.05))
    w = np.asarray(qe_reduced_reweight(_REC, nom._replace(M_A_qe=1.05)))
    direct = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, axial_scale=dip)["amps2"])
    assert np.allclose((w * _A_NOM)[_VALID], direct[_VALID], rtol=1e-9)


def test_qe_reduced_mixed_second_derivative_nonzero_and_correct():
    """d2 log(sum w) / d(vector) d(axial) at nominal: nonzero (the whole point) and == the direct-amps2 value."""
    nom = nominal_knobs()
    keep = jnp.asarray(_VALID)

    def logsumw(vs, as_):
        w = qe_reduced_reweight(_REC, nom._replace(vector_strength=vs, axial_strength=as_))
        return jnp.log(jnp.sum(jnp.where(keep, w, 0.0)))
    d2_ad = float(jax.grad(jax.grad(logsumw, argnums=0), argnums=1)(1.0, 1.0))
    # direct finite-difference of the same quantity through me_cross_section (independent of the M path)
    def logsumw_direct(vs, as_):
        a = jnp.asarray(me_cross_section(_KN, _KM, _PS, _PO, vector_scale=vs, axial_scale=as_)["amps2"])
        return jnp.log(jnp.sum(jnp.where(keep, a / jnp.asarray(_A_NOM), 0.0)))
    e = 1e-3
    d2_fd = (float(logsumw_direct(1 + e, 1 + e)) - float(logsumw_direct(1 + e, 1 - e))
             - float(logsumw_direct(1 - e, 1 + e)) + float(logsumw_direct(1 - e, 1 - e))) / (4 * e * e)
    assert abs(d2_ad) > 1e-6, "mixed 2nd derivative is ~0 -- the cross term is missing"
    assert abs(d2_ad - d2_fd) / max(abs(d2_fd), 1e-30) < 1e-3, (d2_ad, d2_fd)


# ---- wiring (step 1) + first-order invariance (step 2): joint vs the legacy per-knob product ----
from adonis.reweight.amps2_records import (build_qe_ma_records, build_qe_vector_records,
                                        build_qe_ff_records, ma_reweight, strength_reweight)

_MAREC = build_qe_ma_records(_KN, _KM, _PS, _PO)
_VREC = build_qe_vector_records(_KN, _KM, _PS, _PO)
_FF = {k: build_qe_ff_records(_KN, _KM, _PS, _PO, k) for k in ("gmp", "gmn", "gep", "gen")}


def _old_hv_qe(k):
    """The legacy per-knob product (reweight_model._hv_qe, qe_joint=False)."""
    return (ma_reweight(_MAREC, k.M_A_qe) * strength_reweight(_MAREC, k.axial_strength)
            * strength_reweight(_VREC, k.vector_strength) * strength_reweight(_FF["gmp"], k.mu_p)
            * strength_reweight(_FF["gmn"], k.mu_n) * strength_reweight(_FF["gep"], k.gep)
            * strength_reweight(_FF["gen"], k.gen))


_SINGLE = [("vector_strength", 1.2), ("axial_strength", 0.85), ("M_A_qe", 1.05),
           ("gep", 1.1), ("gen", 0.95), ("mu_p", 0.9), ("mu_n", 1.1)]


def test_single_knob_matches_legacy():
    """Moving ONE knob: joint == the old per-knob reweight (the old code is exact single-axis -> no regression)."""
    nom = nominal_knobs()
    for fld, val in _SINGLE:
        w_new = np.asarray(qe_reduced_reweight(_REC, nom._replace(**{fld: val})))
        w_old = np.asarray(_old_hv_qe(nom._replace(**{fld: val})))
        assert np.allclose(w_new[_VALID], w_old[_VALID], rtol=1e-9), fld


def test_first_order_invariance():
    """d(sum w)/d(knob) at nominal == the legacy value, per knob -> the Jacobian/Fisher (sec2/sec3) are unchanged."""
    nom = nominal_knobs(); keep = jnp.asarray(_VALID)
    for fld, _ in _SINGLE:
        x0 = float(getattr(nom, fld))
        f_new = lambda x: jnp.sum(jnp.where(keep, qe_reduced_reweight(_REC, nom._replace(**{fld: x})), 0.0))
        f_old = lambda x: jnp.sum(jnp.where(keep, _old_hv_qe(nom._replace(**{fld: x})), 0.0))
        g_new, g_old = float(jax.grad(f_new)(x0)), float(jax.grad(f_old)(x0))
        assert abs(g_new - g_old) / max(abs(g_old), 1e-30) < 1e-6, (fld, g_new, g_old)


def test_hv_qe_dispatch():
    """build_hv_sf(qe_joint=True) puts 'qe_reduced' in HV and _hv_qe routes to the joint reweight."""
    from adonis.reweight.reweight_model import _hv_qe
    HV = {"qe_reduced": _REC, "qe_probe": "CC", "qe_isp": None}
    k = nominal_knobs()._replace(vector_strength=1.1, axial_strength=0.9, gep=1.05)
    assert np.allclose(np.asarray(_hv_qe(k, HV)), np.asarray(qe_reduced_reweight(_REC, k)), rtol=1e-12)


def test_qe_mij_exact_em():
    """EM probe: F^T M F (vector-only, struck nucleon's own FFs) == direct EM amps2."""
    isp = jnp.asarray(np.arange(N) % 2 == 0)
    rec = build_qe_reduced(_KN, _KM, _PS, _PO, probe="EM", is_proton=isp)
    Mem = jnp.asarray(rec["M"]); Q2 = jnp.asarray(rec["Q2"])
    a_nom = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, probe="EM", is_proton=isp)["amps2"])
    valid = np.isfinite(a_nom) & (a_nom > 0) & (np.asarray(Q2) > 0)   # q2>0: the reduced record's mask
    for v, fs in [(1.0, {}), (1.2, {"gep": 1.1}), (0.8, {"gmp": 0.9, "gep": 1.05})]:
        ff = nucleon_ff(Q2 / 1.0e6, ff_scale=fs)
        f1 = jnp.where(isp, ff["F1p"], ff["F1n"]) * v; f2 = jnp.where(isp, ff["F2p"], ff["F2n"]) * v
        F = jnp.stack([f1, f2, jnp.zeros_like(f1), jnp.zeros_like(f1)], axis=-1)
        lhs = np.asarray(jnp.einsum('ni,nij,nj->n', F, Mem, F))
        rhs = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, vector_scale=v, ff_scale=fs or None,
                                          probe="EM", is_proton=isp)["amps2"])
        assert np.allclose(lhs[valid], rhs[valid], rtol=1e-9), (v, fs)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
