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
_VALID = np.isfinite(_A_NOM) & (_A_NOM > 0)                    # drop rejected/unphysical draws


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
