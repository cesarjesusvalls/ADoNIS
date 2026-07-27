"""Closure for the axial_strength knob (overall axial-current scale), reusing the M_A amps2 records:
1. nominal identity: strength_reweight(rec, 1.0) == 1 exactly;
2. quadratic identity: the (a,b,c) record reproduces amps2(strength) directly (amps2 linear-in-axial);
3. autodiff == finite-difference gradient of the summed reweight.
Same (a,b,c) QE/RES decomposition as M_A (build_qe_ma_records); strength uses the flat scale r=strength."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.channels import qe as qe_xsec
from adonis.channels.currents.matrix_element import me_cross_section
from adonis.reweight.amps2_records import (build_qe_ma_records, build_qe_vector_records,
                                        build_qe_ff_records, strength_reweight)

N = 2000
_QE = qe_xsec.sample_importance(N, seed=5)
_KN, _KM, _PS, _PO = (jnp.asarray(_QE[k]) for k in ("k_nu", "k_mu", "p_struck", "p_out"))
_REC = build_qe_ma_records(_KN, _KM, _PS, _PO)
_VREC = build_qe_vector_records(_KN, _KM, _PS, _PO)
_FFREC = {k: build_qe_ff_records(_KN, _KM, _PS, _PO, k) for k in ("gmp", "gmn", "gep", "gen")}


def test_strength_nominal_identity():
    w = np.asarray(strength_reweight(_REC, 1.0))
    assert np.allclose(w, 1.0, atol=1e-12)


def test_strength_matches_direct_amps2():
    """w(strength) * amps2(nominal) == amps2(strength) per event (the reweight IS the amps2 ratio).
    Exclude the record's identity placeholders (rejected draws -> (a,b,c)=(1,0,0))."""
    a, b, c, _ = (np.asarray(x) for x in _REC)
    a1 = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, axial_scale=1.0)["amps2"])
    valid = np.isfinite(a1) & (a1 > 0) & ~((b == 0.0) & (c == 0.0))    # drop placeholder events
    for s in (0.6, 1.4):
        direct = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, axial_scale=s)["amps2"])
        w = np.asarray(strength_reweight(_REC, s))
        assert np.allclose((w * a1)[valid], direct[valid], rtol=1e-9)


def test_strength_autodiff_equals_fd():
    f = lambda s: jnp.sum(strength_reweight(_REC, s))
    eps = 1e-5
    g_ad = float(jax.grad(f)(1.2))
    g_fd = (float(f(1.2 + eps)) - float(f(1.2 - eps))) / (2 * eps)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-6


# ---- vector_strength (QE vector-current scale): same flat-scale ratio on the vector record ----
def test_vector_nominal_identity():
    assert np.allclose(np.asarray(strength_reweight(_VREC, 1.0)), 1.0, atol=1e-12)


def test_vector_matches_direct_amps2():
    a, b, c, _ = (np.asarray(x) for x in _VREC)
    a1 = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, vector_scale=1.0)["amps2"])
    valid = np.isfinite(a1) & (a1 > 0) & ~((b == 0.0) & (c == 0.0))
    for v in (0.6, 1.4):
        direct = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, vector_scale=v)["amps2"])
        w = np.asarray(strength_reweight(_VREC, v))
        assert np.allclose((w * a1)[valid], direct[valid], rtol=1e-9)


def test_vector_autodiff_equals_fd():
    f = lambda v: jnp.sum(strength_reweight(_VREC, v))
    eps = 1e-5
    g_ad = float(jax.grad(f)(1.2))
    g_fd = (float(f(1.2 + eps)) - float(f(1.2 - eps))) / (2 * eps)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-6


def test_qe_nominal_amps2_unchanged():
    """vector_scale=axial_scale=1, ff_scale=None reproduces the default amps2 bit-for-bit (no-op nominal)."""
    base = np.asarray(me_cross_section(_KN, _KM, _PS, _PO)["amps2"])
    both = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, axial_scale=1.0, vector_scale=1.0, ff_scale=None)["amps2"])
    assert np.array_equal(base, both)


# ---- per-Sachs-FF knobs: mu_p (gmp), mu_n (gmn), gep, gen ----
def test_ff_nominal_identity():
    for k, rec in _FFREC.items():
        assert np.allclose(np.asarray(strength_reweight(rec, 1.0)), 1.0, atol=1e-12), k


def test_ff_matches_direct_amps2():
    for k, rec in _FFREC.items():
        a, b, c, _ = (np.asarray(x) for x in rec)
        a1 = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, ff_scale={k: 1.0})["amps2"])
        valid = np.isfinite(a1) & (a1 > 0) & ~((b == 0.0) & (c == 0.0))
        for s in (0.7, 1.3):
            direct = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, ff_scale={k: s})["amps2"])
            w = np.asarray(strength_reweight(rec, s))
            assert np.allclose((w * a1)[valid], direct[valid], rtol=1e-9), (k, s)


def test_ff_autodiff_equals_fd():
    eps = 1e-5
    for k, rec in _FFREC.items():
        f = lambda s: jnp.sum(strength_reweight(rec, s))
        g_ad = float(jax.grad(f)(1.15))
        g_fd = (float(f(1.15 + eps)) - float(f(1.15 - eps))) / (2 * eps)
        assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-6, k


def test_ff_gmp_nominal_amps2_unchanged():
    base = np.asarray(me_cross_section(_KN, _KM, _PS, _PO)["amps2"])
    g1 = np.asarray(me_cross_section(_KN, _KM, _PS, _PO, ff_scale={"gmp": 1.0})["amps2"])
    assert np.array_equal(base, g1)
