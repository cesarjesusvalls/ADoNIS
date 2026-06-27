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

from adonis.xsec import qe_xsec
from adonis.xsec.backend import me_cross_section
from adonis.analysis.ma_records import build_qe_ma_records, strength_reweight

N = 2000
_QE = qe_xsec.sample_importance(N, seed=5)
_KN, _KM, _PS, _PO = (jnp.asarray(_QE[k]) for k in ("k_nu", "k_mu", "p_struck", "p_out"))
_REC = build_qe_ma_records(_KN, _KM, _PS, _PO)


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
