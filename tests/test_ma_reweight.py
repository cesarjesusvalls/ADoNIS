"""Gates for the exact M_A reweight hooks:
1. nominal identity: axial_scale=1 / r_axial=ones reproduce the default amps2 bit-for-bit;
2. quadratic identity: amps2(r) == a + b r + c r^2 from evals at r = 0, 1, -1 (exact);
3. the dipole-ratio knob is 1 exactly at MA = 1.0 and its gradient is exact (AD == FD).
These underpin scripts/cc0pi_tune_adonis.build_ma_records / ma_reweight."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.channels import qe as qe_xsec
from adonis.channels.currents.matrix_element import me_cross_section
from adonis.channels.dcc.form_factors import axial_reweight_dipole

N = 2000
_QE = qe_xsec.sample_importance(N, seed=3)
_KN, _KM, _PS, _PO = (jnp.asarray(_QE[k]) for k in ("k_nu", "k_lep", "p_struck", "p_out"))


def _amps2(scale):
    return np.asarray(me_cross_section(_KN, _KM, _PS, _PO, axial_scale=scale)["amps2"])


def test_qe_nominal_identity():
    assert np.array_equal(_amps2(1.0), np.asarray(me_cross_section(_KN, _KM, _PS, _PO)["amps2"]))


def test_qe_quadratic_identity():
    a1, a0, am = _amps2(1.0), _amps2(0.0), _amps2(-1.0)
    a, b, c = a0, 0.5 * (a1 - am), 0.5 * (a1 + am) - a0
    ok = np.isfinite(a1)                          # rejected draws (w=0) carry NaN kinematics
    for r in (0.73, 1.31):
        direct = _amps2(r)
        assert np.allclose(direct[ok], (a + b * r + c * r * r)[ok], rtol=1e-12)


def test_dipole_knob_nominal_and_grad():
    q2 = jnp.asarray(np.linspace(1e4, 3e6, 50))   # MeV^2
    assert np.allclose(np.asarray(axial_reweight_dipole(q2, 1.0)), 1.0, atol=1e-14)
    f = lambda ma: jnp.sum(axial_reweight_dipole(q2, ma) ** 2)
    eps = 1e-5
    g_ad = float(jax.grad(f)(1.1))
    g_fd = (float(f(1.1 + eps)) - float(f(1.1 - eps))) / (2 * eps)
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-6
