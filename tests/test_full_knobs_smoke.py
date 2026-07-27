"""FAST analog of test_full_knobs_grad (gated behind --runslow).

Small-N (800) + with_pw=False so it builds in seconds.  The two checks it keeps are
STATISTICS-INDEPENDENT, so they hold at any N: (1) nominal model_hist_full reproduces the legacy
FSI+MA model_hist bit-for-bit; (2) autodiff == finite-difference of a real dsigma/dx quantity w.r.t.
one knob per mechanism (this validates the gradient wiring, not physics accuracy).  The full 27-knob
sweep at N=5000 lives in test_full_knobs_grad (run with `pytest --runslow`)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ["CC0PI_N"] = "800"

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import adonis.reweight.tune as T
from adonis.reweight.reweight_model import nominal_knobs, build_hv_sf, model_hist_full
from adonis.workflow.materials import resolve_targets
from adonis.nuclear.spectral import SpectralFunction

T.NQE = T.NRES = 800
_qe, _qw, _res, _rw = T.build_proposal()
_R = T.build_replica(jax.random.PRNGKey(0), _qe, _qw, _res, _rw)
_M = T.build_ma_records(_qe, _res)
_sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
_HV, _SF = build_hv_sf(_qe, _res, _sf, with_pw=False)   # skip the 14 DCC pw records -> fast (pw not tested here)
_EDGES = np.asarray(T.EDGES); _CONV = T.CONV
_NOM = nominal_knobs()


def _hist(knobs):
    return model_hist_full(knobs, _R, _HV, _SF, _EDGES, _CONV)


def test_nominal_matches_legacy():
    """Nominal full model == legacy FSI+MA model_hist (bit-identity; N-independent)."""
    h_full = np.asarray(_hist(_NOM._replace(Eb_shift=0.0)))
    h_leg = np.asarray(T.model_hist(jnp.array([1.0, 1.0, 1.0]), _R, _M))
    assert np.allclose(h_full, h_leg, rtol=1e-10, atol=1e-30), (h_full[:5], h_leg[:5])


def _grad_fd(setk, v0, ibin=None):
    def f(v):
        h = _hist(setk(v))
        return h[ibin] if ibin is not None else jnp.sum(h)
    g_ad = float(jax.grad(f)(v0))
    eps = 1e-4
    g_fd = (float(f(v0 + eps)) - float(f(v0 - eps))) / (2 * eps)
    return g_ad, g_fd


def test_grad_vector_strength():        # hard-vertex amps2
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(vector_strength=v), 1.1)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-4, (g_ad, g_fd)


def test_grad_s_piN_cex():              # FSI kind-1
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(s_piN_cex=v), 1.2)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)


def test_grad_kF_sf():                  # spectral function, one bin
    nb = len(_EDGES) - 1
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(kF_sf=v), 1.1, ibin=nb // 3)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)
