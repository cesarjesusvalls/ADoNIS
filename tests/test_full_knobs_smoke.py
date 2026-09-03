"""The two checks of test_full_knobs_grad that hold at any statistics, at a size that runs in seconds.

Small N (800) and with_pw=False keep the build cheap.  Both checks are statistics-independent:
(1) at nominal knobs, model_hist_full equals the FSI+MA model_hist bit-for-bit; (2) autodiff equals
the finite difference of a real dsigma/dx with respect to one knob per mechanism, which exercises the
gradient wiring rather than the accuracy of the physics."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ["CC0PI_N"] = "800"

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import analysis.campaign.tune as T
from adonis.reweight.reweight_model import nominal_knobs, build_hv_sf, model_hist_full
from adonis.nuclear.targets import resolve_targets
from adonis.nuclear.spectral import SpectralFunction

T.NQE = T.NRES = 800
_qe, _qw, _res, _rw = T.build_proposal()
_R = T.build_replica(jax.random.PRNGKey(0), _qe, _qw, _res, _rw)
_M = T.build_ma_records(_qe, _res)
_sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
_HV, _SF = build_hv_sf(_qe, _res, _sf, with_pw=False)
_EDGES = np.asarray(T.EDGES); _CONV = T.CONV
_NOM = nominal_knobs()


def _hist(knobs):
    return model_hist_full(knobs, _R, _HV, _SF, _EDGES, _CONV)


def test_nominal_matches_legacy():
    """At nominal knobs the full model equals the FSI+MA model_hist bit for bit, at any N."""
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


def test_grad_vector_strength():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(vector_strength=v), 1.1)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-4, (g_ad, g_fd)


def test_grad_s_piN_cex():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(s_piN_cex=v), 1.2)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)


def test_grad_kF_sf():
    nb = len(_EDGES) - 1
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(kF_sf=v), 1.1, ibin=nb // 3)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)
