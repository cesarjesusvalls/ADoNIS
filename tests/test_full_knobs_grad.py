"""End-to-end: the T2K CC0pi dsigma/dx distribution is DIFFERENTIABLE in the exposed knobs.
Builds a frozen QE+RES proposal + pool walk + hard-vertex amps2 records + SF points, then checks
autodiff == finite-difference of an actual dsigma/dx bin (and the total) w.r.t. a representative knob
from each mechanism: hard-vertex amps2 (vector_strength, pion_pole), FSI kind-1 (s_piN_cex,
s_NN_inelastic), branch (f_NN_cex), spectral function (kF_sf).  Also: nominal model_hist_full reproduces
the legacy FSI-only model_hist (sabs/sscat/MA) -> the new path is a faithful superset."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ["CC0PI_N"] = "5000"

import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import analysis.campaign.tune as T
from adonis.reweight.reweight_model import (nominal_knobs, build_hv_sf, model_hist_full)
from adonis.nuclear.targets import resolve_targets
from adonis.nuclear.spectral import SpectralFunction

T.NQE = T.NRES = 5000
_qe, _qw, _res, _rw = T.build_proposal()
_R = T.build_replica(jax.random.PRNGKey(0), _qe, _qw, _res, _rw)
_M = T.build_ma_records(_qe, _res)
_sf = SpectralFunction(resolve_targets("C")[0][0].spectral_n)
_HV, _SF = build_hv_sf(_qe, _res, _sf)
_EDGES = np.asarray(T.EDGES); _CONV = T.CONV
_NOM = nominal_knobs()


def _hist(knobs):
    return model_hist_full(knobs, _R, _HV, _SF, _EDGES, _CONV)


def test_nominal_matches_legacy_model_hist():
    """All knobs nominal -> identical to the legacy FSI+MA model_hist at theta=(1,1,1).
    Eb_shift is overridden to 0 here: the legacy model has no SF knob (== no removal-energy shift), so the
    bit-identity comparison is against the unshifted SF.  The full-model NOMINAL Eb_shift is a deliberate
    epsilon (_EB_EPS=1e-2 MeV, full_knobs) that anchors the gradient off the Eb=0 corner; it deforms the SF
    by a negligible ~6e-5 (verified in test_sf_reweight), which is exactly the (intended) difference from
    the legacy here -- not a bit-identity failure of the reweight machinery."""
    h_full = np.asarray(_hist(_NOM._replace(Eb_shift=0.0)))
    h_leg = np.asarray(T.model_hist(jnp.array([1.0, 1.0, 1.0]), _R, _M))
    assert np.allclose(h_full, h_leg, rtol=1e-10, atol=1e-30), (h_full[:5], h_leg[:5])


def _grad_fd(setk, v0, ibin=None):
    """autodiff vs FD of (a bin of, or sum of) the histogram w.r.t. a scalar knob."""
    def f(v):
        h = _hist(setk(v))
        return h[ibin] if ibin is not None else jnp.sum(h)
    g_ad = float(jax.grad(f)(v0))
    # Differencing a histogram is cancellation-limited, so the error grows as the step shrinks:
    # a step this large is what makes the finite difference the accurate side of the comparison.
    eps = 1e-2
    g_fd = (float(f(v0 + eps)) - float(f(v0 - eps))) / (2 * eps)
    return g_ad, g_fd


def test_grad_vector_strength():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(vector_strength=v), 1.1)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-4, (g_ad, g_fd)


def test_grad_pion_pole():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(pion_pole=v), 1.1)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-4, (g_ad, g_fd)


def test_grad_s_piN_cex():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(s_piN_cex=v), 1.2)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)


def test_grad_s_NN_inelastic():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(s_NN_inelastic=(v, v, v)), 1.2)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)


def test_grad_f_NN_cex():
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(f_NN_cex=v), 0.6)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)


def test_grad_kF_sf_on_a_bin():
    nb = len(_EDGES) - 1
    g_ad, g_fd = _grad_fd(lambda v: _NOM._replace(kF_sf=v), 1.1, ibin=nb // 3)
    assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_ad, g_fd)
