"""Closure for the spectral-function reweight knobs (kF_sf, Eb_shift, sf_norm, src_tail):
1. nominal identity: sf_reweight(.., all nominal) == 1 exactly (numerator==denominator);
2. autodiff == finite-difference for each knob (on-grid events);
3. norm/tail behave as pure multipliers.
The struck (|p|, removal) is taken from a QE sample (p_struck)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from adonis.channels import qe as qe_xsec
from adonis.nuclear.targets import resolve_targets
from adonis.nuclear.spectral import SpectralFunction
from adonis.reweight.sf_reweight import sf_grids, sf_reweight, removal_from_struck

_tg = resolve_targets("C")[0][0]
_G = sf_grids(SpectralFunction(_tg.spectral_n))
_QE = qe_xsec.sample_importance(4000, seed=2)
_PS = jnp.asarray(_QE["p_struck"])[jnp.asarray(_QE["w"]) != 0]
_PMAG, _EREM = removal_from_struck(_PS)


def test_sf_nominal_identity():
    w = np.asarray(sf_reweight(_G, _PMAG, _EREM))
    assert np.allclose(w, 1.0, atol=1e-12)


def test_sf_norm_is_multiplier():
    w = np.asarray(sf_reweight(_G, _PMAG, _EREM, sf_norm=1.37))
    assert np.allclose(w, 1.37, atol=1e-12)


def test_sf_autodiff_equals_fd():
    # (name, v0, eps, tol, f).  Eb_shift is checked at its OPERATING POINT (nominal _EB_EPS=1e-2 MeV,
    # full_knobs) on the smooth one-sided branch -- NOT at a far-from-nominal v0 that would hide the
    # near-zero corner.  The removal-energy reweight S(p,E-Eb)/S(p,E) is steep there (heavy-tailed SF), so
    # FD closure is ~1% (not 1e-4); the tight 1e-4 machinery check is carried by the other three knobs,
    # which share the same _bspline2d interpolation.
    cases = [
        ("kF_sf", 1.1, 1e-4, 1e-4, lambda v: jnp.sum(sf_reweight(_G, _PMAG, _EREM, kF_sf=v))),
        ("Eb_shift", 1e-2, 5e-4, 1e-2, lambda v: jnp.sum(sf_reweight(_G, _PMAG, _EREM, Eb_shift=v))),
        ("sf_norm", 1.2, 1e-4, 1e-4, lambda v: jnp.sum(sf_reweight(_G, _PMAG, _EREM, sf_norm=v))),
        ("src_tail", 1.3, 1e-4, 1e-4, lambda v: jnp.sum(sf_reweight(_G, _PMAG, _EREM, src_tail=v))),
    ]
    for name, v0, eps, tol, f in cases:
        g_ad = float(jax.grad(f)(v0))
        g_fd = (float(f(v0 + eps)) - float(f(v0 - eps))) / (2 * eps)
        assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-30) < tol, f"{name}: ad={g_ad} fd={g_fd}"


def test_sf_eb_is_one_sided():
    """Eb_shift is one-sided: negative shifts clamp to 0 (== no shift == ACHILLES nominal)."""
    w_neg = np.asarray(sf_reweight(_G, _PMAG, _EREM, Eb_shift=-5.0))
    w_nom = np.asarray(sf_reweight(_G, _PMAG, _EREM, Eb_shift=0.0))
    assert np.allclose(w_neg, w_nom, atol=1e-12) and np.allclose(w_nom, 1.0, atol=1e-12)


def test_sf_kF_moves_weight():
    """A non-trivial kF_sf must actually change the weights (not silently identity)."""
    w = np.asarray(sf_reweight(_G, _PMAG, _EREM, kF_sf=1.2))
    assert np.std(w) > 1e-3
