"""isotropic_two_body_cm reproduces the inline qe/ee two-body formula BIT-FOR-BIT.

The helper's (k1, k2, pcm, sqrts, s, lam) equal the explicit "form s, get pcm from Kallen, draw
isotropic CM, boost" computation exactly, for both the CC (mu, p) and EM (e, N) mass pairs.  The
rest-frame variant used by qe_nc is covered by the free-nucleon sigma gate, not here.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from adonis.channels.twobody import isotropic_two_body_cm
from adonis.kinematics import boost as _boost

_M_MU, _M_P, _M_E, _M_N = 105.6583745, 938.272, 0.5109989, 939.565


def _inline(P, m1, m2, ucos, uphi):
    """The two-body decay-and-boost block written out explicitly, as the samplers express it."""
    s = P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1)
    sqrts = np.sqrt(np.clip(s, 1e-9, None))
    s2, s3 = m1 ** 2, m2 ** 2
    E1 = sqrts / 2 * (1 + s2 / s - s3 / s); E2 = sqrts / 2 * (1 + s3 / s - s2 / s)
    lam = np.sqrt(np.clip((s - s2 - s3) ** 2 - 4 * s2 * s3, 0, None)); pcm = lam / (2 * sqrts)
    cts = 2 * ucos - 1; sts = np.sqrt(np.clip(1 - cts ** 2, 0, None)); php = 2 * np.pi * uphi
    dirn = np.stack([sts * np.cos(php), sts * np.sin(php), cts], axis=1)
    beta = P[:, 1:] / P[:, 0:1]
    k1 = np.asarray(_boost(np.concatenate([E1[:, None], pcm[:, None] * dirn], axis=1), beta))
    k2 = np.asarray(_boost(np.concatenate([E2[:, None], -pcm[:, None] * dirn], axis=1), beta))
    return k1, k2, pcm, sqrts, s, lam


def _random_totals(seed, n=1000):
    rng = np.random.default_rng(seed)
    E = rng.uniform(1000.0, 3000.0, n)
    px = rng.uniform(-80.0, 80.0, n); py = rng.uniform(-80.0, 80.0, n); pz = rng.uniform(300.0, 2200.0, n)
    return np.column_stack([E + _M_N, px, py, pz]), rng.random(n), rng.random(n)


def _assert_bit_identical(P, m1, m2, ucos, uphi):
    h = isotropic_two_body_cm(P, m1, m2, ucos, uphi)
    r = _inline(P, m1, m2, ucos, uphi)
    for name, a, b in zip(("k1", "k2", "pcm", "sqrts", "s", "lam"), h, r):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name} differs (max {np.max(np.abs(a-b)):.2e})"


def test_helper_matches_inline_qe_mu_p():
    P, ucos, uphi = _random_totals(1)
    _assert_bit_identical(P, _M_MU, _M_P, ucos, uphi)


def test_helper_matches_inline_ee_e_N():
    P, ucos, uphi = _random_totals(2)
    _assert_bit_identical(P, _M_E, _M_P, ucos, uphi)
