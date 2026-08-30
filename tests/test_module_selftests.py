"""Per-module self-test gate (gap 1): exercise every chain module's
closure_test() (standalone differentiability) and oracle_test() (physics validity
vs ACHILLES) through the uniform contract on the base classes.

These are the CI-facing gates -- they assert each module's own self-test passes,
so the same checks run locally and in the cloud.  Statistics are deliberately
modest: these gate that each self-test runs and passes, not the physics precision.
"""
import os
import jax

from adonis import ChainConfig, DCCSinglePion
from adonis.nuclear.spectral import SpectralFunction
from adonis.flux.mono import Monochromatic
from adonis.fsi.none import NoFSI

N = 8_000 if os.environ.get("ADONIS_CI_FAST") else 30_000
N_ORACLE = 120_000 if os.environ.get("ADONIS_CI_FAST") else 300_000


def _channel():
    return DCCSinglePion(ChainConfig(spline=False))


def test_channel_closure():
    r = _channel().closure_test(key=jax.random.PRNGKey(0), n=N)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_err"] < 1e-3


def test_detached_modules_closure_skip():
    for m in (SpectralFunction(), Monochromatic(), NoFSI()):
        r = m.closure_test()
        assert r.skipped and r.passed


def test_nuclear_oracle():
    r = SpectralFunction().oracle_test(n=N_ORACLE)
    assert not r.skipped
    assert r.passed, r.detail
