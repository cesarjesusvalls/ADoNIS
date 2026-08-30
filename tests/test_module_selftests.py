"""Every chain module's own self-tests pass through the uniform base-class contract.

Each module exposes closure_test() (standalone differentiability) and oracle_test() (physics
validity against ACHILLES).  These assert that each self-test runs and passes, so the same checks
run locally and in CI; the statistics are modest by design and do not gate physics precision.
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
