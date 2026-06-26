"""Per-module self-test gate (gap 1): exercise every chain module's
closure_test() (standalone differentiability) and oracle_test() (physics validity
vs ACHILLES) through the uniform contract on the base classes.

These are the CI-facing gates -- they assert each module's own self-test passes,
so the same checks run locally and in the cloud.  Statistics are deliberately
modest; the full high-statistics comparison lives in scripts/validate_final_state.py.
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


# -- closure (standalone differentiability) ----------------------------------- #
def test_channel_closure():
    r = _channel().closure_test(key=jax.random.PRNGKey(0), n=N)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_err"] < 1e-3


def test_detached_modules_closure_skip():
    # detached samplers / passthroughs have no differentiable parameter -> skipped
    for m in (SpectralFunction(), Monochromatic(), NoFSI()):
        r = m.closure_test()
        assert r.skipped and r.passed


# -- oracle (physics validity) ------------------------------------------------ #
def test_nuclear_oracle():
    r = SpectralFunction().oracle_test(n=N_ORACLE)
    assert not r.skipped
    assert r.passed, r.detail


def test_channel_oracle(oracle_path):
    r = _channel().oracle_test(oracle_path, key=jax.random.PRNGKey(5000),
                               n=N_ORACLE, chunk=50_000)
    assert not r.skipped
    assert r.passed, r.detail
    # the gated observables should sit near chi2/ndf ~ 1
    for nm in ("cos_theta_star", "ppi_mag", "Q2"):
        assert r.metrics[nm]["chi2_ndf"] < 3.0, (nm, r.metrics[nm])
