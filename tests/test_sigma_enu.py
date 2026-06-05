"""Phase A3 gates: free-nucleon single-pion σ(E_ν).

Closure (differentiability of σ wrt M_A) + forward physics sanity (channel ratios
match the DCC isospin structure; σ rises from threshold to a high-energy plateau).
The absolute-normalisation / ACHILLES-oracle / ANL-BNL gates land with A3.4–A3.5
(see docs/phases/PHASE_A3.md).
"""
import os
import numpy as np
import jax

from adonis.params import PhysicsParams, GenConfig
from adonis.nuclear.free import FreeNucleon
from adonis.primary.dcc.sigma_enu import sigma_vs_enu, sigma_channels_at, dsigma_dMA_closure

N = 8_000 if os.environ.get("ADONIS_CI_FAST") else 40_000


def test_sigma_closure_dMA():
    """d(total σ)/dM_A: autodiff == central finite difference (kind-1 exact gradient)."""
    r = dsigma_dMA_closure(key=jax.random.PRNGKey(0), n=N)
    assert not r.skipped
    assert r.passed, r.detail
    assert r.metrics["rel_err"] < 1e-3


def test_free_nucleon_channel_ratios():
    """Free-nucleon channel split reproduces the DCC isospin fractions (STATUS.md:
    p→pπ⁺ ~0.66, n→pπ⁰ ~0.19, n→nπ⁺ ~0.15 in the Δ region)."""
    _, sc = sigma_channels_at(PhysicsParams(), jax.random.PRNGKey(1), e_nu=1000.0, n=N)
    sc = np.asarray(sc)
    f = sc / sc.sum()
    assert f[2] > f[0] > f[1], f               # p→pπ⁺ largest, n→nπ⁺ smallest
    assert 0.60 < f[2] < 0.75, f               # Δ⁺⁺-like dominance
    assert 0.12 < f[0] < 0.26, f               # n→pπ⁰
    assert f.sum() == np.float64(f.sum())      # finite


def test_sigma_rises_and_plateaus():
    """σ(E_ν) increases monotonically over the rise and is near-flat at high E_ν."""
    E = [500., 700., 1000., 1500., 2000.]
    tot, _ = sigma_vs_enu(PhysicsParams(), jax.random.PRNGKey(2), E, n=N)
    assert np.all(np.diff(tot) > 0), tot       # monotonic rise across this range
    assert tot[0] > 0
    # high-energy flattening: last step's fractional increase < the first step's
    fr = np.diff(tot) / tot[:-1]
    assert fr[-1] < fr[0], fr
