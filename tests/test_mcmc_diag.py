"""MCMC diagnostics against cases with a KNOWN answer.

No arviz in this environment, so these are hand-rolled -- and a silently wrong ESS would not announce
itself, it would just hand back a plausible efficiency number, which is the whole quantity a sampler
comparison reports.  Hence: iid (ESS = N), AR(1) (ESS = N(1-rho)/(1+rho)), offset chains (Rhat >> 1).
"""
import numpy as np
import pytest

from adonis.fit.mcmc_diag import ess_bulk, ess_basic, ess_tail, split_rhat, tau_int


def _ar1(n, rho, rng, m=4):
    x = np.zeros((m, n))
    s = np.sqrt(1 - rho ** 2)
    for i in range(m):
        e = rng.standard_normal(n) * s
        v = 0.0
        for t in range(n):
            v = rho * v + e[t]
            x[i, t] = v
    return x


def test_iid_ess_is_about_n():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 4000))
    for f in (ess_bulk, ess_basic):
        e = f(x)
        assert 0.85 * x.size < e < 1.15 * x.size, (f.__name__, e, x.size)


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.95])
def test_ar1_ess_matches_theory(rho):
    rng = np.random.default_rng(1)
    x = _ar1(20000, rho, rng)
    expect = x.size * (1 - rho) / (1 + rho)
    e = ess_basic(x)
    assert 0.75 * expect < e < 1.3 * expect, (rho, e, expect)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_tau_int_matches_theory(rho):
    rng = np.random.default_rng(2)
    x = _ar1(20000, rho, rng)
    expect = (1 + rho) / (1 - rho)
    t = tau_int(x)
    assert 0.75 * expect < t < 1.35 * expect, (rho, t, expect)


def test_rhat_is_one_for_converged_chains():
    rng = np.random.default_rng(3)
    assert abs(split_rhat(rng.standard_normal((4, 4000))) - 1.0) < 0.01


def test_rhat_detects_offset_chains():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 2000)) + np.array([0.0, 1.0, 2.0, 3.0])[:, None]
    assert split_rhat(x) > 1.2


def test_rhat_detects_within_chain_drift():
    # each chain drifts across its own length: the chains AGREE overall, so only SPLIT Rhat sees it
    rng = np.random.default_rng(5)
    n = 4000
    x = rng.standard_normal((4, n)) + np.linspace(-2, 2, n)[None, :]
    assert split_rhat(x) > 1.2


def test_tail_ess_is_finite_and_below_bulk_for_ar1():
    rng = np.random.default_rng(6)
    x = _ar1(20000, 0.9, rng)
    et, eb = ess_tail(x), ess_bulk(x)
    assert np.isfinite(et) and et > 0
    assert et < 3 * eb                       # tail is never wildly larger than bulk


def test_repeated_values_do_not_break_rank_normalisation():
    # a Metropolis chain REPEATS its state on rejection; ties must not corrupt the normal scores
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 3000))
    keep = rng.random((4, 3000)) < 0.6       # 40% rejections -> long runs of identical values
    for i in range(4):
        for t in range(1, 3000):
            if not keep[i, t]:
                x[i, t] = x[i, t - 1]
    assert np.isfinite(split_rhat(x)) and np.isfinite(ess_bulk(x))
    assert ess_bulk(x) < x.size              # correlated by construction


def test_frozen_chain_has_no_effective_samples():
    """A stuck chain carries no information: ESS must be ~0, never N, and must not depend on the value.

    The old code returned m*n (perfect independence) for a constant chain, and which branch it took was
    decided by FFT round-off: frozen at 3.14 -> ESS = N, frozen at 7.77 -> ESS = 8.
    """
    for c in (0.0, 3.14, 7.77, -2.5):
        x = np.full((4, 4000), c)
        assert ess_basic(x) == 0.0, c
        assert ess_bulk(x) == 0.0, c


def test_rhat_detects_pure_scale_mismatch():
    """Same mean, sd 1:2:4:8 -- unconverged, and invisible without the FOLDED half of Rhat."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 4000)) * np.array([1.0, 2.0, 4.0, 8.0])[:, None]
    assert split_rhat(x) > 1.01


def test_tail_ess_of_iid_is_about_n():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((4, 20000))
    e = ess_tail(x)
    assert 0.85 * x.size < e < 1.15 * x.size, e


def test_single_chain_ess_matches_theory():
    rng = np.random.default_rng(13)
    rho = 0.8
    n = 40000
    e = rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    v = 0.0
    x = np.empty(n)
    for t in range(n):
        v = rho * v + e[t]
        x[t] = v
    expect = n * (1 - rho) / (1 + rho)
    assert 0.7 * expect < ess_basic(x) < 1.4 * expect
