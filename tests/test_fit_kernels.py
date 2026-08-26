"""FitKernel algebra, on a synthetic engine so it runs in seconds without banks or a GPU.

This is NOT the acceptance test for the fused binning -- that needs the real engine and lives in
`analysis.benchmarks.verify_kernels`, which compares against the host np.bincount path.  What is checked here is
everything AROUND the binning, which is where the arithmetic mistakes live: the prior block, the
whitening, the dead-bin mask, reverse-vs-forward consistency, batch invariance, and the event-pass
counters.  Those are exactly the parts that a big GPU run would only reveal as a wrong number hours in.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from adonis.fit.kernels import FitKernel


class _Sample:
    """A block of bins whose model is a genuinely nonlinear, coupled function of theta."""

    def __init__(self, nbin, npar, seed, offset):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, 1, (nbin, npar))
        self.b = rng.uniform(0.5, 1.5, nbin)
        self.off = offset
        self.ds = [dict(nbin=nbin // 2, data=np.zeros(nbin // 2), sigma=np.ones(nbin // 2)),
                   dict(nbin=nbin - nbin // 2, data=np.zeros(nbin - nbin // 2),
                        sigma=np.ones(nbin - nbin // 2))]

    def _m(self, th, xp):
        # nonlinear in theta and coupled across dials, so a Jacobian bug cannot hide
        return self.b * xp.exp(0.3 * (self.A @ th)) + self.off

    def model_blocks_jax(self, th):
        m = self._m(th, jnp)
        k = self.ds[0]["nbin"]
        return [m[:k], m[k:]]

    def model_blocks(self, th):
        m = self._m(np.asarray(th), np)
        k = self.ds[0]["nbin"]
        return [m[:k], m[k:]]


class _Engine:
    def __init__(self, npar=5, seed=0):
        self.samples = [_Sample(6, npar, seed + 1, 0.1), _Sample(8, npar, seed + 2, -0.2)]
        self.ds = [d for s in self.samples for d in s.ds]
        self.th0 = np.linspace(0.8, 1.2, npar)
        self.prior = np.linspace(0.1, 0.4, npar)
        self.pnames = [f"p{i}" for i in range(npar)]

    def model(self, th):
        return np.concatenate([np.asarray(b) for s in self.samples for b in s.model_blocks(th)])


def _fill_data(eng, th_truth, sigmas):
    """Closure data at th_truth, with a caller-supplied sigma per bin (inf = dead)."""
    m = eng.model(th_truth)
    r0 = 0
    for s in eng.samples:
        for d in s.ds:
            n = d["nbin"]
            d["data"] = m[r0:r0 + n].copy()
            d["sigma"] = np.asarray(sigmas[r0:r0 + n], float).copy()
            r0 += n
    return eng


def _setup(npar=5, subset=(0, 2, 3, 4), dead=(1, 9)):
    eng = _Engine(npar)
    nbin = sum(d["nbin"] for d in eng.ds)
    sig = np.linspace(0.05, 0.2, nbin)
    sig[list(dead)] = np.inf                    # dead bins, exactly as set_closure_data marks them
    # The truth moves ONLY the fitted dials.  Displacing an unfitted dial as well makes the closure
    # unreachable -- the fit is then chasing a truth it is not allowed to represent, which is a
    # misspecified test, and it is the same mistake that made the earlier n-scan not apples-to-apples.
    th_t = eng.th0.copy()
    th_t[np.asarray(subset, int)] += 0.15
    _fill_data(eng, th_t, sig)
    return eng, list(subset), np.asarray(subset, int), th_t


def _host(eng, subset):
    """The reference objective, written as trf_fit writes it."""
    idx = np.asarray(subset, int)
    th0 = np.asarray(eng.th0, float)
    data = np.concatenate([d["data"] for d in eng.ds])
    sigma = np.concatenate([d["sigma"] for d in eng.ds])
    ok = np.isfinite(sigma) & (sigma > 0)
    w = np.where(ok, 1.0 / np.where(ok, sigma, 1.0), 0.0)
    pw = 1.0 / np.asarray(eng.prior, float)[idx]

    def resid(x):
        th = th0.copy(); th[idx] = x
        return np.concatenate([(eng.model(th) - data) * w, (np.asarray(x) - th0[idx]) * pw])

    return resid, w, pw


def test_residuals_and_chi2_match_the_host_expression():
    eng, subset, idx, th_t = _setup()
    k = FitKernel(eng, subset).warmup()
    resid, _, _ = _host(eng, subset)
    for x in (k.x0, th_t[idx], 0.5 * (k.x0 + th_t[idx])):
        r_h, r_k = resid(x), k.residuals(x)
        assert r_k.shape == (k.nbin + k.n,)
        assert np.allclose(r_k, r_h, rtol=0, atol=1e-12)
        assert abs(k.chi2(x) - float(r_h @ r_h)) <= 1e-12 * max(1.0, float(r_h @ r_h))


def test_dead_bins_are_dropped_not_merely_downweighted():
    eng, subset, idx, _ = _setup(dead=(1, 9))
    k = FitKernel(eng, subset).warmup()
    r = k.residuals(k.x0 + 0.05)
    assert r[1] == 0.0 and r[9] == 0.0
    assert k.n_live == k.nbin - 2
    assert k.jac(k.x0 + 0.05)[1].max() == 0.0


def test_gradient_equals_two_JT_r_and_a_finite_difference():
    eng, subset, idx, th_t = _setup()
    k = FitKernel(eng, subset).warmup()
    resid, _, _ = _host(eng, subset)
    x = 0.5 * (k.x0 + th_t[idx])
    g = k.grad(x)
    assert np.allclose(g, 2.0 * (k.jac(x).T @ k.residuals(x)), rtol=1e-10, atol=1e-12)
    h = 1e-6
    fd = np.array([(k.chi2(x + h * e) - k.chi2(x - h * e)) / (2 * h)
                   for e in np.eye(k.n)])
    assert np.allclose(g, fd, rtol=2e-6, atol=1e-8)


def test_jacobian_matches_a_finite_difference_of_the_residuals():
    eng, subset, idx, th_t = _setup()
    k = FitKernel(eng, subset).warmup()
    x = 0.5 * (k.x0 + th_t[idx])
    J = k.jac(x)
    h = 1e-6
    for j, e in enumerate(np.eye(k.n)):
        fd = (k.residuals(x + h * e) - k.residuals(x - h * e)) / (2 * h)
        assert np.allclose(J[:, j], fd, rtol=1e-5, atol=1e-8)


def test_prior_block_is_present_and_correct():
    eng, subset, idx, _ = _setup()
    k = FitKernel(eng, subset).warmup()
    x = k.x0 + 0.07
    pw = 1.0 / np.asarray(eng.prior)[idx]
    assert np.allclose(k.residuals(x)[k.nbin:], (x - k.x0) * pw)
    assert np.allclose(k.jac(x)[k.nbin:], np.diag(pw))


def test_batching_changes_dispatches_and_nothing_else():
    eng, subset, idx, th_t = _setup()
    x = th_t[idx]
    ref = FitKernel(eng, subset).warmup().jac(x)
    for B in (1, 2, 3):
        k = FitKernel(eng, subset, jac_batch=B).warmup()
        assert np.allclose(k.jac(x), ref, rtol=0, atol=1e-13)
        assert k.nblk == int(np.ceil(k.n / B))
        k.reset_counts(); k.jac(x)
        # padding is real executed work and is counted as such; it is never quietly dropped
        assert k.counts["tangent"] == k.nblk * B >= k.n
        assert k.counts["tangent_eff"] == k.n
        assert k.counts["primal"] == k.nblk


def test_event_pass_bookkeeping():
    eng, subset, idx, _ = _setup()
    k = FitKernel(eng, subset).warmup()
    k.chi2(k.x0); k.chi2(k.x0); k.grad(k.x0); k.jac(k.x0)
    c = k.counts
    assert (c["chi2"], c["grad"], c["jac"]) == (2, 1, 1)
    assert c["primal"] == 2 + k.nblk and c["vjp"] == 1 and c["tangent"] == k.n
    assert k.event_passes() == pytest.approx(2 + k.nblk + k.n + 2.0)


def test_refresh_data_retargets_without_rebuilding():
    eng, subset, idx, th_t = _setup()
    k = FitKernel(eng, subset).warmup()
    assert k.chi2(th_t[idx]) == pytest.approx(
        float(((th_t[idx] - k.x0) / np.asarray(eng.prior)[idx]) @ ((th_t[idx] - k.x0) / np.asarray(eng.prior)[idx])),
        rel=1e-10)                                   # closure: data residual vanishes at the truth
    th_new = eng.th0.copy(); th_new[idx] += 0.25
    sig = np.concatenate([d["sigma"] for d in eng.ds])
    _fill_data(eng, th_new, sig)
    k.refresh_data()
    assert k.chi2(th_t[idx]) > 1.0                   # the old truth is no longer the solution
    assert k.chi2(th_new[idx]) == pytest.approx(
        float(((th_new[idx] - k.x0) / np.asarray(eng.prior)[idx]) @ ((th_new[idx] - k.x0) / np.asarray(eng.prior)[idx])),
        rel=1e-10)


def test_mask_can_only_remove_bins():
    eng, subset, idx, _ = _setup(dead=(1,))
    nbin = sum(d["nbin"] for d in eng.ds)
    m = np.ones(nbin, bool); m[0] = False          # ask to drop bin 0 as well
    k = FitKernel(eng, subset, mask=m).warmup()
    assert k.n_live == nbin - 2                    # bin 0 by the mask, bin 1 by sigma = inf
    m2 = np.ones(nbin, bool)
    k2 = FitKernel(eng, subset, mask=m2).warmup()
    assert k2.n_live == nbin - 1                   # a mask cannot resurrect a sigma = inf bin
