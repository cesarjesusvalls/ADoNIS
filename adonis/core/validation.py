"""Validation harness (Strategy §2, §12, §13).

Every component is held to the same three gates before it is trusted:

  (F) forward agreement   -- the hard sampler and the differentiable (weighted)
                             estimator must agree in the forward pass, to within
                             Monte-Carlo error.
  (G) gradient check      -- reverse-mode autodiff vs central finite differences,
                             at a fixed PRNG key (common random numbers), in float64.
  (C) closure             -- synthesise data at known theta*, re-init, fit, and
                             recover theta* within statistical error.

Plus measure_snr: gradient signal-to-noise vs cascade depth / sample budget, the
make-or-break property for the deep cascade.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from adonis.core.autodiff import Adam


# --- per-module self-test contract ------------------------------------------- #
# Every module (Channel, NuclearModel, FluxModel, FSIModel) exposes two uniform
# self-tests so the chain can be validated component-by-component (and from CI):
#   closure_test() -- the module's standalone differentiability (autodiff == FD
#                     on its differentiable output), and
#   oracle_test()  -- its physics validity vs the ACHILLES oracle, where applicable.
# Modules with no differentiable parameter (a detached sampler) or no oracle return
# a SKIPPED result rather than failing, so the contract is uniform across the tree.
class TestResult(NamedTuple):
    name: str            # e.g. "DCCSinglePion.closure"
    kind: str            # "closure" | "oracle"
    passed: bool         # True if it ran and met tolerance (True when skipped)
    skipped: bool        # True if not applicable to this module
    detail: str          # one-line human summary
    metrics: dict        # raw numbers (rel errors, chi2/ndf, ...)

    def __bool__(self):
        return self.passed


def skipped_result(name, kind, reason) -> TestResult:
    return TestResult(name, kind, True, True, f"SKIPPED: {reason}", {})


class SelfTestMixin:
    """Default (skipped) self-tests, so every module in the chain exposes the same
    `closure_test` / `oracle_test` contract.  Concrete modules override the ones
    that are meaningful for them."""

    def closure_test(self, key=None, **kw) -> "TestResult":
        return skipped_result(f"{type(self).__name__}.closure", "closure",
                              "no closure_test for this module")

    def oracle_test(self, oracle=None, key=None, **kw) -> "TestResult":
        return skipped_result(f"{type(self).__name__}.oracle", "oracle",
                              "no oracle_test for this module")


def grad_closure(f: Callable[[float], float], name, *, x0=1.0, eps=2e-3,
                 tol=1e-3) -> TestResult:
    """Closure for a scalar differentiable output: autodiff d f/dx vs central FD.

    `f` maps a single physics knob (e.g. M_A) to a scalar (e.g. total xsec, or a
    bin yield).  In the kind-1 reweighting estimator the proposal is detached, so
    the two must agree to ~machine-times-conditioning -- a tight `tol` is expected.
    """
    g_ad = float(jax.grad(lambda x: f(x))(x0))
    g_fd = float((f(x0 + eps) - f(x0 - eps)) / (2 * eps))
    rel = abs(g_ad - g_fd) / (abs(g_ad) + abs(g_fd) + 1e-30)
    return TestResult(
        name, "closure", bool(rel < tol), False,
        f"d/dx autodiff {g_ad:+.4e} vs FD {g_fd:+.4e}  rel {rel:.2e} "
        f"(tol {tol:.0e})",
        {"autodiff": g_ad, "finite_diff": g_fd, "rel_err": rel, "tol": tol},
    )


def chi2_ndf(model, model_err, oracle, oracle_err, *, floor=0.0):
    """Per-bin chi2 and ndf of a normalised model vs oracle, with combined errors.

    Both histograms are normalised to unit area first (shape comparison).  Bins
    with oracle density <= `floor` (relative to the max) are dropped from the ndf.
    """
    m = np.asarray(model, float); me = np.asarray(model_err, float)
    o = np.asarray(oracle, float); oe = np.asarray(oracle_err, float)
    Sm, So = m.sum(), o.sum()
    m, me = m / Sm, me / Sm
    o, oe = o / So, oe / So
    good = o > floor * (o.max() if o.size else 0.0)
    err2 = me ** 2 + oe ** 2
    good &= err2 > 0
    chi2 = float(np.sum((m[good] - o[good]) ** 2 / err2[good]))
    return chi2, int(good.sum())


# --- (F) forward agreement --------------------------------------------------- #
class ForwardAgreement(NamedTuple):
    max_abs_diff: float
    max_rel_diff: float
    mc_tol: float
    passed: bool


def check_forward_agreement(hard_hist, weighted_hist, n_data, n_model, n_sigma=5.0):
    """Compare a hard-sampled histogram (counts) to the differentiable estimator.

    Both are expected counts; the weighted estimator is rescaled outside. The
    tolerance is a Poisson-style MC band: sqrt(count) per bin, scaled by n_sigma.
    """
    hard = np.asarray(hard_hist, dtype=float)
    model = np.asarray(weighted_hist, dtype=float)
    diff = np.abs(hard - model)
    # combined statistical error on the difference (both are MC estimates)
    err = np.sqrt(np.maximum(hard, 1.0) * (1.0 + n_data / n_model))
    mc_tol = float(n_sigma * np.max(err))
    rel = diff / np.maximum(np.abs(hard), 1.0)
    passed = bool(np.all(diff <= n_sigma * err))
    return ForwardAgreement(float(diff.max()), float(rel.max()), mc_tol, passed)


# --- (G) finite-difference gradient check ------------------------------------ #
class GradCheck(NamedTuple):
    autodiff: np.ndarray
    finite_diff: np.ndarray
    max_rel_err: float
    passed: bool


class JacobianCheck(NamedTuple):
    max_rel_err: float
    weighted_rel_err: float
    passed: bool


def check_expected_jacobian(hist_fn, theta, key, eps=3e-3, tol=5e-2,
                            n_keys=64, count_floor_frac=1e-2):
    """Validate d E[hist_b]/d theta_i : autodiff vs finite difference.

    This is the *reliable* gradient gate for a stochastic (score-function)
    estimator. Rather than finite-differencing a noisy scalar loss -- where FD
    amplifies the gradient noise by 1/eps -- we check the gradient of the
    **expected histogram** itself. E[hist] is smooth and can be estimated to high
    precision (large n inside `hist_fn`, averaged over `n_keys` keys with common
    random numbers), so its finite difference is a trustworthy reference.

    `hist_fn(theta, key) -> (n_bins,)` should return the *expected-count* (or
    probability) histogram. Bins with negligible occupancy (< count_floor_frac of
    the max) are ignored in the relative-error summary.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    keys = list(jax.random.split(key, n_keys))

    def mean_hist(t):
        return jnp.mean(jnp.stack([hist_fn(t, k) for k in keys]), axis=0)

    # forward-mode: with few parameters and many output bins this is far cheaper
    # (and far lighter on memory) than reverse-mode, which would vmap one cotangent
    # per bin through the whole trajectory graph.
    jac_fn = jax.jit(jax.jacfwd(lambda t, k: hist_fn(t, k)))
    J_ad = np.mean([np.asarray(jac_fn(theta, k)) for k in keys], axis=0)  # (n_bins, n_params)

    base = np.asarray(theta, dtype=float)
    J_fd = np.zeros_like(J_ad)
    for i in range(base.size):
        tp = base.copy(); tp[i] += eps
        tm = base.copy(); tm[i] -= eps
        J_fd[:, i] = np.asarray((mean_hist(jnp.asarray(tp)) - mean_hist(jnp.asarray(tm))) / (2 * eps))

    ref = np.asarray(mean_hist(theta))
    mask = ref > count_floor_frac * ref.max()
    denom = np.maximum(np.abs(J_ad) + np.abs(J_fd), 1e-12)
    rel = np.abs(J_ad - J_fd) / denom
    # weight the relative error by bin occupancy so tiny bins don't dominate
    w = (ref * mask)[:, None]
    weighted = float((rel * w).sum() / np.maximum(w.sum() * J_ad.shape[1], 1e-12) * J_ad.shape[1])
    max_rel = float(rel[mask].max())
    return JacobianCheck(max_rel, weighted, bool(weighted <= tol))


def check_gradient(loss_fn, theta, key, eps=1e-4, tol=1e-2, n_keys=1):
    """Autodiff vs central finite differences for a scalar `loss_fn(theta, key)`.

    A score-function (REINFORCE) estimator gives the gradient of the *expectation*
    E_key[loss], not of `loss` at one fixed key: for a single key the discrete
    sampled choices jump as theta moves, so a single-key finite difference can
    never match the smooth score gradient. The correct test therefore averages
    BOTH the autodiff gradient and the finite difference over a batch of `n_keys`
    keys (common random numbers: the same batch is reused at theta +/- eps, so the
    jumps average out to the true expected gradient). With `n_keys=1` this reduces
    to the exact check used for deterministic (kind-1-only) estimators like R0.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    keys = ([key] if n_keys == 1
            else list(jax.random.split(key, n_keys)))

    grad_fn = jax.jit(jax.grad(lambda t, k: loss_fn(t, k)))
    g_ad = np.mean([np.asarray(grad_fn(theta, k)) for k in keys], axis=0)

    def mean_loss(t):
        return float(np.mean([float(loss_fn(t, k)) for k in keys]))

    g_fd = np.zeros_like(g_ad)
    base = np.asarray(theta, dtype=float)
    for i in range(base.size):
        tp = base.copy(); tp[i] += eps
        tm = base.copy(); tm[i] -= eps
        g_fd[i] = (mean_loss(jnp.asarray(tp)) - mean_loss(jnp.asarray(tm))) / (2 * eps)

    denom = np.maximum(np.abs(g_ad) + np.abs(g_fd), 1e-12)
    rel = np.abs(g_ad - g_fd) / denom
    max_rel = float(rel.max())
    return GradCheck(g_ad, g_fd, max_rel, bool(max_rel <= tol))


# --- (C) closure test -------------------------------------------------------- #
class Closure(NamedTuple):
    theta_true: np.ndarray
    theta_fit: np.ndarray
    loss_history: np.ndarray
    param_history: np.ndarray
    max_abs_err: float


def run_closure(loss_fn, to_params, init_unconstrained, key,
                iterations=300, learning_rate=0.1, theta_true=None, verbose=True,
                clip_norm=None, final_lr_frac=1.0):
    """Fit unconstrained parameters by Adam on `loss_fn(unconstrained, key)`.

    `to_params(unconstrained)` maps to the physical parameters for reporting.
    `clip_norm`, if set, clips the global gradient norm each step -- this keeps a
    high-variance stochastic gradient from kicking the parameters into pathological
    regions (e.g. mean free paths so short the fixed bounce budget can't deplete).
    `final_lr_frac < 1` exponentially anneals the learning rate from `learning_rate`
    to `learning_rate * final_lr_frac` over the run -- needed when a parameter sits
    in a shallow basin whose restoring force is comparable to the gradient noise, so
    a constant step lets it wander (and drift under Adam momentum) instead of settle.
    Returns the fit history; closure is asserted by the caller against theta_true.
    """
    theta = jnp.asarray(init_unconstrained, dtype=jnp.float64)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    opt = Adam(learning_rate)
    opt_state = opt.init(theta)

    def _clip(g):
        if clip_norm is None:
            return g
        norm = jnp.sqrt(jnp.sum(g ** 2))
        return jnp.where(norm > clip_norm, g * clip_norm / (norm + 1e-30), g)

    loss_hist, param_hist = [], []
    for it in range(iterations):
        key, sub = jax.random.split(key)
        opt.lr = learning_rate * (final_lr_frac ** (it / max(iterations - 1, 1)))
        loss_value, grads = value_and_grad(theta, sub)
        grads = _clip(jax.tree_util.tree_map(jnp.nan_to_num, grads))
        theta, opt_state = opt.update(theta, grads, opt_state)
        param_hist.append(np.asarray(to_params(theta)))
        loss_hist.append(float(loss_value))
        if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
            shown = "  ".join(f"{p:.4f}" for p in np.atleast_1d(param_hist[-1]))
            print(f"  iter {it:4d}  loss={loss_value:.4e}   params=[{shown}]")

    theta_fit = np.asarray(to_params(theta))
    max_err = (float(np.max(np.abs(theta_fit - np.asarray(theta_true))))
               if theta_true is not None else float("nan"))
    return Closure(np.asarray(theta_true) if theta_true is not None else None,
                   theta_fit, np.asarray(loss_hist), np.asarray(param_hist), max_err)


# --- (§13) gradient signal-to-noise ------------------------------------------ #
def measure_snr(grad_fn, theta, keys):
    """Gradient SNR = |E[g]| / std[g] per parameter, over a batch of PRNG keys.

    `grad_fn(theta, key)` returns the gradient vector for one key. High SNR means
    the deep-cascade variance is under control (Strategy §13).
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    grads = np.stack([np.asarray(grad_fn(theta, k)) for k in keys])
    mean = grads.mean(axis=0)
    std = grads.std(axis=0) + 1e-30
    return mean / std, mean, std
