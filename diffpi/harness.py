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

from .kernel import Adam


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

    jac_fn = jax.jit(jax.jacobian(lambda t, k: hist_fn(t, k)))
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
                clip_norm=None):
    """Fit unconstrained parameters by Adam on `loss_fn(unconstrained, key)`.

    `to_params(unconstrained)` maps to the physical parameters for reporting.
    `clip_norm`, if set, clips the global gradient norm each step -- this keeps a
    high-variance stochastic gradient from kicking the parameters into pathological
    regions (e.g. mean free paths so short the fixed bounce budget can't deplete).
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
