"""Gradient-based parameter fitting on differentiable ADoNIS predictions.

`fit_scalar` recovers one physics parameter by Adam on a scalar loss(x), using FORWARD-mode
(`jvp`) gradients -- as cheap as reverse mode for a scalar but far lighter on memory
(reverse-mode through the spline amplitude interp blows up).  Learning-rate annealing settles
the noisy two-replica gradient.  `fit_pytree` is the multi-parameter generalisation
(reverse-mode grad over a PhysicsParams pytree, for when several knobs are fit jointly).
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

from adonis.core.autodiff import Adam


def fit_scalar(loss, x0, lr=0.05, iters=200, final_lr_frac=0.02, verbose=True, label=""):
    """Minimise scalar `loss(x)` (x a float) from x0. Returns (x_final, trajectory, losses).
    Also returns the autodiff-vs-finite-difference grad ratio at x0 as a sanity check."""
    vg = jax.jit(lambda x: jax.jvp(loss, (x,), (1.0,)))     # (value, dvalue/dx)
    g_ad = float(vg(x0)[1])
    eps = 5e-3
    g_fd = float((vg(x0 + eps)[0] - vg(x0 - eps)[0]) / (2 * eps))
    rel = abs(g_ad - g_fd) / (abs(g_ad) + abs(g_fd) + 1e-30)
    if verbose:
        print(f"[fit {label}] grad@x0 AD={g_ad:.4e} FD={g_fd:.4e} rel={rel:.2e}", flush=True)
    x = jnp.asarray(float(x0)); opt = Adam(lr); st = opt.init(x)
    hist = [float(x)]; losses = []; t0 = time.time()
    for it in range(iters):
        opt.lr = lr * (final_lr_frac ** (it / max(iters - 1, 1)))
        l, g = vg(x); x, st = opt.update(x, jnp.asarray(g), st)
        hist.append(float(x)); losses.append(float(l))
        if verbose and (it % 30 == 0 or it == iters - 1):
            r = (it + 1) / (time.time() - t0)
            print(f"  iter {it:3d}/{iters} loss={float(l):.4e} x={float(x):.4f} "
                  f"({r:.1f} it/s)", flush=True)
    return float(x), np.array(hist), np.array(losses), rel


def fit_pytree(loss, params0, lr=0.02, iters=300, final_lr_frac=0.05, verbose=True):
    """Minimise `loss(params)` over a PhysicsParams pytree (reverse-mode grad). Returns
    (params_final, loss_history)."""
    vg = jax.jit(jax.value_and_grad(loss))
    params = params0; opt = Adam(lr); st = opt.init(params)
    losses = []
    for it in range(iters):
        opt.lr = lr * (final_lr_frac ** (it / max(iters - 1, 1)))
        l, g = vg(params)
        params, st = opt.update(params, jax.tree_util.tree_map(jnp.nan_to_num, g), st)
        losses.append(float(l))
        if verbose and (it % 30 == 0 or it == iters - 1):
            print(f"  iter {it:3d}/{iters} loss={float(l):.4e}", flush=True)
    return params, np.array(losses)
