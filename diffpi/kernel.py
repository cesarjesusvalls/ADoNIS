"""Differentiable Monte-Carlo kernel: the score-function weight, parameter
bijections, and a pytree Adam.

This is the reusable core of the project (Strategy §0, §2). It is a lightly
extended port of the reference `differentiable.py` from `differentiable-sampling/`.

The one substantive change is that we enable float64 globally: finite-difference
gradient validation (harness.py) and physics parameter fits are far better
conditioned in double precision than the float32 used by the demo.
"""
from __future__ import annotations

from typing import NamedTuple

import jax

# Enable double precision before any array is created. Every gradient check in
# this project assumes it.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax, tree_util


def score_weight(prob: jax.Array) -> jax.Array:
    """Unit-valued factor whose gradient equals d log(prob).

    Forward value is exactly 1 (the simulation is untouched); the derivative is
    d/dtheta log(prob). Multiply one of these into a trajectory's carried weight
    for each *sampled* random choice whose probability depends on a parameter,
    deposit the product into a hard histogram bin, and the bin contents become
    differentiable with the exact gradient d/dtheta E[counts].
    """
    return prob / lax.stop_gradient(prob)


# --- parameter bijections: optimise in R, evaluate in the valid range -------- #
def to_positive(u):      return jax.nn.softplus(u)          # R   -> (0, inf)
def from_positive(x):    return jnp.log(jnp.expm1(x))
def to_unit(u):          return jax.nn.sigmoid(u)           # R   -> (0, 1)
def from_unit(x):        return jnp.log(x) - jnp.log1p(-x)
def to_signed_unit(u):   return jnp.tanh(u)                 # R   -> (-1, 1)
def from_signed_unit(x): return jnp.arctanh(x)


class _AdamState(NamedTuple):
    mean: object          # pytrees matching the parameters
    variance: object
    count: int


class Adam:
    """Minimal Adam that operates on an arbitrary pytree of parameters."""

    def __init__(self, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = learning_rate, b1, b2, eps

    def init(self, params):
        zeros = tree_util.tree_map(jnp.zeros_like, params)
        return _AdamState(zeros, zeros, 0)

    def update(self, params, grads, state):
        t = state.count + 1
        b1, b2, eps, lr = self.b1, self.b2, self.eps, self.lr
        mean = tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mean, grads)
        var = tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.variance, grads)
        step = tree_util.tree_map(
            lambda m, v: (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps), mean, var)
        params = tree_util.tree_map(lambda p, s: p - lr * s, params, step)
        return params, _AdamState(mean, var, t)
