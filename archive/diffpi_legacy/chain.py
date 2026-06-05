"""Compositional driver for the differentiable cascade (Strategy §9, §11).

Every event carries one scalar **weight** `w` (initialised to 1) and a bundle of
**detached kinematics** (the State). A physics block is a `Step`: a pure function

    step(state, params, key) -> (new_state, weight_terms, deposits)

The *driver* — not the physics code — owns weight accumulation and the geometry
detach, so the two invariants that are easiest to get wrong ("carry exactly one
global weight" and "stop_gradient the geometry every step") are structural.

Weight terms come in the three kinds of Strategy §9:

    RAW(p)        carried as `p` itself        (kind 1: smooth prob, reparameterised)
    CHOICE(pi, c) -> score_weight(pi[c])       (kind 2: sampled categorical)
    SHAPE(p)      -> score_weight(p)            (kind 3: sampled continuous shape;
                                                 `p` is the density at the *detached* sample)

Deposits are expected-value contributions `(bins, fracs)`: each carries the
fraction of the current weight that leaves the system now (escape / absorb), and
the driver adds `w * frac` into a hard bin. The kind-1 gradient flows through
`frac`; the carried-forward weight picks up the kinds-1/2/3 factors.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util

from .kernel import score_weight


# --- weight terms ------------------------------------------------------------ #
class RAW(NamedTuple):
    """A smooth probability carried as itself (kind 1). Shape (n,) or scalar."""
    p: jax.Array


class CHOICE(NamedTuple):
    """A sampled categorical (kind 2). `probs` is (n, n_cat), `chosen` is (n,)."""
    probs: jax.Array
    chosen: jax.Array


class SHAPE(NamedTuple):
    """A sampled continuous shape (kind 3). `density` is (n,), the pdf evaluated
    at the detached sample."""
    density: jax.Array


def _fold(term, w):
    """Multiply weight term into the carried weight `w`."""
    if isinstance(term, RAW):
        return w * term.p
    if isinstance(term, CHOICE):
        n = term.chosen.shape[0]
        chosen_prob = term.probs[jnp.arange(n), term.chosen]
        return w * score_weight(chosen_prob)
    if isinstance(term, SHAPE):
        return w * score_weight(term.density)
    raise TypeError(f"unknown weight term {type(term)}")


class Deposit(NamedTuple):
    bins: jax.Array       # (n,) int   destination bin per particle
    fracs: jax.Array      # (n,) float fraction of weight escaping into that bin now


def run_chain(steps, init_state, params, key, n, n_bins):
    """Run the differentiable (weighted / expected-value) estimator.

    `steps` is a sequence of `Step` callables. Returns the differentiable
    histogram of length `n_bins`. Forward value is an honest expected count;
    gradient is the exact d/dtheta E[histogram].
    """
    state = init_state
    w = jnp.ones(n)
    hist = jnp.zeros(n_bins)
    for step in steps:
        key, sub = jax.random.split(key)
        state, weight_terms, deposits = step(state, params, sub)
        for dep in deposits:
            hist = hist.at[dep.bins].add(w * dep.fracs)
        for term in weight_terms:
            w = _fold(term, w)
        state = tree_util.tree_map(jax.lax.stop_gradient, state)  # geometry detach, enforced centrally
    return hist
