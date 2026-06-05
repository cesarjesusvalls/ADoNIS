"""Component C, rung R0 (Strategy §12): 1-D slab transport, single channel.

A particle enters a homogeneous slab of length `D` at x=0 and steps through it in
`K` steps of length `dx = D/K`. With mean free path `L`, each step it either
interacts (probability `1 - exp(-dx/L)`) -- recorded in that step's depth bin and
removed -- or survives and continues. Particles that survive all `K` steps land in
a terminal "transmitted" bin. The observable is the histogram of interaction depth
(bins 0..K-1) plus the transmitted bin (index K): a truncated geometric law whose
shape depends smoothly on `L`.

Single channel => the gradient w.r.t. `L` is entirely kind-1 (expected-value
deposits, Strategy §9); there is no categorical or shape score weight yet. The
differentiable estimator is therefore *exact* (zero variance) -- which is exactly
what we want from R0: it isolates and validates the deposit machinery, the chain
driver, and the harness, before R1 introduces the sampled (variance-bearing)
channel and angle choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .chain import RAW, Deposit, run_chain
from .kernel import to_positive, from_positive


@dataclass(frozen=True)
class ConfigR0:
    D: float = 10.0          # slab length [fm]
    K: int = 20              # number of transport steps
    L_true: float = 3.0      # true mean free path [fm]
    L_init: float = 6.0      # starting guess
    n_data: int = 200_000
    n_model: int = 50_000    # irrelevant to accuracy here (estimator is exact), kept for uniformity
    iterations: int = 200
    learning_rate: float = 0.1
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def dx(self):
        return self.D / self.K

    @property
    def n_bins(self):
        return self.K + 1            # K depth bins + 1 transmitted bin


def _per_step_probs(L, dx):
    survival = jnp.exp(-dx / L)
    interact = -jnp.expm1(-dx / L)   # 1 - exp(-dx/L), accurately
    return interact, survival


# --- differentiable estimator (expected-value deposits via the chain driver) --- #
def _make_steps(cfg: ConfigR0):
    """One Step per interaction depth, plus a terminal transmitted Step."""
    def interaction_step(k):
        def step(state, params, key, k=k):
            n = state["n"]
            interact, survival = _per_step_probs(params["L"], cfg.dx)
            deposits = [Deposit(bins=jnp.full(n, k, jnp.int32),
                                fracs=jnp.full(n, interact))]
            return state, [RAW(survival)], deposits
        return step

    def transmitted_step(state, params, key):
        n = state["n"]
        deposits = [Deposit(bins=jnp.full(n, cfg.K, jnp.int32), fracs=jnp.ones(n))]
        return state, [], deposits

    return [interaction_step(k) for k in range(cfg.K)] + [transmitted_step]


@partial(jax.jit, static_argnums=(2,))
def weighted_histogram(L, key, cfg: ConfigR0, n=None):
    """Differentiable expected-count histogram of length K+1."""
    n = cfg.n_model if n is None else n
    steps = _make_steps(cfg)
    return run_chain(steps, {"n": n}, {"L": L}, key, n, cfg.n_bins)


# --- hard reference sampler (one count per particle) ------------------------- #
@partial(jax.jit, static_argnums=(2,))
def sampled_histogram(L, key, cfg: ConfigR0, n=None):
    """Honest hard sampler: each particle interacts or transmits exactly once."""
    n = cfg.n_data if n is None else n
    interact, _ = _per_step_probs(L, cfg.dx)
    alive = jnp.ones(n, bool)
    landed = jnp.full(n, cfg.K, jnp.int32)        # default: transmitted
    for k, sub in enumerate(jax.random.split(key, cfg.K)):
        hit = alive & (jax.random.uniform(sub, (n,)) < interact)
        landed = jnp.where(hit, k, landed)
        alive = alive & ~hit
    return jnp.zeros(cfg.n_bins).at[landed].add(1.0)
