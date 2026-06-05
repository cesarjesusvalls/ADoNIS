"""Component A, rung R0 (Strategy §12, §15): the hard electroweak vertex.

The project's namesake piece: charged-current single-pion production,
nu + N -> l^- + N' + pi, via the Delta resonance, at fixed neutrino energy. An
event is characterised by the momentum transfer Q^2 and the hadronic invariant
mass W. The double-differential rate is

    R(Q^2, W) = F_A(Q^2; M_A)^2 * BW(W; m_Delta, Gamma_Delta)

with the dipole axial form factor and a Delta Breit-Wigner,

    F_A(Q^2; M_A) = 1 / (1 + Q^2 / M_A^2)^2            (F_A(0) = 1)
    BW(W; m, G)   = (G/2)^2 / ((W - m)^2 + (G/2)^2).

The Q^2 spectrum's steepness encodes the **axial mass M_A** -- the canonical
tunable parameter of neutrino single-pion production -- while the W spectrum
encodes the Delta mass and width. We recover (M_A, m_Delta, Gamma_Delta) jointly
from the 2-D (Q^2, W) histogram.

This validates the *reweighting* estimator (Strategy §10): the kinematics
(Q^2, W) are sampled uniformly over the phase space and **detached**, and the
matrix element R(Q^2, W; knobs) is carried as the differentiable weight. All
gradient flows through that weight (kind 1, reparameterised) -- there is no
sampled discrete choice here, so no score weight is needed. A future rung adds
the W-dependent kinematic boundary Q^2_max(W) and vector form factors.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ConfigAR0:
    E_nu: float = 1.0                 # neutrino energy [GeV] (informational for R0)
    Q2_max: float = 1.5              # [GeV^2]
    W_min: float = 1.08
    W_max: float = 1.50              # [GeV]
    true_MA: float = 1.00            # axial mass [GeV]
    true_mDelta: float = 1.232
    true_GammaDelta: float = 0.117
    init_MA: float = 1.30
    init_mDelta: float = 1.27
    init_GammaDelta: float = 0.15
    n_Q2: int = 24
    n_W: int = 24
    n_data: int = 400_000
    n_model: int = 200_000
    iterations: int = 400
    learning_rate: float = 0.03
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def n_bins(self):
        return self.n_Q2 * self.n_W


# --------------------------------------------------------------------------- #
# Matrix element
# --------------------------------------------------------------------------- #
def axial_ff(Q2, MA):
    """Dipole axial form factor F_A(Q^2; M_A), normalised to F_A(0) = 1."""
    return 1.0 / (1.0 + Q2 / MA ** 2) ** 2


def breit_wigner(W, m, G):
    return (G / 2) ** 2 / ((W - m) ** 2 + (G / 2) ** 2)


def rate(Q2, W, MA, mD, GD):
    """Double-differential single-pion-production rate R(Q^2, W)."""
    return axial_ff(Q2, MA) ** 2 * breit_wigner(W, mD, GD)


def _rate_ref(cfg):
    # R <= F_A(0)^2 * BW(m_Delta) = 1 * 1 = 1; pad for safety.
    return 1.2


def _bins(cfg, Q2, W):
    qb = jnp.clip((Q2 / cfg.Q2_max * cfg.n_Q2).astype(jnp.int32), 0, cfg.n_Q2 - 1)
    wb = jnp.clip(((W - cfg.W_min) / (cfg.W_max - cfg.W_min) * cfg.n_W).astype(jnp.int32),
                  0, cfg.n_W - 1)
    return qb * cfg.n_W + wb


# --------------------------------------------------------------------------- #
# Differentiable estimator (reweighting) and hard reference
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def weighted_histogram(params, key, cfg: ConfigAR0, n=None):
    n = cfg.n_model if n is None else n
    MA, mD, GD = params
    kq, kw = jax.random.split(key)
    Q2 = jax.lax.stop_gradient(jax.random.uniform(kq, (n,)) * cfg.Q2_max)
    W = jax.lax.stop_gradient(jax.random.uniform(kw, (n,), minval=cfg.W_min, maxval=cfg.W_max))
    weight = rate(Q2, W, MA, mD, GD) / _rate_ref(cfg)        # detached kinematics, weight carries grad
    return jnp.zeros(cfg.n_bins).at[_bins(cfg, Q2, W)].add(weight)


@partial(jax.jit, static_argnums=(2, 3))
def sampled_histogram(params, key, cfg: ConfigAR0, n=None):
    n = cfg.n_data if n is None else n
    MA, mD, GD = params
    kq, kw, ka = jax.random.split(key, 3)
    Q2 = jax.random.uniform(kq, (n,)) * cfg.Q2_max
    W = jax.random.uniform(kw, (n,), minval=cfg.W_min, maxval=cfg.W_max)
    accept = jax.random.uniform(ka, (n,)) < (rate(Q2, W, MA, mD, GD) / _rate_ref(cfg))
    bins = jnp.where(accept, _bins(cfg, Q2, W), 0)
    return jnp.zeros(cfg.n_bins).at[bins].add(accept.astype(jnp.float64))
