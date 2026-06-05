"""Component E, rung R0 (Strategy §12): the propagating Delta resonance.

The GiBUU-style picture where the Delta is a propagating degree of freedom. A
Delta is created off-shell with invariant mass mu drawn from its spectral function

    A_Delta(mu; m, Gamma) = (1/pi) * mu * Gamma / ((m^2 - mu^2)^2 + mu^2 Gamma^2),

with fixed momentum p_Delta, and then propagates through a uniform medium in time
steps dt. At each step it competes between two fates:

  * decay  Delta -> pi N   at lab-frame rate 1/tau, tau = gamma * hbar*c / Gamma
           (paper: P = 1 - exp(-dt/tau), gamma = E/mu the Lorentz factor) -> a pion
           survives into the final state;
  * absorption Delta N -> N N at rate sigma_abs -> no pion (pion absorbed).

The competition between decay and absorption is exactly how the propagating-Delta
mode produces pion absorption. We recover (m_Delta, Gamma, sigma_abs) from the
2-D observable (off-shell mass mu, outcome), where outcome is the decay time step
or "absorbed".

Differentiable handles (Strategy §9):
  * initial weight score_weight(A_Delta(mu; m, Gamma))  -- spectral function (kind 3)
  * expected-value deposits of the decay-now / absorb-now fractions each step,
    with the decay rate 1/tau depending on Gamma (and mu via gamma) -- (kind 1).
  m_Delta enters only through the spectral function; Gamma enters through BOTH the
  spectral function and the decay rate; sigma_abs only through absorption.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .kernel import score_weight

HBARC = 0.1973           # GeV * fm


@dataclass(frozen=True)
class ConfigER0:
    p_Delta: float = 0.30            # Delta momentum [GeV/c]
    mu_min: float = 1.08
    mu_max: float = 1.42             # off-shell mass window [GeV]
    dt: float = 0.4                  # time step [fm/c]
    K: int = 24                      # number of time steps
    true_mDelta: float = 1.232
    true_Gamma: float = 0.112        # vacuum width [GeV]
    true_sigma_abs: float = 0.15     # absorption rate [c/fm]
    init_mDelta: float = 1.27
    init_Gamma: float = 0.150
    init_sigma_abs: float = 0.30
    n_mu: int = 24
    n_data: int = 400_000
    n_model: int = 200_000
    iterations: int = 300
    learning_rate: float = 0.03
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def n_out(self):
        return self.K + 1            # K decay-step bins + 1 absorbed bin

    @property
    def n_bins(self):
        return self.n_mu * self.n_out


# --------------------------------------------------------------------------- #
# Spectral function
# --------------------------------------------------------------------------- #
def A_raw(mu, m, G):
    return (1.0 / jnp.pi) * mu * G / ((m ** 2 - mu ** 2) ** 2 + mu ** 2 * G ** 2)


def _mu_grid(cfg):
    return jnp.linspace(cfg.mu_min, cfg.mu_max, 513)


def A_density(mu, m, G, cfg):
    """Spectral function normalised to a pdf over [mu_min, mu_max]."""
    grid = _mu_grid(cfg)
    Z = jnp.trapezoid(A_raw(grid, m, G), grid)
    return A_raw(mu, m, G) / Z


def _sample_mu(u, m, G, cfg):
    grid = _mu_grid(cfg)
    cdf = jnp.cumsum(A_raw(grid, m, G)); cdf = cdf / cdf[-1]
    return jnp.interp(u, cdf, grid)


def _gamma(mu, p):
    return jnp.sqrt(p ** 2 + mu ** 2) / mu


def _mu_bin(cfg, mu):
    return jnp.clip(((mu - cfg.mu_min) / (cfg.mu_max - cfg.mu_min) * cfg.n_mu).astype(jnp.int32),
                    0, cfg.n_mu - 1)


# --------------------------------------------------------------------------- #
# Differentiable estimator (expected-value deposits) and hard reference
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def weighted_histogram(params, key, cfg: ConfigER0, n=None):
    n = cfg.n_model if n is None else n
    mD, G, sig_abs = params
    ku, = jax.random.split(key, 1)
    mu = jax.lax.stop_gradient(_sample_mu(jax.random.uniform(ku, (n,)), mD, G, cfg))
    mub = _mu_bin(cfg, mu)

    lam_dec = G / (_gamma(mu, cfg.p_Delta) * HBARC)      # decay rate 1/tau [c/fm]
    lam_abs = jnp.full(n, sig_abs)
    lam_tot = lam_dec + lam_abs
    p_something = -jnp.expm1(-lam_tot * cfg.dt)
    survive = jnp.exp(-lam_tot * cfg.dt)
    f_decay = (lam_dec / lam_tot) * p_something          # decay-now fraction
    f_abs = (lam_abs / lam_tot) * p_something            # absorb-now fraction

    weight = score_weight(A_density(mu, mD, G, cfg))     # spectral-function gradient
    hist = jnp.zeros(cfg.n_bins)
    for k in range(cfg.K):
        hist = hist.at[mub * cfg.n_out + k].add(weight * f_decay)        # decay at step k, per mu
        hist = hist.at[mub * cfg.n_out + cfg.K].add(weight * f_abs)      # absorbed, per mu (per particle)
        weight = weight * survive
    hist = hist.at[mub * cfg.n_out + (cfg.K - 1)].add(weight)   # residual -> late decay
    return hist


@partial(jax.jit, static_argnums=(2, 3))
def sampled_histogram(params, key, cfg: ConfigER0, n=None):
    n = cfg.n_data if n is None else n
    mD, G, sig_abs = params
    ku, ks = jax.random.split(key)
    mu = _sample_mu(jax.random.uniform(ku, (n,)), mD, G, cfg)
    mub = _mu_bin(cfg, mu)

    lam_dec = G / (_gamma(mu, cfg.p_Delta) * HBARC)
    lam_tot = lam_dec + sig_abs
    p_something = -jnp.expm1(-lam_tot * cfg.dt)
    p_decay_given = lam_dec / lam_tot

    alive = jnp.ones(n, bool)
    outcome = jnp.full(n, cfg.K - 1, jnp.int32)          # residual -> late decay
    for k, sub in enumerate(jax.random.split(ks, cfg.K)):
        ke, kc = jax.random.split(sub)
        happens = alive & (jax.random.uniform(ke, (n,)) < p_something)
        is_decay = jax.random.uniform(kc, (n,)) < p_decay_given
        outcome = jnp.where(happens & is_decay, k, outcome)          # decay at step k
        outcome = jnp.where(happens & ~is_decay, cfg.K, outcome)     # absorbed
        alive = alive & ~happens
    return jnp.zeros(cfg.n_bins).at[mub * cfg.n_out + outcome].add(1.0)
