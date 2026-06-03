"""Component D, rung R0 (Strategy §12): Oset pion absorption.

The Oset model gives the imaginary part of the Delta self-energy in nuclear matter
as a sum of terms with different powers of the local density (paper Eq. for
Im Sigma_Delta): quasi-elastic (~rho), two-nucleon absorption (~rho^2), and
three-nucleon absorption (~rho^3). Here we model the *absorption* part, with a
per-unit-length absorption rate

    lambda(T_pi, x) = C_A2 * g2(T_pi) * (rho(x)/rho0)^2
                    + C_A3 * g3(T_pi) * (rho(x)/rho0)^3,

where rho(x) is the local density along the pion's path (a smooth nuclear bump)
and g2, g3 are fixed energy profiles -- 2N absorption peaks near the Delta, 3N
absorption rises with energy. A pion of kinetic energy T_pi traverses the medium
in K steps and is either absorbed at some depth or transmitted.

We recover the absorption coefficients (C_A2, C_A3) from the 2-D observable
(T_pi, absorption depth) + a transmitted bin. The two coefficients are separable
because they imprint differently in BOTH dimensions: the rho^3 term concentrates
absorption in the dense core (deeper depth bins) while rho^2 is broader, and g2
vs g3 give different energy dependence.

Gradient is pure kind-1 (expected-value deposits with smooth, parameter-dependent
rates); T_pi is a detached label. Like C-R0 / A-R0 the estimator is exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ConfigDR0:
    L: float = 8.0                   # path length through the nucleus [fm]
    K: int = 20                      # transport steps
    rho_peak: float = 1.30           # peak density / rho0
    rho_w: float = 2.0               # density bump width [fm]
    T_min: float = 0.05
    T_max: float = 0.35              # pion kinetic energy window [GeV]
    true_CA2: float = 0.12           # 2N absorption coefficient [1/fm]
    true_CA3: float = 0.06           # 3N absorption coefficient [1/fm]
    init_CA2: float = 0.06
    init_CA3: float = 0.12
    n_T: int = 20
    n_data: int = 400_000
    n_model: int = 200_000
    iterations: int = 300
    learning_rate: float = 0.03
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def dx(self):
        return self.L / self.K

    @property
    def n_bins(self):
        return self.n_T * (self.K + 1)       # (T bins) x (K depth bins + transmitted)


def density_profile(cfg: ConfigDR0):
    """rho(x)/rho0 on the K-step grid (a smooth nuclear bump)."""
    x = (jnp.arange(cfg.K) + 0.5) * cfg.dx
    return cfg.rho_peak * jnp.exp(-0.5 * ((x - cfg.L / 2) / cfg.rho_w) ** 2)


def g2(T):
    """2N absorption energy profile: peaks near the Delta (~180 MeV)."""
    return jnp.exp(-((T - 0.18) / 0.08) ** 2)


def g3(T):
    """3N absorption energy profile: rises with energy."""
    return T / 0.35


def _T_bin(cfg, T):
    return jnp.clip(((T - cfg.T_min) / (cfg.T_max - cfg.T_min) * cfg.n_T).astype(jnp.int32),
                    0, cfg.n_T - 1)


# --------------------------------------------------------------------------- #
# Differentiable estimator (expected-value deposits) and hard reference
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def weighted_histogram(params, key, cfg: ConfigDR0, n=None):
    n = cfg.n_model if n is None else n
    CA2, CA3 = params
    rho = density_profile(cfg)
    T = jax.lax.stop_gradient(jax.random.uniform(key, (n,), minval=cfg.T_min, maxval=cfg.T_max))
    Tb = _T_bin(cfg, T)
    g2T, g3T = g2(T), g3(T)

    weight = jnp.ones(n)
    hist = jnp.zeros(cfg.n_bins)
    for k in range(cfg.K):
        lam = CA2 * g2T * rho[k] ** 2 + CA3 * g3T * rho[k] ** 3
        absorb = -jnp.expm1(-lam * cfg.dx)
        survive = jnp.exp(-lam * cfg.dx)
        hist = hist.at[Tb * (cfg.K + 1) + k].add(weight * absorb)
        weight = weight * survive
    hist = hist.at[Tb * (cfg.K + 1) + cfg.K].add(weight)         # transmitted
    return hist


@partial(jax.jit, static_argnums=(2, 3))
def sampled_histogram(params, key, cfg: ConfigDR0, n=None):
    n = cfg.n_data if n is None else n
    CA2, CA3 = params
    rho = density_profile(cfg)
    kT, ks = jax.random.split(key)
    T = jax.random.uniform(kT, (n,), minval=cfg.T_min, maxval=cfg.T_max)
    Tb = _T_bin(cfg, T)
    g2T, g3T = g2(T), g3(T)

    alive = jnp.ones(n, bool)
    landed = jnp.full(n, cfg.K, jnp.int32)                       # default: transmitted depth
    for k, sub in enumerate(jax.random.split(ks, cfg.K)):
        lam = CA2 * g2T * rho[k] ** 2 + CA3 * g3T * rho[k] ** 3
        absorb = -jnp.expm1(-lam * cfg.dx)
        hit = alive & (jax.random.uniform(sub, (n,)) < absorb)
        landed = jnp.where(hit, k, landed)
        alive = alive & ~hit
    return jnp.zeros(cfg.n_bins).at[Tb * (cfg.K + 1) + landed].add(1.0)
