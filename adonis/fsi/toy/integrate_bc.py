"""Integrated chain B -> C -> G (Strategy §15 milestone 5, toy version).

A single differentiable simulation that chains production (B-like) into transport
(C-like) and bins a final observable (G), then recovers production AND final-state-
interaction (FSI) parameters *jointly* from the post-FSI observable. This is the
project goal in miniature: tune the underlying parameters of the whole chain from
final-state data.

Physics (a faithful toy of the paper's central story -- a pion production spectrum
distorted by FSI):

  Production:  a pion is created at the nucleus centre with momentum p drawn from a
    production spectrum f_prod(p; p_prod) (a truncated Gaussian peaked at p_prod) and
    an isotropic direction. Knob: p_prod (where pions are made).

  Transport:   the pion propagates through a uniform-density sphere (radius R). The
    FSI cross sections are MOMENTUM-DEPENDENT and resonant (peaked at p_res, the
    Delta): sigma_abs(p) = A_abs * BW(p), sigma_sc(p) = A_sc * (BW(p) + floor). So
    pions near the resonance momentum are preferentially absorbed/scattered out,
    carving a p-dependent distortion into the escaping spectrum. |p| is conserved in
    a scatter (elastic toy); scattering lengthens the path and so enhances absorption
    -- the genuine cascade coupling. Knobs: A_abs, A_sc (FSI strengths).

  Observable:  the escaped-pion momentum spectrum (binned in |p|) plus an absorbed
    overflow bin. FSI carves a dip at p_res; production sets the overall peak at
    p_prod. With p_prod != p_res the two are jointly identifiable.

Differentiable handles (Strategy §9):
  * initial weight score_weight(f_prod(p; p_prod))   -- production knob  (kind 3)
  * expected-value escape/scatter deposits via Lambda(p) -- FSI strengths (kind 1)
  * score_weight(channel_prob) for absorb-vs-scatter     -- FSI split    (kind 2)
  * geometry (positions/directions) detached, |p| carried as a detached label
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from .kernel import score_weight
from .component_c_r1 import _distance_to_boundary, _deflect, hg_sample_cos


@dataclass(frozen=True)
class ConfigBC:
    R: float = 3.0                       # nucleus radius [fm]
    p_min: float = 0.10
    p_max: float = 0.50                  # pion momentum window [GeV/c]
    w_prod: float = 0.08                 # production spectrum width (fixed)
    p_res: float = 0.22                  # FSI resonance momentum (fixed)
    Gamma_res: float = 0.06              # FSI resonance width (fixed)
    sc_floor: float = 0.10               # non-resonant scattering floor
    g_scatter: float = 0.30              # HG asymmetry of FSI scattering (fixed)
    # truth / init for the three recovered knobs (p_prod, A_abs, A_sc)
    true_p_prod: float = 0.35
    true_A_abs: float = 0.40
    true_A_sc: float = 0.50
    init_p_prod: float = 0.25
    init_A_abs: float = 0.20
    init_A_sc: float = 0.25
    n_p: int = 24                        # momentum bins; bin n_p = absorbed overflow
    n_bounces: int = 16
    n_data: int = 400_000
    n_model: int = 300_000
    iterations: int = 500
    learning_rate: float = 0.03
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def n_bins(self):
        return self.n_p + 1


# --------------------------------------------------------------------------- #
# Production spectrum (truncated Gaussian)
# --------------------------------------------------------------------------- #
def f_prod_density(p, p_prod, cfg: ConfigBC):
    """Normalised production pdf in p over [p_min, p_max]."""
    w = cfg.w_prod
    g = jnp.exp(-0.5 * ((p - p_prod) / w) ** 2)
    norm = (w * jnp.sqrt(jnp.pi / 2)
            * (jsp.erf((cfg.p_max - p_prod) / (w * jnp.sqrt(2.0)))
               - jsp.erf((cfg.p_min - p_prod) / (w * jnp.sqrt(2.0)))))
    return g / norm


def _sample_p(u, p_prod, cfg: ConfigBC):
    """Inverse-CDF sample of p ~ f_prod via a grid (p is detached downstream)."""
    pg = jnp.linspace(cfg.p_min, cfg.p_max, 513)
    dens = f_prod_density(pg, p_prod, cfg)
    cdf = jnp.cumsum(dens); cdf = cdf / cdf[-1]
    return jnp.interp(u, cdf, pg)


# --------------------------------------------------------------------------- #
# Momentum-dependent FSI rates
# --------------------------------------------------------------------------- #
def _rates(p, A_abs, A_sc, cfg: ConfigBC):
    bw = (cfg.Gamma_res / 2) ** 2 / ((p - cfg.p_res) ** 2 + (cfg.Gamma_res / 2) ** 2)
    sig_abs = A_abs * bw
    sig_sc = A_sc * (bw + cfg.sc_floor)
    Lam = sig_abs + sig_sc
    return Lam, sig_abs / Lam            # total rate [1/fm], P(absorb | interact)


def _p_bin(p, cfg: ConfigBC):
    return jnp.clip(((p - cfg.p_min) / (cfg.p_max - cfg.p_min) * cfg.n_p).astype(jnp.int32),
                    0, cfg.n_p - 1)


def _isotropic_dirs(key, n):
    k1, k2 = jax.random.split(key)
    cos_t = jax.random.uniform(k1, (n,), minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(k2, (n,)) * 2 * jnp.pi
    sin_t = jnp.sqrt(jnp.maximum(1 - cos_t ** 2, 0.0))
    return jnp.stack([sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), cos_t], axis=1)


# --------------------------------------------------------------------------- #
# Differentiable estimator
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def weighted_histogram(params, key, cfg: ConfigBC, n=None):
    n = cfg.n_model if n is None else n
    p_prod, A_abs, A_sc = params
    R, K, N = cfg.R, cfg.n_bounces, cfg.n_p

    kp, kdir, ksteps = jax.random.split(key, 3)
    p = jax.lax.stop_gradient(_sample_p(jax.random.uniform(kp, (n,)), p_prod, cfg))
    pbin = _p_bin(p, cfg)
    Lam, abs_prob = _rates(p, A_abs, A_sc, cfg)

    pos = jnp.zeros((n, 3))
    direction = _isotropic_dirs(kdir, n)
    weight = score_weight(f_prod_density(p, p_prod, cfg))     # production knob gradient
    alive = jnp.ones(n)
    hist = jnp.zeros(cfg.n_bins)

    for sub in jax.random.split(ksteps, K):
        kc, kd, ka, kb = jax.random.split(sub, 4)
        d = _distance_to_boundary(pos, direction, R)
        reach = jnp.exp(-d * Lam)
        scatter = -jnp.expm1(-d * Lam)

        hist = hist.at[pbin].add(weight * alive * reach)             # escaped now -> its p-bin

        is_abs = jax.random.uniform(kc, (n,)) < abs_prob
        chosen = jnp.where(is_abs, abs_prob, 1.0 - abs_prob)
        w_after = weight * scatter * score_weight(chosen)

        hist = hist.at[N].add(jnp.sum(w_after * alive * is_abs.astype(weight.dtype)))
        alive = alive * jnp.where(is_abs, 0.0, 1.0)

        cos_a = jax.lax.stop_gradient(hg_sample_cos(jax.random.uniform(ka, (n,)), cfg.g_scatter))
        beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
        step = -jnp.log1p(-jax.random.uniform(kd, (n,)) * scatter) / Lam
        pos = jax.lax.stop_gradient(pos + (step * alive)[:, None] * direction)
        direction = jax.lax.stop_gradient(_deflect(direction, cos_a, beta))
        weight = w_after

    hist = hist.at[pbin].add(weight * alive)                          # residual escapes
    return hist


# --------------------------------------------------------------------------- #
# Hard reference sampler
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def sampled_histogram(params, key, cfg: ConfigBC, n=None):
    n = cfg.n_data if n is None else n
    p_prod, A_abs, A_sc = params
    R, K, N = cfg.R, cfg.n_bounces, cfg.n_p

    kp, kdir, ksteps = jax.random.split(key, 3)
    p = _sample_p(jax.random.uniform(kp, (n,)), p_prod, cfg)
    pbin = _p_bin(p, cfg)
    Lam, abs_prob = _rates(p, A_abs, A_sc, cfg)

    pos = jnp.zeros((n, 3))
    direction = _isotropic_dirs(kdir, n)
    alive = jnp.ones(n, bool)
    landed = jnp.zeros(n, jnp.int32)

    for sub in jax.random.split(ksteps, K):
        kr, kc, kd, ka, kb = jax.random.split(sub, 5)
        d = _distance_to_boundary(pos, direction, R)
        reaches = jax.random.uniform(kr, (n,)) < jnp.exp(-d * Lam)
        escaped = alive & reaches
        landed = jnp.where(escaped, pbin, landed)
        alive = alive & ~reaches

        is_abs = jax.random.uniform(kc, (n,)) < abs_prob
        absorbed = alive & is_abs
        landed = jnp.where(absorbed, N, landed)
        alive = alive & ~is_abs

        scatter = -jnp.expm1(-d * Lam)
        step = -jnp.log1p(-jax.random.uniform(kd, (n,)) * scatter) / Lam
        cos_a = hg_sample_cos(jax.random.uniform(ka, (n,)), cfg.g_scatter)
        beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
        moving = alive[:, None]
        pos = jnp.where(moving, pos + step[:, None] * direction, pos)
        direction = jnp.where(moving, _deflect(direction, cos_a, beta), direction)

    landed = jnp.where(alive, pbin, landed)
    return jnp.zeros(cfg.n_bins).at[landed].add(1.0)
