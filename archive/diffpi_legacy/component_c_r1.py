"""Component C, rung R1 (Strategy §12): 3-D sphere, competing channels, absorption.

A pion starts at the centre of a uniform-density sphere of radius `R`, moving along
+z, and propagates until it exits the boundary, where its outgoing polar angle
cos(theta) = dir.z is binned. Inside, it interacts through competing channels, each
with a mean free path `L_c` (rate 1/L_c) and -- for the scattering channels -- a
Henyey-Greenstein angular distribution with asymmetry `g_c`:

    elastic      scattering, asymmetry g0   (g0 is a *tunable shape* parameter)
    cex          scattering, asymmetry g1   (fixed)
    absorption   removes the pion           (deposited into a terminal "absorbed" bin)

This is the 3-D generalisation of `ring_scattering.py` with an added absorption
channel. It is the first rung whose differentiable estimator carries genuine
variance, so it is where the gradient-SNR study (Strategy §13) lives.

Differentiable handles (Strategy §9):
  * expected-value deposits of the escape fraction exp(-d*Lambda)        (kind 1: rate gradient)
  * score_weight(channel_prob[c])                                        (kind 2: which channel)
  * score_weight(HG_density(cos_alpha; g_c))                             (kind 3: asymmetry g0)
  * geometry (positions/directions) detached so gradients ride on weights
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .kernel import score_weight


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConfigR1:
    R: float = 3.0                                   # sphere radius [fm]
    channel_names: tuple = ("scatter", "absorption")
    true_L: tuple = (2.0, 5.0)                        # mean free path per channel [fm]
    true_g: tuple = (0.60, 0.0)                       # HG asymmetry (absorption unused)
    g_fit_index: int = 0                             # which channel's g is a fit parameter
    init_L: tuple = (3.5, 3.5)
    init_g0: float = 0.0
    abs_index: int = 1
    n_angle: int = 40                                # cos(theta) bins; bin n_angle = absorbed
    n_bounces: int = 16
    n_data: int = 300_000
    n_model: int = 150_000
    iterations: int = 400
    learning_rate: float = 0.04
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def n_channels(self):
        return len(self.channel_names)

    @property
    def n_bins(self):
        return self.n_angle + 1


# --------------------------------------------------------------------------- #
# Henyey-Greenstein angular distribution (pdf in cos_alpha over [-1, 1])
# --------------------------------------------------------------------------- #
def hg_sample_cos(u, g):
    """Inverse-CDF sample of cos(alpha) ~ HG(g). g may be a 0-d array."""
    g = jnp.asarray(g)
    safe = jnp.abs(g) > 1e-6
    s = (1 - g ** 2) / (1 - g + 2 * g * u + 1e-30)
    cos_g = (1 + g ** 2 - s ** 2) / (2 * g + 1e-30)
    return jnp.where(safe, jnp.clip(cos_g, -1.0, 1.0), 2 * u - 1.0)


def hg_density(cos_alpha, g):
    """HG pdf in cos(alpha): (1/2)(1-g^2) / (1+g^2-2 g cos)^{3/2}, integrates to 1."""
    g = jnp.asarray(g)
    return 0.5 * (1 - g ** 2) / jnp.power(1 + g ** 2 - 2 * g * cos_alpha, 1.5)


# --------------------------------------------------------------------------- #
# 3-D geometry (batched over particles)
# --------------------------------------------------------------------------- #
def _distance_to_boundary(pos, direction, R):
    proj = jnp.sum(pos * direction, axis=1)
    return -proj + jnp.sqrt(jnp.maximum(proj ** 2 + R ** 2 - jnp.sum(pos ** 2, axis=1), 0.0))


def _bin_exit(direction, n_angle):
    cos_t = direction[:, 2]
    return jnp.clip(((cos_t + 1.0) / 2.0 * n_angle).astype(jnp.int32), 0, n_angle - 1)


def _deflect(direction, cos_alpha, beta):
    """Rotate each unit `direction` by polar angle alpha (cos given) and azimuth beta."""
    sin_alpha = jnp.sqrt(jnp.maximum(1.0 - cos_alpha ** 2, 0.0))
    dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
    denom = jnp.sqrt(jnp.maximum(1.0 - dz ** 2, 1e-12))
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    # general case (away from the poles)
    nx = dx * cos_alpha + sin_alpha * (dx * dz * cb - dy * sb) / denom
    ny = dy * cos_alpha + sin_alpha * (dy * dz * cb + dx * sb) / denom
    nz = dz * cos_alpha - sin_alpha * denom * cb
    gen = jnp.stack([nx, ny, nz], axis=1)
    # pole case (direction ~ +/- z): rotate the canonical frame
    pole = jnp.stack([sin_alpha * cb, sin_alpha * sb, jnp.sign(dz) * cos_alpha], axis=1)
    new = jnp.where((jnp.abs(dz) > 1 - 1e-6)[:, None], pole, gen)
    return new / jnp.linalg.norm(new, axis=1, keepdims=True)


def _start(n):
    pos = jnp.zeros((n, 3))
    direction = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), (n, 3))
    return pos, direction


# --------------------------------------------------------------------------- #
# Differentiable estimator: expected-value deposits + score weights
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(3, 4, 5))
def weighted_histogram(L, gvec, key, cfg: ConfigR1, n=None, detach=True):
    """Differentiable expected-count histogram (n_angle exit bins + 1 absorbed bin).

    `detach=False` lets gradients flow through the random-walk geometry, which is
    used only to *demonstrate* the variance blow-up in the SNR study (Strategy §13).
    """
    n = cfg.n_model if n is None else n
    R, K, N, abs_i = cfg.R, cfg.n_bounces, cfg.n_angle, cfg.abs_index

    rate = 1.0 / L
    total = jnp.sum(rate)
    channel_prob = rate / total

    pos, direction = _start(n)
    weight = jnp.ones(n)
    alive = jnp.ones(n)                         # 1.0 alive, 0.0 removed
    hist = jnp.zeros(cfg.n_bins)

    def maybe_detach(x):
        return jax.lax.stop_gradient(x) if detach else x

    for sub in jax.random.split(key, K):
        kc, kd, ka, kb = jax.random.split(sub, 4)
        d = _distance_to_boundary(pos, direction, R)
        reach_prob = jnp.exp(-d * total)
        scatter_prob = -jnp.expm1(-d * total)

        # (kind 1) fraction that exits straight through now
        hist = hist.at[_bin_exit(direction, N)].add(weight * alive * reach_prob)

        # (kind 2) which channel does the interacting fraction take
        c = jnp.clip(jnp.searchsorted(jnp.cumsum(channel_prob),
                                      jax.random.uniform(kc, (n,))), 0, cfg.n_channels - 1)
        g_c = gvec[c]
        # (kind 3) deflection of the scattering channels. The sampled angle is a
        # *constant* (detached): the gradient w.r.t. g must come only from the
        # explicit density in the score weight, never from the inverse-CDF sampler.
        cos_alpha = jax.lax.stop_gradient(hg_sample_cos(jax.random.uniform(ka, (n,)), g_c))
        dens = hg_density(cos_alpha, g_c)

        is_abs = (c == abs_i)
        # weight after this interaction (scatter fraction, channel split, shape)
        w_after = (weight * scatter_prob
                   * score_weight(channel_prob[c])
                   * jnp.where(is_abs, 1.0, score_weight(dens)))

        # absorbed fraction is deposited and the particle is removed
        hist = hist.at[N].add(jnp.sum(w_after * alive * is_abs.astype(weight.dtype)))
        alive = alive * jnp.where(is_abs, 0.0, 1.0)

        # advance the survivors to the interaction point and deflect (geometry detached)
        step = -jnp.log1p(-jax.random.uniform(kd, (n,)) * scatter_prob) / total
        beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
        pos = maybe_detach(pos + (step * alive)[:, None] * direction)
        direction = maybe_detach(_deflect(direction, maybe_detach(cos_alpha), beta))
        weight = w_after

    # residual: survivors exit straight through
    d = _distance_to_boundary(pos, direction, R)
    hist = hist.at[_bin_exit(direction, N)].add(weight * alive)
    return hist


# --------------------------------------------------------------------------- #
# Hard reference sampler: one count per particle
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(3, 4))
def sampled_histogram(L, gvec, key, cfg: ConfigR1, n=None):
    n = cfg.n_data if n is None else n
    R, K, N, abs_i = cfg.R, cfg.n_bounces, cfg.n_angle, cfg.abs_index

    rate = 1.0 / L
    total = jnp.sum(rate)
    channel_prob = rate / total

    pos, direction = _start(n)
    alive = jnp.ones(n, bool)
    landed = jnp.zeros(n, jnp.int32)

    for sub in jax.random.split(key, K):
        kr, kc, kd, ka, kb = jax.random.split(sub, 5)
        d = _distance_to_boundary(pos, direction, R)
        reaches = jax.random.uniform(kr, (n,)) < jnp.exp(-d * total)
        exit_now = alive & reaches
        landed = jnp.where(exit_now, _bin_exit(direction, N), landed)
        alive = alive & ~reaches

        c = jnp.clip(jnp.searchsorted(jnp.cumsum(channel_prob),
                                      jax.random.uniform(kc, (n,))), 0, cfg.n_channels - 1)
        is_abs = (c == abs_i)
        absorbed_now = alive & is_abs
        landed = jnp.where(absorbed_now, N, landed)
        alive = alive & ~is_abs

        scatter_prob = -jnp.expm1(-d * total)
        step = -jnp.log1p(-jax.random.uniform(kd, (n,)) * scatter_prob) / total
        cos_alpha = hg_sample_cos(jax.random.uniform(ka, (n,)), gvec[c])
        beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
        moving = alive[:, None]
        pos = jnp.where(moving, pos + step[:, None] * direction, pos)
        direction = jnp.where(moving, _deflect(direction, cos_alpha, beta), direction)

    landed = jnp.where(alive, _bin_exit(direction, N), landed)
    return jnp.zeros(cfg.n_bins).at[landed].add(1.0)


# --------------------------------------------------------------------------- #
# Parameter packing helpers (4 params: L0, L1, L2, g0)
# --------------------------------------------------------------------------- #
def gvec_from(cfg: ConfigR1, g0):
    g = jnp.asarray(cfg.true_g)
    return g.at[cfg.g_fit_index].set(g0)
