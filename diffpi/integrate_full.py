"""Full integrated chain A -> (C + B + D) -> G (Strategy §15 milestone 5, full form).

One differentiable simulation that chains every piece and recovers the production
AND final-state-interaction parameters jointly from one observable -- the toy
version of the whole ACHILLES single-pion-production + cascade problem.

Stage 1 -- hard vertex (Component A): nu + N -> l + (Delta -> N pi). Sample
  (Q^2, W) and carry the matrix-element weight R = F_A(Q^2; M_A)^2 * BW(W; m_D, G_D)
  (reweighting). The pion's initial momentum is the two-body decay momentum
  p*(W); its direction is isotropic. Q^2 is a lepton-side label, unchanged by FSI.

Stage 2 -- cascade (Components C/B/D): the pion propagates through a uniform
  sphere. Per step it escapes, scatters (Component B-style: deflect AND lose a
  fixed momentum fraction -- in-medium pi N scattering degrades the pion energy),
  or is absorbed (Component D-style: removed, a CC0pi event). Rates set by the
  cross-section knobs sigma_scatter, sigma_abs.

Stage 3 -- observable (G): a 2-D histogram (Q^2, final pion |p|) for escaped pions
  plus an "absorbed" (CC0pi) bin per Q^2.

Recovered jointly: M_A (axial mass, from Q^2 -- FSI-independent), m_Delta &
Gamma_Delta (resonance, from the pion-momentum peak), sigma_scatter (low-momentum
tail from energy loss) and sigma_abs (the CC0pi fraction). All gradient flows
through one global weight: reweighting for the vertex knobs (kind 1), escape/
scatter deposits for the rates (kind 1), and a channel score weight for absorb-vs-
scatter (kind 2); the geometry and the degraded momentum are detached.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .kernel import score_weight
from .component_c_r1 import _distance_to_boundary, _deflect, hg_sample_cos
from .integrate_bc import _isotropic_dirs
from .component_a_r0 import axial_ff, breit_wigner

M_N = 0.938
M_PI = 0.138


@dataclass(frozen=True)
class ConfigFull:
    # vertex
    Q2_max: float = 1.5
    W_min: float = 1.10
    W_max: float = 1.50
    # geometry / cascade
    R: float = 3.0
    n_bounces: int = 16
    g_scatter: float = 0.4
    mom_loss: float = 0.20           # fractional pion momentum lost per scatter
    p_pi_max: float = 0.55           # momentum-bin range [GeV/c]
    # knobs: truth / init
    true_MA: float = 1.00
    true_mDelta: float = 1.232
    true_GammaDelta: float = 0.117
    true_sigma_scatter: float = 0.30  # [1/fm]
    true_sigma_abs: float = 0.20      # [1/fm]
    init_MA: float = 1.25
    init_mDelta: float = 1.27
    init_GammaDelta: float = 0.15
    init_sigma_scatter: float = 0.15
    init_sigma_abs: float = 0.35
    n_Q2: int = 16
    n_ppi: int = 18
    n_data: int = 500_000
    n_model: int = 150_000
    iterations: int = 250
    learning_rate: float = 0.05
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def n_out(self):
        return self.n_ppi + 1        # pion-momentum bins + absorbed (CC0pi)

    @property
    def n_bins(self):
        return self.n_Q2 * self.n_out


def p_star(W):
    """Two-body Delta -> N pi decay momentum at invariant mass W."""
    lam = (W ** 2 - (M_N + M_PI) ** 2) * (W ** 2 - (M_N - M_PI) ** 2)
    return jnp.sqrt(jnp.maximum(lam, 1e-12)) / (2 * W)


def rate_vertex(Q2, W, MA, mD, GD):
    return axial_ff(Q2, MA) ** 2 * breit_wigner(W, mD, GD)


def _rate_ref():
    return 1.2


def _Q2_bin(cfg, Q2):
    return jnp.clip((Q2 / cfg.Q2_max * cfg.n_Q2).astype(jnp.int32), 0, cfg.n_Q2 - 1)


def _ppi_bin(cfg, p):
    return jnp.clip((p / cfg.p_pi_max * cfg.n_ppi).astype(jnp.int32), 0, cfg.n_ppi - 1)


# --------------------------------------------------------------------------- #
# Differentiable estimator
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def weighted_histogram(params, key, cfg: ConfigFull, n=None):
    n = cfg.n_model if n is None else n
    MA, mD, GD, sig_sc, sig_abs = params
    R, K, no = cfg.R, cfg.n_bounces, cfg.n_out

    kq, kw, kd, ksteps = jax.random.split(key, 4)
    Q2 = jax.lax.stop_gradient(jax.random.uniform(kq, (n,)) * cfg.Q2_max)
    W = jax.lax.stop_gradient(jax.random.uniform(kw, (n,), minval=cfg.W_min, maxval=cfg.W_max))
    Qb = _Q2_bin(cfg, Q2)
    p_pi = jax.lax.stop_gradient(p_star(W))

    pos = jnp.zeros((n, 3))
    direction = _isotropic_dirs(kd, n)
    weight = rate_vertex(Q2, W, MA, mD, GD) / _rate_ref()       # vertex reweighting (kind 1)
    alive = jnp.ones(n)
    hist = jnp.zeros(cfg.n_bins)

    lam_tot = sig_sc + sig_abs
    abs_prob = sig_abs / lam_tot

    for sub in jax.random.split(ksteps, K):
        kc, ks, kb = jax.random.split(sub, 3)
        d = _distance_to_boundary(pos, direction, R)
        reach = jnp.exp(-d * lam_tot)
        scatter = -jnp.expm1(-d * lam_tot)

        hist = hist.at[Qb * no + _ppi_bin(cfg, p_pi)].add(weight * alive * reach)   # escaped now

        is_abs = jax.random.uniform(kc, (n,)) < abs_prob
        chosen = jnp.where(is_abs, abs_prob, 1.0 - abs_prob)
        w_after = weight * scatter * score_weight(chosen)                          # rate + channel grad

        hist = hist.at[Qb * no + cfg.n_ppi].add(w_after * alive * is_abs.astype(weight.dtype))  # CC0pi
        alive = alive * jnp.where(is_abs, 0.0, 1.0)

        cos_a = jax.lax.stop_gradient(hg_sample_cos(jax.random.uniform(ks, (n,)), cfg.g_scatter))
        beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
        direction = jax.lax.stop_gradient(_deflect(direction, cos_a, beta))
        p_pi = jax.lax.stop_gradient(p_pi * (1.0 - cfg.mom_loss))                  # inelastic energy loss
        # straight-line advance to a representative interaction point
        pos = jax.lax.stop_gradient(pos + (0.5 * d * alive)[:, None] * direction)
        weight = w_after

    hist = hist.at[Qb * no + _ppi_bin(cfg, p_pi)].add(weight * alive)              # residual escapes
    return hist


# --------------------------------------------------------------------------- #
# Hard reference sampler
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def sampled_histogram(params, key, cfg: ConfigFull, n=None):
    n = cfg.n_data if n is None else n
    MA, mD, GD, sig_sc, sig_abs = params
    R, K, no = cfg.R, cfg.n_bounces, cfg.n_out

    kq, kw, kacc, kd, ksteps = jax.random.split(key, 5)
    Q2 = jax.random.uniform(kq, (n,)) * cfg.Q2_max
    W = jax.random.uniform(kw, (n,), minval=cfg.W_min, maxval=cfg.W_max)
    Qb = _Q2_bin(cfg, Q2)
    accept = jax.random.uniform(kacc, (n,)) < (rate_vertex(Q2, W, MA, mD, GD) / _rate_ref())
    p_pi = p_star(W)

    pos = jnp.zeros((n, 3))
    direction = _isotropic_dirs(kd, n)
    alive = accept
    absorbed = jnp.zeros(n, bool)
    lam_tot = sig_sc + sig_abs
    abs_prob = sig_abs / lam_tot

    for sub in jax.random.split(ksteps, K):
        kr, kc, ks, kb = jax.random.split(sub, 4)
        d = _distance_to_boundary(pos, direction, R)
        reaches = jax.random.uniform(kr, (n,)) < jnp.exp(-d * lam_tot)
        alive = alive & ~reaches                                 # escaped: stop, keep p_pi
        interacts = alive
        is_abs = interacts & (jax.random.uniform(kc, (n,)) < abs_prob)
        absorbed = absorbed | is_abs
        alive = alive & ~is_abs
        cos_a = hg_sample_cos(jax.random.uniform(ks, (n,)), cfg.g_scatter)
        beta = jax.random.uniform(kb, (n,)) * 2 * jnp.pi
        moving = alive[:, None]
        direction = jnp.where(moving, _deflect(direction, cos_a, beta), direction)
        p_pi = jnp.where(alive, p_pi * (1.0 - cfg.mom_loss), p_pi)
        pos = jnp.where(moving, pos + (0.5 * d)[:, None] * direction, pos)

    out_bin = jnp.where(absorbed, cfg.n_ppi, _ppi_bin(cfg, p_pi))
    bins = jnp.where(accept, Qb * no + out_bin, 0)
    return jnp.zeros(cfg.n_bins).at[bins].add(accept.astype(jnp.float64))
