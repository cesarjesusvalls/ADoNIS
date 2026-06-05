"""Component B, rung R0 (Strategy §12): meson-baryon vertex with real PWAs.

The first component to use the *actual* partial-wave angular distribution from the
paper (App. Meson-baryon amplitudes, Eq. for dsigma/dOmega) rather than a toy
Henyey-Greenstein shape. We model the textbook case: the Delta(1232) resonance in
elastic pi-N scattering, kept to S- and P-waves.

The complex partial-wave amplitudes are
    tau^{0,+}(W) = b                                  (S-wave background, real knob b)
    tau^{1,+}(W) = (Gamma/2) / ((m_R - W) - i Gamma/2)  (P33 resonance: m_R, Gamma)
with tau^{L,-} = 0. With only L = 0, 1 the differential cross section
    dsigma/dOmega ~ |tau0 + 2 tau1 z|^2 + (1 - z^2) |tau1|^2
              = c0 + c1 z + c2 z^2,   z = cos(theta_cm),
    c0 = |tau0|^2 + |tau1|^2,  c1 = 4 Re(tau0 tau1*),  c2 = 3 |tau1|^2,
which reduces to the classic 1 + 3 cos^2(theta) at resonance (b -> 0). The S-P
interference term c1 z is the forward-backward asymmetry that pins down b.

The angle-integrated cross section is
    sigma(W) = (4 pi / k^2) ( |tau0|^2 + 2 |tau1|^2 ),
whose line shape in W pins down (m_R, Gamma).

Observable: a 2-D histogram in (W, cos theta) of scattering events. Incoming W is
flux-uniform, so the W projection traces sigma(W) (the line shape) and the cos(theta)
projection traces the angular distribution. Knobs recovered jointly: (m_R, Gamma, b).

Differentiable handles (Strategy §9):
  * deposit weight sigma(W)/sigma_ref carries the line-shape gradient (kind 1)
  * z sampled (detached) from p(z|W); score_weight(p_density) carries the angular
    gradient (kind 3). m_R, Gamma, b enter both sigma(W) and p(z|W).
Channel selection (kind 2) is already validated in Component C R1 and is added in B-R1.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .kernel import score_weight


M_N = 0.938
M_PI = 0.138


@dataclass(frozen=True)
class ConfigBR0:
    W_min: float = 1.10
    W_max: float = 1.40
    true_mR: float = 1.232
    true_Gamma: float = 0.117
    true_b: float = 0.15
    init_mR: float = 1.275
    init_Gamma: float = 0.160
    init_b: float = 0.00
    n_W: int = 30
    n_z: int = 20
    n_grid: int = 257            # grid for inverse-CDF sampling of z
    n_data: int = 400_000
    n_model: int = 200_000
    iterations: int = 400
    learning_rate: float = 0.03
    data_seed: int = 1
    fit_seed: int = 2

    @property
    def n_bins(self):
        return self.n_W * self.n_z


# --------------------------------------------------------------------------- #
# Physics: kinematics, amplitudes, cross section, angular distribution
# --------------------------------------------------------------------------- #
def k_cm(W):
    """pi-N center-of-mass momentum at total energy W (>= threshold)."""
    lam = (W ** 2 - (M_N + M_PI) ** 2) * (W ** 2 - (M_N - M_PI) ** 2)
    return jnp.sqrt(jnp.maximum(lam, 1e-12)) / (2 * W)


def amplitudes(W, mR, Gamma, b):
    """Complex S- and P-wave partial-wave amplitudes tau0, tau1."""
    tau0 = b + 0j
    tau1 = (Gamma / 2) / ((mR - W) - 1j * (Gamma / 2))
    return tau0, tau1


def ang_coeffs(W, mR, Gamma, b):
    """c0, c1, c2 of dsigma/dOmega ~ c0 + c1 z + c2 z^2."""
    tau0, tau1 = amplitudes(W, mR, Gamma, b)
    a1 = jnp.abs(tau1) ** 2
    c0 = jnp.abs(tau0) ** 2 + a1
    c1 = 4 * jnp.real(tau0 * jnp.conj(tau1))
    c2 = 3 * a1
    return c0, c1, c2


def sigma_tot(W, mR, Gamma, b):
    tau0, tau1 = amplitudes(W, mR, Gamma, b)
    return (4 * jnp.pi / k_cm(W) ** 2) * (jnp.abs(tau0) ** 2 + 2 * jnp.abs(tau1) ** 2)


def p_density(z, c0, c1, c2):
    """Normalised angular pdf in z = cos(theta) over [-1, 1]."""
    norm = 2 * c0 + (2.0 / 3.0) * c2
    return (c0 + c1 * z + c2 * z ** 2) / norm


def _sample_z(u, c0, c1, c2, n_grid):
    """Inverse-CDF sample of z ~ p(z) via a per-event grid (z is detached anyway)."""
    zg = jnp.linspace(-1.0, 1.0, n_grid)                       # (G,)
    norm = 2 * c0 + (2.0 / 3.0) * c2                           # (n,)
    # CDF(z) on the grid for each event's coefficients -> (n, G)
    cdf = (c0[:, None] * (zg[None, :] + 1)
           + c1[:, None] * (zg[None, :] ** 2 - 1) / 2
           + c2[:, None] * (zg[None, :] ** 3 + 1) / 3) / norm[:, None]
    return jax.vmap(lambda uu, cc: jnp.interp(uu, cc, zg))(u, cdf)


def _sigma_ref(cfg):
    """A fixed scale >= max sigma over the window, so deposit weights stay in (0, 1].

    Computed eagerly in NumPy (depends only on the static config) so it is a plain
    Python float -- a constant inside the jitted estimators, not a traced value.
    """
    Wg = np.linspace(cfg.W_min, cfg.W_max, 400)
    lam = (Wg ** 2 - (M_N + M_PI) ** 2) * (Wg ** 2 - (M_N - M_PI) ** 2)
    k2 = np.maximum(lam, 1e-12) / (4 * Wg ** 2)
    a1 = (cfg.true_Gamma / 2) ** 2 / ((cfg.true_mR - Wg) ** 2 + (cfg.true_Gamma / 2) ** 2)
    s = (4 * np.pi / k2) * (cfg.true_b ** 2 + 2 * a1)
    return float(np.max(s) * 1.3)


def _bins(cfg, W, z):
    wb = jnp.clip(((W - cfg.W_min) / (cfg.W_max - cfg.W_min) * cfg.n_W).astype(jnp.int32),
                  0, cfg.n_W - 1)
    zb = jnp.clip(((z + 1) / 2 * cfg.n_z).astype(jnp.int32), 0, cfg.n_z - 1)
    return wb * cfg.n_z + zb


# --------------------------------------------------------------------------- #
# Differentiable estimator and hard reference
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(2, 3))
def weighted_histogram(params, key, cfg: ConfigBR0, n=None):
    n = cfg.n_model if n is None else n
    mR, Gamma, b = params
    kW, kz = jax.random.split(key)
    W = jax.lax.stop_gradient(jax.random.uniform(kW, (n,), minval=cfg.W_min, maxval=cfg.W_max))
    c0, c1, c2 = ang_coeffs(W, mR, Gamma, b)
    z = jax.lax.stop_gradient(_sample_z(jax.random.uniform(kz, (n,)), c0, c1, c2, cfg.n_grid))
    dens = p_density(z, c0, c1, c2)

    weight = (sigma_tot(W, mR, Gamma, b) / _sigma_ref(cfg)) * score_weight(dens)
    return jnp.zeros(cfg.n_bins).at[_bins(cfg, W, z)].add(weight)


@partial(jax.jit, static_argnums=(2, 3))
def sampled_histogram(params, key, cfg: ConfigBR0, n=None):
    n = cfg.n_data if n is None else n
    mR, Gamma, b = params
    kW, kz, ka = jax.random.split(key, 3)
    W = jax.random.uniform(kW, (n,), minval=cfg.W_min, maxval=cfg.W_max)
    c0, c1, c2 = ang_coeffs(W, mR, Gamma, b)
    accept = jax.random.uniform(ka, (n,)) < (sigma_tot(W, mR, Gamma, b) / _sigma_ref(cfg))
    z = _sample_z(jax.random.uniform(kz, (n,)), c0, c1, c2, cfg.n_grid)
    bins = jnp.where(accept, _bins(cfg, W, z), 0)
    return jnp.zeros(cfg.n_bins).at[bins].add(accept.astype(jnp.float64))
