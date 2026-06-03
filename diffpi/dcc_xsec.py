"""Differentiable DCC single-pion cross-section ASSEMBLY (Strategy §14, Phase-2 step 4).

Angle-integrated diagonal-bilinear assembly: turn the knob-scaled DCC partial-wave
amplitudes (diffpi/dcc.py) into the elementary gamma*N -> piN cross section
sigma(W, Q^2; knobs), evaluated DETERMINISTICALLY on a (W, Q^2) grid (no Monte
Carlo at the vertex -> exact, zero-variance gradients).

Physics (from amp_dcc_sl.f, the dsigma/dOmega comment block):

    dsigma/dOmega_pi = sum_{s_i,s_f,lambda} (4 pi^2 alpha / E_gamma)
                       * |zj_mu|^2 * m_N k_pi / (16 pi^3 W) / 4

Integrated over the pion solid angle, partial-wave orthogonality collapses the
helicity current to a DIAGONAL sum over partial waves:

    sigma(W,Q^2) = K * PS(W) * sum_pw (2J+1) * sum_idx |A_pw,idx(W,Q^2; knobs)|^2

with PS(W) = k_pi(W) / (E_gamma(W) * W) the transverse phase-space/flux factor,
(2J+1) the spin multiplicity of each partial wave, and K a single overall
calibration constant (the precise multipole normalization convention is the one
thing we do NOT re-derive from scratch -- we fix it by the oracle total). The
SHAPE (Delta(1232) peak, W- and Q^2-dependence) is PREDICTED from the real DCC
amplitudes; only the absolute scale K is calibrated.

This is the kind-1 / reweighting-free estimator: sigma is a smooth deterministic
function of the knobs, so autodiff gives exact gradients (cf. Component A).

For the electromagnetic (e-) oracle only the VECTOR current contributes; the
axial block (and the axial_strength knob) is exercised against a neutrino oracle
later.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .dcc import DCCAmplitudes, DCCKnobs

M_N = 938.272      # nucleon mass [MeV]
M_PI = 138.0       # representative pion mass [MeV] (pi0/pi+- average)


def two_j_plus_one(labels):
    """Spin multiplicity (2J+1) per partial wave from the spectroscopic label
    L_{2I,2J} (e.g. 'p33' -> 2J=3 -> 4)."""
    out = []
    for lab in labels:
        two_j = int(lab[2])          # third char = 2J
        out.append(two_j + 1.0)
    return jnp.asarray(out)


def pion_cm_momentum(W):
    """Pion CM momentum k_pi(W) [MeV]; 0 below the piN threshold."""
    thr_hi = (M_N + M_PI) ** 2
    thr_lo = (M_N - M_PI) ** 2
    lam = (W ** 2 - thr_hi) * (W ** 2 - thr_lo)
    lam = jnp.clip(lam, 0.0, None)
    return jnp.sqrt(lam) / (2.0 * W)


def e_gamma(W):
    """Equivalent (real-photon) energy E_gamma(W) = (W^2 - m_N^2)/(2 m_N) [MeV]."""
    return (W ** 2 - M_N ** 2) / (2.0 * M_N)


def phase_space(W):
    """Transverse phase-space/flux factor PS(W) = k_pi / (E_gamma * W)."""
    return pion_cm_momentum(W) / (e_gamma(W) * W)


class DCCCrossSection:
    """Differentiable elementary single-pion cross section sigma(W, Q^2; knobs)."""

    def __init__(self, amps: DCCAmplitudes | None = None, calibration: float = 1.0):
        self.D = amps or DCCAmplitudes()
        self.twoJ1 = two_j_plus_one(self.D.labels)     # (n_pw,)
        self.K = calibration

    def sigma(self, W, Q2, knobs: DCCKnobs = DCCKnobs(), current: str = "vec"):
        """Elementary cross section at scalar (W [MeV], Q2 [MeV^2]).

        current: 'vec' (electromagnetic, default), 'axial', or 'all' (vec+axial,
        incoherent -- a placeholder for the weak case pending a neutrino oracle).
        """
        vec, isv, axial = self.D.amplitudes(W, Q2, knobs)     # each (n_idx, n_pw)
        if current == "vec":
            amp2 = jnp.sum(jnp.abs(vec) ** 2, axis=0)          # sum idx -> (n_pw,)
        elif current == "axial":
            amp2 = jnp.sum(jnp.abs(axial) ** 2, axis=0)
        else:
            amp2 = (jnp.sum(jnp.abs(vec) ** 2, axis=0)
                    + jnp.sum(jnp.abs(axial) ** 2, axis=0))
        diag = jnp.sum(self.twoJ1 * amp2)                      # diagonal PW sum
        return self.K * phase_space(W) * diag

    def sigma_per_pw(self, W, Q2, knobs: DCCKnobs = DCCKnobs(), current: str = "vec"):
        """Per-partial-wave contribution (2J+1)|A|^2 * PS, for decomposition plots.
        Returns (n_pw,)."""
        vec, isv, axial = self.D.amplitudes(W, Q2, knobs)
        amp2 = jnp.sum(jnp.abs(vec if current == "vec" else axial) ** 2, axis=0)
        return self.K * phase_space(W) * self.twoJ1 * amp2

    # --- deterministic projections by quadrature ----------------------------- #
    def dsigma_dW(self, W_grid, Q2_grid, knobs=DCCKnobs(), current="vec"):
        """dsigma/dW [arb] on W_grid, integrating sigma(W,Q2) over Q2_grid
        (trapezoid). Q2_grid in MeV^2."""
        def at_W(W):
            sig = jax.vmap(lambda q: self.sigma(W, q, knobs, current))(Q2_grid)
            return jnp.trapezoid(sig, Q2_grid)
        return jax.vmap(at_W)(W_grid)

    def dsigma_dQ2(self, W_grid, Q2_grid, knobs=DCCKnobs(), current="vec"):
        """dsigma/dQ2 [arb] on Q2_grid, integrating over W_grid (trapezoid)."""
        def at_Q2(Q2):
            sig = jax.vmap(lambda w: self.sigma(w, Q2, knobs, current))(W_grid)
            return jnp.trapezoid(sig, W_grid)
        return jax.vmap(at_Q2)(Q2_grid)
